# ================================================================
# mpj_spark/core/root_process.py
#
# Root process coordinator for all benchmark models:
#
#   B1  Single-driver Spark  local[N]              (baseline_logreg.py)
#   B2  Single-driver Spark  spark://master:7077   (baseline_logreg.py)
#   M1  Multi-driver, NO sync    sync_mode='none'  (nosync_run.py)
#   M2  Multi-driver, FedAvg     sync_mode='queue' (queue_run.py)
#   M3  Multi-driver, MPI Allreduce                (allreduce.py / main.py)
#
# run_root() dispatches M1 vs M2 via sync_mode parameter.
# B1/B2 are invoked when compare=True (inside run_root phase B).
# M3 is dispatched directly from main.py via _run_mpi_logreg().
# ================================================================
from __future__ import annotations

import math
import os
import threading
import time
from datetime import datetime, timezone
from multiprocessing import Process, Queue

from mpj_spark.core.file_manager import MPJSparkFileManager
from mpj_spark.core.sync_modes import (
    MODE_NONE,
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    normalize_sync_mode,
)
from mpj_spark.utils.dev_logger import DevLogger
from mpj_spark.workers.worker_process import worker_process

SEP = "=" * 70
DASH = "─" * 70


def _hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _phase(n, label):
    print(f"\n── Phase {n}: {label}")


def _ok(msg):
    print(f"  ✓  {msg}")


def _info(msg):
    print(f"     {msg}")


# ──────────────────────────────────────────────────────────────────
# Isolated seeding subprocess (K-Means global seed centroids)
# ──────────────────────────────────────────────────────────────────


def _seeding_worker(result_q, input_file, k, total_cores, sample_fraction, seed):
    try:
        from pyspark.ml.clustering import KMeans
        from pyspark.ml.feature import VectorAssembler
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("MPJ-Global-Seeding")
            .master(f"local[{total_cores}]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", str(total_cores * 2))
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

        df_raw = spark.read.csv(input_file, inferSchema=True, header=False)
        df_samp = df_raw.sample(fraction=sample_fraction, seed=seed).dropna()
        assembler = VectorAssembler(
            inputCols=df_raw.columns, outputCol="features", handleInvalid="skip"
        )
        df_vec = assembler.transform(df_samp).select("features")
        model = KMeans(
            k=k, maxIter=20, seed=seed, featuresCol="features", initMode="k-means||"
        ).fit(df_vec)
        centres = [c.tolist() for c in model.clusterCenters]
        spark.stop()
        result_q.put({"status": "ok", "centres": centres})
    except Exception as exc:
        result_q.put({"status": "error", "msg": str(exc)})


def compute_global_seed_centres(input_file, k, total_cores, sample_fraction=0.05, seed=42):
    print(
        f"  [Seeding] Sampling {sample_fraction * 100:.0f}% of dataset "
        f"for global seed centroids (isolated subprocess) ..."
    )
    t0 = time.perf_counter()
    result_q = Queue()
    p = Process(
        target=_seeding_worker,
        args=(result_q, input_file, k, total_cores, sample_fraction, seed),
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"[Seeding] subprocess exited with code {p.exitcode}.")
    result = result_q.get_nowait()
    if result["status"] != "ok":
        raise RuntimeError(f"[Seeding] seeding worker failed: {result['msg']}")
    centres = result["centres"]
    elapsed = time.perf_counter() - t0
    print(f"  [Seeding] {k} seed centroids computed in {elapsed:.3f}s")
    for i, c in enumerate(centres):
        preview = ", ".join(f"{v:.3f}" for v in c[:4])
        print(f"  [Seeding] Seed C{i}: [{preview}...]")
    return centres


# ──────────────────────────────────────────────────────────────────
# Partition helper
# ──────────────────────────────────────────────────────────────────


def dynamic_partition(input_path, num_partitions, output_dir):
    fm = MPJSparkFileManager(output_dir)
    return fm.dynamic_partition(input_path, num_partitions)


# ──────────────────────────────────────────────────────────────────
# K-Means helpers
# ──────────────────────────────────────────────────────────────────


def align_centres_hungarian(reference, candidate):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    cost_matrix = np.array(
        [[np.linalg.norm(np.array(r) - np.array(c)) for c in candidate] for r in reference]
    )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [candidate[i] for i in col_ind.tolist()], col_ind.tolist()


def resolve_worker_count(num_workers: int | None) -> int:
    """Return a safe positive worker count for orchestration paths."""
    if num_workers is None:
        return 1
    return max(1, int(num_workers))


def aggregate_kmeans_results(worker_results):
    import numpy as np

    total_rows = sum(r["row_count"] for r in worker_results)
    k = len(worker_results[0]["centres"])
    num_dims = len(worker_results[0]["centres"][0])
    reference_centres = worker_results[0]["centres"]
    aligned_results = [worker_results[0]]
    for w_idx, r in enumerate(worker_results[1:], start=1):
        aligned_centres, perm = align_centres_hungarian(reference_centres, r["centres"])
        _info(f"Worker {w_idx} centroid alignment (Hungarian): {perm}")
        aligned_results.append({**r, "centres": aligned_centres})
    merged = []
    for c_idx in range(k):
        ws = np.zeros(num_dims)
        for r in aligned_results:
            ws += (r["row_count"] / total_rows) * np.array(r["centres"][c_idx])
        merged.append(ws.tolist())
    total_wcss = sum(r["wcss"] for r in worker_results)
    print("\n  K-Means aggregation complete")
    _info(f"Total rows : {total_rows:,}")
    _info(f"Total WCSS : {total_wcss:.4f}")
    print("  Global centres:")
    for i, c in enumerate(merged):
        preview = ", ".join(f"{v:.3f}" for v in c[:4])
        _info(f"  C{i}: [{preview}{'...' if len(c) > 4 else ''}]")
    return {
        "centres": merged,
        "total_wcss": total_wcss,
        "total_rows": total_rows,
        "num_workers": len(worker_results),
    }


# ──────────────────────────────────────────────────────────────────
# LogReg Allreduce coordinator (M2 — Queue/FedAvg, root side)
# ──────────────────────────────────────────────────────────────────


def run_logreg_allreduce(up_queue, down_queue, num_workers, num_iterations, num_features):
    """
    Root-side FedAvg coordinator for M2 (Queue/FedAvg sync).
    TWO-QUEUE DESIGN — eliminates single-queue livelock.
    """
    import numpy as np

    print(
        f"  [LogReg Allreduce] Starting — {num_workers} workers × {num_iterations} iterations"
    )
    final_weights = None
    final_intercept = 0.0
    for iteration in range(num_iterations):
        t_iter = time.perf_counter()
        msgs = []
        while len(msgs) < num_workers:
            msg = up_queue.get(timeout=300)
            msgs.append(msg)
        total_rows = sum(m["row_count"] for m in msgs)
        avg_w = np.zeros(num_features)
        avg_intercept = 0.0
        for m in msgs:
            frac = m["row_count"] / total_rows
            avg_w += frac * np.array(m["weights"])
            avg_intercept += frac * m["intercept"]
        final_weights = avg_w.tolist()
        final_intercept = float(avg_intercept)
        for _ in range(num_workers):
            down_queue.put(
                {
                    "type": "avg_weights",
                    "iteration": iteration,
                    "weights": final_weights,
                    "intercept": final_intercept,
                }
            )
        iter_time = time.perf_counter() - t_iter
        weight_norm = float(np.linalg.norm(avg_w))
        print(
            f"  [LogReg Allreduce] iter {iteration + 1}/{num_iterations}  "
            f"({iter_time:.3f}s)  |w|={weight_norm:.4f}"
        )
    print("  [LogReg Allreduce] Complete")
    return {
        "weight_vector": final_weights,
        "intercept": final_intercept,
        "iterations_done": num_iterations,
    }


# ──────────────────────────────────────────────────────────────────
# LogReg metrics CSV writer
# ──────────────────────────────────────────────────────────────────


def _write_merged_iter_metrics(
    worker_results,
    results_dir,
    run_id,
    num_workers,
    reg_param,
    num_features,
    sync_mode=MODE_PS_SYNC_FEDAVG_MPI,
):
    import csv as _csv

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "logreg_iter_metrics.csv")
    file_exists = os.path.isfile(out_path)
    fieldnames = [
        "run_id",
        "num_workers",
        "sync_mode",
        "reg_param",
        "num_features",
        "worker_id",
        "iteration",
        "iter_time_s",
        "weight_norm",
        "weight_delta",
        "local_weight_norm",
        "intercept",
        "row_count",
    ]
    rows_written = 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in worker_results:
            for rec in r.get("iter_metrics", []):
                row = {
                    "run_id": run_id,
                    "num_workers": num_workers,
                    "sync_mode": rec.get("sync_mode", sync_mode),
                    "reg_param": reg_param,
                    "num_features": num_features,
                    "local_weight_norm": rec.get("local_weight_norm", rec.get("weight_norm", "")),
                }
                row.update({k: v for k, v in rec.items() if k not in ("local_weight_norm", "sync_mode")})
                writer.writerow(row)
                rows_written += 1
    return out_path, rows_written


def aggregate_logreg_results(
    worker_results,
    allreduce_result=None,
    results_dir="results",
    run_id=None,
    num_workers=None,
    reg_param=None,
    num_features=None,
    sync_mode=MODE_PS_SYNC_FEDAVG_QUEUE,
):
    """
    Aggregate LogReg worker results.

    M2 (sync_mode='queue' or 'ps_sync_fedavg_queue'): allreduce_result holds FedAvg model.
    P3-08 (sync_mode='ps_sync_fedavg_mpi'): workers already aggregated via MPI gather/bcast.
    M1 (sync_mode='none'): post-hoc row-weighted average of worker models.
    """
    import numpy as np

    sync_mode = normalize_sync_mode(sync_mode)

    total_rows = sum(r["row_count"] for r in worker_results)
    avg_accuracy = sum(r["train_accuracy"] * (r["row_count"] / total_rows) for r in worker_results)

    if allreduce_result is not None:
        final_weights = allreduce_result["weight_vector"]
        final_intercept = allreduce_result["intercept"]
        agg_mode = "Allreduce — FedAvg per-iteration (M2)"
    elif sync_mode == MODE_PS_SYNC_FEDAVG_MPI:
        final_weights = worker_results[0]["weight_vector"]
        final_intercept = worker_results[0].get("intercept", 0.0)
        agg_mode = "Native MPI Collectives FedAvg (P3-08)"
    else:
        num_feat = len(worker_results[0]["weight_vector"])
        avg_w = np.zeros(num_feat)
        avg_intercept = 0.0
        for r in worker_results:
            frac = r["row_count"] / total_rows
            avg_w += frac * np.array(r["weight_vector"])
            avg_intercept += frac * r["intercept"]
        final_weights = avg_w.tolist()
        final_intercept = float(avg_intercept)
        agg_mode = "Post-hoc row-weighted average — no sync (M1)"

    weight_norm = float(sum(w**2 for w in final_weights) ** 0.5)

    print("\n  Logistic Regression aggregation complete")
    _info(f"Total rows       : {total_rows:,}")
    _info(f"Weighted accuracy: {avg_accuracy:.4f}")
    _info(f"Final |w|        : {weight_norm:.4f}")
    _info(f"Final intercept  : {final_intercept:.4f}")
    w_preview = ", ".join(f"{v:.4f}" for v in final_weights[:5])
    _info(f"Weight preview   : [{w_preview}{'...' if len(final_weights) > 5 else ''}]")

    _run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _nw = num_workers or len(worker_results)
    _rp = reg_param or 0.0
    _nf = num_features or len(final_weights)
    csv_path, n_written = _write_merged_iter_metrics(
        worker_results,
        results_dir,
        _run_id,
        _nw,
        _rp,
        _nf,
        sync_mode=sync_mode,
    )
    _info(f"Iter metrics CSV : {csv_path} ({n_written} rows appended)")

    return {
        "weight_vector": final_weights,
        "intercept": final_intercept,
        "avg_accuracy": avg_accuracy,
        "total_rows": total_rows,
        "num_workers": _nw,
        "weight_norm": weight_norm,
        "agg_mode": agg_mode,
        "sync_mode": sync_mode,
    }


# ──────────────────────────────────────────────────────────────────
# Reassignment pass (K-Means gossip correction)
# ──────────────────────────────────────────────────────────────────


def reassign_pass_root(processes_alive, gossip_centres, reassign_queue, num_workers, k, dims):
    import numpy as np

    print(f"  [Reassign] Broadcasting gossip-final centroids to {num_workers} workers ...")
    for _ in range(num_workers):
        reassign_queue.put({"type": "reassign", "centres": gossip_centres})
    all_sums = np.zeros((k, dims))
    all_counts = np.zeros(k, dtype=np.int64)
    total_rows = 0
    received = 0
    deadline = time.monotonic() + 180.0
    while received < num_workers:
        try:
            msg = reassign_queue.get_nowait()
        except Exception as err:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"[Reassign] timed out waiting for worker stats "
                    f"({received}/{num_workers} received)"
                ) from err
            time.sleep(0.01)
            continue
        if msg.get("type") == "stats":
            for j in range(k):
                all_sums[j] += np.array(msg["cluster_sums"][j])
                all_counts[j] += msg["cluster_counts"][j]
            total_rows += msg["row_count"]
            received += 1
        elif msg.get("type") == "reassign":
            reassign_queue.put(msg)
            time.sleep(0.001)
    corrected = []
    for j in range(k):
        if all_counts[j] > 0:
            corrected.append((all_sums[j] / all_counts[j]).tolist())
        else:
            corrected.append(gossip_centres[j])
    print(f"  [Reassign] Recomputed {k} exact global centroids from {total_rows:,} rows")
    for i, c in enumerate(corrected):
        preview = ", ".join(f"{v:.3f}" for v in c[:4])
        _info(f"  C{i}: [{preview}{'...' if dims > 4 else ''}]")
    return corrected


def _print_comparison(
    multi_timing,
    baseline_timing,
    num_workers,
    app,
    baseline_threads=None,
    parity_iter=None,
    model_label=None,
):
    note_parts = []
    if baseline_threads:
        note_parts.append(f"baseline-threads={baseline_threads}")
    if parity_iter is not None:
        note_parts.append(f"baseline-iter={parity_iter} (parity)")
    if model_label:
        note_parts.append(model_label)
    note = f"  [{'  |  '.join(note_parts)}]" if note_parts else ""
    bmode = baseline_timing.get("mode", "local[N]")

    print(f"\n{SEP}")
    print(f"  Multi-Driver vs Baseline  |  app={app}  |  workers={num_workers}{note}")
    print(f"  Baseline mode: {bmode}")
    print(SEP)
    print(f"  {'Metric':<28} {'Multi-Driver':>13} {'Baseline':>13} {'Speedup':>9}")
    print(f"  {'-' * 28} {'-' * 13} {'-' * 13} {'-' * 9}")
    rows = [
        ("load_time", "Load Time (s)"),
        ("processing_time", "Proc Time — fit only (s)"),
    ]
    reassign_val = multi_timing.get("reassign_time")
    if reassign_val is not None:
        rows.append((\"_reassign\", \"Re-assign Pass (s)\"))
    rows.append((\"total_time\", \"Total Time (s)\"))
    for key, label in rows:
        if key == \"_reassign\":
            m = reassign_val
            b = 0.0
        else:
            m = multi_timing.get(key, 0.0)
            b = baseline_timing.get(key, 0.0)
        sp = b / m if m > 0 else 0.0
        flag = \"  ⚡\" if sp >= 1.5 else (\"  ⚠\" if sp < 1.0 else \"\")
        print(f\"  {label:<28} {m:>12.4f}s {b:>12.4f}s {sp:>8.2f}x{flag}\")
    print(SEP)


def _print_timing_summary(
    load_time,
    avg_proc,
    agg_time,
    t_wall,
    prewarm_init=None,
    gossip_info=None,
    seed_time=None,
    reassign_time=None,
    model_label=None,
):
    print(f\"\\n{DASH}\")
    label = \"Timing Summary\" + (f\"  [{model_label}]\" if model_label else \"\")
    print(f\"  {label}\")
    print(DASH)
    if seed_time is not None:
        print(f\"  {'Global Seed Sampling':<28} {seed_time:>8.4f} s\")
    print(f\"  {'Partition Load Time':<28} {load_time:>8.4f} s\")
    if prewarm_init is not None:
        print(f\"  {'Pre-warm JVM Init (avg)':<28} {prewarm_init:>8.4f} s\")
    print(f\"  {'Processing Time (avg fit)':<28} {avg_proc:>8.4f} s\")
    if reassign_time is not None:
        print(f\"  {'Re-assignment Pass':<28} {reassign_time:>8.4f} s\")
    print(f\"  {'Aggregation Time':<28} {agg_time:>8.4f} s\")
    if gossip_info:
        if \"rounds_run\" in gossip_info:
            print(f\"  {'Gossip Rounds':<28} {gossip_info['rounds_run']:>8d}\")
            print(f\"  {'Gossip Converged':<28} {str(gossip_info['converged']):>8}\")
        elif \"iterations_done\" in gossip_info:
            print(f\"  {'Allreduce Rounds':<28} {gossip_info['iterations_done']:>8d}\")
    print(f\"  {'Total Wall-clock Time':<28} {t_wall:>8.4f} s\")
    print(DASH)


def run_root(
    input_file,
    num_workers=2,
    compare=False,
    prewarm=True,
    cores_override=None,
    app=\"wordcount\",
    kmeans_k=3,
    kmeans_iter=20,
    baseline_threads=None,
    baseline_master=None,
    use_gossip=False,
    gossip_threshold=1e-3,
    gossip_max_rounds=10,
    gossip_fanout=2,
    use_global_seed=True,
    use_reassign=True,
    logreg_iter=10,
    logreg_reg_param=0.01,
    logreg_features=10,
    results_dir=\"results\",
    sync_mode=MODE_PS_SYNC_FEDAVG_QUEUE,
    **_extra_kwargs,
):
    from mpj_spark.config import DATA_DIR, TOTAL_CORES

    logger = DevLogger(worker_id=\"root\")

    if _extra_kwargs:
        ignored = \", \".join(_extra_kwargs.keys())
        print(f\"  [run_root] Extra kwargs ignored: {ignored}\")

    sync_mode = normalize_sync_mode(sync_mode)

    do_seed = use_global_seed and use_gossip and app == \"kmeans\"
    do_reassign = use_reassign and use_gossip and app == \"kmeans\"
    do_logreg_allreduce = app == \"logreg\" and sync_mode == MODE_PS_SYNC_FEDAVG_QUEUE
    do_logreg_nosync = app == \"logreg\" and sync_mode == MODE_NONE

    if app == \"logreg\":
        if sync_mode == MODE_NONE:
            model_label = \"M1 — Multi-driver, NO sync\"
        elif sync_mode == MODE_PS_SYNC_FEDAVG_MPI:
            model_label = \"P3-08 — Multi-driver, Native MPI FedAvg\"
        else:
            model_label = \"M2 — Multi-driver, Queue/FedAvg\"
    else:
        model_label = None

    agg_mode = (\n        f\"Adaptive Gossip (threshold={gossip_threshold}, max_rounds={gossip_max_rounds})\"\n        if (use_gossip and app == \"kmeans\")\n        else \"Batch Hungarian\"\n        if app == \"kmeans\"\n        else f\"Queue FedAvg, {logreg_iter} iters [M2]\"\n        if do_logreg_allreduce\n        else \"Post-hoc row-weighted average [M1]\"\n        if do_logreg_nosync\n        else \"N/A\"\n    )\n\n    if app == \"logreg\":\n        title_extra = (\n            f\"  iter={logreg_iter}  reg_param={logreg_reg_param}  features={logreg_features}\\n\"\n            f\"  Aggregation : {agg_mode}\"\n        )\n    elif app == \"kmeans\":\n        correctness_flags = \"\"\n        if use_gossip:\n            correctness_flags = (\n                f\"  Global Seed     : {'ON' if do_seed else 'OFF'}\\n\"\n                f\"  Re-assign Pass  : {'ON' if do_reassign else 'OFF'}\"\n            )\n        title_extra = f\"  k={kmeans_k}  max_iter={kmeans_iter}\\n  Aggregation : {agg_mode}\\n{correctness_flags}\"\n    else:\n        title_extra = \"\"\n\n    num_workers = resolve_worker_count(num_workers)\\n    _hdr(f\"MPJ-Spark Multi-Driver  |  app={app}  |  workers={num_workers}\\n{title_extra}\")\n\n    cores = max(1, cores_override) if cores_override else max(1, math.ceil(TOTAL_CORES / max(1, num_workers)))\n    print(f\"  Core budget : local[{cores}]  ({TOTAL_CORES} total ÷ {num_workers} workers)\")\n    if model_label:\n        print(f\"  Benchmark   : {model_label}\")\n\n    seed_centres = None\n    seed_time = None\n    if do_seed:\n        _phase(\"1b\", \"Computing global seed centroids (isolated subprocess)\")\n        t_seed = time.perf_counter()\n        seed_centres = compute_global_seed_centres(\n            input_file=input_file,\n            k=kmeans_k,\n            total_cores=TOTAL_CORES,\n            sample_fraction=0.05,\n            seed=42,\n        )\n        seed_time = time.perf_counter() - t_seed\n        _ok(f\"Global seed centroids ready ({seed_time:.3f}s)\")\n\n    worker_cfg = {\n        \"app\": app,\n        \"cores_override\": cores,\n        \"kmeans_k\": kmeans_k,\n        \"kmeans_max_iter\": kmeans_iter,\n        \"num_workers\": num_workers,\n        \"seed_centres\": seed_centres,\n        \"logreg_iter\": logreg_iter,\n        \"logreg_reg_param\": logreg_reg_param,\n        \"logreg_features\": logreg_features,\n        \"results_dir\": results_dir,\n        \"sync_mode\": sync_mode,\n    }\n\n    _phase(1, \"Partitioning dataset\")\n    t_load_start = time.perf_counter()\n    partition_paths = dynamic_partition(input_file, num_workers, DATA_DIR)\n    load_time = time.perf_counter() - t_load_start\n    _ok(f\"Split into {num_workers} partitions ({load_time:.3f}s)\")\n\n    _phase(2, f\"Spawning {num_workers} worker processes\")\n    processes = []\n    result_queue = Queue()\n    timing_queue = Queue()\n    from multiprocessing import Event\n\n    ready_signals = [Event() for _ in range(num_workers)]\n    go_signals = [Event() for _ in range(num_workers)]\n\n    gossip_queue = Queue() if (use_gossip and app == \"kmeans\") else None\n    reassign_queue = Queue() if do_reassign else None\n    allreduce_up_queue = Queue() if do_logreg_allreduce else None\n    allreduce_down_queue = Queue() if do_logreg_allreduce else None\n\n    for i in range(num_workers):\n        p = Process(\n            target=worker_process,\n            args=(\n                i,\n                partition_paths[i],\n                result_queue,\n                go_signals[i],\n                ready_signals[i],\n                timing_queue,\n                worker_cfg,\n            ),\n            kwargs={\n                \"allreduce_up_queue\": gossip_queue or allreduce_up_queue,\n                \"reassign_queue\": reassign_queue,\n                \"allreduce_down_queue\": allreduce_down_queue,\n            },\n        )\n        p.start()\n        processes.append(p)\n        _info(f\"Worker {i} (PID {p.pid}) spawned for {partition_paths[i]}\")\n\n    _phase(\"3a\", \"Waiting for JVM-ready signals from all workers\")\n    for sig in ready_signals:\n        sig.wait()\n    _ok(f\"All {num_workers} workers initialized Spark and are ready\")\n\n    _phase(\"3b\", \"Firing all workers simultaneously\")\n    t_proc_start = time.perf_counter()\n    for sig in go_signals:\n        sig.set()\n\n    allreduce_result = None\n    allreduce_thread = None\n\n    if do_logreg_allreduce:\n        _allreduce_store = []\n\n        def _allreduce_thread_fn():\n            res = run_logreg_allreduce(\n                up_queue=allreduce_up_queue,\n                down_queue=allreduce_down_queue,\n                num_workers=num_workers,\n                num_iterations=logreg_iter,\n                num_features=logreg_features,\n            )\n            _allreduce_store.append(res)\n\n        allreduce_thread = threading.Thread(target=_allreduce_thread_fn, daemon=True)\n        allreduce_thread.start()\n        print(\"  [M2] LogReg FedAvg coordinator started in background thread\")\n    elif do_logreg_nosync:\n        print(\"  [M1] Workers training independently — no sync coordinator\")\n\n    _phase(4, \"Collecting results\")\n    worker_results = []\n    worker_timings = []\n    errors = []\n\n    for _ in range(num_workers):\n        res = result_queue.get()\n        timing = timing_queue.get()\n        if res.get(\"status\") == \"success\":\n            worker_results.append(res[\"result\"])\n        else:\n            errors.append(res)\n            print(f\"  ✗  Worker {res.get('worker_id')} FAILED: {res.get('error')}\")\n        worker_timings.append(timing)\n\n    for p in processes:\n        p.join()\n\n    proc_time = time.perf_counter() - t_proc_start\n    if errors:\n        print(f\"  {len(errors)} worker(s) failed. Aborting.\")\n        return\n\n    _ok(f\"All {num_workers} workers completed\")\n\n    if allreduce_thread is not None:\n        allreduce_thread.join(timeout=60)\n        if _allreduce_store:\n            allreduce_result = _allreduce_store[0]\n\n    _phase(5, \"Aggregating results\")\n    t_agg_start = time.perf_counter()\n    gossip_info = None\n    agg = None\n\n    if app == \"wordcount\":\n        from mpj_spark.core.key_value import KeyValueStructure\n\n        kv = KeyValueStructure()\n        for r in worker_results:\n            kv.merge(KeyValueStructure.from_serializable(r))\n        top = kv.get_top_n(20)\n        print(\"\\n  Top-20 words:\")\n        for word, count in top:\n            print(f\"    {word:<22} {count:>12,}\")\n\n    elif app == \"kmeans\":\n        if use_gossip:\n            from mpj_spark.core.gossip_aggregator import GossipAggregator\n\n            gagg = GossipAggregator(\n                num_workers=num_workers,\n                convergence_threshold=gossip_threshold,\n                max_rounds=gossip_max_rounds,\n                initial_fanout=gossip_fanout,\n                verbose=True,\n            )\n            agg = gagg.aggregate(\n                gossip_queue,\n                timeout_per_worker=120.0,\n                seed_centres=seed_centres,\n            )\n            gossip_info = agg\n            _ok(f\"Gossip done  rounds={agg['rounds_run']}  converged={agg['converged']}\")\n        else:\n            agg = aggregate_kmeans_results(worker_results)\n        _info(f\"Total rows : {agg['total_rows']:,}\")\n        _info(f\"Total WCSS : {agg['total_wcss']:.4f}\")\n\n    elif app == \"logreg\":\n        _run_id = datetime.now(timezone.utc).strftime(\"%Y%m%dT%H%M%SZ\")\n        agg = aggregate_logreg_results(\n            worker_results,\n            allreduce_result=allreduce_result,\n            results_dir=results_dir,\n            run_id=_run_id,\n            num_workers=num_workers,\n            reg_param=logreg_reg_param,\n            num_features=logreg_features,\n            sync_mode=sync_mode,\n        )\n        gossip_info = {"iterations_done": logreg_iter}\n        _info(f\"Total rows       : {agg['total_rows']:,}\")\n        _info(f\"Weighted accuracy: {agg['avg_accuracy']:.4f}\")\n        _info(f\"Agg mode         : {agg['agg_mode']}\")\n\n    agg_time = time.perf_counter() - t_agg_start\n    reassign_time = None\n\n    if do_reassign and agg is not None and app == \"kmeans\":\n        _phase(\"5b\", \"Re-assignment pass — exact global centroid correction\")\n        t_reassign = time.perf_counter()\n        gossip_centres = agg[\"centres\"]\n        k_val = len(gossip_centres)\n        d_val = len(gossip_centres[0])\n        corrected_centres = reassign_pass_root(\n            processes_alive=processes,\n            gossip_centres=gossip_centres,\n            reassign_queue=reassign_queue,\n            num_workers=num_workers,\n            k=k_val,\n            dims=d_val,\n        )\n        reassign_time = time.perf_counter() - t_reassign\n        agg[\"centres\"] = corrected_centres\n        _ok(f\"Re-assignment done ({reassign_time:.3f}s)\")\n\n    t_wall = load_time + proc_time + agg_time + (reassign_time or 0.0)\n    avg_proc = sum(t[\"processing_time\"] for t in worker_timings) / num_workers\n    avg_init = sum(t[\"init_time\"] for t in worker_timings) / num_workers if prewarm else None\n\n    _print_timing_summary(\n        load_time,\n        avg_proc,\n        agg_time,\n        t_wall,\n        prewarm_init=avg_init,\n        gossip_info=gossip_info,\n        seed_time=seed_time,\n        reassign_time=reassign_time,\n        model_label=model_label,\n    )\n\n    logger.log_run(\n        app=app,\n        num_workers=num_workers,\n        cores=cores,\n        load_time=load_time,\n        proc_time=avg_proc,\n        agg_time=agg_time,\n        total_time=t_wall,\n    )\n\n    if compare:\n        logreg_parity_iter = num_workers * logreg_iter if app == \"logreg\" else None\n        b_label = \"B2 — Standalone cluster\" if baseline_master else \"B1 — local[N]\"\n        _phase(\n            \"B\",\n            f\"Running {app} baseline ({b_label}) for comparison\"\n            + (\n                f\"  [parity maxIter={logreg_parity_iter} = {num_workers} workers × {logreg_iter} iters]\"\n                if logreg_parity_iter is not None\n                else \"\"\n            ),\n        )\n\n        if app == \"wordcount\":\n            from mpj_spark.applications.baseline_spark import run_baseline\n            _, baseline_timing = run_baseline(input_file, num_workers, cores_override)\n        elif app == \"kmeans\":\n            from mpj_spark.applications.baseline_kmeans import run_baseline_kmeans\n            _, baseline_timing = run_baseline_kmeans(\n                input_file,\n                num_workers,\n                cores_override,\n                kmeans_k,\n                kmeans_iter,\n                baseline_threads=baseline_threads,\n            )\n        elif app == \"logreg\":\n            from mpj_spark.applications.baseline_logreg import run_baseline_logreg\n            _, baseline_timing = run_baseline_logreg(\n                input_file,\n                num_workers,\n                cores_override,\n                logreg_iter,\n                logreg_reg_param,\n                logreg_features,\n                baseline_threads=baseline_threads,\n                parity_iter=logreg_parity_iter,\n                baseline_master=baseline_master,\n            )\n\n        multi_timing = {\n            \"load_time\": load_time,\n            \"processing_time\": avg_proc,\n            \"reassign_time\": reassign_time,\n            \"total_time\": t_wall,\n        }\n        _print_comparison(\n            multi_timing,\n            baseline_timing,\n            num_workers,\n            app,\n            baseline_threads=baseline_threads,\n            parity_iter=logreg_parity_iter,\n            model_label=model_label,\n        )\n