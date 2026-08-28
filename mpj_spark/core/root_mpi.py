# ================================================================
# mpj_spark/core/root_mpi.py  -  MPI Root Coordinator  (rank 0)
# MPJ-SPARK Multi-Driver Architecture  (mpi4py + OpenMPI)
# University of Jaffna  -  2022/E/033 & 2022/E/090
# ================================================================

import math
import threading
import time
from datetime import UTC

from mpi4py import MPI

from mpj_spark.core.sync_modes import (
    MODE_HYBRID_PS_ALLREDUCE,
    MODE_PS_ASYNC,
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    normalize_sync_mode,
)

TAG_CONFIG = 10
TAG_RESULT = 20
TAG_TIMING = 21
TAG_ALLREDUCE_UP = 30
TAG_ALLREDUCE_DOWN = 31
TAG_REASSIGN_BCAST = 40
TAG_REASSIGN_STATS = 41
TAG_READY = 50
TAG_GO = 60

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


def run_logreg_allreduce_mpi(
    comm,
    num_workers: int,
    num_iterations: int,
    num_features: int,
) -> dict:
    """Root-side MPI coordinator for per-iteration FedAvg weight synchronisation (Point-to-Point legacy)."""
    import numpy as np

    print(
        f"  [LogReg Allreduce MPI] Starting — {num_workers} workers × {num_iterations} iterations"
    )

    final_weights = None
    final_intercept = 0.0

    for iteration in range(num_iterations):
        t_iter = time.perf_counter()
        msgs = []
        for w_rank in range(1, num_workers + 1):
            msg = comm.recv(source=w_rank, tag=TAG_ALLREDUCE_UP)
            msgs.append(msg)

        total_rows = sum(m["row_count"] for m in msgs)
        avg_w = np.zeros(num_features)
        avg_intercept = 0.0
        for m in msgs:
            frac = m["row_count"] / total_rows if total_rows > 0 else (1.0 / len(msgs))
            avg_w += frac * np.array(m["weights"])
            avg_intercept += frac * m["intercept"]

        final_weights = avg_w.tolist()
        final_intercept = float(avg_intercept)

        payload = {
            "type": "avg_weights",
            "iteration": iteration,
            "weights": final_weights,
            "intercept": final_intercept,
        }
        for w_rank in range(1, num_workers + 1):
            comm.send(payload, dest=w_rank, tag=TAG_ALLREDUCE_DOWN)

        iter_time = time.perf_counter() - t_iter
        weight_norm = float(np.linalg.norm(avg_w))
        print(
            f"  [LogReg Allreduce MPI] iter {iteration+1}/{num_iterations}  ({iter_time:.3f}s)  |w|={weight_norm:.4f}"
        )

    print("  [LogReg Allreduce MPI] Complete")
    return {
        "weight_vector": final_weights,
        "intercept": final_intercept,
        "iterations_done": num_iterations,
    }


def run_root_mpi(
    comm,
    input_file,
    num_workers=None,
    compare=False,
    prewarm=True,
    cores_override=None,
    app="wordcount",
    sync_mode=MODE_PS_SYNC_FEDAVG_MPI,
    kmeans_k=3,
    kmeans_iter=20,
    baseline_threads=None,
    use_gossip=False,
    gossip_threshold=1e-3,
    gossip_max_rounds=10,
    gossip_fanout=2,
    use_global_seed=True,
    use_reassign=True,
    logreg_iter=10,
    logreg_reg_param=0.01,
    logreg_features=10,
    results_dir="results",
):
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert rank == 0, f"run_root_mpi() called on rank {rank}"
    assert size >= 2, f"Need at least 2 MPI ranks, got size={size}"

    if num_workers is None:
        num_workers = size - 1

    sync_mode = normalize_sync_mode(sync_mode)

    do_seed = use_global_seed and use_gossip and app == "kmeans"
    do_reassign = use_reassign and use_gossip and app == "kmeans"
    do_logreg_allreduce_p2p = app == "logreg" and sync_mode == MODE_PS_SYNC_FEDAVG_QUEUE
    do_logreg_async_ps = app == "logreg" and sync_mode == MODE_PS_ASYNC
    do_logreg_hybrid = app == "logreg" and sync_mode == MODE_HYBRID_PS_ALLREDUCE

    from mpj_spark.config import DATA_DIR, TOTAL_CORES
    from mpj_spark.core.key_value import KeyValueStructure
    from mpj_spark.core.root_process import (
        _print_comparison,
        _print_timing_summary,
        aggregate_kmeans_results,
        aggregate_logreg_results,
        compute_global_seed_centres,
        dynamic_partition,
    )
    from mpj_spark.utils.dev_logger import DevLogger

    logger = DevLogger(worker_id="root-mpi")

    agg_mode = (
        f"Adaptive Gossip (threshold={gossip_threshold}, max_rounds={gossip_max_rounds})"
        if (use_gossip and app == "kmeans")
        else "Batch Hungarian"
        if app == "kmeans"
        else f"Native MPI FedAvg ({logreg_iter} iters)"
        if (app == "logreg" and sync_mode == MODE_PS_SYNC_FEDAVG_MPI)
        else f"Native MPI FedAvg ({logreg_iter} iters)"
        if (app == "logreg" and sync_mode == MODE_PS_SYNC_FEDAVG_MPI)
        else f"Hybrid PS+Allreduce ({logreg_iter} iters)"
        if (app == "logreg" and sync_mode == MODE_HYBRID_PS_ALLREDUCE)
        else f"Async Parameter Server ({logreg_iter} rounds, FedAsync)"
        if (app == "logreg" and sync_mode == MODE_PS_ASYNC)
        else f"Allreduce FedAvg MPI ({logreg_iter} iters)"
        if app == "logreg"
        else "N/A"
    )

    title_extra = ""
    if app == "logreg":
        title_extra = (
            f"  iter={logreg_iter}  reg_param={logreg_reg_param}  features={logreg_features}\n"
            f"  Aggregation : {agg_mode}  [sync_mode={sync_mode}]"
        )
    elif app == "kmeans":
        cf = f"  Global Seed    : {'ON' if do_seed else 'OFF'}\n  Re-assign Pass : {'ON' if do_reassign else 'OFF'}"
        cf = f"  Global Seed    : {'ON' if do_seed else 'OFF'}\n  Re-assign Pass : {'ON' if do_reassign else 'OFF'}"
        title_extra = f"  k={kmeans_k}  max_iter={kmeans_iter}\n  Aggregation : {agg_mode}\n{cf}"

    _hdr(
        f"MPJ-Spark Multi-Driver [MPI]  |  app={app}  |  workers={num_workers}\n"
        f"  MPI_COMM_WORLD size={size}  root=rank-0  workers=ranks-1..{size-1}\n"
        f"{title_extra}"
    )

    cores = (
        max(1, cores_override) if cores_override else max(1, math.ceil(TOTAL_CORES / num_workers))
    )
    print(f"  Core budget : local[{cores}]  ({TOTAL_CORES} total ÷ {num_workers} workers)")

    seed_centres = None
    seed_time = None
    if do_seed:
        _phase("0", "Computing global seed centroids (isolated subprocess)")
        t_seed = time.perf_counter()
        seed_centres = compute_global_seed_centres(
            input_file=input_file,
            k=kmeans_k,
            total_cores=TOTAL_CORES,
            sample_fraction=0.05,
            seed=42,
        )
        seed_time = time.perf_counter() - t_seed
        _ok(f"Global seed centroids ready ({seed_time:.3f}s)")

    _phase(1, "Partitioning dataset")
    t_load_start = time.perf_counter()
    partition_paths = dynamic_partition(input_file, num_workers, DATA_DIR)
    load_time = time.perf_counter() - t_load_start
    _ok(f"Split into {num_workers} partitions ({load_time:.3f}s)")

    _phase(2, f"Distributing config to {num_workers} worker ranks")
    worker_cfg_base = {
        "app": app,
        "cores_override": cores,
        "kmeans_k": kmeans_k,
        "kmeans_max_iter": kmeans_iter,
        "num_workers": num_workers,
        "seed_centres": seed_centres,
        "sync_mode": sync_mode,
        "logreg_iter": logreg_iter,
        "logreg_reg_param": logreg_reg_param,
        "logreg_features": logreg_features,
        "results_dir": results_dir,
    }

    for i in range(num_workers):
        w_rank = i + 1
        cfg = {**worker_cfg_base, "partition_path": partition_paths[i], "worker_id": i}
        comm.send(cfg, dest=w_rank, tag=TAG_CONFIG)
        _info(f"Config sent to rank {w_rank} (worker {i} | {partition_paths[i]})")
        _info(f"Config sent to rank {w_rank} (worker {i} | {partition_paths[i]})")

    _phase("3a", "Waiting for JVM-ready signals from all workers")
    for i in range(num_workers):
        w_rank = i + 1
        comm.recv(source=w_rank, tag=TAG_READY)
        _ok(f"Worker rank {w_rank} (worker {i}) JVM ready")

    _phase("3b", "Firing all workers simultaneously")
    t_proc_start = time.perf_counter()
    for i in range(num_workers):
        w_rank = i + 1
        comm.send(True, dest=w_rank, tag=TAG_GO)
    _ok(f"Go signal sent to {num_workers} workers")

    # ── Match the workers' comm.Split(color=1) collective ─────────────
    # worker_mpi.py calls comm.Split(color=1, key=rank) immediately after
    # the go-signal to build the worker-only sub-communicator used by the
    # K-Means / LogReg MPI collectives.  MPI_Comm_split is COLLECTIVE over
    # COMM_WORLD: every rank must call it.  Root joins no worker subgroup
    # (color=MPI.UNDEFINED -> returns COMM_NULL here).  Without this call
    # workers block forever inside Split right after the go-signal.
    comm.Split(color=MPI.UNDEFINED, key=rank)

    allreduce_result = None
    allreduce_thread = None
    _allreduce_store = []

    if do_logreg_allreduce_p2p:

        def _allreduce_thread_fn():
            res = run_logreg_allreduce_mpi(
                comm=comm,
                num_workers=num_workers,
                num_iterations=logreg_iter,
                num_features=logreg_features,
            )
            _allreduce_store.append(res)

        allreduce_thread = threading.Thread(
            target=_allreduce_thread_fn, daemon=True, name="logreg-allreduce-mpi"
        )
        allreduce_thread.start()
        print("  [LogReg Allreduce MPI] Coordinator thread started (P2P Queue-fallback)")

    if do_logreg_async_ps:
        from mpj_spark.core.async_ps import run_logreg_async_ps

        def _async_ps_thread_fn():
            res = run_logreg_async_ps(
                comm,  # COMM_WORLD on root — P2P with worker ranks 1..N
                num_workers=num_workers,
                num_iterations=logreg_iter,
                num_features=logreg_features,
                results_dir=results_dir,
            )
            _allreduce_store.append(res)  # same result shape as run_logreg_allreduce_mpi()

        allreduce_thread = threading.Thread(
            target=_async_ps_thread_fn, daemon=True, name="logreg-async-ps"
        )
        allreduce_thread.start()
        print("  [LogReg Async PS] Coordinator thread started (P3-09, non-blocking P2P)")

    if do_logreg_hybrid:
        from mpj_spark.core.hybrid_ps import run_logreg_hybrid_scalar_ps

        def _hybrid_ps_thread_fn():
            res = run_logreg_hybrid_scalar_ps(
                comm,  # COMM_WORLD on root — scalar P2P with worker ranks 1..N
                num_workers=num_workers,
                num_iterations=logreg_iter,
                results_dir=results_dir,
            )
            _allreduce_store.append(res)  # intercept-only result; weights via Allreduce

        allreduce_thread = threading.Thread(
            target=_hybrid_ps_thread_fn, daemon=True, name="logreg-hybrid-ps"
        )
        allreduce_thread.start()
        print("  [LogReg Hybrid PS] Scalar coordinator thread started (P3-10)")

    _phase(4, "Collecting results from worker ranks")
    worker_results = []
    worker_timings = []
    errors = []

    for i in range(num_workers):
        w_rank = i + 1
        res = comm.recv(source=w_rank, tag=TAG_RESULT)
        timing = comm.recv(source=w_rank, tag=TAG_TIMING)
        if res.get("status") == "success":
            worker_results.append(res.get("result"))
        else:
            errors.append(res)
            print(f"  ✗  Worker rank {w_rank} (worker {i}) FAILED: {res.get('error')}")
        worker_timings.append(timing)

    proc_time = time.perf_counter() - t_proc_start

    if errors:
        print(f"  {len(errors)} worker(s) failed. Aborting.")
        return

    _ok(f"All {num_workers} workers completed")

    if allreduce_thread is not None:
        allreduce_thread.join(timeout=60)
        if _allreduce_store:
            allreduce_result = _allreduce_store[0]

    _phase(5, "Aggregating results")
    t_agg_start = time.perf_counter()
    gossip_info = None
    agg = None

    if app == "wordcount":
        kv = KeyValueStructure()
        for r in worker_results:
            kv.merge(KeyValueStructure.from_serializable(r))
        top = kv.get_top_n(20)
        print("\n  Top-20 words:")
        for word, count in top:
            print(f"    {word:<22} {count:>12,}")

    elif app == "kmeans":
        if use_gossip:
            from mpj_spark.core.gossip_aggregator import GossipAggregator
            from mpj_spark_mpi import MpiRootFanoutQueue

            gossip_q = MpiRootFanoutQueue(tag=TAG_ALLREDUCE_UP, num_workers=num_workers)
            gagg = GossipAggregator(
                num_workers=num_workers,
                convergence_threshold=gossip_threshold,
                max_rounds=gossip_max_rounds,
                initial_fanout=gossip_fanout,
                verbose=True,
            )
            agg = gagg.aggregate(
                gossip_q,
                timeout_per_worker=120.0,
                seed_centres=seed_centres,
            )
            gossip_info = agg
            _ok(f"Gossip done rounds={agg['rounds_run']} converged={agg['converged']}")
        else:
            agg = aggregate_kmeans_results(worker_results)

        _info(f"Total rows : {agg['total_rows']:,}")
        _info(f"Total WCSS : {agg['total_wcss']:.4f}")

    elif app == "logreg":
        from datetime import datetime

        _run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        agg = aggregate_logreg_results(
            worker_results,
            allreduce_result=allreduce_result,
            results_dir=results_dir,
            run_id=_run_id,
            num_workers=num_workers,
            reg_param=logreg_reg_param,
            num_features=logreg_features,
            sync_mode=sync_mode,
        )
        gossip_info = {"iterations_done": agg.get("iterations_done", logreg_iter)}
        _info(f"Total rows       : {agg['total_rows']:,}")
        _info(f"Weighted accuracy: {agg['avg_accuracy']:.4f}")
        _info(f"Agg mode         : {agg['agg_mode']} [sync_mode={sync_mode}]")

    agg_time = time.perf_counter() - t_agg_start
    reassign_time = None

    if do_reassign and agg is not None and app == "kmeans":
        _phase("5b", "Re-assignment pass — exact global centroid correction")
        t_reassign = time.perf_counter()
        gossip_centres = agg["centres"]
        k_val = len(gossip_centres)
        d_val = len(gossip_centres[0])

        for i in range(num_workers):
            w_rank = i + 1
            comm.send(
                {"type": "reassign", "centres": gossip_centres}, dest=w_rank, tag=TAG_REASSIGN_BCAST
            )

        import numpy as np

        all_sums = np.zeros((k_val, d_val))
        all_counts = np.zeros(k_val, dtype=np.int64)
        total_rows = 0
        for i in range(num_workers):
            w_rank = i + 1
            msg = comm.recv(source=w_rank, tag=TAG_REASSIGN_STATS)
            for j in range(k_val):
                all_sums[j] += np.array(msg["cluster_sums"][j])
                all_counts[j] += msg["cluster_counts"][j]
            total_rows += msg["row_count"]

        corrected = [
            (all_sums[j] / all_counts[j]).tolist() if all_counts[j] > 0 else gossip_centres[j]
            for j in range(k_val)
        ]
        reassign_time = time.perf_counter() - t_reassign
        agg["centres"] = corrected
        _ok(f"Re-assignment done ({reassign_time:.3f}s) from {total_rows:,} rows")

    t_wall = load_time + proc_time + agg_time + (reassign_time or 0.0)
    avg_proc = sum(t["processing_time"] for t in worker_timings) / num_workers
    avg_init = sum(t["init_time"] for t in worker_timings) / num_workers if prewarm else None

    _print_timing_summary(
        load_time,
        avg_proc,
        agg_time,
        t_wall,
        prewarm_init=avg_init,
        gossip_info=gossip_info,
        seed_time=seed_time,
        reassign_time=reassign_time,
    )

    logger.log_run(
        app=app,
        num_workers=num_workers,
        cores=cores,
        load_time=load_time,
        proc_time=avg_proc,
        agg_time=agg_time,
        total_time=t_wall,
    )

    if compare:
        logreg_parity_iter = num_workers * logreg_iter if app == "logreg" else None
        _phase("B", f"Running {app} baseline for comparison")

        if app == "wordcount":
            from mpj_spark.applications.baseline_spark import run_baseline

            _, baseline_timing = run_baseline(input_file, num_workers, cores_override)
        elif app == "kmeans":
            from mpj_spark.applications.baseline_kmeans import run_baseline_kmeans

            _, baseline_timing = run_baseline_kmeans(
                input_file,
                num_workers,
                cores_override,
                kmeans_k,
                kmeans_iter,
                baseline_threads=baseline_threads,
            )
        elif app == "logreg":
            from mpj_spark.applications.baseline_logreg import run_baseline_logreg

            _, baseline_timing = run_baseline_logreg(
                input_file,
                num_workers,
                cores_override,
                logreg_iter,
                logreg_reg_param,
                logreg_features,
                baseline_threads=baseline_threads,
                parity_iter=logreg_parity_iter,
            )

        multi_timing = {
            "load_time": load_time,
            "processing_time": avg_proc,
            "reassign_time": reassign_time,
            "total_time": t_wall,
        }
        _print_comparison(
            multi_timing,
            baseline_timing,
            num_workers,
            app,
            baseline_threads=baseline_threads,
            parity_iter=logreg_parity_iter,
        )
