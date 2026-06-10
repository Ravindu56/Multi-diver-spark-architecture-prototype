# ================================================================
# mpj_spark/core/root_mpi.py  -  MPI Root Coordinator  (rank 0)
# MPJ-SPARK Multi-Driver Architecture  (mpi4py + OpenMPI)
# University of Jaffna  -  2022/E/033 & 2022/E/090
#
# PURPOSE
# -------
# MPI-native root coordinator.  Replaces every multiprocessing.Process,
# Queue, and Event used by run_root() in root_process.py with direct
# MPI point-to-point calls over MPI_COMM_WORLD.
#
# P3-02 ACCEPTANCE CRITERION
# ---------------------------
#   MPI_COMM_WORLD replaces multiprocessing.Process; root is rank 0.
#
#   Verified by:
#     - assert comm.Get_rank() == 0  at entry
#     - grep multiprocessing mpj_spark/core/root_mpi.py  =>  no output
#
# MPI TAG ALLOCATION  (consistent with mpj_spark_mpi.py)
# -------------------------------------------------------
#   TAG_CONFIG          = 10   root -> worker  (partition path + cfg)
#   TAG_RESULT          = 20   worker -> root  (app result dict)
#   TAG_TIMING          = 21   worker -> root  (timing dict)
#   TAG_ALLREDUCE_UP    = 30   worker -> root  (logreg weights, gossip)
#   TAG_ALLREDUCE_DOWN  = 31   root -> worker  (averaged weights)
#   TAG_REASSIGN_BCAST  = 40   root -> worker  (gossip centroids)
#   TAG_REASSIGN_STATS  = 41   worker -> root  (cluster sums/counts)
#   TAG_READY           = 50   worker -> root  (JVM-ready sentinel)
#   TAG_GO              = 60   root -> worker  (go-signal sentinel)
# ================================================================

import math
import threading
import time
from datetime import UTC

# ── MPI tag constants (mirrors mpj_spark_mpi.py tag allocation) ──
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


# ──────────────────────────────────────────────────────────────────
# Internal print helpers  (consistent with root_process.py)
# ──────────────────────────────────────────────────────────────────


def _hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _phase(n, label):
    print(f"\n── Phase {n}: {label}")


def _ok(msg):
    print(f"  ✓  {msg}")


def _info(msg):
    print(f"     {msg}")


# ──────────────────────────────────────────────────────────────────
# Phase 2 → Phase 3 LogReg Allreduce coordinator (MPI point-to-point)
# ──────────────────────────────────────────────────────────────────
# TODO P3-04: Replace this function body with comm.Allreduce(
#     local_weights, global_weights, op=MPI.SUM) per iteration.
#     All ranks (including rank 0) must participate simultaneously.
#     Consider splitting the communicator so rank 0 joins as a
#     compute peer, or use a dedicated MPI.Intracomm for the
#     Allreduce collective.


def run_logreg_allreduce_mpi(
    comm,
    num_workers: int,
    num_iterations: int,
    num_features: int,
) -> dict:
    """
    Root-side MPI coordinator for per-iteration FedAvg weight synchronisation.

    Replaces the two-Queue / background-thread design from root_process.py
    with direct MPI point-to-point:
      - Collect weight vectors from all workers via TAG_ALLREDUCE_UP.
      - Compute FedAvg (weighted mean by partition size).
      - Broadcast averaged weights back via TAG_ALLREDUCE_DOWN.

    The message protocol per iteration is identical to the Queue version
    so worker_process.py requires no changes for this step.
    """
    import numpy as np

    print(
        f"  [LogReg Allreduce MPI] Starting — "
        f"{num_workers} workers × {num_iterations} iterations"
    )

    final_weights = None
    final_intercept = 0.0

    for iteration in range(num_iterations):
        t_iter = time.perf_counter()

        # ── 1. Collect weight vectors from every worker ───────────
        msgs = []
        for w_rank in range(1, num_workers + 1):
            msg = comm.recv(source=w_rank, tag=TAG_ALLREDUCE_UP)
            msgs.append(msg)

        # ── 2. FedAvg: weighted mean proportional to partition size ──
        total_rows = sum(m["row_count"] for m in msgs)
        avg_w = np.zeros(num_features)
        avg_intercept = 0.0
        for m in msgs:
            frac = m["row_count"] / total_rows
            avg_w += frac * np.array(m["weights"])
            avg_intercept += frac * m["intercept"]

        final_weights = avg_w.tolist()
        final_intercept = float(avg_intercept)

        # ── 3. Broadcast averaged weights back to every worker ────
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
            f"  [LogReg Allreduce MPI] iter {iteration + 1}/{num_iterations}  "
            f"({iter_time:.3f}s)  |w|={weight_norm:.4f}"
        )

    print("  [LogReg Allreduce MPI] Complete")
    return {
        "weight_vector": final_weights,
        "intercept": final_intercept,
        "iterations_done": num_iterations,
    }


# ──────────────────────────────────────────────────────────────────
# Main MPI root entry point
# ──────────────────────────────────────────────────────────────────


def run_root_mpi(
    comm,  # MPI communicator (MPI_COMM_WORLD)
    input_file,
    num_workers=None,  # defaults to comm.Get_size() - 1
    compare=False,
    prewarm=True,
    cores_override=None,
    app="wordcount",
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
    """
    MPI-native root coordinator.  Must only be called by rank 0.

    Replaces the multiprocessing.Process / Queue / Event scaffolding
    in run_root() (root_process.py) with MPI point-to-point primitives.
    All aggregation helpers are reused unchanged from root_process.py.

    P3-02 acceptance criterion:
      MPI_COMM_WORLD replaces multiprocessing.Process; root is rank 0.
    """
    # ── Rank / size guards ────────────────────────────────────────────
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert rank == 0, f"run_root_mpi() must only be called by rank 0 (called on rank {rank})."
    assert size >= 2, (
        f"Need at least 2 MPI ranks (1 root + 1 worker); got size={size}. "
        "Increase -np in your mpirun command."
    )

    if num_workers is None:
        num_workers = size - 1
    assert num_workers == size - 1, (
        f"num_workers={num_workers} must equal MPI size - 1 = {size - 1}. "
        "Adjust -np in your mpirun command to change parallelism."
    )

    # ── Derived flags (identical to run_root logic) ───────────────────
    do_seed = use_global_seed and use_gossip and app == "kmeans"
    do_reassign = use_reassign and use_gossip and app == "kmeans"
    do_logreg_allreduce = app == "logreg"

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

    # ── Header ────────────────────────────────────────────────────────
    agg_mode = (
        f"Adaptive Gossip  (threshold={gossip_threshold}, "
        f"max_rounds={gossip_max_rounds}, fanout={gossip_fanout})"
        if (use_gossip and app == "kmeans")
        else "Batch Hungarian"
        if app == "kmeans"
        else f"Allreduce FedAvg MPI ({logreg_iter} iters)"
        if app == "logreg"
        else "N/A"
    )
    title_extra = ""
    if app == "logreg":
        title_extra = (
            f"  iter={logreg_iter}  reg_param={logreg_reg_param}  "
            f"features={logreg_features}\n  Aggregation : {agg_mode}"
        )
    elif app == "kmeans":
        cf = (
            f"  Global Seed    : {'ON' if do_seed else 'OFF'}\n"
            f"  Re-assign Pass : {'ON' if do_reassign else 'OFF'}"
        )
        title_extra = f"  k={kmeans_k}  max_iter={kmeans_iter}\n  Aggregation : {agg_mode}\n{cf}"

    _hdr(
        f"MPJ-Spark Multi-Driver [MPI]  |  app={app}  |  workers={num_workers}\n"
        f"  MPI_COMM_WORLD size={size}  root=rank-0  "
        f"workers=ranks-1..{size - 1}\n"
        f"{title_extra}"
    )

    cores = (
        max(1, cores_override) if cores_override else max(1, math.ceil(TOTAL_CORES / num_workers))
    )
    print(f"  Core budget : local[{cores}]  ({TOTAL_CORES} total ÷ {num_workers} workers)")

    # ═══════════════════════════════════════════════════════════════
    # Phase 0  (optional): Global seed centroids for K-Means
    # ═══════════════════════════════════════════════════════════════
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
        _ok(f"Global seed centroids ready  ({seed_time:.3f}s)")

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Partition dataset (root only)
    # ═══════════════════════════════════════════════════════════════
    _phase(1, "Partitioning dataset")
    t_load_start = time.perf_counter()
    partition_paths = dynamic_partition(input_file, num_workers, DATA_DIR)
    load_time = time.perf_counter() - t_load_start
    _ok(f"Split into {num_workers} partitions  ({load_time:.3f}s)")

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Distribute config to worker ranks via MPI send
    #          (replaces multiprocessing.Process(args=(...)))
    # ═══════════════════════════════════════════════════════════════
    _phase(2, f"Distributing config to {num_workers} worker ranks")

    worker_cfg_base = {
        "app": app,
        "cores_override": cores,
        "kmeans_k": kmeans_k,
        "kmeans_max_iter": kmeans_iter,
        "num_workers": num_workers,
        "seed_centres": seed_centres,
        "logreg_iter": logreg_iter,
        "logreg_reg_param": logreg_reg_param,
        "logreg_features": logreg_features,
        "results_dir": results_dir,
    }

    for i in range(num_workers):
        w_rank = i + 1  # MPI rank of this worker (1-indexed)
        cfg = {
            **worker_cfg_base,
            "partition_path": partition_paths[i],
            "worker_id": i,
        }
        comm.send(cfg, dest=w_rank, tag=TAG_CONFIG)
        _info(f"Config sent to rank {w_rank}  (worker {i}  |  {partition_paths[i]})")

    # ═══════════════════════════════════════════════════════════════
    # Phase 3a: JVM-ready barrier
    #           Replaces:  for sig in ready_signals: sig.wait()
    # ═══════════════════════════════════════════════════════════════
    _phase("3a", "Waiting for JVM-ready signals from all workers")
    for i in range(num_workers):
        w_rank = i + 1
        comm.recv(source=w_rank, tag=TAG_READY)
        _ok(f"Worker rank {w_rank} (worker {i}) JVM ready")

    # ═══════════════════════════════════════════════════════════════
    # Phase 3b: Fire all workers simultaneously
    #           Replaces:  for sig in go_signals: sig.set()
    # ═══════════════════════════════════════════════════════════════
    _phase("3b", "Firing all workers simultaneously")
    t_proc_start = time.perf_counter()
    for i in range(num_workers):
        w_rank = i + 1
        comm.send(True, dest=w_rank, tag=TAG_GO)
    _ok(f"Go signal sent to {num_workers} workers")

    # ═══════════════════════════════════════════════════════════════
    # Phase 3c: LogReg Allreduce coordinator (MPI point-to-point)
    #           Replaces:  threading.Thread(target=_allreduce_thread_fn)
    # TODO P3-04: Upgrade to comm.Allreduce collective.
    # ═══════════════════════════════════════════════════════════════
    allreduce_result = None
    allreduce_thread = None
    _allreduce_store = []

    if do_logreg_allreduce:

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
        print("  [LogReg Allreduce MPI] Coordinator thread started")

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Collect results and timings from each worker rank
    #          Replaces:  result_queue.get() / timing_queue.get() loops
    # ═══════════════════════════════════════════════════════════════
    _phase(4, "Collecting results from worker ranks")
    worker_results = []
    worker_timings = []
    errors = []

    for i in range(num_workers):
        w_rank = i + 1
        res = comm.recv(source=w_rank, tag=TAG_RESULT)
        timing = comm.recv(source=w_rank, tag=TAG_TIMING)
        if res["status"] == "success":
            worker_results.append(res["result"])
        else:
            errors.append(res)
            print(f"  ✗  Worker rank {w_rank} (worker {i}) FAILED: {res.get('error')}")
        worker_timings.append(timing)

    proc_time = time.perf_counter() - t_proc_start

    if errors:
        print(f"  {len(errors)} worker(s) failed. Aborting.")
        return

    _ok(f"All {num_workers} workers completed")

    # Join Allreduce coordinator thread before aggregation
    if allreduce_thread is not None:
        allreduce_thread.join(timeout=60)
        if _allreduce_store:
            allreduce_result = _allreduce_store[0]

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Aggregate results
    #          Aggregation helpers reused unchanged from root_process.py
    #          (pure Python/NumPy — no MPI changes needed at this stage)
    # TODO P3-03: Gossip aggregator will be replaced with
    #             comm.Allreduce(local_centroids, global_centroids)
    # ═══════════════════════════════════════════════════════════════
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
            print("  Using Adaptive Gossip aggregation ...")
            from mpj_spark.core.gossip_aggregator import GossipAggregator

            # Gossip queue: collect per-worker gossip messages
            # Workers send gossip payloads on TAG_ALLREDUCE_UP (tag=30)
            # We wrap them in a queue-compatible object for GossipAggregator
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
            _ok(f"Gossip done  rounds={agg['rounds_run']}  converged={agg['converged']}")
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
        )
        gossip_info = {"iterations_done": agg.get("iterations_done", logreg_iter)}
        _info(f"Total rows       : {agg['total_rows']:,}")
        _info(f"Weighted accuracy: {agg['avg_accuracy']:.4f}")
        _info(f"Agg mode         : {agg['agg_mode']}")

    agg_time = time.perf_counter() - t_agg_start
    reassign_time = None

    # ═══════════════════════════════════════════════════════════════
    # Phase 5b (optional): Re-assignment pass for K-Means
    # ═══════════════════════════════════════════════════════════════
    if do_reassign and agg is not None and app == "kmeans":
        _phase("5b", "Re-assignment pass — exact global centroid correction")
        t_reassign = time.perf_counter()
        gossip_centres = agg["centres"]
        k_val = len(gossip_centres)
        d_val = len(gossip_centres[0])

        # Broadcast gossip centroids to all workers (TAG_REASSIGN_BCAST)
        for i in range(num_workers):
            w_rank = i + 1
            comm.send(
                {"type": "reassign", "centres": gossip_centres},
                dest=w_rank,
                tag=TAG_REASSIGN_BCAST,
            )

        # Collect per-worker cluster sums + counts (TAG_REASSIGN_STATS)
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

        print(f"\n{DASH}")
        print("  Final Corrected Centres (post re-assignment):")
        for i, c in enumerate(corrected):
            preview = ", ".join(f"{v:.3f}" for v in c[:4])
            _info(f"C{i}: [{preview}{'...' if d_val > 4 else ''}]")
        print(DASH)
        _ok(f"Re-assignment done  ({reassign_time:.3f}s)  from {total_rows:,} rows")

    # ═══════════════════════════════════════════════════════════════
    # Phase 6: Timing summary + optional baseline comparison
    # ═══════════════════════════════════════════════════════════════
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

        _phase(
            "B",
            (
                f"Running {app} baseline for comparison"
                + (
                    f"  [parity maxIter={logreg_parity_iter} "
                    f"= {num_workers} workers × {logreg_iter} iters]"
                    if logreg_parity_iter is not None
                    else ""
                )
            ),
        )

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
