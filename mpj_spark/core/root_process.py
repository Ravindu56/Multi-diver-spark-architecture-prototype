# ================================================================
# mpj_spark/core/root_process.py
#
# Changes from feature/ml-kmeans-workload:
#
#   GOSSIP EXTENSION (feature/adaptive-gossip-aggregation):
#     - Added `use_gossip` parameter to run_root().
#     - When use_gossip=True AND app=='kmeans':
#         * A multiprocessing.Queue (gossip_queue) is created.
#         * gossip_queue is passed into every worker_process call.
#         * Phase 5 calls GossipAggregator.aggregate() instead of
#           the Hungarian batch aggregation.
#         * Round diagnostics are printed and included in timing.
#     - When use_gossip=False (default), behaviour is byte-for-byte
#       identical to feature/ml-kmeans-workload.
#
# Original changes (kept):
#   Fix #4 — math.ceil core allocation
#   Fix #5 — Hungarian centroid alignment in batch aggregation
# ================================================================
import math
import os
import time
from multiprocessing import Queue, Process, Event

from mpj_spark.core.file_manager import MPJSparkFileManager
from mpj_spark.core.key_value    import KeyValueStructure
from mpj_spark.workers.worker_process import worker_process
from mpj_spark.utils.dev_logger  import DevLogger


def dynamic_partition(input_path, num_partitions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manager = MPJSparkFileManager(shared_storage_path=output_dir)
    raw = manager.dynamic_partition(input_path, num_partitions)
    paths = []
    for item in raw:
        if isinstance(item, dict):
            p = item.get('path') or item.get('file_path') or item.get('partition_path')
            if p is None:
                p = list(item.values())[0]
            paths.append(str(p))
        else:
            paths.append(str(item))
    return paths


def align_centres_hungarian(reference, candidate):
    """
    Re-order candidate centroid list to optimally match the reference
    ordering using the Hungarian algorithm.
    Fix #5 (Issue #5) — Exact centroid label matching.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    ref  = np.array(reference)
    cand = np.array(candidate)
    diff = ref[:, np.newaxis, :] - cand[np.newaxis, :, :]
    cost = np.linalg.norm(diff, axis=2)
    _, col_ind = linear_sum_assignment(cost)
    permutation = col_ind.tolist()
    aligned     = [candidate[i] for i in permutation]
    return aligned, permutation


def aggregate_kmeans_results(worker_results):
    """
    Aggregate per-worker K-Means results into global cluster centres.
    Fix #5: Hungarian centroid alignment before weighted averaging.
    Used when use_gossip=False (original batch aggregation path).
    """
    import numpy as np

    total_rows = sum(r['row_count'] for r in worker_results)
    k          = worker_results[0]['k']
    num_dims   = len(worker_results[0]['centres'][0])

    reference_centres = worker_results[0]['centres']
    aligned_results   = [worker_results[0]]

    for w_idx, r in enumerate(worker_results[1:], start=1):
        aligned_centres, perm = align_centres_hungarian(
            reference_centres, r['centres']
        )
        print(f'[Root] Worker {w_idx} centroid alignment permutation (Hungarian): {perm}')
        aligned_results.append({
            **r,
            'centres': aligned_centres,
        })

    merged = []
    for c_idx in range(k):
        ws = np.zeros(num_dims)
        for r in aligned_results:
            ws += (r['row_count'] / total_rows) * np.array(r['centres'][c_idx])
        merged.append(ws.tolist())

    total_wcss = sum(r['wcss'] for r in worker_results)

    print('\n[Root] K-Means Aggregation Complete')
    print(f'       Total rows  : {total_rows:,}')
    print(f'       Total WCSS  : {total_wcss:.4f}')
    print(f'       Workers     : {len(worker_results)}')
    print('       Global Centres:')
    for i, c in enumerate(merged):
        preview = ', '.join(f'{v:.3f}' for v in c[:4])
        print(f'         C{i}: [{preview}{"..." if len(c) > 4 else ""}]')

    return {
        'centres'    : merged,
        'total_wcss' : total_wcss,
        'total_rows' : total_rows,
        'num_workers': len(worker_results),
    }


def _print_table(multi_timing, baseline_timing, num_workers, app, baseline_threads=None):
    SEP = '=' * 70
    thread_note = f'  [baseline local[{baseline_threads}] — fair mode]' if baseline_threads else ''
    print(f'\n{SEP}')
    print(f'  Comparison: Multi-Driver vs Baseline ({app}, {num_workers} workers){thread_note}')
    print(SEP)
    print(f'  {"Metric":<25} {"Multi-Driver":>14} {"Baseline":>14} {"Speedup":>10}')
    print(f'  {"-"*25} {"-"*14} {"-"*14} {"-"*10}')
    for key, label in [
        ('load_time',       'Load Time (s)'),
        ('processing_time', 'Proc Time (s)'),
        ('total_time',      'Total Time (s)'),
    ]:
        m = multi_timing[key]; b = baseline_timing[key]
        sp = b / m if m > 0 else 0
        print(f'  {label:<25} {m:>14.4f} {b:>14.4f} {sp:>9.2f}x')
    print(SEP)


def run_root(
    input_file,
    num_workers=2,
    compare=False,
    prewarm=True,
    cores_override=None,
    app='wordcount',
    kmeans_k=3,
    kmeans_iter=20,
    baseline_threads=None,
    use_gossip=False,             # NEW: enable adaptive gossip aggregation
    gossip_threshold=1e-3,        # NEW: convergence drift threshold
    gossip_max_rounds=10,         # NEW: hard cap on gossip rounds
    gossip_fanout=2,              # NEW: initial peer fan-out per round
):
    """
    Root orchestrator for MPJ-Spark.

    New parameters (gossip extension):
    ------------------------------------
    use_gossip       : bool  — if True, Phase 5 uses GossipAggregator
                               instead of batch Hungarian aggregation.
                               Only active when app='kmeans'.
    gossip_threshold : float — convergence criterion (max centroid drift)
    gossip_max_rounds: int   — maximum gossip rounds before forced stop
    gossip_fanout    : int   — initial fan-out (adaptive thereafter)
    """
    from mpj_spark.config import TOTAL_CORES, DATA_DIR
    logger = DevLogger(worker_id='root')
    SEP = '=' * 70
    print(f'\n{SEP}')
    print(f'  MPJ-Spark Multi-Driver  |  app={app}  |  workers={num_workers}')
    if app == 'kmeans':
        print(f'  k={kmeans_k}  max_iter={kmeans_iter}')
    if use_gossip and app == 'kmeans':
        print(f'  Aggregation  : Adaptive Gossip  '
              f'(threshold={gossip_threshold}, max_rounds={gossip_max_rounds}, '
              f'fanout={gossip_fanout})')
    else:
        print(f'  Aggregation  : Batch Hungarian (default)')
    print(SEP)

    if cores_override:
        cores = max(1, cores_override)
    else:
        cores = max(1, math.ceil(TOTAL_CORES / num_workers))
    print(f'  Core budget per worker : local[{cores}]  ({TOTAL_CORES} total \u00f7 {num_workers} workers)')

    worker_cfg = {
        'app'            : app,
        'cores_override' : cores,
        'kmeans_k'       : kmeans_k,
        'kmeans_max_iter': kmeans_iter,
        'num_workers'    : num_workers,
    }

    # Phase 1 — Partition
    print(f'\n[Root] Phase 1 — Partitioning into {num_workers} parts ...')
    t_load_start = time.perf_counter()
    partition_paths = dynamic_partition(input_file, num_workers, DATA_DIR)
    t_load_end = time.perf_counter()
    load_time  = t_load_end - t_load_start
    print(f'[Root] Partitioning done in {load_time:.3f}s')
    print(f'[Root] Partition paths: {partition_paths}')

    # Phase 2 — Launch
    print(f'\n[Root] Phase 2 — Launching {num_workers} workers ...')
    result_queue  = Queue()
    timing_queue  = Queue()
    go_signals    = [Event() for _ in range(num_workers)]
    ready_signals = [Event() for _ in range(num_workers)]
    processes     = []

    # ── GOSSIP: create shared gossip_queue if enabled ─────────────
    gossip_queue = Queue() if (use_gossip and app == 'kmeans') else None
    # ─────────────────────────────────────────────────────────────

    for i in range(num_workers):
        p = Process(
            target=worker_process,
            args=(i, partition_paths[i], result_queue,
                  go_signals[i], ready_signals[i], timing_queue, worker_cfg,
                  gossip_queue),   # GOSSIP: pass queue (None if disabled)
            daemon=True)
        p.start()
        processes.append(p)
        print(f'[Root] Worker {i} started (PID {p.pid})')

    print('[Root] Waiting for all workers to be JVM-ready ...')
    for i, sig in enumerate(ready_signals):
        sig.wait()
        print(f'[Root] Worker {i} ready \u2713')

    # Phase 3 — Fire
    print('\n[Root] Phase 3 — Firing all workers simultaneously ...')
    t_proc_start = time.perf_counter()
    for sig in go_signals:
        sig.set()

    # Phase 4 — Collect
    print('[Root] Phase 4 — Collecting results ...')
    worker_results = []
    worker_timings = []
    errors         = []
    for _ in range(num_workers):
        res = result_queue.get(timeout=600)
        if res['status'] == 'success':
            worker_results.append(res['result'])
        else:
            errors.append(res)
            print(f"[Root] Worker {res['worker_id']} ERROR: {res.get('error')}")
    for _ in range(num_workers):
        worker_timings.append(timing_queue.get(timeout=60))
    t_proc_end = time.perf_counter()
    proc_time  = t_proc_end - t_proc_start
    for p in processes:
        p.join(timeout=30)
    if errors:
        print(f'[Root] {len(errors)} worker(s) failed. Aborting.')
        return

    # Phase 5 — Aggregate
    print('\n[Root] Phase 5 — Aggregating results ...')
    t_agg_start = time.perf_counter()
    agg = None

    if app == 'wordcount':
        kv = KeyValueStructure()
        for r in worker_results:
            kv.merge(r)
        top = kv.get_top_n(20)
        print('\n  Top-20 words:')
        for word, count in top:
            print(f'    {word:<20} {count:>10,}')

    elif app == 'kmeans':
        if use_gossip and gossip_queue is not None:
            # ── GOSSIP AGGREGATION PATH ───────────────────────────
            # gossip_queue is already populated by each worker via
            # the gossip hook in worker_process.py.
            print('[Root] Using Adaptive Gossip aggregation ...')
            from mpj_spark.core.gossip_aggregator import GossipAggregator
            gagg = GossipAggregator(
                num_workers           = num_workers,
                convergence_threshold = gossip_threshold,
                max_rounds            = gossip_max_rounds,
                initial_fanout        = gossip_fanout,
                verbose               = True,
            )
            agg = gagg.aggregate(gossip_queue, timeout_per_worker=120.0)
            print(f'[Root] Gossip converged={agg["converged"]} '
                  f'rounds={agg["rounds_run"]} '
                  f'agg_time={agg["agg_time_s"]}s')
            # ── END GOSSIP PATH ───────────────────────────────────
        else:
            # ── ORIGINAL BATCH HUNGARIAN PATH (unchanged) ─────────
            agg = aggregate_kmeans_results(worker_results)

        print(f'\n  Total rows processed : {agg["total_rows"]:,}')
        print(f'  Total WCSS (inertia) : {agg["total_wcss"]:.4f}')

    t_agg_end = time.perf_counter()
    agg_time  = t_agg_end - t_agg_start
    t_wall    = load_time + proc_time + agg_time
    avg_proc  = sum(t['processing_time'] for t in worker_timings) / num_workers

    print(f'\n{chr(8212)*70}')
    print('  MPJ Multi-Driver Timing Summary')
    print(f'{chr(8212)*70}')
    print(f'  Partition / Load Time    : {load_time:.4f} s')
    print(f'  Avg Worker Process Time  : {avg_proc:.4f} s')
    print(f'  Aggregation Time         : {agg_time:.4f} s')
    print(f'  Total Wall-clock Time    : {t_wall:.4f} s')
    if use_gossip and agg and 'round_log' in agg:
        print(f'  Gossip Rounds            : {agg["rounds_run"]}')
        print(f'  Gossip Converged         : {agg["converged"]}')

    logger.log_run(app=app, num_workers=num_workers, cores=cores,
                   load_time=load_time, proc_time=avg_proc,
                   agg_time=agg_time, total_time=t_wall)

    if compare:
        print(f'\n[Root] Running baseline ({app}) for comparison ...')
        if app == 'wordcount':
            from mpj_spark.applications.baseline_spark import run_baseline
            _, baseline_timing = run_baseline(input_file, num_workers, cores_override)
        elif app == 'kmeans':
            from mpj_spark.applications.baseline_kmeans import run_baseline_kmeans
            _, baseline_timing = run_baseline_kmeans(
                input_file, num_workers, cores_override,
                kmeans_k, kmeans_iter,
                baseline_threads=baseline_threads)
        multi_timing = {
            'load_time'       : load_time,
            'processing_time' : avg_proc,
            'total_time'      : t_wall,
        }
        _print_table(multi_timing, baseline_timing, num_workers, app,
                     baseline_threads=baseline_threads)


mpj_root_process = run_root
