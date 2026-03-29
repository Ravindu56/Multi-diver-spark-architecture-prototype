# ================================================================
# mpj_spark/core/root_process.py
#
# Root process — orchestrates the 5-phase MPJ-Spark pipeline
# Supports: wordcount | kmeans
# ================================================================

import os
import time
import multiprocessing as mp
from multiprocessing import Queue, Process, Event

from mpj_spark.core.file_manager import MPJSparkFileManager
from mpj_spark.core.key_value    import KeyValueStructure
from mpj_spark.workers.worker_process import worker_process
from mpj_spark.utils.dev_logger  import DevLogger


# ── Adapter: wraps MPJSparkFileManager into a simple function call ──────
def dynamic_partition(input_path: str, num_partitions: int, output_dir: str) -> list:
    """
    Stream-split input file into N partition files using MPJSparkFileManager.
    Returns a list of partition file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    manager = MPJSparkFileManager(
        input_file=input_path,
        num_partitions=num_partitions,
        output_dir=output_dir,
    )
    return manager.partition()


# ── K-Means centroid aggregation ─────────────────────────────────────────
def aggregate_kmeans_results(worker_results: list) -> dict:
    """
    Merge K-Means results from multiple workers via weighted centroid average.

    Each worker trained on a different data partition. Centroids are merged
    by weighting each worker's contribution by its row count relative to
    total rows. Total WCSS is summed (WCSS is additive across partitions).

    Parameters
    ----------
    worker_results : list[dict]
        Each dict: {'centres': list[list[float]], 'wcss': float,
                    'k': int, 'row_count': int}

    Returns
    -------
    dict:
        centres     : list[list[float]]
        total_wcss  : float
        total_rows  : int
        num_workers : int
    """
    import numpy as np

    total_rows  = sum(r['row_count'] for r in worker_results)
    k           = worker_results[0]['k']
    num_dims    = len(worker_results[0]['centres'][0])
    num_workers = len(worker_results)

    merged_centres = []
    for c_idx in range(k):
        weighted_sum = np.zeros(num_dims)
        for r in worker_results:
            weight        = r['row_count'] / total_rows
            weighted_sum += weight * np.array(r['centres'][c_idx])
        merged_centres.append(weighted_sum.tolist())

    total_wcss = sum(r['wcss'] for r in worker_results)

    print('\n[Root] K-Means Aggregation Complete')
    print(f'       Total rows  : {total_rows:,}')
    print(f'       Total WCSS  : {total_wcss:.4f}')
    print(f'       Workers     : {num_workers}')
    print('       Global Centres:')
    for i, c in enumerate(merged_centres):
        preview = ', '.join(f'{v:.3f}' for v in c[:4])
        print(f'         C{i}: [{preview}{"..." if len(c) > 4 else ""}]')

    return {
        'centres'    : merged_centres,
        'total_wcss' : total_wcss,
        'total_rows' : total_rows,
        'num_workers': num_workers,
    }


# ── Root orchestrator ─────────────────────────────────────────────────────
def run_root(
    input_file:     str,
    num_workers:    int  = 2,
    compare:        bool = False,
    prewarm:        bool = True,
    cores_override: int  = None,
    app:            str  = 'wordcount',
    kmeans_k:       int  = 3,
    kmeans_iter:    int  = 20,
):
    """
    Main Root process — 5-phase MPJ-Spark pipeline.

    Phases
    ------
    1. Partition  — O(1) RAM stream-split into N partition files
    2. Launch     — spawn N worker processes, build SparkSessions
    3. Fire       — simultaneous go-signal (barrier sync)
    4. Collect    — gather results + timings from queues
    5. Aggregate  — WordCount reduce OR K-Means weighted centroid merge
    """
    from mpj_spark.config import TOTAL_CORES, DATA_DIR

    logger = DevLogger(worker_id='root')

    SEP = '=' * 70
    print(f'\n{SEP}')
    print(f'  MPJ-Spark Multi-Driver  |  app={app}  |  workers={num_workers}')
    if app == 'kmeans':
        print(f'  k={kmeans_k}  max_iter={kmeans_iter}')
    print(SEP)

    cores = max(1, cores_override) if cores_override else max(1, TOTAL_CORES // num_workers)
    print(f'  Core budget per worker : local[{cores}]  '
          f'({TOTAL_CORES} total ÷ {num_workers} workers)')

    worker_cfg = {
        'app'            : app,
        'cores_override' : cores,
        'kmeans_k'       : kmeans_k,
        'kmeans_max_iter': kmeans_iter,
    }

    # ════════════════════════════════════════════════════════════
    # Phase 1 — Partition
    # ════════════════════════════════════════════════════════════
    print(f'\n[Root] Phase 1 — Partitioning into {num_workers} parts ...')
    t_load_start = time.perf_counter()

    partition_paths = dynamic_partition(
        input_path=input_file,
        num_partitions=num_workers,
        output_dir=DATA_DIR,
    )

    t_load_end = time.perf_counter()
    load_time  = t_load_end - t_load_start
    print(f'[Root] Partitioning done in {load_time:.3f}s')

    # ════════════════════════════════════════════════════════════
    # Phase 2 — Launch Workers
    # ════════════════════════════════════════════════════════════
    print(f'\n[Root] Phase 2 — Launching {num_workers} workers ...')

    result_queue  = Queue()
    timing_queue  = Queue()
    go_signals    = [Event() for _ in range(num_workers)]
    ready_signals = [Event() for _ in range(num_workers)]
    processes     = []

    for i in range(num_workers):
        p = Process(
            target=worker_process,
            args=(i, partition_paths[i], result_queue,
                  go_signals[i], ready_signals[i], timing_queue, worker_cfg),
            daemon=True,
        )
        p.start()
        processes.append(p)
        print(f'[Root] Worker {i} started (PID {p.pid})')

    print('[Root] Waiting for all workers to be JVM-ready ...')
    for i, sig in enumerate(ready_signals):
        sig.wait()
        print(f'[Root] Worker {i} ready ✓')

    # ════════════════════════════════════════════════════════════
    # Phase 3 — Fire
    # ════════════════════════════════════════════════════════════
    print('\n[Root] Phase 3 — Firing all workers simultaneously ...')
    t_proc_start = time.perf_counter()
    for sig in go_signals:
        sig.set()

    # ════════════════════════════════════════════════════════════
    # Phase 4 — Collect
    # ════════════════════════════════════════════════════════════
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
        print(f'[Root] {len(errors)} worker(s) failed. Aborting aggregation.')
        return

    # ════════════════════════════════════════════════════════════
    # Phase 5 — Aggregate
    # ════════════════════════════════════════════════════════════
    print('\n[Root] Phase 5 — Aggregating results ...')
    t_agg_start = time.perf_counter()

    if app == 'wordcount':
        kv = KeyValueStructure()
        for r in worker_results:
            kv.merge(r)
        final_result = kv.get_top_n(20)
        print(f'\n  Top-20 words:')
        for word, count in final_result:
            print(f'    {word:<20} {count:>10,}')

    elif app == 'kmeans':
        agg = aggregate_kmeans_results(worker_results)
        print(f'\n  Total rows processed : {agg["total_rows"]:,}')
        print(f'  Total WCSS (inertia) : {agg["total_wcss"]:.4f}')

    t_agg_end = time.perf_counter()
    agg_time  = t_agg_end - t_agg_start
    t_wall    = load_time + proc_time + agg_time
    avg_proc  = sum(t['processing_time'] for t in worker_timings) / num_workers

    print(f'\n{"─"*70}')
    print(f'  MPJ Multi-Driver Timing Summary')
    print(f'{"─"*70}')
    print(f'  Partition / Load Time    : {load_time:.4f} s')
    print(f'  Avg Worker Process Time  : {avg_proc:.4f} s')
    print(f'  Aggregation Time         : {agg_time:.4f} s')
    print(f'  Total Wall-clock Time    : {t_wall:.4f} s')

    logger.log_run(
        app=app, num_workers=num_workers, cores=cores,
        load_time=load_time, proc_time=avg_proc,
        agg_time=agg_time, total_time=t_wall,
    )

    # ── Optional baseline comparison ─────────────────────────────────────
    if compare:
        print(f'\n[Root] Running baseline ({app}) for comparison ...')

        if app == 'wordcount':
            from mpj_spark.applications.baseline_spark import run_baseline
            _, baseline_timing = run_baseline(
                input_file_path=input_file,
                num_workers=num_workers,
                cores_override=cores_override,
            )
        elif app == 'kmeans':
            from mpj_spark.applications.baseline_kmeans import run_baseline_kmeans
            _, baseline_timing = run_baseline_kmeans(
                input_file_path=input_file,
                num_workers=num_workers,
                cores_override=cores_override,
                k=kmeans_k,
                max_iter=kmeans_iter,
            )

        from mpj_spark.benchmarks.reporter import print_comparison_table
        multi_timing = {
            'load_time'      : load_time,
            'processing_time': avg_proc,
            'total_time'     : t_wall,
        }
        print_comparison_table(
            multi_timing=multi_timing,
            baseline_timing=baseline_timing,
            num_workers=num_workers,
            app=app,
        )


# Backwards-compatibility alias
mpj_root_process = run_root