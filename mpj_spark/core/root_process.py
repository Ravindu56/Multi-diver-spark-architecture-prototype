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
#         * Round diagnostics printed and included in timing.
#     - When use_gossip=False (default), behaviour is unchanged.
#
#   LOGGING REFACTOR:
#     - Phase banners use consistent ── Phase N ── style
#     - Timing summary uses fixed-width box-drawing table
#     - DevLogger.log_run() writes silently to file (no console noise)
#     - Partition path list suppressed behind a single summary line
# ================================================================
import math
import os
import time
from multiprocessing import Queue, Process, Event

from mpj_spark.core.file_manager import MPJSparkFileManager
from mpj_spark.core.key_value    import KeyValueStructure
from mpj_spark.workers.worker_process import worker_process
from mpj_spark.utils.dev_logger  import DevLogger

SEP  = '=' * 70
DASH = '─' * 70


def _hdr(title):
    print(f'\n{SEP}\n  {title}\n{SEP}')


def _phase(n, label):
    print(f'\n── Phase {n}: {label}')


def _ok(msg):
    print(f'  ✓  {msg}')


def _info(msg):
    print(f'     {msg}')


def dynamic_partition(input_path, num_partitions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manager = MPJSparkFileManager(shared_storage_path=output_dir)
    raw     = manager.dynamic_partition(input_path, num_partitions)
    paths   = []
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
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    ref  = np.array(reference)
    cand = np.array(candidate)
    diff = ref[:, np.newaxis, :] - cand[np.newaxis, :, :]
    cost = np.linalg.norm(diff, axis=2)
    _, col_ind = linear_sum_assignment(cost)
    return [candidate[i] for i in col_ind.tolist()], col_ind.tolist()


def aggregate_kmeans_results(worker_results):
    import numpy as np
    total_rows        = sum(r['row_count'] for r in worker_results)
    k                 = worker_results[0]['k']
    num_dims          = len(worker_results[0]['centres'][0])
    reference_centres = worker_results[0]['centres']
    aligned_results   = [worker_results[0]]

    for w_idx, r in enumerate(worker_results[1:], start=1):
        aligned_centres, perm = align_centres_hungarian(reference_centres, r['centres'])
        _info(f'Worker {w_idx} centroid alignment (Hungarian): {perm}')
        aligned_results.append({**r, 'centres': aligned_centres})

    merged = []
    for c_idx in range(k):
        ws = np.zeros(num_dims)
        for r in aligned_results:
            ws += (r['row_count'] / total_rows) * np.array(r['centres'][c_idx])
        merged.append(ws.tolist())

    total_wcss = sum(r['wcss'] for r in worker_results)
    print(f'\n  K-Means aggregation complete')
    _info(f'Total rows : {total_rows:,}')
    _info(f'Total WCSS : {total_wcss:.4f}')
    _info(f'Workers    : {len(worker_results)}')
    print('  Global centres:')
    for i, c in enumerate(merged):
        preview = ', '.join(f'{v:.3f}' for v in c[:4])
        _info(f'  C{i}: [{preview}{"..." if len(c) > 4 else ""}]')
    return {'centres': merged, 'total_wcss': total_wcss,
            'total_rows': total_rows, 'num_workers': len(worker_results)}


def _print_comparison(multi_timing, baseline_timing, num_workers, app, baseline_threads=None):
    note = f'  [baseline-threads={baseline_threads}]' if baseline_threads else ''
    print(f'\n{SEP}')
    print(f'  Multi-Driver vs Baseline  |  app={app}  |  workers={num_workers}{note}')
    print(SEP)
    print(f'  {"Metric":<26} {"Multi-Driver":>13} {"Baseline":>13} {"Speedup":>9}')
    print(f'  {"-"*26} {"-"*13} {"-"*13} {"-"*9}')
    for key, label in [
        ('load_time',       'Load Time (s)'),
        ('processing_time', 'Proc Time (s)'),
        ('total_time',      'Total Time (s)'),
    ]:
        m  = multi_timing[key]
        b  = baseline_timing[key]
        sp = b / m if m > 0 else 0.0
        flag = '  ⚡' if sp >= 1.5 else ('  ⚠' if sp < 1.0 else '')
        print(f'  {label:<26} {m:>12.4f}s {b:>12.4f}s {sp:>8.2f}x{flag}')
    print(SEP)


def _print_timing_summary(load_time, avg_proc, agg_time, t_wall,
                          prewarm_init=None, gossip_info=None):
    print(f'\n{DASH}')
    print('  Timing Summary')
    print(DASH)
    print(f'  {"Partition / Load":<28} {load_time:>8.4f} s')
    print(f'  {"Avg Worker Proc":<28} {avg_proc:>8.4f} s')
    print(f'  {"Aggregation":<28} {agg_time:>8.4f} s')
    if prewarm_init is not None:
        print(f'  {"Avg Worker JVM Init (excl.)":<28} {prewarm_init:>8.4f} s')
    if gossip_info:
        print(f'  {"Gossip Rounds":<28} {gossip_info["rounds_run"]:>8}')
        print(f'  {"Gossip Converged":<28} {str(gossip_info["converged"]):>8}')
    print(f'  {"-"*40}')
    print(f'  {"Total Wall-clock":<28} {t_wall:>8.4f} s')
    print(DASH)


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
    use_gossip=False,
    gossip_threshold=1e-3,
    gossip_max_rounds=10,
    gossip_fanout=2,
):
    from mpj_spark.config import TOTAL_CORES, DATA_DIR
    logger = DevLogger(worker_id='root')

    # ── Header ────────────────────────────────────────────────────────
    agg_mode = (
        f'Adaptive Gossip  (threshold={gossip_threshold}, '
        f'max_rounds={gossip_max_rounds}, fanout={gossip_fanout})'
        if (use_gossip and app == 'kmeans')
        else 'Batch Hungarian'
    )
    title_extra = f'  k={kmeans_k}  max_iter={kmeans_iter}\n' if app == 'kmeans' else ''
    _hdr(
        f'MPJ-Spark Multi-Driver  |  app={app}  |  workers={num_workers}\n'
        f'{title_extra}'
        f'  Aggregation : {agg_mode}'
    )

    cores = max(1, cores_override) if cores_override else max(1, math.ceil(TOTAL_CORES / num_workers))
    print(f'  Core budget : local[{cores}]  ({TOTAL_CORES} total ÷ {num_workers} workers)')

    worker_cfg = {
        'app'            : app,
        'cores_override' : cores,
        'kmeans_k'       : kmeans_k,
        'kmeans_max_iter': kmeans_iter,
        'num_workers'    : num_workers,
    }

    # ── Phase 1: Partition ─────────────────────────────────────────────
    _phase(1, 'Partitioning')
    t_load_start    = time.perf_counter()
    partition_paths = dynamic_partition(input_file, num_workers, DATA_DIR)
    load_time       = time.perf_counter() - t_load_start
    _ok(f'Split into {num_workers} partitions  ({load_time:.3f}s)')

    # ── Phase 2: Launch workers ────────────────────────────────────────
    _phase(2, f'Launching {num_workers} workers')
    result_queue  = Queue()
    timing_queue  = Queue()
    go_signals    = [Event() for _ in range(num_workers)]
    ready_signals = [Event() for _ in range(num_workers)]
    processes     = []
    gossip_queue  = Queue() if (use_gossip and app == 'kmeans') else None

    for i in range(num_workers):
        p = Process(
            target=worker_process,
            args=(i, partition_paths[i], result_queue,
                  go_signals[i], ready_signals[i], timing_queue, worker_cfg,
                  gossip_queue),
            daemon=True)
        p.start()
        processes.append(p)
        _info(f'Worker {i} started  (PID {p.pid})')

    print('  Waiting for JVM barrier ...')
    for i, sig in enumerate(ready_signals):
        sig.wait()
        _ok(f'Worker {i} JVM ready')

    # ── Phase 3: Fire ─────────────────────────────────────────────────
    _phase(3, 'Firing all workers simultaneously')
    t_proc_start = time.perf_counter()
    for sig in go_signals:
        sig.set()

    # ── Phase 4: Collect ──────────────────────────────────────────────
    _phase(4, 'Collecting results')
    worker_results = []
    worker_timings = []
    errors         = []
    for _ in range(num_workers):
        res = result_queue.get(timeout=600)
        if res['status'] == 'success':
            worker_results.append(res['result'])
        else:
            errors.append(res)
            print(f"  ✗  Worker {res['worker_id']} FAILED: {res.get('error')}")
    for _ in range(num_workers):
        worker_timings.append(timing_queue.get(timeout=60))
    proc_time = time.perf_counter() - t_proc_start
    for p in processes:
        p.join(timeout=30)
    if errors:
        print(f'  {len(errors)} worker(s) failed. Aborting.')
        return
    _ok(f'All {num_workers} workers completed')

    # ── Phase 5: Aggregate ────────────────────────────────────────────
    _phase(5, 'Aggregating results')
    t_agg_start  = time.perf_counter()
    gossip_info  = None
    agg          = None

    if app == 'wordcount':
        kv = KeyValueStructure()
        for r in worker_results:
            kv.merge(r)
        top = kv.get_top_n(20)
        print('\n  Top-20 words:')
        for word, count in top:
            print(f'    {word:<22} {count:>12,}')

    elif app == 'kmeans':
        if use_gossip and gossip_queue is not None:
            print('  Using Adaptive Gossip aggregation ...')
            from mpj_spark.core.gossip_aggregator import GossipAggregator
            gagg = GossipAggregator(
                num_workers=num_workers,
                convergence_threshold=gossip_threshold,
                max_rounds=gossip_max_rounds,
                initial_fanout=gossip_fanout,
                verbose=True,
            )
            agg         = gagg.aggregate(gossip_queue, timeout_per_worker=120.0)
            gossip_info = agg
            _ok(f'Gossip done  rounds={agg["rounds_run"]}  converged={agg["converged"]}')
        else:
            agg = aggregate_kmeans_results(worker_results)

        _info(f'Total rows : {agg["total_rows"]:,}')
        _info(f'Total WCSS : {agg["total_wcss"]:.4f}')

    agg_time  = time.perf_counter() - t_agg_start
    t_wall    = load_time + proc_time + agg_time
    avg_proc  = sum(t['processing_time'] for t in worker_timings) / num_workers
    avg_init  = sum(t['init_time'] for t in worker_timings) / num_workers if prewarm else None

    _print_timing_summary(load_time, avg_proc, agg_time, t_wall,
                          prewarm_init=avg_init, gossip_info=gossip_info)

    # Silent file log
    logger.log_run(app=app, num_workers=num_workers, cores=cores,
                   load_time=load_time, proc_time=avg_proc,
                   agg_time=agg_time, total_time=t_wall)

    # ── Baseline comparison ───────────────────────────────────────────
    if compare:
        _phase('B', f'Running {app} baseline for comparison')
        if app == 'wordcount':
            from mpj_spark.applications.baseline_spark import run_baseline
            _, baseline_timing = run_baseline(input_file, num_workers, cores_override)
        elif app == 'kmeans':
            from mpj_spark.applications.baseline_kmeans import run_baseline_kmeans
            _, baseline_timing = run_baseline_kmeans(
                input_file, num_workers, cores_override,
                kmeans_k, kmeans_iter, baseline_threads=baseline_threads)
        multi_timing = {
            'load_time'       : load_time,
            'processing_time' : avg_proc,
            'total_time'      : t_wall,
        }
        _print_comparison(multi_timing, baseline_timing, num_workers, app,
                          baseline_threads=baseline_threads)


mpj_root_process = run_root
