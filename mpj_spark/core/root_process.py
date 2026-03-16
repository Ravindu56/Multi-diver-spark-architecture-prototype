# ============================================================
# core/root_process.py
# MPJ Root Process — orchestrates the full multi-driver pipeline
# Paper Reference: Section IV.A + Algorithm 1
# ============================================================
import time
from collections import defaultdict
from multiprocessing import Process, Queue

from mpj_spark.core.file_manager       import MPJSparkFileManager
from mpj_spark.core.key_value          import KeyValueStructure
from mpj_spark.workers.worker_process  import mpj_worker_process
from mpj_spark.benchmarks.timing       import TimingCollector
from mpj_spark.benchmarks.reporter     import print_results, print_timing


def mpj_root_process(input_file_path: str,
                     num_workers:     int,
                     app:             str  = 'wordcount',
                     prewarm:         bool = True) -> tuple:
    """
    Root Process (Paper §IV.A + Algorithm 1).

    Phases:
      1. Partition input via MPJSparkFileManager
      2. Launch workers (each spins up its own SparkSession)
      2a. [prewarm=True] Wait for ALL JVMs to initialise before
          starting the parallel computation timer, then fire
          go-signals to all workers simultaneously.
      3. Wait for workers to complete computation
      4. Collect & aggregate results from Queue
      5. Print final results + timing report

    Parameters
    ----------
    input_file_path : str   — path to input text file
    num_workers     : int   — number of parallel MPJ workers
    app             : str   — application key ('wordcount', future: 'kmeans')
    prewarm         : bool  — if True (default), exclude JVM init from the
                              parallel computation timer. Mirrors HPC behaviour
                              where Spark drivers are already live on nodes.

    Returns
    -------
    (sorted_results, timing_dict)
      timing_dict keys: load_time, processing_time (avg per-worker T_Proc),
                        total_time, parallel_time, avg_init_time, agg_time
    """
    tc = TimingCollector()

    mode_label = 'pre-warmed' if prewarm else 'cold-start'
    print('=' * 70)
    print('  MPJ-SPARK Multi-Driver Prototype  v2.0')
    print(f'  Workers: {num_workers} | Input: {input_file_path}')
    print(f'  App: {app} | JVM mode: {mode_label}')
    print('=' * 70)

    tc.start('total')

    # ── Phase 1: Partition ────────────────────────────────────────────────
    print('\n[ROOT] Phase 1: Partitioning input...')
    fm = MPJSparkFileManager()
    tc.start('load')
    metadata_list = fm.dynamic_partition(input_file_path, num_workers)
    tc.stop('load')

    for m in metadata_list:
        print(f"  Partition {m['partition_id']}: "
              f"{m['num_lines']:,} lines \u2192 {m['partition_path']}")
    print(f"[ROOT] {num_workers} partitions created in {tc.elapsed('load'):.3f}s")

    # ── Phase 2: Launch workers ───────────────────────────────────────────
    print(f'\n[ROOT] Phase 2: Launching {num_workers} MPJ Workers (JVM init)...')
    result_q = Queue()
    timing_q = Queue()
    procs    = []

    # Per-worker go-queues: each worker gets its own Queue so Root can
    # fire them individually or all at once after the barrier.
    go_queues    = [Queue() for _ in range(num_workers)] if prewarm else [None] * num_workers
    ready_queue  = Queue() if prewarm else None

    tc.start('jvm_init')   # track how long it takes all JVMs to warm up
    for i, meta in enumerate(metadata_list):
        p = Process(
            target=mpj_worker_process,
            args=(i, meta, result_q, timing_q, app,
                  ready_queue, go_queues[i])
        )
        procs.append(p)
        p.start()
        print(f'  [ROOT] Worker {i} launched (PID {p.pid}) — warming JVM...')

    # ── Phase 2a: Pre-warm barrier (wait for all JVMs) ─────────────────────
    if prewarm:
        print(f'\n[ROOT] Phase 2a: Waiting for all {num_workers} JVMs to warm up...')
        ready_count  = 0
        init_reports = {}
        while ready_count < num_workers:
            report = ready_queue.get()   # blocks until next worker is ready
            wid    = report['worker_id']
            init_reports[wid] = report['init_time']
            ready_count += 1
            print(f'  [ROOT] Worker {wid} JVM ready '
                  f'(init={report["init_time"]:.2f}s) '
                  f'[{ready_count}/{num_workers}]')

        tc.stop('jvm_init')
        avg_jvm = sum(init_reports.values()) / len(init_reports)
        print(f'[ROOT] All JVMs ready. Avg init: {avg_jvm:.2f}s  '
              f'Total wait: {tc.elapsed("jvm_init"):.2f}s')

        # Fire go-signals to ALL workers simultaneously → true parallel start
        print(f'[ROOT] Firing go-signals to all {num_workers} workers...')
        tc.start('parallel')   # ← parallel timer starts HERE (pure computation)
        for gq in go_queues:
            gq.put('GO')
    else:
        tc.stop('jvm_init')
        tc.start('parallel')   # legacy: timer starts with process launch

    # ── Phase 3: Wait for computation to finish ─────────────────────────
    print(f'\n[ROOT] Phase 3: Waiting for {num_workers} workers to complete...')
    for p in procs:
        p.join()
    tc.stop('parallel')

    # ── Phase 4: Collect ──────────────────────────────────────────────────
    print('\n[ROOT] Phase 4: Collecting results...')
    all_results, worker_timings = [], []

    while not result_q.empty():
        r = result_q.get()
        if 'error' in r:
            print(f"  [ERROR] Worker {r['worker_id']}: {r['error']}")
        else:
            all_results.append(r)
            print(f"  Received {r['num_items']:,} items from Worker {r['worker_id']}")

    while not timing_q.empty():
        worker_timings.append(timing_q.get())

    # ── Phase 5: Aggregate ────────────────────────────────────────────────
    print('\n[ROOT] Phase 5: Aggregating results...')
    tc.start('agg')
    final_counts: dict = defaultdict(int)
    for wr in all_results:
        kv = KeyValueStructure.from_serializable(wr['results'])
        for key, val in kv.data:
            final_counts[key] += val
    tc.stop('agg')
    tc.stop('total')

    sorted_results = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)

    # ── Report ────────────────────────────────────────────────────────────
    print_results(sorted_results)
    print_timing(tc, worker_timings)

    fm.cleanup()

    return sorted_results, tc.summary(worker_timings)
