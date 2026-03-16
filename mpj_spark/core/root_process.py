# ============================================================
# core/root_process.py
# MPJ Root Process — orchestrates the full multi-driver pipeline
# Paper Reference: Section IV.A + Algorithm 1
# ============================================================
import time
from collections import defaultdict
from multiprocessing import Process, Queue

from mpj_spark.core.file_manager import MPJSparkFileManager
from mpj_spark.core.key_value    import KeyValueStructure
from mpj_spark.workers.worker_process import mpj_worker_process
from mpj_spark.benchmarks.timing  import TimingCollector
from mpj_spark.benchmarks.reporter import print_results, print_timing


def mpj_root_process(input_file_path: str,
                     num_workers: int,
                     app: str = 'wordcount') -> tuple:
    """
    Root Process (Paper §IV.A + Algorithm 1).

    Phases:
      1. Partition input via MPJSparkFileManager
      2. Launch one MPJ worker Process per partition
      3. Wait for all workers
      4. Collect & aggregate results from Queue
      5. Print final results + timing report

    Parameters
    ----------
    input_file_path : str  — path to input text file
    num_workers     : int  — number of parallel MPJ workers
    app             : str  — application key ('wordcount', future: 'kmeans')

    Returns
    -------
    (sorted_results, timing_dict)
    """
    tc = TimingCollector()

    print('=' * 70)
    print('  MPJ-SPARK Multi-Driver Prototype  v2.0')
    print(f'  Workers: {num_workers} | Input: {input_file_path} | App: {app}')
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
              f"{m['num_lines']:,} lines → {m['partition_path']}")
    print(f"[ROOT] {num_workers} partitions created in {tc.elapsed('load'):.3f}s")

    # ── Phase 2: Launch workers ───────────────────────────────────────────
    print(f'\n[ROOT] Phase 2: Launching {num_workers} MPJ Workers...')
    result_q = Queue()
    timing_q = Queue()
    procs = []

    tc.start('parallel')
    for i, meta in enumerate(metadata_list):
        p = Process(
            target=mpj_worker_process,
            args=(i, meta, result_q, timing_q, app)
        )
        procs.append(p)
        p.start()
        print(f'  [ROOT] Worker {i} launched (PID {p.pid})')

    # ── Phase 3: Wait ─────────────────────────────────────────────────────
    print(f'\n[ROOT] Phase 3: Waiting for {num_workers} workers...')
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

    return sorted_results, tc.summary()
