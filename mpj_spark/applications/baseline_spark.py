# ============================================================
# applications/baseline_spark.py
# Standard single-driver Spark WordCount — comparison baseline
# Paper Reference: Section VI.B — Spark Cluster baseline
# ============================================================
# Fair-comparison note
# --------------------
# The baseline is constrained to the same per-entity thread budget
# as each MPJ worker:  cores = TOTAL_CORES // num_workers
# When num_workers=2 on a 22-core machine, each worker and the
# baseline both get local[11] — not local[*] (22 cores).
# This prevents the baseline from having an unfair 2x thread
# advantage over the multi-driver workers.
# ============================================================
import time

from mpj_spark.workers.spark_session import build_spark_session


def run_baseline(input_file_path: str,
                 num_workers:    int = 1,
                 cores_override: int = None) -> tuple:
    """
    Standard single-driver Spark WordCount.
    Used as the comparison baseline against multi-driver.

    Parameters
    ----------
    input_file_path : str — path to full (un-partitioned) input file
    num_workers     : int — number of MPJ workers in the comparison run.
                           Used to compute fair per-entity core budget:
                           baseline_cores = TOTAL_CORES // num_workers
                           Defaults to 1 (baseline gets all cores).
    cores_override  : int — bypass the formula, force this exact core
                           count (from --cores CLI flag). None = auto.

    Returns
    -------
    (sorted_results, timing_dict)
    """
    from mpj_spark.config import TOTAL_CORES

    if cores_override is not None:
        cores = max(1, cores_override)
    else:
        cores = max(1, TOTAL_CORES // num_workers)

    print('\n' + '=' * 70)
    print('  Standard Spark (Single Driver) — BASELINE')
    print(f'  Thread budget: local[{cores}]  '
          f'({TOTAL_CORES} total cores ÷ {num_workers} workers)')
    print('=' * 70)

    t_total_start = time.time()

    spark = build_spark_session('Baseline-SingleDriver',
                                cores_override=cores)
    sc    = spark.sparkContext

    # Load
    t_load_start = time.time()
    text_rdd     = sc.textFile(input_file_path)
    text_rdd.count()          # force materialisation
    t_load_end   = time.time()

    # Process
    t_proc_start = time.time()
    results = (
        text_rdd
        .flatMap(lambda line: line.lower().split())
        .filter(lambda word: len(word) > 0)
        .map(lambda word: (word, 1))
        .reduceByKey(lambda a, b: a + b)
        .collect()
    )
    t_proc_end = time.time()

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    t_total_end    = time.time()

    load_time = t_load_end  - t_load_start
    proc_time = t_proc_end  - t_proc_start
    total     = t_total_end - t_total_start

    print(f'  Unique words     : {len(sorted_results):,}')
    print('  Top 10 words:')
    for word, cnt in sorted_results[:10]:
        print(f'    {word:20s} -> {cnt:,}')
    print(f'\n  Load Time        : {load_time:.4f} s')
    print(f'  Processing Time  : {proc_time:.4f} s')
    print(f'  Total Execution  : {total:.4f} s')

    spark.stop()

    return sorted_results, {
        'load_time':       load_time,
        'processing_time': proc_time,
        'total_time':      total,
    }
