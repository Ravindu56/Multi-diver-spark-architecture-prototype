# ============================================================
# applications/baseline_spark.py
# Standard single-driver Spark WordCount — comparison baseline
# Paper Reference: Section VI.B — Spark Cluster baseline
# ============================================================
import time

from mpj_spark.workers.spark_session import build_spark_session


def run_baseline(input_file_path: str) -> tuple:
    """
    Standard single-driver Spark WordCount.
    Used as the comparison baseline against multi-driver.

    Returns
    -------
    (sorted_results, timing_dict)
    """
    print('\n' + '=' * 70)
    print('  Standard Spark (Single Driver) — BASELINE')
    print('=' * 70)

    t_total_start = time.time()

    spark = build_spark_session('Baseline-SingleDriver')
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

    load_time = t_load_end - t_load_start
    proc_time = t_proc_end - t_proc_start
    total     = t_total_end - t_total_start

    print(f'  Unique words     : {len(sorted_results):,}')
    print(f'  Top 10 words:')
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
