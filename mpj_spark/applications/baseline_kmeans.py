# ================================================================
# mpj_spark/applications/baseline_kmeans.py
# Standard single-driver Spark K-Means — fair comparison baseline
# ================================================================
import time

try:
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.feature import VectorAssembler
except ImportError:  # pragma: no cover
    KMeans = None
    VectorAssembler = None

from mpj_spark.workers.spark_session import build_spark_session


def run_baseline_kmeans(
    input_file_path:  str,
    num_workers:      int = 1,
    cores_override:   int = None,
    k:                int = 3,
    max_iter:         int = 20,
    baseline_threads: int = None,
) -> tuple:
    """
    Single-driver Spark K-Means. Fair baseline for multi-driver comparison.

    Parameters
    ----------
    baseline_threads : int, optional
        When provided, the baseline SparkSession is given exactly this many
        threads (local[N]), regardless of num_workers or cores_override.
        Use this to give the baseline the same *total* thread count as all
        MPJ workers combined, e.g.:
            --workers 4 --cores 5  →  MPJ uses 4×local[5] = 20 threads
            --baseline-threads 20  →  baseline uses local[20]  (fair)
        When omitted the baseline uses the same per-worker budget as each
        individual MPJ worker (existing behaviour, conservative comparison).

    Returns
    -------
    (result_dict, timing_dict)
        result_dict : {'centres': list[list[float]], 'wcss': float}
        timing_dict : {'load_time': float, 'processing_time': float, 'total_time': float}
    """
    from mpj_spark.config import TOTAL_CORES

    # Thread budget resolution (priority order):
    #   1. baseline_threads explicitly passed  →  fair comparison mode
    #   2. cores_override passed               →  manual override
    #   3. default: TOTAL_CORES // num_workers →  same per-worker budget
    if baseline_threads is not None:
        cores = max(1, baseline_threads)
        budget_label = f'local[{cores}]  [fair: total MPJ threads = {cores}]'
    elif cores_override:
        cores = max(1, cores_override)
        budget_label = f'local[{cores}]  (cores_override)'
    else:
        cores = max(1, TOTAL_CORES // num_workers)
        budget_label = f'local[{cores}]  ({TOTAL_CORES} total ÷ {num_workers} workers)'

    print('\n' + '=' * 70)
    print('  Standard Spark K-Means (Single Driver) — BASELINE')
    print(f'  Thread budget : {budget_label}')
    print(f'  k={k}  max_iter={max_iter}')
    print('=' * 70)

    t_total_start = time.perf_counter()

    # ── Build SparkSession ────────────────────────────────────────────
    try:
        spark = build_spark_session('Baseline-KMeans', cores_override=cores, num_workers=1)
    except TypeError:
        spark = build_spark_session('Baseline-KMeans', cores)

    # ── Load ─────────────────────────────────────────────────────────
    t_load_start = time.perf_counter()
    raw_rdd      = spark.sparkContext.textFile(input_file_path)
    first_row    = raw_rdd.first()
    num_features = len(first_row.strip().split(','))
    feature_cols = [f'f{i}' for i in range(num_features)]

    def parse_row(line):
        from pyspark.sql import Row
        try:
            vals = [float(x) for x in line.strip().split(',') if x.strip()]
            return Row(**dict(zip(feature_cols, vals))) if len(vals) == num_features else None
        except ValueError:
            return None

    df        = spark.createDataFrame(raw_rdd.map(parse_row).filter(lambda r: r is not None))
    row_count = df.count()
    t_load_end = time.perf_counter()

    # ── Assemble + Train ─────────────────────────────────────────────
    t_proc_start = time.perf_counter()
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol='features',
        handleInvalid='skip'
    )
    df_vec = assembler.transform(df).select('features').cache()
    model  = KMeans(
        k=k, maxIter=max_iter, seed=42,
        featuresCol='features', initMode='k-means||'
    ).fit(df_vec)
    t_proc_end  = time.perf_counter()
    t_total_end = time.perf_counter()

    centres   = [c.tolist() for c in model.clusterCenters]
    wcss      = float(model.summary.trainingCost)
    load_time = t_load_end  - t_load_start
    proc_time = t_proc_end  - t_proc_start
    total     = t_total_end - t_total_start

    print(f'\n  Rows processed : {row_count:,}')
    print(f'  WCSS (inertia) : {wcss:.4f}')
    print(f'  Cluster centres:')
    for i, c in enumerate(centres):
        preview = ', '.join(f'{v:.3f}' for v in c[:4])
        print(f'    C{i}: [{preview}{"..." if len(c) > 4 else ""}]')
    print(f'\n  Load Time      : {load_time:.4f} s')
    print(f'  Proc Time      : {proc_time:.4f} s')
    print(f'  Total          : {total:.4f} s')

    df_vec.unpersist()
    spark.stop()

    return (
        {'centres': centres, 'wcss': wcss},
        {'load_time': load_time, 'processing_time': proc_time, 'total_time': total},
    )
