# ================================================================
# mpj_spark/applications/baseline_kmeans.py
# Standard single-driver Spark K-Means — fair comparison baseline
# ================================================================
import time


def run_baseline_kmeans(
    input_file_path: str,
    num_workers:     int = 1,
    cores_override:  int = None,
    k:               int = 3,
    max_iter:        int = 20,
) -> tuple:
    """
    Single-driver Spark K-Means. Fair baseline for multi-driver comparison.

    Returns
    -------
    (result_dict, timing_dict)
        result_dict : {'centres': list[list[float]], 'wcss': float}
        timing_dict : {'load_time': float, 'processing_time': float, 'total_time': float}
    """
    from mpj_spark.config import TOTAL_CORES
    from mpj_spark.workers.spark_session import build_spark_session
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.feature import VectorAssembler

    cores = max(1, cores_override) if cores_override else max(1, TOTAL_CORES // num_workers)

    print('\n' + '=' * 70)
    print('  Standard Spark K-Means (Single Driver) — BASELINE')
    print(f'  Thread budget : local[{cores}]  ({TOTAL_CORES} total ÷ {num_workers} workers)')
    print(f'  k={k}  max_iter={max_iter}')
    print('=' * 70)

    t_total_start = time.perf_counter()

    # ── Build SparkSession ────────────────────────────────────────────
    # Try passing cores_override; fall back if the function doesn't accept it
    try:
        spark = build_spark_session('Baseline-KMeans', cores_override=cores)
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

    centres   = [c.tolist() for c in model.clusterCenters()]
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