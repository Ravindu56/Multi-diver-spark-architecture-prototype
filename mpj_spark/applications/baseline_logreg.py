# ================================================================
# mpj_spark/applications/baseline_logreg.py
#
# Single-driver LogisticRegression baseline for --compare mode.
# Mirrors baseline_kmeans.py structure.
# ================================================================
import time


def run_baseline_logreg(
    input_file: str,
    num_workers: int,
    cores_override,
    max_iter: int = 10,
    reg_param: float = 0.01,
    num_features: int = 10,
    baseline_threads: int = None,
):
    """
    Single-driver Spark LogisticRegression baseline.

    Fits on the full dataset in a single Spark session.
    Returns (model_result, timing_dict) to match baseline_kmeans API.
    """
    import math
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import VectorAssembler
    from mpj_spark.config import TOTAL_CORES

    if baseline_threads is not None:
        thread_count = baseline_threads
    elif cores_override is not None:
        thread_count = cores_override
    else:
        thread_count = max(1, math.ceil(TOTAL_CORES / num_workers))

    print(f'  [Baseline-LogReg] local[{thread_count}]  '
          f'max_iter={max_iter}  reg_param={reg_param}')

    t_load_start = time.perf_counter()
    spark = (
        SparkSession.builder
        .appName('MPJ-Baseline-LogReg')
        .master(f'local[{thread_count}]')
        .config('spark.ui.enabled', 'false')
        .config('spark.sql.shuffle.partitions', str(thread_count))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel('ERROR')

    df_raw       = spark.read.csv(input_file, inferSchema=True, header=True)
    df           = df_raw.dropna()
    feature_cols = [c for c in df.columns if c != 'label']
    row_count    = df.count()
    load_time    = time.perf_counter() - t_load_start

    print(f'  [Baseline-LogReg] {row_count:,} rows loaded  ({load_time:.3f}s)')

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol='features', handleInvalid='skip')
    df_vec = assembler.transform(df).select('features', 'label').cache()

    t_proc_start = time.perf_counter()
    lr = LogisticRegression(
        featuresCol='features',
        labelCol='label',
        maxIter=max_iter,
        regParam=reg_param,
        elasticNetParam=0.0,
        family='binomial',
        fitIntercept=True,
        standardization=True,
    )
    model        = lr.fit(df_vec)
    proc_time    = time.perf_counter() - t_proc_start
    accuracy     = float(model.summary.accuracy)
    weight_norm  = float(model.coefficients.norm(2))

    print(f'  [Baseline-LogReg] Accuracy={accuracy:.4f}  |w|={weight_norm:.4f}  '
          f'({proc_time:.3f}s)')

    df_vec.unpersist()
    spark.stop()

    total_time = load_time + proc_time
    timing = {
        'load_time'       : load_time,
        'processing_time' : proc_time,
        'total_time'      : total_time,
    }
    result = {
        'weight_vector' : model.coefficients.toArray().tolist(),
        'intercept'     : float(model.intercept),
        'accuracy'      : accuracy,
        'row_count'     : row_count,
    }
    return result, timing
