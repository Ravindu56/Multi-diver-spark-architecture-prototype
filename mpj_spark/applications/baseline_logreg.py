# ================================================================
# mpj_spark/applications/baseline_logreg.py
#
# Single-driver LogisticRegression baseline for --compare mode.
# Mirrors baseline_kmeans.py structure.
# ================================================================
import math
import time

try:
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import VectorAssembler
except ImportError:  # pragma: no cover
    SparkSession = None
    LogisticRegression = None
    VectorAssembler = None


def _baseline_heap_gb(thread_count: int) -> int:
    """
    Compute a safe driver heap size for the baseline Spark session.

    Formula: 512 MB base + 256 MB per thread, rounded up to the next
    integer GB, capped at 80% of system RAM, minimum 2 GB.

    This is needed because the baseline runs on the full (unpartitioned)
    dataset with potentially many threads doing treeAggregate passes.
    The multi-driver workers each see only 1/N of the data, so their
    per-JVM memory pressure is much lower.
    """
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        cap_gb = max(2, int(total_ram_gb * 0.80))
    except ImportError:
        cap_gb = 8

    raw_gb = math.ceil(0.5 + 0.25 * thread_count)
    return min(max(2, raw_gb), cap_gb)


def run_baseline_logreg(
    input_file: str,
    num_workers: int,
    cores_override,
    max_iter: int = 10,
    reg_param: float = 0.01,
    num_features: int = 10,
    baseline_threads: int = None,
    parity_iter: int = None,
):
    """
    Single-driver Spark LogisticRegression baseline for --compare mode.

    Returns (result_dict, timing_dict).  On OOM or Spark failure the fit
    is skipped and result_dict contains accuracy=None with an oom_error key
    so the comparison table can print a meaningful fallback rather than
    crashing the full run.

    parity_iter
    -----------
    When provided, overrides max_iter so the baseline performs the same
    total number of gradient steps as the multi-driver framework:

        parity_iter = num_workers × logreg_iter

    This ensures the comparison is fair on compute: the multi-driver run
    distributes (num_workers × logreg_iter) gradient steps across workers
    (one per Allreduce round on a 1/num_workers data shard), so the
    baseline must also perform that many steps on the full dataset.

    Memory note — why .cache() is NOT used here
    --------------------------------------------
    Calling df_vec.cache() forces the full VectorAssembler output (dense
    feature matrix + label column, ~4 bytes × rows × features per column)
    into the JVM MemoryStore via putIteratorAsValues.  On a 500 MB CSV
    with 20 features this can exceed 2–3 GB of heap, causing OOM during
    the first treeAggregate pass inside LogisticRegression.train().

    MLlib's L-BFGS solver re-scans the RDD on every iteration anyway via
    treeAggregate — explicit caching provides no speed benefit for a
    single sequential fit call and only adds memory pressure.  The
    workers cache their (smaller) partitions safely because each sees
    only 1/N of the data.

    IMPORTANT: all JVM-backed model attributes (coefficients, intercept)
    must be materialised as plain Python objects BEFORE spark.stop().
    """
    from mpj_spark.config import TOTAL_CORES

    if baseline_threads is not None:
        thread_count = baseline_threads
    elif cores_override is not None:
        thread_count = cores_override
    else:
        thread_count = max(1, math.ceil(TOTAL_CORES / num_workers))

    heap_gb = _baseline_heap_gb(thread_count)

    # Parity-adjusted iteration count
    effective_iter = parity_iter if parity_iter is not None else max_iter
    parity_label   = (
        f'  [parity: {num_workers}×{max_iter}={parity_iter}]'
        if parity_iter is not None else ''
    )

    print(f'  [Baseline-LogReg] local[{thread_count}]  '
          f'max_iter={effective_iter}  reg_param={reg_param}  '
          f'[heap={heap_gb}g]{parity_label}')

    t_load_start = time.perf_counter()
    spark = (
        SparkSession.builder
        .appName('MPJ-Baseline-LogReg')
        .master(f'local[{thread_count}]')
        .config('spark.ui.enabled', 'false')
        .config('spark.sql.shuffle.partitions', str(thread_count * 2))
        .config('spark.driver.memory', f'{heap_gb}g')
        # Push more heap toward execution (gradient treeAggregate),
        # less toward storage (RDD cache) — consistent with no-cache strategy.
        .config('spark.memory.fraction', '0.8')
        .config('spark.memory.storageFraction', '0.2')
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
    # NOTE: intentionally NOT calling .cache() here — see docstring above.
    df_vec = assembler.transform(df).select('features', 'label')

    t_proc_start   = time.perf_counter()
    weight_vector  = None
    intercept_val  = None
    accuracy       = None
    oom_error      = None

    try:
        lr = LogisticRegression(
            featuresCol='features',
            labelCol='label',
            maxIter=effective_iter,
            regParam=reg_param,
            elasticNetParam=0.0,
            family='binomial',
            fitIntercept=True,
            standardization=True,
        )
        model         = lr.fit(df_vec)
        accuracy      = float(model.summary.accuracy)
        weight_norm   = float(model.coefficients.norm(2))
        # Materialise JVM-backed values into plain Python BEFORE spark.stop().
        weight_vector = model.coefficients.toArray().tolist()
        intercept_val = float(model.intercept)

        print(f'  [Baseline-LogReg] Accuracy={accuracy:.4f}  '
              f'|w|={weight_norm:.4f}  ({time.perf_counter() - t_proc_start:.3f}s)')

    except Exception as exc:
        oom_error = str(exc)[:200]
        print(f'\n  [Baseline-LogReg] [WARN] fit() failed — '
              f'comparison table will show N/A for baseline.\n'
              f'  Cause: {oom_error}\n'
              f'  Tip  : reduce --generate size, or set '
              f'SPARK_DRIVER_MEMORY={heap_gb + 2}g before running.\n')

    proc_time = time.perf_counter() - t_proc_start
    spark.stop()

    timing = {
        'load_time'       : load_time,
        'processing_time' : proc_time,
        'total_time'      : load_time + proc_time,
        'effective_iter'  : effective_iter,
        'parity_iter'     : parity_iter,
    }
    result = {
        'weight_vector' : weight_vector,
        'intercept'     : intercept_val,
        'accuracy'      : accuracy,
        'row_count'     : row_count,
        'oom_error'     : oom_error,
    }
    return result, timing
