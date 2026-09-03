# ================================================================
# mpj_spark/applications/baseline_logreg.py
#
# Single-driver LogisticRegression baseline for --compare mode.
# Mirrors baseline_kmeans.py structure.
# ================================================================
import math
import time

try:
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import SparkSession
    from pyspark.sql.types import DoubleType, StructField, StructType
except ImportError:  # pragma: no cover
    SparkSession = None
    LogisticRegression = None
    VectorAssembler = None
    StructType = StructField = DoubleType = None


def _sniff_csv_header(input_file: str) -> tuple[bool, int]:
    """
    Peek at the first non-blank line of *input_file* to decide whether
    the CSV has a named header row.

    Returns
    -------
    has_header : bool  — True when the first line contains non-numeric text
    n_cols     : int   — total number of comma-separated columns in that line
                         (features + 1 label column)

    Strategy
    --------
    Try to parse every token in the first line as float.
    - All-float  → headerless data row  → has_header=False
    - Any non-float → named header row  → has_header=True
    """
    with open(input_file, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split(",")
            all_numeric = all(_is_float(tok) for tok in tokens)
            return (not all_numeric), len(tokens)
    raise RuntimeError(f"Cannot sniff CSV header from '{input_file}': file appears empty.")


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _baseline_heap_gb(thread_count: int) -> int:
    """
    Compute a safe driver heap size for the baseline Spark session.

    Formula: 512 MB base + 256 MB per thread, rounded up to the next
    integer GB, capped at 80% of system RAM, minimum 2 GB.
    """
    try:
        import psutil

        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        cap_gb = max(2, int(total_ram_gb * 0.80))
    except ImportError:
        cap_gb = 8
    raw_gb = math.ceil(0.5 + 0.25 * thread_count)
    return min(max(2, raw_gb), cap_gb)


def _executor_memory_gb(num_workers: int) -> int:
    """
    Per-executor heap for a standalone-cluster baseline.

    Mirrors build_spark_session() logic: allocate 75% of total RAM
    divided equally across workers, floored at 2 GB.
    """
    try:
        import psutil

        total_ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        total_ram_gb = 16.0
    per_worker_gb = max(2, int(total_ram_gb * 0.75 / num_workers))
    return per_worker_gb


def run_baseline_logreg(
    input_file: str,
    num_workers: int,
    cores_override,
    max_iter: int = 10,
    reg_param: float = 0.01,
    num_features: int = 10,
    baseline_threads: int = None,
    parity_iter: int = None,
    baseline_master: str = None,
):
    """
    Single-driver Spark LogisticRegression baseline for --compare mode.

    baseline_master
    ---------------
    When None (default), the session uses local[N] — suitable for
    single-machine development but NOT a fair academic benchmark.

    For fair comparison against the multi-driver Docker cluster, pass
    a Spark standalone master URL::

        baseline_master="spark://spark-master:7077"

    In standalone mode the session is configured with:
        spark.executor.instances = num_workers
        spark.executor.cores     = cores_per_worker
        spark.executor.memory    = heap_per_worker

    This gives the baseline exactly the same hardware budget as the
    multi-driver framework (same worker count, same cores, same RAM),
    so the only variable is the execution architecture.

    parity_iter
    -----------
    When provided, overrides max_iter so the baseline performs the same
    total number of gradient steps as the multi-driver framework:

        parity_iter = num_workers × logreg_iter

    Header detection
    ----------------
    Auto-detects whether the input CSV has a named header row by peeking
    at the first line (_sniff_csv_header).  Robust to both header=True
    and header=False datasets.

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

    # Parity-adjusted iteration count
    effective_iter = parity_iter if parity_iter is not None else max_iter
    parity_label = (
        f"  [parity: {num_workers}×{max_iter}={parity_iter}]" if parity_iter is not None else ""
    )

    # ------------------------------------------------------------------ #
    # Detect whether the CSV has a named header row BEFORE starting Spark #
    # ------------------------------------------------------------------ #
    has_header, n_cols = _sniff_csv_header(input_file)

    # ------------------------------------------------------------------ #
    # Build SparkSession — local[N] for dev, standalone for benchmark     #
    # ------------------------------------------------------------------ #
    use_standalone = baseline_master is not None and baseline_master.strip().startswith(
        ("spark://", "yarn", "k8s://")
    )

    if use_standalone:
        executor_mem_gb = _executor_memory_gb(num_workers)
        master_url = baseline_master.strip()
        mode_tag = f"standalone({master_url})"
        print(
            f"  [Baseline-LogReg] {mode_tag}  "
            f"executors={num_workers}  cores/exec={thread_count}  "
            f"mem/exec={executor_mem_gb}g  "
            f"max_iter={effective_iter}  reg_param={reg_param}"
            f"{parity_label}"
        )
        builder = (
            SparkSession.builder.appName("MPJ-Baseline-LogReg")
            .master(master_url)
            .config("spark.ui.enabled", "false")
            # Match multi-driver worker budget exactly
            .config("spark.executor.instances", str(num_workers))
            .config("spark.executor.cores", str(thread_count))
            .config("spark.executor.memory", f"{executor_mem_gb}g")
            .config("spark.sql.shuffle.partitions", str(num_workers * thread_count * 2))
            .config("spark.memory.fraction", "0.8")
            .config("spark.memory.storageFraction", "0.2")
        )
    else:
        # local[N] — single-machine dev / single-node benchmark
        heap_gb = _baseline_heap_gb(thread_count)
        mode_tag = f"local[{thread_count}]"
        print(
            f"  [Baseline-LogReg] {mode_tag}  "
            f"max_iter={effective_iter}  reg_param={reg_param}  "
            f"[heap={heap_gb}g]{parity_label}"
        )
        builder = (
            SparkSession.builder.appName("MPJ-Baseline-LogReg")
            .master(f"local[{thread_count}]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", str(thread_count * 2))
            .config("spark.driver.memory", f"{heap_gb}g")
            .config("spark.memory.fraction", "0.8")
            .config("spark.memory.storageFraction", "0.2")
        )

    t_load_start = time.perf_counter()
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # ------------------------------------------------------------------ #
    # Load dataset with correct header mode                               #
    # ------------------------------------------------------------------ #
    if has_header:
        df_raw = spark.read.csv(input_file, inferSchema=True, header=True)
        df = df_raw.dropna()
        feature_cols = [c for c in df.columns if c != "label"]
    else:
        n_features = n_cols - 1
        schema_fields = [StructField(f"f{i}", DoubleType(), True) for i in range(n_features)]
        schema_fields.append(StructField("label", DoubleType(), True))
        schema = StructType(schema_fields)
        df_raw = spark.read.csv(input_file, schema=schema, header=False)
        df = df_raw.dropna()
        feature_cols = [f"f{i}" for i in range(n_features)]

    row_count = df.count()
    load_time = time.perf_counter() - t_load_start
    header_mode = "with header" if has_header else "headerless (schema synthesised)"
    print(f"  [Baseline-LogReg] {row_count:,} rows loaded  ({load_time:.3f}s)  [{header_mode}]")

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    # NOTE: intentionally NOT calling .cache() — see module docstring.
    df_vec = assembler.transform(df).select("features", "label")

    t_proc_start = time.perf_counter()
    weight_vector = None
    intercept_val = None
    accuracy = None
    oom_error = None

    try:
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=effective_iter,
            regParam=reg_param,
            elasticNetParam=0.0,
            family="binomial",
            fitIntercept=True,
            standardization=True,
        )
        model = lr.fit(df_vec)
        accuracy = float(model.summary.accuracy)
        weight_norm = float(model.coefficients.norm(2))
        # Materialise JVM-backed values into plain Python BEFORE spark.stop().
        weight_vector = model.coefficients.toArray().tolist()
        intercept_val = float(model.intercept)

        print(
            f"  [Baseline-LogReg] Accuracy={accuracy:.4f}  "
            f"|w|={weight_norm:.4f}  "
            f"({time.perf_counter() - t_proc_start:.3f}s)"
        )

    except Exception as exc:
        oom_error = str(exc)[:200]
        hint = (
            f"SPARK_EXECUTOR_MEMORY={executor_mem_gb + 2}g"
            if use_standalone
            else f"SPARK_DRIVER_MEMORY={_baseline_heap_gb(thread_count) + 2}g"
        )
        print(
            f"\n  [Baseline-LogReg] [WARN] fit() failed — "
            f"comparison table will show N/A for baseline.\n"
            f"  Cause: {oom_error}\n"
            f"  Tip  : {hint}\n"
        )

    proc_time = time.perf_counter() - t_proc_start
    spark.stop()

    timing = {
        "load_time": load_time,
        "processing_time": proc_time,
        "total_time": load_time + proc_time,
        "effective_iter": effective_iter,
        "parity_iter": parity_iter,
        "mode": mode_tag,
    }
    result = {
        "weight_vector": weight_vector,
        "intercept": intercept_val,
        "accuracy": accuracy,
        "row_count": row_count,
        "oom_error": oom_error,
    }
    return result, timing
