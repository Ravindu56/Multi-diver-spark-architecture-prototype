# ================================================================
# mpj_spark/applications/kmeans.py
#
# K-Means MLlib pipeline — per-worker ML workload
#
# Research Objective 1:
#   "Adopt state-of-the-art architecture for Machine Learning
#    workload" — each worker trains K-Means on its partition.
#
# Architecture
# ------------
# Each worker receives its CSV partition as a PySpark text RDD.
# Trains a local KMeans model using PySpark MLlib on that partition.
# Returns centres, WCSS (inertia), k, row_count.
# Root aggregates via weighted centroid average (root_process.py).
#
# Why WCSS?
#   WCSS proves multi-driver achieves EQUIVALENT ML quality
#   to single-driver Spark — key thesis claim for Objective 1.
#
# FIXES APPLIED:
#   FIX 1 — Replace Python UDF parse_row() + createDataFrame(rdd)
#            with spark.read.csv() — fully native JVM columnar reader.
#            Eliminates Python<->JVM serialisation on every row.
#
#   FIX 2 — Remove standalone df_vec.count() before model.fit().
#            The extra count() action forced a full DAG scan just to
#            count rows, triggering the BlockManager lock warnings.
#            row_count is now read from model.summary after fitting.
#
#   FIX 3 (Issue #1) — Reproducibility: fixed seed=42.
#            With initMode='random' and seed=42 the initialisation is
#            fully deterministic and has zero multi-pass RDD sampling
#            overhead. 'k-means||' runs multiple distributed sampling
#            passes before fitting — measurable overhead on the small
#            ~50 MB partitions used in this benchmark. 'random' with
#            a fixed seed is the correct choice for partitioned data.
#
#   FIX 6 (Issue #6) — Changed initMode from 'k-means||' back to
#            'random'. k-means|| performs log(k) rounds of distributed
#            RDD sampling before the actual fit() begins. On partitions
#            of ~50 MB this sampling overhead is disproportionately
#            large relative to the fit time. 'random' with seed=42
#            is deterministic, zero-overhead, and appropriate for
#            benchmarking reproducibility on partitioned datasets.
# ================================================================

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


def run(partition_path: str, k: int = 3, max_iter: int = 20, seed: int = 42) -> dict:
    """
    K-Means clustering pipeline on a numeric CSV partition file.

    Parameters
    ----------
    partition_path : str   — absolute path to the worker's CSV partition file
    k              : int   — number of clusters (default 3)
    max_iter       : int   — maximum KMeans iterations (default 20)
    seed           : int   — random seed for reproducibility (default 42)

    Returns
    -------
    dict:
        centres   : list[list[float]]  — cluster centroid coordinates
        wcss      : float              — within-cluster sum of squares (inertia)
        k         : int                — clusters used
        row_count : int                — rows processed in this partition
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("[KMeans] No active SparkSession found in worker.")

    # -- 1. Load CSV directly via native Spark reader ---------------------
    # FIX 1: spark.read.csv stays entirely inside the JVM tungsten engine.
    # No Python UDF, no RDD<->DataFrame conversion overhead.
    df_raw = spark.read.csv(
        partition_path,
        inferSchema=True,   # infers column types in a single JVM pass
        header=False,       # partition files have no header
    )

    # Column names from inferSchema are _c0, _c1, ..., _cN
    feature_cols = df_raw.columns  # e.g. ['_c0', '_c1', ..., '_c9']
    num_features = len(feature_cols)

    # Drop any rows that contain nulls
    df = df_raw.dropna()

    # -- 2. Assemble feature vector column --------------------------------
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol='features',
        handleInvalid='skip',
    )
    df_vec = assembler.transform(df).select('features').cache()

    # -- 3. Train KMeans model --------------------------------------------
    # FIX 2: Do NOT call df_vec.count() here.
    # row_count is retrieved from model.summary after a single fit() pass.
    #
    # FIX 6 (Issue #6): Use initMode='random' with seed=42.
    # 'k-means||' runs log(k) rounds of distributed RDD sampling passes
    # before fit() begins. On ~50 MB partitions this sampling overhead
    # is disproportionately large. 'random' with a fixed seed is:
    #   - Fully deterministic (reproducible results across runs)
    #   - Zero init overhead (single pass, no extra RDD actions)
    #   - Appropriate for pre-partitioned benchmark datasets where
    #     each partition already has a balanced cluster distribution.
    effective_k = k

    print(f"[KMeans Worker] k={effective_k} | max_iter={max_iter} | seed={seed} | loading...")

    model = KMeans(
        k=effective_k,
        maxIter=max_iter,
        seed=seed,
        featuresCol='features',
        initMode='random',  # FIX 6: zero-overhead, deterministic with fixed seed
    ).fit(df_vec)

    # -- 4. Extract results -----------------------------------------------
    centres   = [c.tolist() for c in model.clusterCenters()]
    wcss      = float(model.summary.trainingCost)
    row_count = model.summary.predictions.count()  # cheap: already materialised

    print(f"[KMeans Worker] Rows: {row_count:,} | k={effective_k} | max_iter={max_iter}")
    print(f"[KMeans Worker] WCSS = {wcss:.4f}")
    for i, c in enumerate(centres):
        preview = ', '.join(f'{v:.3f}' for v in c[:4])
        print(f"[KMeans Worker] C{i}: [{preview}{'...' if num_features > 4 else ''}]")

    df_vec.unpersist()

    return {
        'centres'  : centres,
        'wcss'     : wcss,
        'k'        : effective_k,
        'row_count': row_count,
    }
