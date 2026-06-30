# ================================================================
# mpj_spark/applications/kmeans.py
#
# K-Means MLlib pipeline — per-worker ML workload
#
# INIT STRATEGY when seed_centres are provided (Global Seed ON)
# ─────────────────────────────────────────────────────────────
# PySpark KMeans has no setInitialModel() in Python bindings.
# The Scala API exposes it but it is inaccessible from PySpark
# without Java reflection (which breaks across Spark versions).
#
# Default k-means|| uses initSteps=2 (default) extra candidate
# passes, meaning each worker scans the full partition ~3-4 times
# before fit() even begins. For a 1.4M-row partition this is the
# dominant cost (~10-15s of ~19s total worker time).
#
# Fix — Single-pass init (initSteps=1) + minimal anchors:
#   When seed_centres are available from Phase 1b global seeding:
#     1. Prepend exactly k synthetic anchor rows (one per centroid)
#        into the partition DataFrame. These bias k-means||
#        candidate sampling toward the known seed positions.
#     2. Set initSteps=1 (was: default 2). This collapses k-means||
#        init to a single oversampling pass instead of log(k)+1
#        passes. Combined with the anchors, the first pass reliably
#        selects near-seed positions.
#     3. initMode stays 'k-means||' — random init is non-
#        deterministic and provides no correctness guarantee.
#
#   When seed_centres are absent (global seeding OFF):
#     Unchanged behaviour — initMode='k-means||', initSteps=2,
#     no anchors.
#
# Init cost comparison:
#   Before: initSteps=2, 500 anchors/centroid → ~3-4 data passes
#   After:  initSteps=1, 1 anchor/centroid   → ~1 data pass
# ================================================================

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType

# One anchor per centroid is sufficient when initSteps=1:
# the single oversampling pass already concentrates candidates
# near the highest-weight (most isolated) points in the data,
# and with k anchors placed exactly at the seed positions they
# are the most isolated points from each other (max pairwise
# distance), so k-means|| selection is deterministic in practice.
_ANCHOR_WEIGHT = 1


def _inject_seed_anchors(spark, df_vec, seed_centres: list) -> object:
    """
    Return a new DataFrame = df_vec UNION (seed anchors).

    Prepends exactly k rows (one per seed centroid) so that
    k-means|| init with initSteps=1 reliably selects near-seed
    initial centres. The anchor rows are numerically negligible
    (k rows vs ~1.4M partition rows).
    """
    schema = StructType([StructField("features", VectorUDT(), False)])
    anchor_rows = [(Vectors.dense(list(c)),) for c in seed_centres for _ in range(_ANCHOR_WEIGHT)]
    df_anchors = spark.createDataFrame(anchor_rows, schema)
    return df_anchors.union(df_vec)  # anchors FIRST so they are sampled early


def run(
    partition_path: str,
    k: int = 3,
    max_iter: int = 20,
    seed: int = 42,
    seed_centres: list = None,
) -> dict:
    """
    K-Means clustering pipeline on a numeric CSV partition file.

    Parameters
    ----------
    partition_path : str         — absolute path to the worker's CSV partition
    k              : int         — number of clusters (default 3)
    max_iter       : int         — maximum KMeans iterations (default 20)
    seed           : int         — random seed (default 42)
    seed_centres   : list|None   — k×d list of floats from global seeding
                                   (Option 1). If None, uses standard k-means||.

    Returns
    -------
    dict:
        centres         : list[list[float]]  — cluster centroid coordinates
        wcss            : float              — within-cluster sum of squares
        k               : int                — clusters used
        row_count       : int                — rows processed (excl. anchors)
        partition_path  : str                — path for re-assignment pass
        used_global_seed: bool               — whether global seeding was used
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("[KMeans] No active SparkSession found in worker.")

    # ─ 1. Load CSV --------------------------------------------------------
    df_raw = spark.read.csv(partition_path, inferSchema=True, header=False)
    feature_cols = df_raw.columns
    num_features = len(feature_cols)
    df = df_raw.dropna()

    # ─ 2. Assemble feature vector -----------------------------------------
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    df_vec = assembler.transform(df).select("features")
    row_count = df_vec.count()  # real row count before any anchors

    # ─ 3. Choose init strategy based on seed availability -----------------
    use_global_seed = seed_centres is not None and len(seed_centres) == k

    print(
        f"[KMeans Worker] k={k} | max_iter={max_iter} | seed={seed} | "
        f"global_seed={'YES' if use_global_seed else 'NO'} | "
        f"rows={row_count:,}"
    )

    if use_global_seed:
        # Single-pass init: anchors first + initSteps=1
        df_train = _inject_seed_anchors(spark, df_vec, seed_centres)
        init_steps = 1
        print(
            f"[KMeans Worker] Single-pass init: {k} anchor rows prepended "
            f"(1 per centroid), initSteps=1  "
            f"[was: {k * 500} anchors, initSteps=2]"
        )
    else:
        # Standard k-means|| — no prior knowledge of centres
        df_train = df_vec
        init_steps = 2
        print("[KMeans Worker] Standard init: k-means|| initSteps=2 (no seed)")

    df_train = df_train.cache()

    # ─ 4. Build KMeans estimator ------------------------------------------
    kmeans_est = KMeans(
        k=k,
        maxIter=max_iter,
        seed=seed,
        featuresCol="features",
        initMode="k-means||",
        initSteps=init_steps,
    )

    # ─ 5. Fit -------------------------------------------------------------
    model = kmeans_est.fit(df_train)
    centres = [c.tolist() for c in model.clusterCenters()]
    wcss = float(model.summary.trainingCost)

    print(f"[KMeans Worker] WCSS = {wcss:.4f}")
    for i, c in enumerate(centres):
        preview = ", ".join(f"{v:.3f}" for v in c[:4])
        print(f"[KMeans Worker] C{i}: [{preview}{'...' if num_features > 4 else ''}]")

    df_train.unpersist()

    return {
        "centres": centres,
        "wcss": wcss,
        "k": k,
        "row_count": row_count,
        "partition_path": partition_path,
        "used_global_seed": use_global_seed,
    }
