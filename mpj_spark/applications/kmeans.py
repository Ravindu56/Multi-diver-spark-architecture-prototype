# ================================================================
# mpj_spark/applications/kmeans.py
#
# K-Means MLlib pipeline — per-worker ML workload
#
# OPTION 1 — Global Seeding (pure-Python, no Java reflection):
#
#   Problem: PySpark's KMeans estimator does NOT expose setInitialModel()
#   (that is Scala-only). clusterCenters on a fitted model is also
#   read-only. There is no direct API to inject starting centroids.
#
#   Solution — Weighted Anchor Rows:
#     Prepend ANCHOR_WEIGHT (500) copies of each seed centroid as
#     synthetic rows into the partition DataFrame before fit().
#     k-means|| selects initial centres proportional to squared
#     distance from already-selected centres. The heavily-weighted
#     anchor points dominate this sampling, so k-means|| reliably
#     picks the broadcast seed centroids as its k initial centres.
#     Anchors are ~500 rows vs ~1.4M partition rows (≈1:2800 ratio),
#     so they are numerically negligible in the final centroid update.
#
#   After convergence, all workers have started from the same
#   initial positions — eliminating the dominant source of
#   cross-worker divergence before gossip aggregation.
# ================================================================

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField

# How many times each seed centroid is repeated as an anchor row.
# 500 is large enough to dominate k-means|| sampling but negligible
# as a fraction of a ~1.4M-row partition (ratio ~1:2800).
_ANCHOR_WEIGHT = 500


def _inject_seed_anchors(spark, df_vec, seed_centres: list) -> object:
    """
    Return a new DataFrame = df_vec UNION (seed anchors).

    Each seed centroid is repeated _ANCHOR_WEIGHT times so that
    k-means|| init sampling is biased toward those positions.
    The anchor rows share the same schema as df_vec ('features' column).
    """
    schema = StructType([StructField('features', VectorUDT(), False)])
    anchor_rows = [
        (Vectors.dense(list(c)),)
        for c in seed_centres
        for _ in range(_ANCHOR_WEIGHT)
    ]
    df_anchors = spark.createDataFrame(anchor_rows, schema)
    return df_vec.union(df_anchors)


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
                                   (Option 1). If None, uses k-means|| only.

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
        raise RuntimeError('[KMeans] No active SparkSession found in worker.')

    # ─ 1. Load CSV --------------------------------------------------------
    df_raw       = spark.read.csv(partition_path, inferSchema=True, header=False)
    feature_cols = df_raw.columns
    num_features = len(feature_cols)
    df           = df_raw.dropna()

    # ─ 2. Assemble feature vector -----------------------------------------
    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol='features', handleInvalid='skip')
    df_vec    = assembler.transform(df).select('features')
    row_count = df_vec.count()   # real row count before any anchors

    # ─ 3. Optionally inject seed anchor rows (Option 1) -------------------
    use_global_seed = seed_centres is not None and len(seed_centres) == k

    print(f'[KMeans Worker] k={k} | max_iter={max_iter} | seed={seed} | '
          f"global_seed={'YES' if use_global_seed else 'NO'} | "
          f'rows={row_count:,} | loading...')

    if use_global_seed:
        df_train = _inject_seed_anchors(spark, df_vec, seed_centres)
        print(f'[KMeans Worker] Injected {k * _ANCHOR_WEIGHT:,} anchor rows '
              f'({_ANCHOR_WEIGHT}× per centroid) to bias k-means|| init')
    else:
        df_train = df_vec

    df_train = df_train.cache()

    # ─ 4. Build KMeans estimator (always k-means|| — PySpark has no
    #       setInitialModel() in Python bindings) --------------------------
    kmeans_est = KMeans(
        k=k,
        maxIter=max_iter,
        seed=seed,
        featuresCol='features',
        initMode='k-means||',
    )

    # ─ 5. Fit -------------------------------------------------------------
    model   = kmeans_est.fit(df_train)
    centres = [c.tolist() for c in model.clusterCenters()]
    wcss    = float(model.summary.trainingCost)

    print(f'[KMeans Worker] WCSS = {wcss:.4f}')
    for i, c in enumerate(centres):
        preview = ', '.join(f'{v:.3f}' for v in c[:4])
        print(f"[KMeans Worker] C{i}: [{preview}{'...' if num_features > 4 else ''}]")

    df_train.unpersist()

    return {
        'centres'         : centres,
        'wcss'            : wcss,
        'k'               : k,
        'row_count'       : row_count,
        'partition_path'  : partition_path,
        'used_global_seed': use_global_seed,
    }
