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
# Root aggregates via gossip (root_process.py).
#
# Why WCSS?
#   WCSS proves multi-driver achieves EQUIVALENT ML quality
#   to single-driver Spark — key thesis claim for Objective 1.
#
# FIXES APPLIED:
#   FIX 1 — Replace Python UDF parse_row() + createDataFrame(rdd)
#            with spark.read.csv() — fully native JVM columnar reader.
#
#   FIX 2 — Remove standalone df_vec.count() before model.fit().
#
#   FIX 3 / FIX 6 — Use initMode='k-means||' as default for robust
#            distributed seeding at large partition sizes.
#
# OPTION 1 — Global Seeding (correctness improvement):
#   Root samples ~5% of full dataset, runs lightweight KMeans to
#   obtain k global seed centroids, broadcasts them via worker_config.
#   Workers receive seed_centres and warm-start via setInitialModel().
#   All workers begin from the SAME initial centroid positions —
#   the dominant source of post-aggregation divergence is eliminated.
#   Falls back to standard k-means|| when seed_centres is None.
# ================================================================

import numpy as np
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField


def _make_seed_model(spark, seed_centres: list, k: int) -> KMeansModel:
    """
    Build a minimal KMeansModel from pre-computed seed centroids so that
    KMeans.setInitialModel() can be used for warm-starting.

    Spark MLlib's setInitialModel() accepts a KMeansModel whose
    clusterCenters() are used directly as the first iteration's centroids,
    bypassing the k-means|| or random init pass entirely.
    """
    schema = StructType([StructField("features", VectorUDT(), False)])
    rows   = [(Vectors.dense(c),) for c in seed_centres]
    df     = spark.createDataFrame(rows, schema)

    # Fit a trivial 1-iteration model to get a valid KMeansModel shell
    seed_model = KMeans(
        k=k,
        maxIter=1,
        seed=42,
        featuresCol='features',
        initMode='random',
    ).fit(df)

    # Patch clusterCenters directly via the Java object
    centres_java = [Vectors.dense(c)._java_obj for c in seed_centres]
    seed_model._java_obj.setCenters(
        spark._jvm.scala.collection.JavaConverters
             .asScalaIteratorConverter(iter(centres_java))
             .asScala().toArray(
                 spark._jvm.scala.reflect.ClassTag
                      .apply(spark._jvm.org.apache.spark.ml.linalg.Vector)
             )
    )
    return seed_model


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
                                   (Option 1). If None, uses k-means||.

    Returns
    -------
    dict:
        centres         : list[list[float]]  — cluster centroid coordinates
        wcss            : float              — within-cluster sum of squares
        k               : int                — clusters used
        row_count       : int                — rows processed
        partition_path  : str                — path for re-assignment pass
        used_global_seed: bool               — whether global seeding was used
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("[KMeans] No active SparkSession found in worker.")

    # -- 1. Load CSV via native Spark reader ------------------------------
    df_raw = spark.read.csv(
        partition_path,
        inferSchema=True,
        header=False,
    )
    feature_cols = df_raw.columns
    num_features = len(feature_cols)
    df           = df_raw.dropna()

    # -- 2. Assemble feature vector column --------------------------------
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol='features',
        handleInvalid='skip',
    )
    df_vec = assembler.transform(df).select('features').cache()

    # -- 3. Build KMeans estimator ----------------------------------------
    use_global_seed = seed_centres is not None and len(seed_centres) == k

    print(f"[KMeans Worker] k={k} | max_iter={max_iter} | seed={seed} | "
          f"global_seed={'YES' if use_global_seed else 'NO'} | loading...")

    if use_global_seed:
        # Option 1: warm-start from broadcast global centroids.
        # initMode='random' with setInitialModel() makes Spark skip its own
        # init pass and use our pre-set centres as iteration-0 centroids.
        kmeans_est = KMeans(
            k=k,
            maxIter=max_iter,
            seed=seed,
            featuresCol='features',
            initMode='random',   # required when using setInitialModel()
        )
        seed_model = _make_seed_model(spark, seed_centres, k)
        kmeans_est.setInitialModel(seed_model)
    else:
        # Fallback: standard k-means|| seeding (robust at large partitions)
        kmeans_est = KMeans(
            k=k,
            maxIter=max_iter,
            seed=seed,
            featuresCol='features',
            initMode='k-means||',
        )

    # -- 4. Fit -----------------------------------------------------------
    model     = kmeans_est.fit(df_vec)
    centres   = [c.tolist() for c in model.clusterCenters()]
    wcss      = float(model.summary.trainingCost)
    row_count = model.summary.predictions.count()

    print(f"[KMeans Worker] Rows: {row_count:,} | k={k} | max_iter={max_iter}")
    print(f"[KMeans Worker] WCSS = {wcss:.4f}")
    for i, c in enumerate(centres):
        preview = ', '.join(f'{v:.3f}' for v in c[:4])
        print(f"[KMeans Worker] C{i}: [{preview}{'...' if num_features > 4 else ''}]")

    df_vec.unpersist()

    return {
        'centres'         : centres,
        'wcss'            : wcss,
        'k'               : k,
        'row_count'       : row_count,
        'partition_path'  : partition_path,
        'used_global_seed': use_global_seed,
    }
