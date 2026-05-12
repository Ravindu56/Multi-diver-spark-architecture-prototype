# ================================================================
# mpj_spark/applications/kmeans.py
#
# K-Means MLlib pipeline — per-worker ML workload
#
# OPTION 1 — Global Seeding:
#   Root broadcasts k seed centroids via worker_config['seed_centres'].
#   Workers warm-start via _make_seed_model() which builds a valid
#   KMeansModel from those centroids using a pure-Python synthetic
#   DataFrame (no Java reflection, no _java_obj calls).
# ================================================================

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField


def _make_seed_model(spark, seed_centres: list, k: int):
    """
    Build a KMeansModel whose clusterCenters() are exactly `seed_centres`.

    Strategy: create a DataFrame of exactly k rows where row i = seed_centres[i]
    (each centroid is its own data point). Fitting KMeans(k, maxIter=1) on this
    degenerate k-row dataset forces the model to adopt those exact centroids,
    because with k distinct points there is one cluster per point and Lloyd's
    single pass leaves the centroids unchanged.

    No Java reflection, no _java_obj, no Scala converters required.
    Works on any PySpark 3.x version with local or cluster master.
    """
    schema = StructType([StructField('features', VectorUDT(), False)])
    # Create one row per seed centroid
    rows = [(Vectors.dense(list(c)),) for c in seed_centres]
    df_seed = spark.createDataFrame(rows, schema)

    seed_model = KMeans(
        k=k,
        maxIter=1,          # single pass on k rows → centroids = seed points
        seed=42,
        featuresCol='features',
        initMode='k-means||',
    ).fit(df_seed)

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
        raise RuntimeError('[KMeans] No active SparkSession found in worker.')

    # ─ 1. Load CSV --------------------------------------------------------
    df_raw       = spark.read.csv(partition_path, inferSchema=True, header=False)
    feature_cols = df_raw.columns
    num_features = len(feature_cols)
    df           = df_raw.dropna()

    # ─ 2. Assemble feature vector -----------------------------------------
    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol='features', handleInvalid='skip')
    df_vec = assembler.transform(df).select('features').cache()

    # ─ 3. Build KMeans estimator ------------------------------------------
    use_global_seed = seed_centres is not None and len(seed_centres) == k

    print(f'[KMeans Worker] k={k} | max_iter={max_iter} | seed={seed} | '
          f"global_seed={'YES' if use_global_seed else 'NO'} | loading...")

    if use_global_seed:
        seed_model = _make_seed_model(spark, seed_centres, k)
        kmeans_est = KMeans(
            k=k, maxIter=max_iter, seed=seed,
            featuresCol='features', initMode='random',
        )
        kmeans_est.setInitialModel(seed_model)
    else:
        kmeans_est = KMeans(
            k=k, maxIter=max_iter, seed=seed,
            featuresCol='features', initMode='k-means||',
        )

    # ─ 4. Fit -------------------------------------------------------------
    model     = kmeans_est.fit(df_vec)
    centres   = [c.tolist() for c in model.clusterCenters()]
    wcss      = float(model.summary.trainingCost)
    row_count = model.summary.predictions.count()

    print(f'[KMeans Worker] Rows: {row_count:,} | k={k} | max_iter={max_iter}')
    print(f'[KMeans Worker] WCSS = {wcss:.4f}')
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
