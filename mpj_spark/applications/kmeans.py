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
# ================================================================

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


def run(text_rdd, k: int = 3, max_iter: int = 20, seed: int = 42) -> dict:
    """
    K-Means clustering pipeline on a numeric CSV partition RDD.

    Parameters
    ----------
    text_rdd : pyspark.rdd.RDD  — CSV lines (no header) from worker partition
    k        : int              — number of clusters (default 3)
    max_iter : int              — maximum KMeans iterations (default 20)
    seed     : int              — random seed for reproducibility (default 42)

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

    # ── 1. Infer feature count from first row ─────────────────────────
    first_row    = text_rdd.first()
    num_features = len(first_row.strip().split(','))
    feature_cols = [f'f{i}' for i in range(num_features)]

    # ── 2. Parse CSV RDD → Spark DataFrame ───────────────────────────
    def parse_row(line):
        from pyspark.sql import Row
        try:
            vals = [float(x) for x in line.strip().split(',') if x.strip()]
            if len(vals) != num_features:
                return None
            return Row(**dict(zip(feature_cols, vals)))
        except ValueError:
            return None

    parsed_rdd = text_rdd.map(parse_row).filter(lambda r: r is not None)
    df         = spark.createDataFrame(parsed_rdd)

    # ── 3. Assemble feature vector column ────────────────────────────
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol='features',
        handleInvalid='skip'
    )
    df_vec    = assembler.transform(df).select('features').cache()
    row_count = df_vec.count()

    if row_count == 0:
        raise ValueError("[KMeans] Worker partition is empty after parsing.")

    effective_k = min(k, row_count)
    print(f"[KMeans Worker] Rows: {row_count:,} | k={effective_k} | max_iter={max_iter}")

    # ── 4. Train KMeans model ────────────────────────────────────────
    model = KMeans(
        k=effective_k,
        maxIter=max_iter,
        seed=seed,
        featuresCol='features',
        initMode='k-means||',
    ).fit(df_vec)

    # ── 5. Extract results ───────────────────────────────────────────
    centres = [c.tolist() for c in model.clusterCenters()]
    wcss    = float(model.summary.trainingCost)

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