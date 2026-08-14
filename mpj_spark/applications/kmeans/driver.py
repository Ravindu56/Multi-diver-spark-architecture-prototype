# =============================================================================
# mpj_spark/applications/kmeans/driver.py
#
# Thin facade exposing run_kmeans_driver() — the interface expected by
# scripts/validate_parity.py (Issue #10).
#
# Wraps run_kmeans_allreduce() and normalises its return dict into the
# parity-check contract:
#   {'centres': list[list[float]], 'wcss': float}
#
# INIT STRATEGY when seed_centres are provided (Global Seed ON)
# ─────────────────────────────────────────────────────────────
# Fix for P2-03/P2-04 (non-deterministic K-Means, centroid label misalignment):
#   1. Prepend exactly k synthetic anchor rows (one per seed centroid)
#   2. Set initSteps=1 (was: default 2) — single oversampling pass
#   3. k-means|| selection is deterministic with anchors at seed positions
#
# When seed_centres are absent: standard k-means||, initSteps=2, no anchors.
# =============================================================================
from __future__ import annotations

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType

_ANCHOR_WEIGHT = 1


def _inject_seed_anchors(spark, df_vec, seed_centres: list):
    """
    Return a new DataFrame = df_vec UNION (seed anchors).

    Prepends exactly k rows (one per seed centroid) so that
    k-means|| init with initSteps=1 reliably selects near-seed
    initial centres.
    """
    schema = StructType([StructField("features", VectorUDT(), False)])
    anchor_rows = [(Vectors.dense(list(c)),) for c in seed_centres for _ in range(_ANCHOR_WEIGHT)]
    df_anchors = spark.createDataFrame(anchor_rows, schema)
    return df_anchors.union(df_vec)


def _run_local_kmeans(
    partition_path: str, k: int, max_iter: int, seed: int = 42, seed_centres=None
) -> dict:
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession found for local K-Means fallback.")

    df_raw = spark.read.csv(partition_path, inferSchema=True, header=False)
    feature_cols = df_raw.columns
    df = df_raw.dropna()

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip",
    )
    df_vec = assembler.transform(df).select("features")
    row_count = df_vec.count()

    # Choose init strategy based on seed availability
    use_global_seed = seed_centres is not None and len(seed_centres) == k

    if use_global_seed:
        # Single-pass init: anchors first + initSteps=1
        df_train = _inject_seed_anchors(spark, df_vec, seed_centres)
        init_steps = 1
    else:
        # Standard k-means|| — no prior knowledge of centres
        df_train = df_vec
        init_steps = 2

    df_train = df_train.cache()

    model = KMeans(
        k=k,
        maxIter=max_iter,
        seed=seed,
        featuresCol="features",
        initMode="k-means||",
        initSteps=init_steps,
    ).fit(df_train)

    centres = [c.tolist() for c in model.clusterCenters]
    wcss = float(model.summary.trainingCost)
    df_train.unpersist()
    return {"centres": centres, "wcss": wcss, "row_count": row_count}


def run_kmeans_driver(
    rank: int | None = None,
    size: int | None = None,
    comm=None,
    dataset_path: str | None = None,
    partition_path: str | None = None,
    worker_id: int | None = None,
    num_workers: int | None = None,
    k: int = 3,
    max_iter: int = 20,
    tol: float = 1e-4,
    seed: int = 42,
    metrics_output_dir: str = "./kmeans_results",
    gossip_queue=None,
    reassign_queue=None,
    seed_centres=None,
) -> dict:
    """
    Parity-check facade for the multi-driver K-Means Allreduce runner.

    Parameters
    ----------
    rank, size, comm : MPI rank, world size, and communicator.
    dataset_path     : Shared-storage path to the input CSV (all ranks must
                       be able to read this path).
    k                : Number of clusters.
    max_iter         : Maximum iterations.
    tol              : Centroid-shift convergence tolerance.
    seed             : Random seed for centroid initialisation.
    metrics_output_dir : Directory for per-rank metrics CSV/JSON output.

    Returns  (all ranks return the same dict after root Bcast)
    -------
    {
        'centres' : list[list[float]]   # global centroids  (k x d)
        'wcss'    : float               # final global WCSS / inertia
    }
    """
    actual_dataset_path = dataset_path or partition_path
    actual_rank = rank if rank is not None else worker_id if worker_id is not None else 0
    actual_size = size if size is not None else num_workers if num_workers is not None else 1

    if comm is None:
        local_result = _run_local_kmeans(
            actual_dataset_path, k=k, max_iter=max_iter, seed=seed, seed_centres=seed_centres
        )
        return {
            "centres": local_result["centres"],
            "wcss": float(local_result["wcss"]),
        }

    from mpj_spark.applications.kmeans.allreduce import run_kmeans_allreduce

    raw = run_kmeans_allreduce(
        comm=comm,
        rank=actual_rank,
        size=actual_size,
        input_file=actual_dataset_path,
        k=k,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        metrics_output_dir=metrics_output_dir,
    )

    result = {
        "centres": raw.get("global_centroids", []),
        "wcss": float(raw.get("run_summary", {}).get("final_wcss", 0.0) or _extract_last_wcss(raw)),
    }

    if hasattr(comm, "bcast"):
        result = comm.bcast(result, root=0)
    return result


def _extract_last_wcss(raw: dict) -> float:
    """Fallback: pull WCSS from the last row of the per-iteration metrics."""
    metrics = raw.get("metrics", [])
    if metrics:
        last = metrics[-1]
        # KMeansMetricsCollector columns: iteration, spark_s, sync_s, iter_s,
        # centroid_shift, global_wcss
        return float(last.get("global_wcss", 0.0))
    return 0.0
