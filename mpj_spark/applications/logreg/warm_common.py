"""mpj_spark/applications/logreg/warm_common.py
Shared building blocks for the warm-started LogReg sync modes
(gossip, async parameter server, hybrid PS+Allreduce).

PySpark's LogisticRegression has no initial-weights setter (PySpark 4.2.0), so
these modes used to refit from zero every round inside a swallowed
``try/except``.  They now share one local step: warm-started L-BFGS-B
(mpj_spark.core.warm_start_lr) on an RDD of (features, label) rows.
Same objective in every mode: mean log-loss + 0.5*reg*||w||^2, no feature
standardisation - compare modes by loss, not by weight norm.
"""

from __future__ import annotations

import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from mpj_spark.core.warm_start_lr import warm_lbfgs

SOLVER = "warm_lbfgs"


def load_points(spark, partition_path: str, num_features: int, who: str):
    """Read + clean one partition; return (df_base, points_rdd, row_count).

    ``points_rdd`` holds (features ndarray, label float) pairs, cached.
    """
    from mpj_spark.applications.logreg.fedavg_mpi_run import _build_schema

    schema = _build_schema(num_features)
    df_raw = spark.read.csv(partition_path, schema=schema, header=False)
    df_base = (
        df_raw.filter(F.col("f0").isNotNull())
        .dropna()
        .withColumn("label", F.col("label").cast(IntegerType()))
        .cache()
    )
    feature_cols = [f"f{i}" for i in range(num_features)]
    row_count = df_base.count()
    if row_count == 0:
        raise RuntimeError(f"[{who}] Partition is empty after cleaning: {partition_path}")

    nf = num_features
    points = (
        df_base.select(*feature_cols, "label")
        .rdd.map(lambda row: (np.asarray(row[:nf], dtype=np.float64), float(row[nf])))
        .cache()
    )
    points.count()
    return df_base, points, row_count


def local_step(loss_grad, weights, intercept, num_features: int, local_epochs: int):
    """Warm-started local solve. Returns (weights ndarray, intercept, info, weight_norm)."""
    w, b, info = warm_lbfgs(loss_grad, weights, intercept, num_features, local_epochs)
    return w, b, info, float(np.linalg.norm(w))


def model_loss(loss_grad, weights, intercept) -> float:
    """Objective of (weights, intercept) on this worker's partition (one Spark pass)."""
    theta = np.concatenate([np.asarray(weights, dtype=np.float64), [float(intercept)]])
    return float(loss_grad(theta)[0])
