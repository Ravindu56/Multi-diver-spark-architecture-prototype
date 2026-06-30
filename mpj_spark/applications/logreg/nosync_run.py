# =============================================================================
# mpj_spark/applications/logreg/nosync_run.py
#
# Benchmark Model M1 — Multi-Driver, NO Synchronisation
# ======================================================
# Each Spark driver fits a full LogisticRegression model on its own
# local partition independently.  There is NO cross-driver communication
# of any kind during training — no queues, no weight broadcast, no
# parameter server.
#
# After all workers finish, the root collects the final weight vectors
# and computes a row-weighted average (post-hoc FedAvg) as the global
# model.  This is the correct M1 condition for Objective 2d benchmarking:
#
#   B1  Single-driver Spark  local[N]             (baseline_logreg.py)
#   B2  Single-driver Spark  spark://master:7077  (baseline_logreg.py)
#   M1  Multi-driver, NO sync                     << THIS FILE >>
#   M2  Multi-driver, Queue/FedAvg per-iter sync  (queue_run.py)
#   M3  Multi-driver, MPI Allreduce               (allreduce.py)
#
# The only difference between M1 and M2 is synchronisation strategy.
# Hardware budget, dataset, worker count, and iteration count are
# identical — making the comparison architecturally controlled.
# =============================================================================

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

# ================================================================
# Helpers
# ================================================================


def _build_schema(num_features: int) -> StructType:
    fields = [StructField(f"f{i}", DoubleType(), nullable=True) for i in range(num_features)]
    fields.append(StructField("label", DoubleType(), nullable=True))
    return StructType(fields)


def _weight_norm(weights) -> float:
    return math.sqrt(sum(w * w for w in weights))


def _write_worker_metrics(worker_id: int, records: list, results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"worker_{worker_id}_nosync_logreg_metrics.csv")
    fieldnames = [
        "worker_id",
        "iteration",
        "iter_time_s",
        "weight_norm",
        "weight_delta",
        "intercept",
        "row_count",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


# ================================================================
# Main entry point
# ================================================================


def run(
    partition_path: str,
    max_iter: int = 10,
    reg_param: float = 0.01,
    num_features: int = 10,
    seed: int = 42,
    worker_id: int = 0,
    num_workers: int = 1,
    results_dir: str = "results",
    local_epochs: int = 5,
    # Accept (and ignore) queue kwargs so worker_process can call
    # nosync_run.run() and queue_run.run() with the same signature.
    allreduce_up_queue=None,
    allreduce_down_queue=None,
    allreduce_queue=None,
) -> dict:
    """
    M1: Independent local training — NO cross-driver synchronisation.

    Each worker runs max_iter rounds of full local MLlib LR fitting on
    its own partition.  Weights are NEVER shared between drivers during
    training.  The root post-processes the final weight vectors from all
    workers using row-weighted averaging.

    Parameters
    ----------
    partition_path : str
        Path to this worker's CSV partition.
    max_iter : int
        Number of independent local training rounds (no sync between rounds).
    reg_param : float
        L2 regularisation parameter.
    num_features : int
        Number of feature columns (must match the dataset).
    worker_id : int
        Zero-based worker index — used for logging and metrics output.
    local_epochs : int
        Number of MLlib LBFGS iterations per round.

    Returns
    -------
    dict with keys:
        weight_vector, intercept, train_accuracy, row_count,
        iterations_done, partition_path, iter_metrics
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(f"[NoSync Worker {worker_id}] No active SparkSession found.")

    # ── 1. Load partition ─────────────────────────────────────────────
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
        raise RuntimeError(
            f"[NoSync Worker {worker_id}] Partition is empty after cleaning. "
            f"Check --logreg-features matches the dataset "
            f"(expected {num_features} features)."
        )

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    df_vec = assembler.transform(df_base).select("features", "label").cache()
    df_vec.count()  # materialise cache

    print(
        f"[NoSync Worker {worker_id}] M1 — independent training  "
        f"reg_param={reg_param} | rounds={max_iter} | "
        f"local_epochs={local_epochs} | rows={row_count:,} | "
        f"features={num_features}  [NO SYNC]"
    )

    # ── 2. Independent training loop — zero cross-driver communication ──
    current_weights = None
    current_intercept = 0.0
    prev_weights_vec = np.zeros(num_features)
    iter_metrics = []

    for iteration in range(max_iter):
        t_iter = time.perf_counter()

        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=local_epochs,
            regParam=reg_param,
            elasticNetParam=0.0,
            family="binomial",
            fitIntercept=True,
            standardization=True,
        )

        model = lr.fit(df_vec)

        # No warm-start between rounds in M1 — each round is a fresh fit.
        # (Warm-start would implicitly carry information from prior rounds,
        # which could be confused with a form of self-synchronisation.)
        current_weights = model.coefficients.toArray().tolist()
        current_intercept = float(model.intercept)

        iter_time = time.perf_counter() - t_iter
        global_norm = _weight_norm(current_weights)
        cur_vec = np.array(current_weights)
        weight_delta = float(np.linalg.norm(cur_vec - prev_weights_vec))
        prev_weights_vec = cur_vec.copy()

        iter_metrics.append(
            {
                "worker_id": worker_id,
                "iteration": iteration + 1,
                "iter_time_s": round(iter_time, 6),
                "weight_norm": round(global_norm, 8),
                "weight_delta": round(weight_delta, 8),
                "intercept": round(current_intercept, 8),
                "row_count": row_count,
            }
        )

        print(
            f"[NoSync Worker {worker_id}] round {iteration + 1}/{max_iter}  "
            f"({iter_time:.3f}s)  |w|={global_norm:.4f}  "
            f"\u0394={weight_delta:.6f}  [no sync]"
        )

    # ── 3. Final accuracy on local partition ──────────────────────────
    if current_weights is not None:
        w_list = current_weights
        b_val = current_intercept

        from pyspark.sql.functions import udf as _udf
        from pyspark.sql.types import IntegerType as _IT

        @_udf(returnType=_IT())
        def predict(features):
            import math as _m

            arr = features.toArray()
            logit = sum(w_list[i] * arr[i] for i in range(len(w_list))) + b_val
            return int(1.0 / (1.0 + _m.exp(-logit)) >= 0.5)

        correct = (
            df_vec.withColumn("pred", predict(F.col("features")))
            .filter(F.col("pred") == F.col("label"))
            .count()
        )
        train_accuracy = correct / row_count if row_count > 0 else 0.0
        weight_vector = current_weights
        intercept_final = current_intercept
    else:
        train_accuracy = 0.0
        weight_vector = [0.0] * num_features
        intercept_final = 0.0

    print(
        f"[NoSync Worker {worker_id}] Final local accuracy : {train_accuracy:.4f}  "
        f"[no sync — local partition only]"
    )
    print(
        f"[NoSync Worker {worker_id}] Local weight norm    : " f"{_weight_norm(weight_vector):.4f}"
    )

    df_vec.unpersist()
    df_base.unpersist()

    # ── 4. Write per-worker metrics CSV ──────────────────────────────
    worker_csv_path = _write_worker_metrics(worker_id, iter_metrics, results_dir)
    print(
        f"[NoSync Worker {worker_id}] Iter metrics → {worker_csv_path} "
        f"({len(iter_metrics)} rows)"
    )

    return {
        "weight_vector": weight_vector,
        "intercept": intercept_final,
        "train_accuracy": train_accuracy,
        "row_count": row_count,
        "iterations_done": max_iter,
        "partition_path": partition_path,
        "iter_metrics": iter_metrics,
    }
