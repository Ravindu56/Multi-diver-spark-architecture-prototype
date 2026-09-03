"""mpj_spark/applications/logreg/fedavg_mpi_run.py
Phase 3 - LogReg Periodic FedAvg over Native MPI Collectives.

Implements P3-08 (Issue #65):
Runs periodic FedAvg parameter synchronization over native mpi4py
collectives (comm.gather / comm.bcast), replacing multiprocessing.Queue.
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Any

import numpy as np
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from mpj_spark.core.sync_modes import MODE_PS_SYNC_FEDAVG_MPI


def _build_schema(num_features: int) -> StructType:
    fields = [StructField(f"f{i}", DoubleType(), nullable=True) for i in range(num_features)]
    fields.append(StructField("label", DoubleType(), nullable=True))
    return StructType(fields)


def _weight_norm(weights: list[float] | np.ndarray) -> float:
    return float(math.sqrt(sum(w * w for w in weights)))


def _write_worker_metrics(worker_id: int, records: list[dict[str, Any]], results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"worker_{worker_id}_fedavg_mpi_metrics.csv")
    fieldnames = [
        "worker_id",
        "sync_mode",
        "iteration",
        "iter_time_s",
        "weight_norm",
        "weight_delta",
        "local_weight_norm",
        "intercept",
        "row_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


def run(
    partition_path: str,
    comm,
    rank: int,
    num_workers: int,
    max_iter: int = 10,
    reg_param: float = 0.01,
    num_features: int = 10,
    seed: int = 42,
    results_dir: str = "results",
    local_epochs: int = 5,
) -> dict[str, Any]:
    """Execute LogReg training with periodic FedAvg over MPI collectives.

    Parameters
    ----------
    partition_path : str
        Path to local worker CSV partition.
    comm : MPI communicator
        Active MPI communicator across workers and root.
    rank : int
        Current MPI rank (0 is root, 1..N are workers).
    num_workers : int
        Number of active worker ranks.
    max_iter : int
        Number of global synchronization rounds.
    reg_param : float
        L2 regularization parameter.
    num_features : int
        Feature count.
    seed : int
        Random seed.
    results_dir : str
        Directory to store worker CSV metrics.
    local_epochs : int
        Number of local L-BFGS gradient steps before synchronization.

    Returns
    -------
    dict with model results, accuracy, and per-round convergence records.
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(f"[FedAvg MPI Worker {rank}] No active SparkSession found.")

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
            f"[FedAvg MPI Worker {rank}] Partition is empty after cleaning: {partition_path}"
        )

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    df_vec = assembler.transform(df_base).select("features", "label").cache()
    df_vec.count()

    current_weights: list[float] | None = None
    current_intercept = 0.0
    prev_weights_vec = np.zeros(num_features)
    iter_metrics: list[dict[str, Any]] = []
    iterations_done = 0

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

        if current_weights is not None:
            try:
                lr = lr.setInitialWeights(Vectors.dense(current_weights))
                lr = lr.setInitialIntercept(float(current_intercept))
            except Exception:
                pass

        model = lr.fit(df_vec)
        local_weights_arr = model.coefficients.toArray()
        local_weights = local_weights_arr.tolist()
        local_intercept = float(model.intercept)
        local_norm = float(np.linalg.norm(local_weights_arr))

        local_payload = {
            "weights": local_weights,
            "intercept": local_intercept,
            "row_count": row_count,
            "rank": rank,
        }

        # Step 2: MPI Collective Sync (Gather -> FedAvg -> Broadcast)
        gathered = comm.gather(local_payload, root=0)

        global_payload: dict[str, Any] | None = None
        if rank == 0 and gathered is not None:
            total_rows = sum(m["row_count"] for m in gathered)
            avg_w = np.zeros(num_features)
            avg_b = 0.0
            for m in gathered:
                frac = m["row_count"] / total_rows if total_rows > 0 else (1.0 / len(gathered))
                avg_w += frac * np.array(m["weights"])
                avg_b += frac * m["intercept"]

            global_payload = {
                "weights": avg_w.tolist(),
                "intercept": float(avg_b),
                "iteration": iteration,
            }

        global_model = comm.bcast(global_payload, root=0)
        current_weights = global_model["weights"]
        current_intercept = global_model["intercept"]

        iterations_done += 1
        iter_time = time.perf_counter() - t_iter
        global_norm = _weight_norm(current_weights)

        cur_vec = np.array(current_weights)
        weight_delta = float(np.linalg.norm(cur_vec - prev_weights_vec))
        prev_weights_vec = cur_vec.copy()

        iter_metrics.append(
            {
                "worker_id": rank,
                "sync_mode": MODE_PS_SYNC_FEDAVG_MPI,
                "iteration": iteration + 1,
                "iter_time_s": round(iter_time, 6),
                "weight_norm": round(global_norm, 8),
                "weight_delta": round(weight_delta, 8),
                "local_weight_norm": round(local_norm, 8),
                "intercept": round(current_intercept, 8),
                "row_count": row_count,
            }
        )

    # Compute final accuracy
    w_list = current_weights or [0.0] * num_features
    b_val = current_intercept

    @F.udf(returnType=IntegerType())
    def predict(features):
        arr = features.toArray()
        logit = sum(w_list[i] * arr[i] for i in range(len(w_list))) + b_val
        return int(1.0 / (1.0 + math.exp(-logit)) >= 0.5)

    correct = (
        df_vec.withColumn("pred", predict(F.col("features")))
        .filter(F.col("pred") == F.col("label"))
        .count()
    )
    train_accuracy = correct / row_count if row_count > 0 else 0.0

    df_vec.unpersist()
    df_base.unpersist()

    _write_worker_metrics(rank, iter_metrics, results_dir)

    return {
        "weight_vector": current_weights or [0.0] * num_features,
        "intercept": current_intercept,
        "train_accuracy": train_accuracy,
        "row_count": row_count,
        "iterations_done": iterations_done,
        "partition_path": partition_path,
        "iter_metrics": iter_metrics,
        "sync_mode": MODE_PS_SYNC_FEDAVG_MPI,
    }
