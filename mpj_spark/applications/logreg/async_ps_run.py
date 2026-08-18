"""mpj_spark/applications/logreg/async_ps_run.py
Phase 3 - P3-09: LogReg over an asynchronous Parameter Server (MPI P2P).

Worker-side training loop for sync_mode="ps_async" (Issue #61).

Each synchronisation round the worker:
  1. performs `local_epochs` L-BFGS steps on its own partition,
     warm-started from the latest global model (when available),
  2. sends its local model to the root parameter server
     (dest=0, TAG_ALLREDUCE_UP=30),
  3. blocks on the PS reply (TAG_ALLREDUCE_DOWN=31) - the root answers
     immediately upon receiving the update, so the wait is bounded by
     the PS apply time, NOT by a barrier across workers.

No collectives are used and workers never synchronise with each other,
so stragglers do not stall fast workers (in contrast to the P3-08
gather/bcast FedAvg in fedavg_mpi_run.py).

IMPORTANT: `comm` must be the COMM_WORLD communicator (root PS is rank 0,
workers are ranks 1..N) - not the worker-only sub-communicator created
for the collective sync modes.  This mode is therefore only available on
the MPI execution path (python -m mpj_spark.core.main_mpi).
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Any

import numpy as np
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from mpj_spark.applications.logreg.fedavg_mpi_run import _build_schema, _weight_norm
from mpj_spark.core.sync_modes import MODE_PS_ASYNC

TAG_ALLREDUCE_UP = 30
TAG_ALLREDUCE_DOWN = 31


def _write_async_worker_metrics(
    worker_id: int, records: list[dict[str, Any]], results_dir: str
) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"worker_{worker_id}_async_ps_metrics.csv")
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
        "staleness",
        "global_version",
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
    tag_up: int = TAG_ALLREDUCE_UP,
    tag_down: int = TAG_ALLREDUCE_DOWN,
) -> dict[str, Any]:
    """Execute LogReg training against the asynchronous root parameter server.

    Parameters
    ----------
    partition_path : str
        Path to local worker CSV partition.
    comm : MPI communicator
        COMM_WORLD (root PS is rank 0; this worker is `rank`).
    rank : int
        This worker's COMM_WORLD rank (1..N).
    num_workers : int
        Number of active worker ranks.
    max_iter : int
        Number of local training / sync rounds.
    reg_param : float
        L2 regularization parameter.
    num_features : int
        Feature count.
    seed : int
        Random seed.
    results_dir : str
        Directory to store worker CSV metrics.
    local_epochs : int
        Number of local L-BFGS gradient steps before each push.
    tag_up / tag_down : int
        MPI tags for the PS push / reply channel.

    Returns
    -------
    dict with model results, accuracy, and per-round convergence records
    (same shape as fedavg_mpi_run.run(), plus per-round staleness).
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(f"[Async PS Worker {rank}] No active SparkSession found.")

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
            f"[Async PS Worker {rank}] Partition is empty after cleaning: {partition_path}"
        )

    from pyspark.ml.feature import VectorAssembler

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    df_vec = assembler.transform(df_base).select("features", "label").cache()
    df_vec.count()

    current_weights: list[float] | None = None
    current_intercept = 0.0
    base_version = 0  # global_version this worker's model is built on
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
                from pyspark.ml.linalg import Vectors

                lr = lr.setInitialWeights(Vectors.dense(current_weights))
                lr = lr.setInitialIntercept(float(current_intercept))
            except Exception:
                pass

        model = lr.fit(df_vec)
        local_weights_arr = model.coefficients.toArray()
        local_weights = local_weights_arr.tolist()
        local_intercept = float(model.intercept)
        local_norm = float(np.linalg.norm(local_weights_arr))

        # ── Async PS push/reply (point-to-point, no collectives) ─────
        payload = {
            "weights": local_weights,
            "intercept": local_intercept,
            "row_count": row_count,
            "rank": rank,
            "worker_round": iteration,
            "base_version": base_version,
        }
        comm.send(payload, dest=0, tag=tag_up)
        reply = comm.recv(source=0, tag=tag_down)

        current_weights = reply["weights"]
        current_intercept = float(reply["intercept"])
        base_version = int(reply["global_version"])
        staleness = int(reply.get("staleness", 0))

        iterations_done += 1
        iter_time = time.perf_counter() - t_iter
        global_norm = _weight_norm(current_weights)

        cur_vec = np.array(current_weights)
        weight_delta = float(np.linalg.norm(cur_vec - prev_weights_vec))
        prev_weights_vec = cur_vec.copy()

        iter_metrics.append(
            {
                "worker_id": rank,
                "sync_mode": MODE_PS_ASYNC,
                "iteration": iteration + 1,
                "iter_time_s": round(iter_time, 6),
                "weight_norm": round(global_norm, 8),
                "weight_delta": round(weight_delta, 8),
                "local_weight_norm": round(local_norm, 8),
                "intercept": round(current_intercept, 8),
                "row_count": row_count,
                "staleness": staleness,
                "global_version": base_version,
            }
        )
        print(
            f"[Async PS Worker {rank}] round {iteration + 1}/{max_iter}  "
            f"({iter_time:.3f}s)  |w|={global_norm:.4f}  "
            f"staleness={staleness}  v{base_version}"
        )

    # Compute final accuracy on the local partition
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

    _write_async_worker_metrics(rank, iter_metrics, results_dir)

    return {
        "weight_vector": current_weights or [0.0] * num_features,
        "intercept": current_intercept,
        "train_accuracy": train_accuracy,
        "row_count": row_count,
        "iterations_done": iterations_done,
        "partition_path": partition_path,
        "iter_metrics": iter_metrics,
        "sync_mode": MODE_PS_ASYNC,
    }
