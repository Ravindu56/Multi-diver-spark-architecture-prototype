"""mpj_spark/applications/logreg/fedavg_mpi_run.py
Phase 3 - LogReg Periodic FedAvg over Native MPI Collectives.

Implements P3-08 (Issue #65):
Runs periodic FedAvg parameter synchronization over native mpi4py
collectives (comm.gather / comm.bcast), replacing multiprocessing.Queue.

CHANGE LOG
----------
fix(warm-start): the local step used pyspark.ml LogisticRegression and called
  ``setInitialWeights`` inside ``try/except Exception: pass``.  PySpark's
  LogisticRegression has no such setter (PySpark 4.2.0), so the AttributeError
  was swallowed and EVERY round refit from zero: identical local models, zero
  weight change after round 1.  The local step now runs warm-started L-BFGS-B
  (mpj_spark.core.warm_start_lr) from the current global model, and nothing is
  swallowed.  The objective is mean log-loss + 0.5*reg*||w||^2 with NO feature
  standardisation, so weights/norms are not comparable with the earlier MLlib
  (standardised) runs - compare modes by loss.
  New per-round metrics: global_loss (objective of the model ENTERING the round,
  exact, from the loss each worker already computes), loss_start/loss_end of the
  local step, n_eval, solver.
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Any

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from mpj_spark.core.sync_modes import MODE_PS_SYNC_FEDAVG_MPI
from mpj_spark.core.warm_start_lr import spark_loss_grad, warm_lbfgs

SOLVER = "warm_lbfgs"


def _build_schema(num_features: int) -> StructType:
    fields = [StructField(f"f{i}", DoubleType(), nullable=True) for i in range(num_features)]
    fields.append(StructField("label", DoubleType(), nullable=True))
    return StructType(fields)


def _weight_norm(weights: list[float] | np.ndarray) -> float:
    return float(math.sqrt(sum(w * w for w in weights)))


def _correct_count(rdd, weights: list[float], intercept: float) -> int:
    """Number of rows classified correctly by (weights, intercept); partition-vectorised."""
    w_arr = np.asarray(weights, dtype=np.float64)
    b = float(intercept)

    def per_partition(it):
        rows = list(it)
        if not rows:
            yield 0
            return
        X = np.asarray([r[0] for r in rows], dtype=np.float64)
        y = np.asarray([r[1] for r in rows], dtype=np.float64)
        yield int(np.sum((X @ w_arr + b >= 0.0) == (y >= 0.5)))

    return int(rdd.mapPartitions(per_partition).sum())


def _weighted_mean(values: list[float], rows: list[int]) -> float:
    total = sum(rows)
    if total <= 0:
        return float(sum(values) / len(values))
    return float(sum(v * r for v, r in zip(values, rows, strict=True)) / total)


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
        "global_loss",
        "loss_start",
        "loss_end",
        "n_eval",
        "solver",
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
        Worker sub-communicator (rank 0 aggregates).
    rank : int
        This worker's rank within ``comm``.
    num_workers : int
        Number of active worker ranks.
    max_iter : int
        Number of global synchronization rounds.
    reg_param : float
        L2 regularization parameter.
    num_features : int
        Feature count.
    seed : int
        Unused (kept for interface compatibility; the solver is deterministic).
    results_dir : str
        Directory to store worker CSV metrics.
    local_epochs : int
        Max warm-started L-BFGS-B iterations per round on the local partition.

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

    nf = num_features
    points = (
        df_base.select(*feature_cols, "label")
        .rdd.map(lambda row: (np.asarray(row[:nf], dtype=np.float64), float(row[nf])))
        .cache()
    )
    points.count()
    loss_grad = spark_loss_grad(points, num_features, reg_param, row_count)

    current_weights: list[float] | None = None
    current_intercept = 0.0
    prev_weights_vec = np.zeros(num_features)
    iter_metrics: list[dict[str, Any]] = []
    iterations_done = 0

    for iteration in range(max_iter):
        t_iter = time.perf_counter()

        # Local step: warm-started from the global model (cold only in round 1).
        w_local, b_local, info = warm_lbfgs(
            loss_grad, current_weights, current_intercept, num_features, local_epochs
        )
        local_norm = float(np.linalg.norm(w_local))

        local_payload = {
            "weights": w_local.tolist(),
            "intercept": b_local,
            "row_count": row_count,
            "rank": rank,
            "loss_start": info["loss_start"],
        }

        # MPI collective sync (Gather -> FedAvg -> Broadcast)
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
                # objective of the model that ENTERED this round (row-weighted
                # mean of each worker's loss at that model = exact global loss)
                "global_loss": _weighted_mean(
                    [m["loss_start"] for m in gathered], [m["row_count"] for m in gathered]
                ),
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
                "global_loss": round(float(global_model["global_loss"]), 8),
                "loss_start": round(info["loss_start"], 8),
                "loss_end": round(info["loss_end"], 8),
                "n_eval": info["n_eval"],
                "solver": SOLVER,
            }
        )
        print(
            f"[FedAvg MPI Worker {rank}] round {iteration + 1}/{max_iter}  "
            f"({iter_time:.3f}s)  |w|={global_norm:.4f}  delta={weight_delta:.6f}  "
            f"global_loss={global_model['global_loss']:.6f}"
        )

    # Final accuracy on the local partition
    w_list = current_weights or [0.0] * num_features
    correct = _correct_count(points, w_list, current_intercept)
    train_accuracy = correct / row_count if row_count > 0 else 0.0

    points.unpersist()
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
