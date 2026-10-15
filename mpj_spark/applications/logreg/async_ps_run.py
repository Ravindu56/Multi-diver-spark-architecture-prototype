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

CHANGE LOG
----------
fix(warm-start): the local step used pyspark.ml LogisticRegression and a
  swallowed ``setInitialWeights`` call that does not exist in PySpark, so every
  round refit from zero.  It now runs warm-started L-BFGS-B from the latest
  global model (mpj_spark.applications.logreg.warm_common).
  New per-round metrics: loss_start (loss of the global model this worker holds
  when the round starts, on its partition), loss_end, n_eval, solver;
  ``final_loss`` is returned.  No early stop: the root PS waits for exactly
  num_iterations updates per worker, so a worker leaving early would stall it.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any

import numpy as np
from pyspark.sql import SparkSession

from mpj_spark.applications.logreg.fedavg_mpi_run import _correct_count, _weight_norm
from mpj_spark.applications.logreg.warm_common import SOLVER, load_points, local_step, model_loss
from mpj_spark.core.sync_modes import MODE_PS_ASYNC
from mpj_spark.core.warm_start_lr import spark_loss_grad

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
        Max warm-started L-BFGS-B iterations before each push.
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

    df_base, points, row_count = load_points(
        spark, partition_path, num_features, f"Async PS Worker {rank}"
    )
    loss_grad = spark_loss_grad(points, num_features, reg_param, row_count)

    current_weights: list[float] | None = None
    current_intercept = 0.0
    base_version = 0  # global_version this worker's model is built on
    prev_weights_vec = np.zeros(num_features)
    iter_metrics: list[dict[str, Any]] = []
    iterations_done = 0

    for iteration in range(max_iter):
        t_iter = time.perf_counter()

        w_local, b_local, info, local_norm = local_step(
            loss_grad, current_weights, current_intercept, num_features, local_epochs
        )

        # ── Async PS push/reply (point-to-point, no collectives) ─────
        payload = {
            "weights": w_local.tolist(),
            "intercept": b_local,
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
                "loss_start": round(info["loss_start"], 8),
                "loss_end": round(info["loss_end"], 8),
                "n_eval": info["n_eval"],
                "solver": SOLVER,
            }
        )
        print(
            f"[Async PS Worker {rank}] round {iteration + 1}/{max_iter}  "
            f"({iter_time:.3f}s)  |w|={global_norm:.4f}  "
            f"staleness={staleness}  v{base_version}"
        )

    # Final loss and accuracy of the last received global model on this partition
    w_list = current_weights or [0.0] * num_features
    final_loss = model_loss(loss_grad, w_list, current_intercept)
    correct = _correct_count(points, w_list, current_intercept)
    train_accuracy = correct / row_count if row_count > 0 else 0.0

    points.unpersist()
    df_base.unpersist()

    _write_async_worker_metrics(rank, iter_metrics, results_dir)

    return {
        "weight_vector": current_weights or [0.0] * num_features,
        "intercept": current_intercept,
        "train_accuracy": train_accuracy,
        "final_loss": final_loss,
        "row_count": row_count,
        "iterations_done": iterations_done,
        "partition_path": partition_path,
        "iter_metrics": iter_metrics,
        "sync_mode": MODE_PS_ASYNC,
    }
