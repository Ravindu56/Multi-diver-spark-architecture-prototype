"""mpj_spark/applications/logreg/hybrid_run.py
Phase 3 - P3-10: Hybrid PS + Allreduce synchronization for LogReg (Issue #62).

Per-round two-channel synchronization:
  1. DENSE weight vector  -> comm.Allreduce(SUM) over the WORKER
     sub-communicator.  Each worker contributes frac_i * w_i with
     frac_i = rows_i / total_rows, so the collective directly yields the
     row-weighted global mean.  Row counts are static per run, so they
     are exchanged ONCE at init via allgather - zero per-round PS traffic
     for row counts.
  2. SCALAR metadata (intercept) -> root parameter server over COMM_WORLD
     P2P (TAG_ALLREDUCE_UP=30 / TAG_ALLREDUCE_DOWN=31), row-weighted by
     the root and replied per round.

This mirrors production hybrid stacks (MindSpore, PyTorch DDP+RPC):
large/dense tensors ride the collective, small/sparse state rides the PS.
The mode is SYNCHRONOUS on both channels (bulk-synchronous rounds), in
contrast to the non-blocking ps_async mode (P3-09).

IMPORTANT: two communicators are required -
  comm      = worker-only sub-communicator (Allreduce channel)
  root_comm = COMM_WORLD (root PS is rank 0; workers are ranks 1..N)
Both are provided by run_worker_core() on the MPI execution path.

Import-safe on CI runners without a system MPI library: mpi4py is
imported lazily inside run(); unit tests exercise helpers and the
dispatch contract without libmpi.so.

CHANGE LOG
----------
fix(warm-start): the local step used pyspark.ml LogisticRegression and a
  swallowed ``setInitialWeights`` call that does not exist in PySpark, so every
  round refit from zero.  It now runs warm-started L-BFGS-B from the current
  global model (mpj_spark.applications.logreg.warm_common).
  New per-round metrics: loss_start (loss of the global model entering the
  round, on this worker's partition), loss_end, n_eval, solver; ``final_loss``
  is returned.  No early stop: the root scalar PS serves exactly
  num_iterations rounds, so a worker leaving early would stall it.
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
from mpj_spark.core.sync_modes import MODE_HYBRID_PS_ALLREDUCE
from mpj_spark.core.warm_start_lr import spark_loss_grad

TAG_ALLREDUCE_UP = 30
TAG_ALLREDUCE_DOWN = 31


def _write_hybrid_worker_metrics(
    world_rank: int, records: list[dict[str, Any]], results_dir: str
) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"worker_{world_rank}_hybrid_metrics.csv")
    fieldnames = [
        "worker_id",
        "sync_mode",
        "iteration",
        "iter_time_s",
        "allreduce_time_s",
        "ps_time_s",
        "weight_norm",
        "weight_delta",
        "local_weight_norm",
        "intercept",
        "row_count",
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
    root_comm=None,
    world_rank: int | None = None,
    max_iter: int = 10,
    reg_param: float = 0.01,
    num_features: int = 10,
    seed: int = 42,
    results_dir: str = "results",
    local_epochs: int = 5,
    tag_up: int = TAG_ALLREDUCE_UP,
    tag_down: int = TAG_ALLREDUCE_DOWN,
) -> dict[str, Any]:
    """Execute LogReg training with hybrid PS+Allreduce synchronization.

    Parameters
    ----------
    partition_path : str
        Path to local worker CSV partition.
    comm : MPI communicator
        Worker-only sub-communicator (dense-weight Allreduce channel).
    rank : int
        This worker's rank within the sub-communicator (0-based).
    num_workers : int
        Number of active worker ranks.
    root_comm : MPI communicator
        COMM_WORLD (scalar PS channel; root is rank 0).  Required.
    world_rank : int
        This worker's COMM_WORLD rank (1..N).  Defaults to rank + 1.
    max_iter, reg_param, num_features, seed, results_dir, local_epochs :
        Same semantics as fedavg_mpi_run.run().
    tag_up / tag_down : int
        MPI tags for the scalar PS channel.

    Returns
    -------
    dict with model results, accuracy, and per-round convergence records
    (same shape as fedavg_mpi_run.run(), plus per-round allreduce/ps
    split timings for the communication-cost comparison, Issue #62).
    """
    if comm is None or root_comm is None:
        raise RuntimeError(
            "[Hybrid Worker] hybrid_ps_allreduce requires both the worker "
            "sub-communicator (comm) and COMM_WORLD (root_comm)."
        )
    if world_rank is None:
        world_rank = rank + 1

    from mpi4py import MPI  # lazy: module must import without libmpi (CI)

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(f"[Hybrid Worker {world_rank}] No active SparkSession found.")

    df_base, points, row_count = load_points(
        spark, partition_path, num_features, f"Hybrid Worker {world_rank}"
    )
    loss_grad = spark_loss_grad(points, num_features, reg_param, row_count)

    # One-time row-count exchange (static per run): afterwards the
    # Allreduce of frac-weighted vectors directly yields the global mean.
    all_rows = comm.allgather(row_count)
    total_rows = int(sum(all_rows))
    frac = row_count / total_rows if total_rows > 0 else 1.0 / max(1, len(all_rows))

    current_weights: list[float] | None = None
    current_intercept = 0.0
    prev_weights_vec = np.zeros(num_features)
    iter_metrics: list[dict[str, Any]] = []
    iterations_done = 0

    print(
        f"[Hybrid Worker {world_rank}] reg_param={reg_param} | rounds={max_iter} | "
        f"local_epochs={local_epochs} | rows={row_count:,} | features={num_features} | "
        "mode=hybrid_ps_allreduce (dense=Allreduce, scalars=PS)"
    )

    for iteration in range(max_iter):
        t_iter = time.perf_counter()

        local_weights_arr, local_intercept, info, local_norm = local_step(
            loss_grad, current_weights, current_intercept, num_features, local_epochs
        )

        # ── Channel 1: dense weights via Allreduce over the worker sub-comm ──
        t_allreduce = time.perf_counter()
        send_w = np.ascontiguousarray(frac * local_weights_arr, dtype=np.float64)
        global_w = np.zeros(num_features, dtype=np.float64)
        comm.Allreduce(send_w, global_w, op=MPI.SUM)
        allreduce_time = time.perf_counter() - t_allreduce

        # ── Channel 2: intercept scalar via root PS (COMM_WORLD P2P) ──
        t_ps = time.perf_counter()
        root_comm.send(
            {
                "intercept": local_intercept,
                "row_count": row_count,
                "rank": world_rank,
                "worker_round": iteration,
            },
            dest=0,
            tag=tag_up,
        )
        reply = root_comm.recv(source=0, tag=tag_down)
        ps_time = time.perf_counter() - t_ps

        current_weights = global_w.tolist()
        current_intercept = float(reply["intercept"])

        iterations_done += 1
        iter_time = time.perf_counter() - t_iter
        global_norm = _weight_norm(current_weights)

        cur_vec = np.array(current_weights)
        weight_delta = float(np.linalg.norm(cur_vec - prev_weights_vec))
        prev_weights_vec = cur_vec.copy()

        iter_metrics.append(
            {
                "worker_id": world_rank,
                "sync_mode": MODE_HYBRID_PS_ALLREDUCE,
                "iteration": iteration + 1,
                "iter_time_s": round(iter_time, 6),
                "allreduce_time_s": round(allreduce_time, 6),
                "ps_time_s": round(ps_time, 6),
                "weight_norm": round(global_norm, 8),
                "weight_delta": round(weight_delta, 8),
                "local_weight_norm": round(local_norm, 8),
                "intercept": round(current_intercept, 8),
                "row_count": row_count,
                "loss_start": round(info["loss_start"], 8),
                "loss_end": round(info["loss_end"], 8),
                "n_eval": info["n_eval"],
                "solver": SOLVER,
            }
        )
        print(
            f"[Hybrid Worker {world_rank}] round {iteration + 1}/{max_iter}  "
            f"({iter_time:.3f}s)  |w|={global_norm:.4f}  "
            f"allreduce={allreduce_time:.4f}s  ps={ps_time:.4f}s"
        )

    # Final loss and accuracy of the global model on this worker's partition
    w_list = current_weights or [0.0] * num_features
    final_loss = model_loss(loss_grad, w_list, current_intercept)
    correct = _correct_count(points, w_list, current_intercept)
    train_accuracy = correct / row_count if row_count > 0 else 0.0

    points.unpersist()
    df_base.unpersist()

    _write_hybrid_worker_metrics(world_rank, iter_metrics, results_dir)

    return {
        "weight_vector": current_weights or [0.0] * num_features,
        "intercept": current_intercept,
        "train_accuracy": train_accuracy,
        "final_loss": final_loss,
        "row_count": row_count,
        "iterations_done": iterations_done,
        "partition_path": partition_path,
        "iter_metrics": iter_metrics,
        "sync_mode": MODE_HYBRID_PS_ALLREDUCE,
    }
