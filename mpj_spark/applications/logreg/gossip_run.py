"""mpj_spark/applications/logreg/gossip_run.py
Phase 3 - P3-11: Decentralized gossip synchronization for LogReg (Issue #63).

D-PSGD-style decentralized training: per round, each worker performs
`local_epochs` L-BFGS steps on its own partition (warm-started from its
current consensus model), then exchanges {weights, intercept, row_count}
with its fixed-ring neighbours over the WORKER sub-communicator and
mixes states via row-count-weighted diffusion (consensus_mix).

No root coordinator, no collectives, no global barrier: each worker
syncs only with its ring neighbours.  With tol > 0, a worker whose
round delta falls below tol freezes its local refit but KEEPS
participating in exchanges (neighbours must never block on a stopped
peer) - the convergence config mirrors the K-Means gossip knobs
(gossip_threshold / gossip_max_rounds) used by GossipAggregator.

Communicator: the worker-only sub-communicator (comm), 0-based ranks.
COMM_WORLD / root is not involved in the sync path at all - root only
collects final results, and aggregation is the post-hoc row-weighted
mean of the (near-consensus) worker models.

Import-safe on CI: this module never imports mpi4py - the ring exchange
uses Python-object sendrecv only, so no MPI constants are needed.

CHANGE LOG
----------
fix(warm-start): the local step used pyspark.ml LogisticRegression and a
  swallowed ``setInitialWeights`` call that does not exist in PySpark, so every
  round refit from zero.  It now runs warm-started L-BFGS-B from the worker's
  current consensus model (mpj_spark.applications.logreg.warm_common).
  New per-round metrics: loss_start (loss of the model entering the round, on
  this worker's partition), loss_end, n_eval, solver; ``final_loss`` is
  returned.  There is no global early stop: the mode is decentralized by design
  (no collective to agree on one), so it always runs max_iter rounds.
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
from mpj_spark.core.gossip_protocol import (
    TAG_GOSSIP_EXCHANGE,
    consensus_mix,
    gossip_exchange,
    ring_neighbors,
)
from mpj_spark.core.sync_modes import MODE_GOSSIP
from mpj_spark.core.warm_start_lr import spark_loss_grad


def _write_gossip_worker_metrics(rank: int, records: list[dict[str, Any]], results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"worker_{rank}_gossip_metrics.csv")
    fieldnames = [
        "worker_id",
        "sync_mode",
        "iteration",
        "iter_time_s",
        "gossip_time_s",
        "peers_contacted",
        "weight_norm",
        "weight_delta",
        "local_weight_norm",
        "intercept",
        "row_count",
        "frozen",
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
    fanout: int = 1,
    tol: float = 0.0,
    tag: int = TAG_GOSSIP_EXCHANGE,
) -> dict[str, Any]:
    """Execute LogReg training with decentralized gossip synchronization.

    Parameters
    ----------
    partition_path : str
        Path to local worker CSV partition.
    comm : MPI communicator
        Worker-only sub-communicator (ring exchange channel).  Required.
    rank : int
        This worker's rank within the sub-communicator (0-based).
    num_workers : int
        Number of active worker ranks (ring size).
    fanout : int
        Ring distance contacted per round (1 = immediate neighbours).
    tol : float
        If > 0, freeze the local refit once the per-round weight delta
        drops below tol; exchanges continue so neighbours never block.
    max_iter, reg_param, num_features, seed, results_dir, local_epochs :
        Same semantics as fedavg_mpi_run.run().

    Returns
    -------
    dict with model results, accuracy, and per-round convergence records
    (same shape as fedavg_mpi_run.run(), plus gossip_time_s /
    peers_contacted per round for the #64 communication-cost analysis).
    """
    if comm is None:
        raise RuntimeError(
            "[Gossip Worker] sync_mode='gossip' requires the MPI worker "
            "sub-communicator (comm) - run via python -m mpj_spark.core.main_mpi."
        )

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(f"[Gossip Worker {rank}] No active SparkSession found.")

    df_base, points, row_count = load_points(
        spark, partition_path, num_features, f"Gossip Worker {rank}"
    )
    loss_grad = spark_loss_grad(points, num_features, reg_param, row_count)

    peers = ring_neighbors(rank, num_workers, fanout)
    if not peers and num_workers > 1:
        raise RuntimeError(
            f"[Gossip Worker {rank}] ring topology produced no neighbours "
            f"(size={num_workers}, fanout={fanout})"
        )

    current_weights: list[float] | None = None
    current_intercept = 0.0
    frozen = False
    prev_weights_vec = np.zeros(num_features)
    iter_metrics: list[dict[str, Any]] = []
    iterations_done = 0

    print(
        f"[Gossip Worker {rank}] reg_param={reg_param} | rounds={max_iter} | "
        f"local_epochs={local_epochs} | rows={row_count:,} | features={num_features} | "
        f"peers={peers} | mode=gossip (decentralized)"
    )

    for iteration in range(max_iter):
        t_iter = time.perf_counter()

        loss_start: float | str = ""
        loss_end: float | str = ""
        n_eval: int | str = ""
        if not frozen:
            w_local, b_local, info, local_norm = local_step(
                loss_grad, current_weights, current_intercept, num_features, local_epochs
            )
            current_weights = w_local.tolist()
            current_intercept = b_local
            loss_start, loss_end, n_eval = info["loss_start"], info["loss_end"], info["n_eval"]
        else:
            local_norm = _weight_norm(current_weights or [0.0] * num_features)

        # ── Decentralized gossip exchange over the worker sub-comm ──
        t_gossip = time.perf_counter()
        payload = {
            "weights": current_weights,
            "intercept": current_intercept,
            "row_count": row_count,
            "rank": rank,
            "worker_round": iteration,
        }
        received = gossip_exchange(comm, payload, rank, num_workers, fanout=fanout, tag=tag)
        mixed_w, mixed_b = consensus_mix(payload, received)
        gossip_time = time.perf_counter() - t_gossip

        current_weights = mixed_w.tolist()
        current_intercept = mixed_b

        iterations_done += 1
        iter_time = time.perf_counter() - t_iter
        global_norm = _weight_norm(current_weights)

        cur_vec = np.array(current_weights)
        weight_delta = float(np.linalg.norm(cur_vec - prev_weights_vec))
        prev_weights_vec = cur_vec.copy()

        if tol > 0.0 and not frozen and weight_delta < tol:
            frozen = True
            print(
                f"[Gossip Worker {rank}] converged at round {iteration + 1} "
                f"(delta={weight_delta:.6f} < tol={tol}) - local refit frozen, "
                "exchanges continue"
            )

        iter_metrics.append(
            {
                "worker_id": rank,
                "sync_mode": MODE_GOSSIP,
                "iteration": iteration + 1,
                "iter_time_s": round(iter_time, 6),
                "gossip_time_s": round(gossip_time, 6),
                "peers_contacted": len(received),
                "weight_norm": round(global_norm, 8),
                "weight_delta": round(weight_delta, 8),
                "local_weight_norm": round(local_norm, 8),
                "intercept": round(current_intercept, 8),
                "row_count": row_count,
                "frozen": int(frozen),
                "loss_start": loss_start if loss_start == "" else round(loss_start, 8),
                "loss_end": loss_end if loss_end == "" else round(loss_end, 8),
                "n_eval": n_eval,
                "solver": SOLVER,
            }
        )
        print(
            f"[Gossip Worker {rank}] round {iteration + 1}/{max_iter}  "
            f"({iter_time:.3f}s)  |w|={global_norm:.4f}  "
            f"peers={len(received)}  gossip={gossip_time:.4f}s"
            f"{'  [frozen]' if frozen else ''}"
        )

    # Final loss and accuracy of this worker's model on its own partition
    w_list = current_weights or [0.0] * num_features
    final_loss = model_loss(loss_grad, w_list, current_intercept)
    correct = _correct_count(points, w_list, current_intercept)
    train_accuracy = correct / row_count if row_count > 0 else 0.0

    points.unpersist()
    df_base.unpersist()

    _write_gossip_worker_metrics(rank, iter_metrics, results_dir)

    return {
        "weight_vector": current_weights or [0.0] * num_features,
        "intercept": current_intercept,
        "train_accuracy": train_accuracy,
        "final_loss": final_loss,
        "row_count": row_count,
        "iterations_done": iterations_done,
        "partition_path": partition_path,
        "iter_metrics": iter_metrics,
        "sync_mode": MODE_GOSSIP,
    }
