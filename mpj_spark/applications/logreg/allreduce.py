# =============================================================================
# mpj_spark/applications/logreg/allreduce.py
# Phase 3 — Issue #9 — Steps 5 & 6: Allreduce Weight Sync + Full Runner
#
# STEP 5 — allreduce_gradients()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synchronous SGD Allreduce for the LogReg MPI-Allreduce architecture:
#
#   comm.Allreduce(grad_local, global_grad, op=MPI.SUM)
#   global_grad /= size                    ← average over all ranks
#   w -= learning_rate * global_grad       ← in-place weight update
#
# Every rank passes its normalised local gradient (from local_gradient.py).
# After the Allreduce the weight vector is identical on all ranks —
# this is the Synchronous Allreduce guarantee.
#
# STEP 5 — check_loss_convergence()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# After each weight update, check whether training has converged:
#
#   comm.Allreduce(local_loss_arr, global_loss_arr, op=MPI.SUM)
#   global_loss = global_loss_arr[0] / size   ← average cross-entropy
#   [rank 0 only] converged = abs(prev_loss - global_loss) < tol
#   comm.bcast(converged, root=0)             ← all ranks get same flag
#
# Rank 0 owns the convergence decision; bcast ensures all ranks stop
# simultaneously (no rank exits the loop early).
#
# STEP 6 — run_logreg_allreduce()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Full orchestration runner that calls Steps 2–6 in sequence:
#   partition_and_init_spark()     (Step 2)
#   load_and_cache_rdd()           (Step 4a)
#   weight init + comm.Barrier()
#   per-epoch loop:
#     compute_gradient_spark()     (Step 4b)  ← spark_time window
#     allreduce_gradients()        (Step 5)   ← sync_time window
#     check_loss_convergence()     (Step 5)
#     collector.record_epoch()     (Step 6)
#   collector.record_run() + to_csv() + to_json()
#   aggregate_across_ranks() on rank 0
#   data_rdd.unpersist() + spark.stop()
#
# TIMING BRACKETS (mirror kmeans/allreduce.py exactly)
# -----------------------------------------------------
#   spark_time_s : time.perf_counter() around compute_gradient_spark() only
#   sync_time_s  : time.perf_counter() from allreduce_gradients() start
#                  to end of check_loss_convergence() (full MPI window)
#   epoch_time_s : full wall-clock for the epoch
#
# CLI ENTRY-POINT
# ---------------
#   mpirun -n 3 python -m mpj_spark.applications.logreg.allreduce \
#       --input /path/to/data.csv --epochs 30 --lr 0.01
# =============================================================================

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# STEP 5a — Allreduce gradient synchronisation + weight update
# ===========================================================================


def allreduce_gradients(
    comm,
    size: int,
    w: np.ndarray,
    grad_local: np.ndarray,
    learning_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synchronous SGD weight update via MPI Allreduce.

    All ranks call this simultaneously with their local gradient.  After
    the Allreduce, every rank holds the same global gradient and applies
    the same weight update — this is the Synchronous Allreduce guarantee.

    MPI call
    --------
    Buffer-level Allreduce (avoids pickle overhead for numpy arrays):
        comm.Allreduce([grad_local, MPI.DOUBLE], [global_grad, MPI.DOUBLE], op=MPI.SUM)

    Parameters
    ----------
    comm          : mpi4py.MPI.Intracomm — COMM_WORLD
    size          : int — MPI world size (comm.Get_size())
    w             : np.ndarray shape (D,) — current weight vector
    grad_local    : np.ndarray shape (D,) — normalised local gradient
                    from compute_gradient_spark() (already divided by n_local)
    learning_rate : float — SGD step size

    Returns
    -------
    w_new        : np.ndarray shape (D,) — updated weight vector (same on all ranks)
    global_grad  : np.ndarray shape (D,) — averaged global gradient (for logging)
    """
    from mpi4py import MPI

    global_grad = np.zeros_like(grad_local)

    # Buffer-level Allreduce: sums grad_local across all ranks into global_grad
    comm.Allreduce(
        [grad_local, MPI.DOUBLE],
        [global_grad, MPI.DOUBLE],
        op=MPI.SUM,
    )

    # Average over ranks: equivalent to synchronous mini-batch SGD where
    # each rank contributes an equal-sized shard of the global mini-batch
    global_grad /= float(size)

    # In-place gradient descent step
    w_new = w - learning_rate * global_grad

    return w_new, global_grad


# ===========================================================================
# STEP 5b — Loss Allreduce + convergence broadcast
# ===========================================================================


def check_loss_convergence(
    comm,
    rank: int,
    size: int,
    data_rdd,
    w: np.ndarray,
    prev_loss: float,
    tol: float,
    epoch: int,
) -> Tuple[bool, float]:
    """
    Compute global cross-entropy loss and broadcast a convergence flag.

    Each rank computes its local average cross-entropy over its shard.
    comm.Allreduce(MPI.SUM) aggregates local losses; rank 0 checks whether
    abs(prev_loss - global_loss) < tol and broadcasts the stop boolean.

    Cross-entropy per sample
    ------------------------
        loss_i = -(y*log(p) + (1-y)*log(1-p))
        local_loss = mean over shard of loss_i
        global_loss = sum of local_loss across ranks / size

    Numerical stability
    -------------------
    log(p) is clipped at log(eps=1e-15) to avoid log(0) = -inf.

    Parameters
    ----------
    comm      : mpi4py.MPI.Intracomm
    rank      : int — this rank's index
    size      : int — MPI world size
    data_rdd  : cached RDD of (np.ndarray, float) pairs
    w         : np.ndarray shape (D,) — current weight vector (post-update)
    prev_loss : float — global loss from the previous epoch
    tol       : float — convergence tolerance
    epoch     : int — current epoch (for logging only)

    Returns
    -------
    converged   : bool  — True if abs(prev_loss - global_loss) < tol
    global_loss : float — current global average cross-entropy
    """
    from mpi4py import MPI

    _w = w
    _eps = 1e-15

    def local_loss_fn(row: Tuple[np.ndarray, float]) -> float:
        x, y = row
        p = 1.0 / (1.0 + np.exp(-float(np.dot(x, _w))))
        # Clip to avoid log(0)
        p = float(np.clip(p, _eps, 1.0 - _eps))
        return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    # Compute local mean cross-entropy via Spark action
    local_loss_sum = float(
        data_rdd.map(local_loss_fn).reduce(lambda a, b: a + b)
    )
    n_local = data_rdd.count()
    local_loss = local_loss_sum / float(n_local) if n_local > 0 else 0.0

    # Allreduce to get global sum, then average over ranks
    local_arr = np.array([local_loss], dtype=np.float64)
    global_arr = np.zeros(1, dtype=np.float64)
    comm.Allreduce([local_arr, MPI.DOUBLE], [global_arr, MPI.DOUBLE], op=MPI.SUM)
    global_loss = float(global_arr[0]) / float(size)

    # Convergence decision on rank 0, broadcast to all
    if rank == 0:
        delta = abs(prev_loss - global_loss)
        converged = bool(delta < tol)
        logger.info(
            "[rank 0] epoch=%d  global_loss=%.6f  delta=%.6f  converged=%s",
            epoch,
            global_loss,
            delta,
            converged,
        )
    else:
        converged = False  # placeholder before bcast

    # All ranks must agree on the stop flag — no rank exits early
    converged = comm.bcast(converged, root=0)

    return converged, global_loss


# ===========================================================================
# STEP 6 — Full LogReg Allreduce runner (Steps 2–6 orchestration)
# ===========================================================================


def run_logreg_allreduce(
    comm,
    rank: int,
    size: int,
    input_file: str,
    max_epochs: int = 30,
    learning_rate: float = 0.01,
    tol: float = 1e-4,
    seed: int = 42,
    cores_override: int | None = None,
    metrics_output_dir: str = "./metrics",
) -> Dict:
    """
    Full multi-driver Logistic Regression with synchronous Allreduce
    gradient sync and per-epoch metrics collection.

    Orchestration order (Steps 2–6)
    --------------------------------
    Step 2 : partition_and_init_spark()    — scatter + SparkSession
    Step 4a: load_and_cache_rdd()          — warm cache
    Setup  : weight init (np.zeros) + comm.Barrier()
    Loop   :
      Step 4b: compute_gradient_spark()   [spark_time window]
      Step 5a: allreduce_gradients()      [sync_time window]
      Step 5b: check_loss_convergence()   [sync_time window]
      Step 6 : collector.record_epoch()
    Post   : record_run + to_csv + to_json + aggregate_across_ranks
             data_rdd.unpersist() + spark.stop()
    """
    from mpj_spark.applications.logreg.partition import partition_and_init_spark
    from mpj_spark.applications.logreg.local_gradient import (
        load_and_cache_rdd,
        compute_gradient_spark,
        cores_per_worker,
    )
    from mpj_spark.applications.logreg.metrics import LogRegMetricsCollector

    t_total_start = time.perf_counter()

    collector = LogRegMetricsCollector(rank=rank, output_dir=metrics_output_dir)

    # ------------------------------------------------------------------ #
    # Step 2: partition + scatter + Spark session                         #
    # ------------------------------------------------------------------ #
    _cores = cores_override if cores_override is not None else cores_per_worker(size)
    partition_path, spark, num_features = partition_and_init_spark(
        comm=comm,
        rank=rank,
        size=size,
        input_file=input_file,
        num_workers=size,
        cores_override=_cores,
    )

    # ------------------------------------------------------------------ #
    # Step 4a: load and warm-cache the RDD                                #
    # ------------------------------------------------------------------ #
    data_rdd = load_and_cache_rdd(spark, partition_path, num_features)
    total_points = data_rdd.count()

    # ------------------------------------------------------------------ #
    # Weight initialisation (same seed on all ranks → same starting w)   #
    # comm.Barrier(): ensure all ranks have loaded data before epoch 1    #
    # ------------------------------------------------------------------ #
    np.random.seed(seed)
    w = np.zeros(num_features, dtype=np.float64)  # zero init for reproducibility

    comm.Barrier()
    logger.info("[rank %d] Barrier passed — entering epoch loop", rank)

    prev_loss = float("inf")
    converged = False

    # ------------------------------------------------------------------ #
    # Main epoch loop: Steps 4b + 5a + 5b + 6                            #
    # ------------------------------------------------------------------ #
    for epoch in range(1, max_epochs + 1):
        t_epoch_start = time.perf_counter()

        # -- spark_time window: local gradient only -------------------- #
        t_spark_start = time.perf_counter()
        grad_local, _n_local = compute_gradient_spark(data_rdd, w)
        spark_time = time.perf_counter() - t_spark_start

        # -- sync_time window: full MPI-collective region -------------- #
        t_sync_start = time.perf_counter()

        w, global_grad = allreduce_gradients(
            comm=comm,
            size=size,
            w=w,
            grad_local=grad_local,
            learning_rate=learning_rate,
        )

        converged, global_loss = check_loss_convergence(
            comm=comm,
            rank=rank,
            size=size,
            data_rdd=data_rdd,
            w=w,
            prev_loss=prev_loss,
            tol=tol,
            epoch=epoch,
        )

        sync_time = time.perf_counter() - t_sync_start
        epoch_time = time.perf_counter() - t_epoch_start

        # -- Step 6: record epoch metrics ------------------------------ #
        grad_norm = float(np.linalg.norm(global_grad))
        weight_norm = float(np.linalg.norm(w))

        collector.record_epoch(
            epoch=epoch,
            spark_time_s=spark_time,
            sync_time_s=sync_time,
            epoch_time_s=epoch_time,
            grad_norm=grad_norm,
            global_loss=global_loss,
            weight_norm=weight_norm,
        )

        logger.info(
            "[rank %d] epoch=%d  spark=%.4fs  sync=%.4fs  epoch=%.4fs  "
            "grad_norm=%.6f  loss=%.6f  |w|=%.4f",
            rank,
            epoch,
            spark_time,
            sync_time,
            epoch_time,
            grad_norm,
            global_loss,
            weight_norm,
        )

        prev_loss = global_loss

        if converged:
            logger.info(
                "[rank %d] Converged at epoch %d (loss_delta < tol=%.2e)",
                rank,
                epoch,
                tol,
            )
            break

    # ------------------------------------------------------------------ #
    # Step 6: run-level record + file output                              #
    # ------------------------------------------------------------------ #
    total_time = time.perf_counter() - t_total_start

    collector.record_run(
        total_time_s=total_time,
        epochs_run=len(collector._epochs),
        converged=converged,
        dataset_size=total_points,
        num_ranks=size,
        learning_rate=learning_rate,
        tol=tol,
    )
    collector.to_csv()
    collector.to_json()

    comm.Barrier()
    if rank == 0:
        try:
            LogRegMetricsCollector.aggregate_across_ranks(
                output_dir=metrics_output_dir,
                num_ranks=size,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Aggregation failed (non-fatal): %s", exc)

    data_rdd.unpersist()
    spark.stop()

    return {
        "final_weights": w.tolist(),
        "epochs_run": len(collector._epochs),
        "converged": converged,
        "final_loss": prev_loss,
        "metrics": collector.summary_table(),
        "run_summary": collector._run,
        "rank": rank,
        "total_time_s": round(total_time, 4),
    }


# ===========================================================================
# CLI entry-point
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import os
    import sys

    from mpi4py import MPI

    # Step 1: MPI init FIRST — before any imports that touch Spark
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()

    parser = argparse.ArgumentParser(
        prog="python -m mpj_spark.applications.logreg.allreduce",
        description="Multi-driver Logistic Regression with synchronous MPI "
        "Allreduce gradient sync (Phase 3 — Issue #9).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file (shared/NFS path visible to all ranks).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="SGD learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance for loss delta (default: 1e-4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight initialisation (default: 42).",
    )
    parser.add_argument(
        "--output",
        default="./logreg_results",
        help="Directory for per-rank metrics CSV/JSON and aggregated CSV "
        "(default: ./logreg_results).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for all ranks (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=f"%(asctime)s [rank {_rank}] %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    os.makedirs(args.output, exist_ok=True)

    if _rank == 0:
        print(
            f"\n{'='*60}\n"
            "  LogReg Allreduce — Phase 3 / Issue #9\n"
            f"  ranks={_size}  epochs={args.epochs}  lr={args.lr}  "
            f"tol={args.tol}  seed={args.seed}\n"
            f"  input  : {args.input}\n"
            f"  output : {args.output}\n"
            f"{'='*60}\n",
            flush=True,
        )

    result = run_logreg_allreduce(
        comm=_comm,
        rank=_rank,
        size=_size,
        input_file=args.input,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        tol=args.tol,
        seed=args.seed,
        metrics_output_dir=args.output,
    )

    if _rank == 0:
        print("\n" + "=" * 60)
        print("  Run complete — rank 0 summary")
        print(f"  converged    : {result['converged']}")
        print(f"  epochs_run   : {result['epochs_run']}")
        print(f"  final_loss   : {result['final_loss']:.6f}")
        print(f"  total_time_s : {result['total_time_s']}s")
        print(f"  output dir   : {args.output}")
        print("=" * 60 + "\n")

        print("Per-epoch metrics (rank 0):")
        table = result["metrics"]
        if table:
            headers = list(table[0].keys())
            col_w = {
                h: max(len(h), max(len(str(r[h])) for r in table))
                for h in headers
            }
            header_line = "  ".join(h.ljust(col_w[h]) for h in headers)
            print(header_line)
            print("-" * len(header_line))
            for row in table:
                print("  ".join(str(row[h]).ljust(col_w[h]) for h in headers))
        print()
