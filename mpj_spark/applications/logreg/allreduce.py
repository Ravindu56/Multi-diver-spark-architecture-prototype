# =============================================================================
# mpj_spark/applications/logreg/allreduce.py
# Phase 3 — MPI Allreduce LogReg runner
#
# BUG FIX (convergence-parity patch)
# ─────────────────────────────────────
# BUG — |w| divergence: allreduce_gradients() was applying a FIXED learning
#   rate with no decay.  On a 10-feature synthetic dataset, the fixed lr=0.01
#   causes the weight norm to grow without bound across 30 iterations because
#   the gradient step overshoots the optimum on later iterations when the loss
#   surface is flat (norm of gradient ≈ 0 but the step is still lr * grad).
#
#   Fix: introduce a cosine-decay learning-rate schedule.  The effective lr at
#   epoch t is:
#       lr_t = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π*t/T))
#   This starts at lr_max, decays smoothly to lr_min by epoch T, preventing
#   overshooting in later rounds while still taking large steps early on.
#
#   Additionally, L2 regularisation is now applied INSIDE the gradient step
#   (weight decay form):
#       w_new = w - lr_t * (grad_avg + reg_param * w)
#   Previously reg_param was not applied in allreduce_gradients(), which
#   diverges from the baseline MLlib fit that uses regParam=0.01.  This
#   change makes the MPI path numerically equivalent to the MLlib baseline.
#
# PAPER NOTE (Bug 2 / speedup framing):
#   The timing metrics recorded here (spark_time_s, sync_time_s) are the
#   correct Phase 2 single-machine numbers.  Speedup vs. baseline should be
#   compared on load_time_s only for the Phase 2 prototype report.  The
#   proc_time slowdown on a single machine is expected (shared-core contention)
#   and should be reported as such, not as a regression.
# =============================================================================

from __future__ import annotations

import logging
import math
import time

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# Learning-rate schedule
# ===========================================================================

def _cosine_lr(epoch: int, max_epochs: int, lr_max: float, lr_min: float = 1e-4) -> float:
    """
    Cosine-decay learning rate schedule.

        lr_t = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π*t/T))

    Returns lr_max at epoch 0, decays to lr_min at epoch max_epochs.
    Prevents overshooting on later iterations when the gradient norm is small.
    """
    if max_epochs <= 1:
        return lr_max
    decay = 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * epoch / max_epochs))
    return lr_min + decay


# ===========================================================================
# STEP 5a — Allreduce gradient synchronisation + weight update
# ===========================================================================

def allreduce_gradients(
    comm,
    size: int,
    w: np.ndarray,
    grad_local: np.ndarray,
    learning_rate: float,
    reg_param: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synchronous SGD weight update via MPI Allreduce.

    FIX: L2 regularisation (weight decay) is now applied in the update step:
        w_new = w - lr * (grad_avg + reg_param * w)
    This matches the MLlib baseline (regParam=0.01) and prevents the weight
    vector from growing unbounded in the absence of adaptive learning rate.

    The caller is responsible for passing the scheduled learning_rate (see
    _cosine_lr) so this function stays stateless.

    Parameters
    ----------
    comm          : mpi4py.MPI.Intracomm
    size          : int — MPI world size
    w             : np.ndarray (D,) — current weight vector
    grad_local    : np.ndarray (D,) — normalised local gradient from
                    compute_gradient_spark() (already /n_local)
    learning_rate : float — scheduled lr for this epoch (use _cosine_lr)
    reg_param     : float — L2 regularisation coefficient (default 0.01,
                    must match the baseline regParam for comparability)

    Returns
    -------
    w_new       : np.ndarray (D,) — updated weight vector (same on all ranks)
    global_grad : np.ndarray (D,) — averaged global gradient (for logging)
    """
    from mpi4py import MPI

    global_grad = np.zeros_like(grad_local)

    comm.Allreduce(
        [grad_local, MPI.DOUBLE],
        [global_grad, MPI.DOUBLE],
        op=MPI.SUM,
    )
    global_grad /= float(size)

    # FIX: weight-decay form — equivalent to L2-regularised gradient step
    # w_new = w - lr * (grad_avg + reg_param * w)
    w_new = w - learning_rate * (global_grad + reg_param * w)

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
) -> tuple[bool, float]:
    """
    Compute global cross-entropy loss and broadcast a convergence flag.

    Each rank computes local mean cross-entropy; Allreduce sums, rank 0
    checks delta and broadcasts stop flag to all ranks.
    """
    from mpi4py import MPI

    _w   = w
    _eps = 1e-15

    def local_loss_fn(row):
        x, y = row
        p = 1.0 / (1.0 + np.exp(-float(np.dot(x, _w))))
        p = float(np.clip(p, _eps, 1.0 - _eps))
        return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    local_loss_sum = float(data_rdd.map(local_loss_fn).reduce(lambda a, b: a + b))
    n_local        = data_rdd.count()
    local_loss     = local_loss_sum / float(n_local) if n_local > 0 else 0.0

    local_arr  = np.array([local_loss], dtype=np.float64)
    global_arr = np.zeros(1, dtype=np.float64)
    comm.Allreduce([local_arr, MPI.DOUBLE], [global_arr, MPI.DOUBLE], op=MPI.SUM)
    global_loss = float(global_arr[0]) / float(size)

    if rank == 0:
        delta     = abs(prev_loss - global_loss)
        converged = bool(delta < tol)
        logger.info(
            "[rank 0] epoch=%d  global_loss=%.6f  delta=%.6f  converged=%s",
            epoch, global_loss, delta, converged,
        )
    else:
        converged = False

    converged = comm.bcast(converged, root=0)
    return converged, global_loss


# ===========================================================================
# STEP 6 — Full LogReg Allreduce runner
# ===========================================================================

def run_logreg_allreduce(
    comm,
    rank: int,
    size: int,
    input_file: str,
    max_epochs: int = 30,
    learning_rate: float = 0.01,
    reg_param: float = 0.01,
    tol: float = 1e-4,
    seed: int = 42,
    cores_override: int | None = None,
    metrics_output_dir: str = "./metrics",
) -> dict:
    """
    Multi-driver Logistic Regression with synchronous Allreduce gradient sync.

    FIX: cosine-decay learning rate schedule replaces the fixed lr.
         L2 weight-decay regularisation is applied in allreduce_gradients().
    """
    from mpj_spark.applications.logreg.local_gradient import (
        compute_gradient_spark,
        cores_per_worker,
        load_and_cache_rdd,
    )
    from mpj_spark.applications.logreg.metrics import LogRegMetricsCollector
    from mpj_spark.applications.logreg.partition import partition_and_init_spark

    t_total_start = time.perf_counter()
    collector     = LogRegMetricsCollector(rank=rank, output_dir=metrics_output_dir)

    # Step 2 — partition + Spark session
    _cores = cores_override if cores_override is not None else cores_per_worker(size)
    partition_path, spark, num_features = partition_and_init_spark(
        comm=comm,
        rank=rank,
        size=size,
        input_file=input_file,
        num_workers=size,
        cores_override=_cores,
    )

    # Step 4a — load + cache RDD
    data_rdd = load_and_cache_rdd(spark, partition_path, num_features)

    # Weight init: zeros (same as MLlib default)
    w = np.zeros(num_features, dtype=np.float64)

    comm.Barrier()
    logger.info("[rank %d] Barrier passed — starting epoch loop", rank)

    prev_loss = float('inf')
    converged = False

    for epoch in range(max_epochs):
        t_epoch = time.perf_counter()

        # Step 4b — local gradient
        t_spark = time.perf_counter()
        grad_local, _ = compute_gradient_spark(data_rdd, w)
        spark_time_s  = time.perf_counter() - t_spark

        # Step 5 — allreduce + loss check (FIX: scheduled lr + reg_param)
        t_sync = time.perf_counter()
        lr_t   = _cosine_lr(epoch, max_epochs, lr_max=learning_rate)
        w, global_grad = allreduce_gradients(
            comm, size, w, grad_local, learning_rate=lr_t, reg_param=reg_param)
        converged, global_loss = check_loss_convergence(
            comm, rank, size, data_rdd, w, prev_loss, tol, epoch)
        sync_time_s = time.perf_counter() - t_sync

        epoch_time_s = time.perf_counter() - t_epoch
        w_norm       = float(np.linalg.norm(w))
        grad_norm    = float(np.linalg.norm(global_grad))
        prev_loss    = global_loss

        collector.record_epoch(
            epoch=epoch,
            w_norm=w_norm,
            grad_norm=grad_norm,
            loss=global_loss,
            lr=lr_t,
            spark_time_s=spark_time_s,
            sync_time_s=sync_time_s,
            epoch_time_s=epoch_time_s,
        )

        if rank == 0:
            logger.info(
                "epoch %d/%d  lr=%.5f  |w|=%.4f  |grad|=%.6f  loss=%.6f  "
                "spark=%.3fs  sync=%.3fs",
                epoch + 1, max_epochs, lr_t, w_norm, grad_norm, global_loss,
                spark_time_s, sync_time_s,
            )

        if converged:
            if rank == 0:
                logger.info("Converged at epoch %d (tol=%.1e)", epoch + 1, tol)
            break

    total_time_s = time.perf_counter() - t_total_start
    result       = collector.record_run(w=w, total_time_s=total_time_s,
                                        epochs_run=epoch + 1, converged=converged)
    collector.to_csv()
    collector.to_json()

    data_rdd.unpersist()
    spark.stop()

    return result
