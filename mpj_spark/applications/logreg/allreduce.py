# =============================================================================
# mpj_spark/applications/logreg/allreduce.py
# Phase 3 — MPI Allreduce LogReg runner
#
# CHANGE LOG (global standardisation fix — call-site update)
# -----------------------------------------------------------
# load_and_cache_rdd() now requires comm and rank parameters so it can call
# _broadcast_global_stats() before caching the RDD.  The call site here is
# updated from:
#     data_rdd = load_and_cache_rdd(spark, partition_path, num_features)
# to:
#     data_rdd = load_and_cache_rdd(spark, partition_path, num_features, comm, rank)
# No other logic in this file is changed.
#
# CHANGE LOG (convergence-parity patch)
# ───────────────────────────────────────
# BUG — |w| divergence: allreduce_gradients() was applying a FIXED learning
#   rate with no decay.  Fix: cosine-decay LR schedule + L2 weight-decay
#   regularisation applied inside the gradient step.
#
# RETURN VALUE FIX (Issue #10 / validate_parity.py integration)
# ───────────────────────────────────────────────────────────────
# run_logreg_allreduce() returns explicit dict:
#   { 'weights', 'intercept', 'run_summary' }
#
# METRICS KWARG FIX
# ─────────────────
# record_epoch() kwargs corrected to match LogRegMetricsCollector signature.
#
# SYNC TIMING FIX (P5-05 / #30)
# ─────────────────────────────
# sync_time_s used to include check_loss_convergence(), which runs two Spark
# actions over the whole partition (loss map/reduce and count()) before the
# loss Allreduce. On the f10 100 MB smoke run that made sync_time_s ~2 s per
# epoch on every rank. The loss pass is now timed as loss_time_s, sync_time_s
# covers only the MPI calls (gradient Allreduce, loss Allreduce, convergence
# bcast, including any wait for slower ranks), and the loss pass reuses the
# row count from compute_gradient_spark(). The arithmetic is unchanged.
# =============================================================================

from __future__ import annotations

import logging
import math
import time

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# Learning-rate schedule  (unchanged)
# ===========================================================================


def _cosine_lr(epoch: int, max_epochs: int, lr_max: float, lr_min: float = 1e-4) -> float:
    """
    Cosine-decay learning rate schedule.

        lr_t = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*t/T))
    """
    if max_epochs <= 1:
        return lr_max
    decay = 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * epoch / max_epochs))
    return lr_min + decay


# ===========================================================================
# STEP 5a — Allreduce gradient synchronisation + weight update  (unchanged)
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

    w_new = w - lr * (grad_avg + reg_param * w)
    """
    from mpi4py import MPI

    global_grad = np.zeros_like(grad_local)
    comm.Allreduce(
        [grad_local, MPI.DOUBLE],
        [global_grad, MPI.DOUBLE],
        op=MPI.SUM,
    )
    global_grad /= float(size)
    w_new = w - learning_rate * (global_grad + reg_param * w)
    return w_new, global_grad


# ===========================================================================
# STEP 5b — Loss computation (Spark) + loss Allreduce / convergence (MPI)
# ===========================================================================
# The loss check is split into a Spark part and an MPI part so the runner can
# time them separately: compute_local_loss() is a full pass over the rank's
# partition and belongs in loss_time_s, not sync_time_s. The arithmetic is
# unchanged from the previous single function, so losses, convergence epochs
# and weights are identical.


def compute_local_loss(data_rdd, w: np.ndarray, n_local: int | None = None) -> float:
    """
    Mean cross-entropy loss over this rank's partition (one Spark pass).

    n_local is the partition's row count. Pass the value already returned by
    compute_gradient_spark() to avoid a second count() pass; if None, it is
    counted here.
    """
    _w = w
    _eps = 1e-15

    def local_loss_fn(row):
        x, y = row
        p = 1.0 / (1.0 + np.exp(-float(np.dot(x, _w))))
        p = float(np.clip(p, _eps, 1.0 - _eps))
        return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    local_loss_sum = float(data_rdd.map(local_loss_fn).reduce(lambda a, b: a + b))
    if n_local is None:
        n_local = data_rdd.count()
    return local_loss_sum / float(n_local) if n_local > 0 else 0.0


def allreduce_loss_and_check(
    comm,
    rank: int,
    size: int,
    local_loss: float,
    prev_loss: float,
    tol: float,
    epoch: int,
) -> tuple[bool, float]:
    """
    Average the per-rank losses with MPI Allreduce and broadcast rank 0's
    convergence decision (|prev_loss - global_loss| < tol).

    The global loss is the unweighted mean of per-rank mean losses. It equals
    the mean over all rows only when partitions have equal row counts, which
    partition_and_init_spark() produces to within one row.
    """
    from mpi4py import MPI

    local_arr = np.array([local_loss], dtype=np.float64)
    global_arr = np.zeros(1, dtype=np.float64)
    comm.Allreduce([local_arr, MPI.DOUBLE], [global_arr, MPI.DOUBLE], op=MPI.SUM)
    global_loss = float(global_arr[0]) / float(size)

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
        converged = False

    converged = comm.bcast(converged, root=0)
    return converged, global_loss


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

    Kept for callers of the previous API: compute_local_loss() followed by
    allreduce_loss_and_check(). The runner calls the two parts directly so it
    can time them separately.
    """
    local_loss = compute_local_loss(data_rdd, w)
    return allreduce_loss_and_check(comm, rank, size, local_loss, prev_loss, tol, epoch)


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

    FIX (this commit): load_and_cache_rdd() now standardises features using
    global mean/std broadcast from rank 0, matching MLlib's default
    StandardScaler behaviour.

    Returns
    -------
    dict with keys:
        'weights'    : list[float]  -- final weight vector
        'intercept'  : float        -- 0.0
        'run_summary': dict         -- epochs_run, converged, total_time_s
    """
    from mpj_spark.applications.logreg.local_gradient import (
        compute_gradient_spark,
        cores_per_worker,
        load_and_cache_rdd,
    )
    from mpj_spark.applications.logreg.metrics import LogRegMetricsCollector
    from mpj_spark.applications.logreg.partition import partition_and_init_spark

    t_total_start = time.perf_counter()
    collector = LogRegMetricsCollector(rank=rank, output_dir=metrics_output_dir)

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

    # Step 4a — load, standardise (global stats via Bcast), and cache RDD
    # FIX: pass comm and rank so _broadcast_global_stats() can be called
    data_rdd = load_and_cache_rdd(spark, partition_path, num_features, comm, rank)

    # Weight init: zeros in normalised feature space
    w = np.zeros(num_features, dtype=np.float64)

    comm.Barrier()
    logger.info("[rank %d] Barrier passed — starting epoch loop", rank)

    prev_loss = float("inf")
    converged = False
    epoch = 0

    for epoch in range(max_epochs):
        t_epoch = time.perf_counter()

        # Step 4b — local gradient (Spark)
        t_spark = time.perf_counter()
        grad_local, n_local = compute_gradient_spark(data_rdd, w)
        spark_time_s = time.perf_counter() - t_spark

        lr_t = _cosine_lr(epoch, max_epochs, lr_max=learning_rate)

        # Step 5a — gradient Allreduce + weight update (MPI)
        t_sync = time.perf_counter()
        w, global_grad = allreduce_gradients(
            comm, size, w, grad_local, learning_rate=lr_t, reg_param=reg_param
        )
        sync_time_s = time.perf_counter() - t_sync

        # Step 5b — local loss on the updated weights (Spark). Reuses the row
        # count from compute_gradient_spark() instead of a second count() pass.
        t_loss = time.perf_counter()
        local_loss = compute_local_loss(data_rdd, w, n_local)
        loss_time_s = time.perf_counter() - t_loss

        # Step 5c — loss Allreduce + convergence broadcast (MPI)
        t_sync = time.perf_counter()
        converged, global_loss = allreduce_loss_and_check(
            comm, rank, size, local_loss, prev_loss, tol, epoch
        )
        sync_time_s += time.perf_counter() - t_sync

        epoch_time_s = time.perf_counter() - t_epoch
        grad_norm = float(np.linalg.norm(global_grad))
        prev_loss = global_loss

        collector.record_epoch(
            epoch=epoch,
            spark_time_s=spark_time_s,
            sync_time_s=sync_time_s,
            epoch_time_s=epoch_time_s,
            grad_norm=grad_norm,
            global_loss=global_loss,
            weight_norm=float(np.linalg.norm(w)),
            loss_time_s=loss_time_s,
        )

        if rank == 0:
            logger.info(
                "epoch %d/%d  lr=%.5f  |w|=%.4f  |grad|=%.6f  loss=%.6f  "
                "spark=%.3fs  loss_pass=%.3fs  sync=%.3fs",
                epoch + 1,
                max_epochs,
                lr_t,
                float(np.linalg.norm(w)),
                grad_norm,
                global_loss,
                spark_time_s,
                loss_time_s,
                sync_time_s,
            )

        if converged:
            if rank == 0:
                logger.info("Converged at epoch %d (tol=%.1e)", epoch + 1, tol)
            break

    total_time_s = time.perf_counter() - t_total_start

    collector.record_run(
        total_time_s=total_time_s,
        epochs_run=epoch + 1,
        converged=converged,
        dataset_size=data_rdd.count(),
        num_ranks=size,
        learning_rate=learning_rate,
        tol=tol,
    )
    collector.to_csv()
    collector.to_json()

    data_rdd.unpersist()
    spark.stop()

    return {
        "weights": w.tolist(),
        "intercept": 0.0,
        "run_summary": {
            "epochs_run": epoch + 1,
            "converged": converged,
            "total_time_s": round(total_time_s, 4),
        },
    }
