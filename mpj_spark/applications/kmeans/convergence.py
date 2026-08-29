# =============================================================================
# mpj_spark/applications/kmeans/convergence.py
# Phase 3 — Issue #8 — Step 5: Convergence Check and Broadcast
#
# PURPOSE
# -------
# Own the stopping criterion for the K-Means Allreduce iteration loop.
# Provides three functions:
#
#   frobenius_shift(new, old)           — compute centroid movement
#   broadcast_convergence(comm, rank, shift, tol, iteration)
#                                       — rank 0 decides; bcast to all
#   check_and_broadcast(...)            — single combined call for the
#                                         loop body in allreduce.py
#
# WHY A SEPARATE MODULE?
# ----------------------
# The Step 4 commit (allreduce.py) inlined the convergence check directly
# in run_kmeans_allreduce(). Extracting it here provides:
#
#   1. TESTABILITY: convergence logic can be unit-tested without Spark
#      or a real MPI environment (FakeComm stub is sufficient).
#   2. REUSE: logreg.py (Issue #9, Step 4) needs the same broadcast
#      stopping-criterion pattern. Sharing this module avoids duplication.
#   3. CLARITY: allreduce.py's loop body becomes 3 function calls
#      (compute_local_stats → allreduce_centroids → check_and_broadcast)
#      with no convergence arithmetic scattered inline.
#
# MPI COLLECTIVE CONTRACT
# -----------------------
# broadcast_convergence() issues ONE buffer-level comm.Bcast([flag, MPI.INT])
# from root=0.  This call is a COLLECTIVE — every rank must call it on
# every iteration.  The caller (run_kmeans_allreduce) guarantees this by
# placing check_and_broadcast() unconditionally inside the loop body, never
# inside an if/else branch that only some ranks execute.
#
# ITERATION-1 GUARD
# -----------------
# On the very first iteration, prev_centroids is all-zeros (the initialisation
# value before any Allreduce result is available).  The Frobenius norm of
# (new_centroids - zeros) would therefore be large regardless of dataset
# properties, which is numerically correct but would falsely trigger
# tol-based convergence if tol is large.  The guard simply prevents
# convergence from being declared on iteration 1 — the loop always runs
# at least 2 full iterations before it can stop.
#
# BOUNDARY
# --------
#   Caller (allreduce.py)     : passes new_centroids, prev_centroids, comm,
#                               rank, tol, iteration
#   This module               : computes shift, decides converged on rank 0,
#                               broadcasts flag to all ranks
#   Returns to caller         : (converged: bool, shift: float)
#   Caller uses converged to  : break the iteration loop
# =============================================================================

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Frobenius norm of centroid shift
# ---------------------------------------------------------------------------


def frobenius_shift(
    new_centroids: np.ndarray,
    prev_centroids: np.ndarray,
) -> float:
    """
    Compute the Frobenius norm of the centroid displacement matrix.

    The Frobenius norm treats the K×D centroid array as a single matrix
    and measures its element-wise Euclidean distance from the previous
    iteration's centroids:

        shift = ||new_centroids - prev_centroids||_F
               = sqrt( sum_{j,d} (new[j,d] - old[j,d])^2 )

    This is the standard stopping criterion for K-Means because it captures
    movement in ALL centroids simultaneously — a single centroid that is
    still moving will keep the shift above tol even if the others converged.

    Alternative: per-centroid max shift (inf-norm).  Not used here because
    it is more sensitive to outlier cluster oscillation on unbalanced data.

    Parameters
    ----------
    new_centroids  : np.ndarray (K, D) — centroids after current Allreduce
    prev_centroids : np.ndarray (K, D) — centroids from previous iteration
                     (zeros on iteration 1 — see iteration-1 guard below)

    Returns
    -------
    float — Frobenius norm of the displacement, >= 0.0
    """
    return float(np.linalg.norm(new_centroids - prev_centroids, ord="fro"))


# ---------------------------------------------------------------------------
# 2. Convergence broadcast
# ---------------------------------------------------------------------------


def broadcast_convergence(
    comm,
    rank: int,
    shift: float,
    tol: float,
    iteration: int,
) -> bool:
    """
    Rank 0 evaluates the stopping criterion and broadcasts the result
    to all ranks.  All ranks return the same boolean.

    Stopping criterion
    ------------------
    Convergence is declared when BOTH conditions hold:
      (a) shift < tol      — centroids have stabilised
      (b) iteration > 1    — iteration-1 guard (see module docstring)

    Collective protocol
    -------------------
    This function issues exactly ONE buffer-level comm.Bcast call:

        converged_flag : np.ndarray([0 or 1], dtype=np.int32)

    Buffer-level Bcast (uppercase B) is used instead of object-level
    bcast (lowercase b) for dtype consistency with the rest of the
    MPI layer.  int32 is the narrowest type that MPI.INT maps to on
    all platforms (avoids platform-dependent int width issues).

    All ranks — including rank 0 — call comm.Bcast.  On rank 0 the send
    buffer IS the receive buffer (MPI in-place for Bcast from root).  On
    non-root ranks the buffer is zeroed before the call and filled by MPI.

    Parameters
    ----------
    comm      : mpi4py MPI communicator
    rank      : this process's rank
    shift     : Frobenius norm from frobenius_shift() — only used on rank 0
    tol       : convergence threshold (same value on all ranks)
    iteration : current 1-based iteration number

    Returns
    -------
    bool — True if all ranks should stop; False if loop should continue
    """
    from mpi4py import MPI

    # ---- Rank 0: evaluate criterion and set the flag -------------------
    if rank == 0:
        converged = (iteration > 1) and (shift < tol)
        converged_flag = np.array([1 if converged else 0], dtype=np.int32)
        if converged:
            logger.info(
                "[rank 0] Convergence declared at iteration %d (shift=%.8f < tol=%.2e)",
                iteration,
                shift,
                tol,
            )
        else:
            logger.debug(
                "[rank 0] Not converged at iteration %d (shift=%.8f, tol=%.2e, iter_guard=%s)",
                iteration,
                shift,
                tol,
                iteration <= 1,
            )
    else:
        # Non-root ranks allocate a zeroed buffer; MPI fills it from root
        converged_flag = np.zeros(1, dtype=np.int32)

    # ---- Collective: broadcast to all ranks ----------------------------
    # Every rank must call this on every iteration — no conditional.
    comm.Bcast([converged_flag, MPI.INT], root=0)

    return bool(converged_flag[0])


# ---------------------------------------------------------------------------
# 3. Combined check-and-broadcast  (used by allreduce.py loop body)
# ---------------------------------------------------------------------------


def check_and_broadcast(
    comm,
    rank: int,
    new_centroids: np.ndarray,
    prev_centroids: np.ndarray,
    tol: float,
    iteration: int,
) -> tuple[bool, float]:
    """
    Compute centroid shift and broadcast the convergence decision.

    This is the single call that replaces the inlined convergence block
    in run_kmeans_allreduce().  It wraps frobenius_shift() and
    broadcast_convergence() into one step with a consistent return type.

    Call site in allreduce.py (loop body, after Allreduce):

        converged, shift = check_and_broadcast(
            comm, rank, new_centroids, prev_centroids, tol, iteration
        )
        if converged:
            break

    Parameters
    ----------
    comm           : mpi4py MPI communicator
    rank           : this process's rank
    new_centroids  : np.ndarray (K, D) — centroids after this Allreduce
    prev_centroids : np.ndarray (K, D) — centroids from previous iteration
    tol            : convergence threshold (Frobenius norm)
    iteration      : current 1-based iteration counter

    Returns
    -------
    (converged, shift) : (bool, float)
        converged — True if loop should stop (all ranks agree)
        shift     — Frobenius norm value for metrics recording
    """
    shift = frobenius_shift(new_centroids, prev_centroids)
    converged = broadcast_convergence(comm, rank, shift, tol, iteration)
    return converged, shift
