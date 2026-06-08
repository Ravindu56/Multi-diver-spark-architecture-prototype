# =============================================================================
# tests/phase3/test_kmeans_convergence.py
# Phase 3 — Issue #8 — Step 5: Convergence Check Unit Tests
#
# WHAT IS TESTED
# --------------
#   1. frobenius_shift() arithmetic on a known (K, D) array pair
#   2. frobenius_shift() is zero when new == old (already converged)
#   3. broadcast_convergence() returns False on iteration 1 (guard)
#   4. broadcast_convergence() returns False when shift >= tol
#   5. broadcast_convergence() returns True when shift < tol AND iter > 1
#   6. broadcast_convergence() issues exactly 1 comm.Bcast call
#      regardless of convergence outcome
#   7. check_and_broadcast() returns (converged: bool, shift: float) tuple
#      and the shift value matches frobenius_shift() directly
#
# MPI MOCKING
# -----------
# FakeComm: Bcast is a no-op (rank 0 is always the caller; the buffer
# already contains the value set by rank 0).  Bcast call count is tracked.
#
# HOW TO RUN  (no mpirun needed)
# --------------------------------
#   python -m pytest tests/phase3/test_kmeans_convergence.py -v
# =============================================================================

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# FakeComm stub
# ---------------------------------------------------------------------------
class FakeComm:
    def __init__(self):
        self.bcast_calls = 0

    def Bcast(self, buf, root=0):
        # No-op: rank 0 sets the buffer value before calling; no MPI needed.
        self.bcast_calls += 1

    def Barrier(self):
        pass


# ---------------------------------------------------------------------------
# Shared centroid fixtures
# ---------------------------------------------------------------------------
# Two 3-cluster 2D centroid arrays with a known Frobenius distance.
# new_centroids[j] = old_centroids[j] + delta for all j
# delta = 0.1 in every element → shift = sqrt(K * D * delta^2)
#       = sqrt(3 * 2 * 0.01) = sqrt(0.06) ≈ 0.244949
_OLD = np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]], dtype=np.float64)
_NEW = np.array([[1.1, 1.1], [5.1, 5.1], [9.1, 9.1]], dtype=np.float64)
_EXPECTED_SHIFT = float(np.linalg.norm(_NEW - _OLD, ord='fro'))  # ≈ 0.244949


# ---------------------------------------------------------------------------
# Test 1 — frobenius_shift arithmetic
# ---------------------------------------------------------------------------
def test_frobenius_shift_arithmetic():
    from mpj_spark.applications.kmeans.convergence import frobenius_shift
    result = frobenius_shift(_NEW, _OLD)
    assert abs(result - _EXPECTED_SHIFT) < 1e-10, (
        f"Expected {_EXPECTED_SHIFT:.10f}, got {result:.10f}"
    )


# ---------------------------------------------------------------------------
# Test 2 — frobenius_shift is zero when centroids are unchanged
# ---------------------------------------------------------------------------
def test_frobenius_shift_zero_when_unchanged():
    from mpj_spark.applications.kmeans.convergence import frobenius_shift
    result = frobenius_shift(_OLD, _OLD)
    assert result == 0.0, (
        f"Expected 0.0 for identical arrays, got {result}"
    )


# ---------------------------------------------------------------------------
# Test 3 — broadcast_convergence returns False on iteration 1 (guard)
# ---------------------------------------------------------------------------
def test_broadcast_convergence_iteration_1_guard():
    """
    Even with shift=0.0 (below any tol), iteration 1 must NOT converge.
    This prevents the loop from exiting before any real Allreduce has run.
    """
    from mpj_spark.applications.kmeans.convergence import broadcast_convergence
    comm = FakeComm()
    result = broadcast_convergence(
        comm=comm, rank=0,
        shift=0.0,
        tol=1.0,     # very loose tolerance — would converge without the guard
        iteration=1,
    )
    assert result is False, (
        "Convergence must not be declared on iteration 1 regardless of shift"
    )


# ---------------------------------------------------------------------------
# Test 4 — broadcast_convergence returns False when shift >= tol
# ---------------------------------------------------------------------------
def test_broadcast_convergence_not_converged_shift_above_tol():
    from mpj_spark.applications.kmeans.convergence import broadcast_convergence
    comm = FakeComm()
    result = broadcast_convergence(
        comm=comm, rank=0,
        shift=0.5,   # above tol
        tol=1e-4,
        iteration=5,
    )
    assert result is False, (
        f"Expected False (shift=0.5 >= tol=1e-4), got {result}"
    )


# ---------------------------------------------------------------------------
# Test 5 — broadcast_convergence returns True when shift < tol AND iter > 1
# ---------------------------------------------------------------------------
def test_broadcast_convergence_converged():
    from mpj_spark.applications.kmeans.convergence import broadcast_convergence
    comm = FakeComm()
    result = broadcast_convergence(
        comm=comm, rank=0,
        shift=1e-6,  # well below tol
        tol=1e-4,
        iteration=10,
    )
    assert result is True, (
        f"Expected True (shift=1e-6 < tol=1e-4, iter=10), got {result}"
    )


# ---------------------------------------------------------------------------
# Test 6 — comm.Bcast is called exactly once per broadcast_convergence call
# ---------------------------------------------------------------------------
def test_bcast_called_exactly_once_per_call():
    """
    comm.Bcast must be called unconditionally — once per iteration —
    regardless of whether convergence is True or False.  Calling it inside
    a conditional branch would violate MPI collective correctness.
    """
    from mpj_spark.applications.kmeans.convergence import broadcast_convergence

    for shift, iteration, expected_converged in [
        (0.5,  5,  False),   # not converged
        (1e-6, 10, True),    # converged
        (0.0,  1,  False),   # iteration-1 guard
    ]:
        comm = FakeComm()
        broadcast_convergence(comm=comm, rank=0, shift=shift, tol=1e-4, iteration=iteration)
        assert comm.bcast_calls == 1, (
            f"Expected exactly 1 Bcast call, got {comm.bcast_calls} "
            f"(shift={shift}, iteration={iteration})"
        )


# ---------------------------------------------------------------------------
# Test 7 — check_and_broadcast returns (bool, float) with correct values
# ---------------------------------------------------------------------------
def test_check_and_broadcast_return_type_and_values():
    """
    check_and_broadcast() must return (converged: bool, shift: float).
    The shift value must equal frobenius_shift(new, prev) exactly.
    """
    from mpj_spark.applications.kmeans.convergence import (
        check_and_broadcast,
        frobenius_shift,
    )
    comm = FakeComm()
    converged, shift = check_and_broadcast(
        comm=comm, rank=0,
        new_centroids=_NEW,
        prev_centroids=_OLD,
        tol=1e-4,
        iteration=5,
    )
    assert isinstance(converged, bool), (
        f"converged must be bool, got {type(converged)}"
    )
    assert isinstance(shift, float), (
        f"shift must be float, got {type(shift)}"
    )
    expected_shift = frobenius_shift(_NEW, _OLD)
    assert abs(shift - expected_shift) < 1e-10, (
        f"shift mismatch: check_and_broadcast={shift:.10f}, "
        f"frobenius_shift={expected_shift:.10f}"
    )
    # shift ≈ 0.245 > tol=1e-4 → not converged
    assert converged is False, (
        f"Expected False (shift={shift:.4f} > tol=1e-4), got {converged}"
    )
