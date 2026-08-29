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
_EXPECTED_SHIFT = float(np.linalg.norm(_NEW - _OLD, ord="fro"))  # ≈ 0.244949


# ---------------------------------------------------------------------------
# Test 1 — frobenius_shift arithmetic
# ---------------------------------------------------------------------------
def test_frobenius_shift_arithmetic():
    from mpj_spark.applications.kmeans.convergence import frobenius_shift

    result = frobenius_shift(_NEW, _OLD)
    assert (
        abs(result - _EXPECTED_SHIFT) < 1e-10
    ), f"Expected {_EXPECTED_SHIFT:.10f}, got {result:.10f}"


# ---------------------------------------------------------------------------
# Test 2 — frobenius_shift is zero when centroids are unchanged
# ---------------------------------------------------------------------------
def test_frobenius_shift_zero_when_unchanged():
    from mpj_spark.applications.kmeans.convergence import frobenius_shift

    result = frobenius_shift(_OLD, _OLD)
    assert result == 0.0, f"Expected 0.0 for identical arrays, got {result}"


# ---------------------------------------------------------------------------
# Test 3 — broadcast_convergence returns False on iteration 1 (guard)
# ---------------------------------------------------------------------------
def test_broadcast_convergence_iteration_1_guard():
    """
    Even with shift=0.0 (below any tol), iteration 1 must NOT converge.
    This prevents the loop from exiting before a single full pass is
    completed — a single-iteration run cannot be called converged.
    """
    from mpj_spark.applications.kmeans.convergence import broadcast_convergence

    comm = FakeComm()
    result = broadcast_convergence(comm=comm, rank=0, shift=0.0, tol=1e-4, iteration=1)
    assert result is False, f"Expected False on iteration 1 (guard), got {result}"


# ---------------------------------------------------------------------------
# Test 4 — broadcast_convergence returns False when shift >= tol
# ---------------------------------------------------------------------------
def test_broadcast_convergence_not_converged():
    from mpj_spark.applications.kmeans.convergence import broadcast_convergence

    comm = FakeComm()
    result = broadcast_convergence(comm=comm, rank=0, shift=0.5, tol=1e-4, iteration=5)
    assert result is False, f"Expected False (shift=0.5 >= tol=1e-4), got {result}"


# ---------------------------------------------------------------------------
# Test 5 — broadcast_convergence returns True when shift < tol AND iter > 1
# ---------------------------------------------------------------------------
def test_broadcast_convergence_converged():
    from mpj_spark.applications.kmeans.convergence import broadcast_convergence

    comm = FakeComm()
    result = broadcast_convergence(comm=comm, rank=0, shift=1e-6, tol=1e-4, iteration=3)
    assert result is True, f"Expected True (shift=1e-6 < tol=1e-4, iter=3), got {result}"


# ---------------------------------------------------------------------------
# Test 6 — broadcast_convergence issues exactly 1 Bcast call
# ---------------------------------------------------------------------------
def test_broadcast_convergence_bcast_call_count():
    """
    broadcast_convergence must call comm.Bcast exactly once per call,
    regardless of whether the convergence condition is met.  The Bcast
    propagates the converged flag to all non-root ranks.
    """
    from mpj_spark.applications.kmeans.convergence import broadcast_convergence

    comm = FakeComm()
    broadcast_convergence(comm=comm, rank=0, shift=0.5, tol=1e-4, iteration=2)
    broadcast_convergence(comm=comm, rank=0, shift=1e-6, tol=1e-4, iteration=3)

    assert (
        comm.bcast_calls == 2
    ), f"Expected 2 Bcast calls (one per broadcast_convergence call), got {comm.bcast_calls}"


# ---------------------------------------------------------------------------
# Test 7 — check_and_broadcast returns (bool, float) and shift matches
# ---------------------------------------------------------------------------
def test_check_and_broadcast_return_type_and_shift():
    """
    check_and_broadcast() must:
      (a) return a 2-tuple (converged: bool, shift: float)
      (b) shift == frobenius_shift(new, old) exactly
      (c) converged is True iff shift < tol AND iteration > 1
    """
    from mpj_spark.applications.kmeans.convergence import (
        check_and_broadcast,
        frobenius_shift,
    )

    comm = FakeComm()

    converged, shift = check_and_broadcast(
        comm=comm,
        rank=0,
        new_centroids=_NEW,
        prev_centroids=_OLD,
        tol=1.0,  # tol large enough that shift < tol → should converge
        iteration=2,
    )

    expected_shift = frobenius_shift(_NEW, _OLD)

    assert isinstance(converged, bool), f"converged must be bool, got {type(converged)}"
    assert isinstance(shift, float), f"shift must be float, got {type(shift)}"
    assert (
        abs(shift - expected_shift) < 1e-10
    ), f"shift={shift:.10f} != frobenius_shift={expected_shift:.10f}"
    assert (
        converged is True
    ), f"Expected True (shift={shift:.6f} < tol=1.0, iter=2), got {converged}"
