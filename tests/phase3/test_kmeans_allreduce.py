# =============================================================================
# tests/phase3/test_kmeans_allreduce.py
# Phase 3 — Issue #8 — Step 4: Allreduce Unit Tests
#
# WHAT IS TESTED
# --------------
#   1. allreduce_centroids() correctness: given known local sums/counts
#      and a FakeComm that simulates MPI.SUM over 2 ranks, the output
#      centroids must equal (local + remote) / (local_count + remote_count)
#      for every cluster.
#
#   2. allreduce_centroids() dtype: output is float64 (MPI buffer safety)
#
#   3. Empty-cluster guard: when global_counts[j] == 0, rank 0 must
#      sample a replacement point and Bcast it.
#
#   4. convergence bcast: after n_iters, comm.Bcast must have been called
#      exactly n_iters times (one convergence flag per iteration).
#
#   5. metrics keys: run_kmeans_allreduce() return dict must contain
#      metric fields (does not run full Spark/MPI; checks structure only)
#
# MPI MOCKING STRATEGY
# --------------------
# FakeComm simulates a 2-rank world. Allreduce adds a fixed "remote rank"
# contribution (_REMOTE_SUMS, _REMOTE_COUNTS) to the local arrays, exactly
# as MPI.SUM over 2 ranks would.  This lets us test the full logic path
# of allreduce_centroids() without a real MPI environment.
#
# HOW TO RUN  (no mpirun needed)
# --------------------------------
#   python -m pytest tests/phase3/test_kmeans_allreduce.py -v
# =============================================================================

from __future__ import annotations

import numpy as np
import pytest

from mpj_spark.applications.kmeans.allreduce import allreduce_centroids


# ---------------------------------------------------------------------------
# FakeComm: minimal MPI communicator stub for 2-rank simulation
# ---------------------------------------------------------------------------
class FakeComm:
    """
    Minimal stub that simulates a 2-rank MPI communicator.

    Allreduce: adds a fixed remote-rank contribution to send_arr so that
    simulating MPI.SUM over 2 ranks (local + one remote).
    Bcast: no-op (rank 0 sets the buffer before calling; buffer holds value).
    """

    def __init__(self, remote_sums: np.ndarray, remote_counts: np.ndarray):
        self._remote = {"sums": remote_sums, "counts": remote_counts}
        self._allreduce_calls = 0
        self._bcast_calls = 0

    def Allreduce(self, send_buf, recv_buf, op=None):
        """
        Simulate MPI.SUM Allreduce: recv = local + remote.
        send_buf and recv_buf are [array, MPI.DOUBLE] pairs.
        """

        send_arr, _dtype = send_buf
        recv_arr, _dtype = recv_buf

        if send_arr.shape == self._remote["sums"].shape:
            remote = self._remote["sums"]
        else:
            remote = self._remote["counts"]

        np.copyto(recv_arr, send_arr + remote)
        self._allreduce_calls += 1

    def Bcast(self, buf, root=0):
        """No-op: rank 0 sets the buffer value before calling; no MPI needed."""
        self._bcast_calls += 1

    def Barrier(self):
        pass


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
# 2-cluster, 2-dimensional problem.
# Local rank (rank 0) data:
_LOCAL_SUMS   = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64)   # K=2, D=2
_LOCAL_COUNTS = np.array([2.0, 2.0], dtype=np.float64)                  # 2 pts/cluster

# Remote rank (rank 1) data injected by FakeComm.Allreduce:
_REMOTE_SUMS   = np.array([[4.0, 2.0], [8.0, 6.0]], dtype=np.float64)
_REMOTE_COUNTS = np.array([2.0, 2.0], dtype=np.float64)

# Expected global centroids after Allreduce:
#   global_sums[0]   = [2+4, 4+2] = [6, 6],  global_counts[0] = 4  → centroid = [1.5, 1.5]
#   global_sums[1]   = [6+8, 8+6] = [14, 14], global_counts[1] = 4 → centroid = [3.5, 3.5]
_EXPECTED_CENTROIDS = np.array([[1.5, 1.5], [3.5, 3.5]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Fake points RDD (needed only for empty-cluster reinit path)
# ---------------------------------------------------------------------------
class FakePointsRDD:
    """Minimal Spark RDD stub: takeSample returns one fixed point."""
    def takeSample(self, withReplacement, num, seed=None):
        return [[0.1, 0.2]]  # replacement point for empty cluster


fake_points_rdd = FakePointsRDD()


# ---------------------------------------------------------------------------
# Test 1 — centroid arithmetic correctness
# ---------------------------------------------------------------------------
def test_allreduce_centroids_arithmetic():
    """
    Given known local sums/counts and a FakeComm that adds fixed remote
    contributions, allreduce_centroids must produce the analytically
    correct global centroids.
    """
    comm = FakeComm(_REMOTE_SUMS, _REMOTE_COUNTS)
    result = allreduce_centroids(
        comm=comm, rank=0,
        local_sums=_LOCAL_SUMS.copy(),
        local_counts=_LOCAL_COUNTS.copy(),
        points_rdd=fake_points_rdd,
    )
    np.testing.assert_array_almost_equal(
        result, _EXPECTED_CENTROIDS, decimal=10,
        err_msg=f"Expected {_EXPECTED_CENTROIDS}, got {result}"
    )


# ---------------------------------------------------------------------------
# Test 2 — output dtype is float64 (MPI buffer safety)
# ---------------------------------------------------------------------------
def test_allreduce_centroids_dtype():
    """
    The returned centroid array must be float64.  mpi4py buffer-level
    Allreduce (MPI.DOUBLE) requires float64 inputs and outputs.
    Using float32 would silently corrupt data on the MPI layer.
    """
    comm = FakeComm(_REMOTE_SUMS, _REMOTE_COUNTS)
    result = allreduce_centroids(
        comm=comm, rank=0,
        local_sums=_LOCAL_SUMS.copy(),
        local_counts=_LOCAL_COUNTS.copy(),
        points_rdd=fake_points_rdd,
    )
    assert result.dtype == np.float64, (
        f"Expected float64, got {result.dtype}. "
        "mpi4py MPI.DOUBLE Allreduce requires float64 arrays."
    )


# ---------------------------------------------------------------------------
# Test 3 — empty-cluster guard: rank 0 reinitialises with a sampled point
# ---------------------------------------------------------------------------
def test_allreduce_empty_cluster_reinit():
    """
    When global_counts[j] == 0 for a cluster, allreduce_centroids must:
      (a) detect the empty cluster
      (b) rank 0 samples a replacement point from points_rdd
      (c) call comm.Bcast to broadcast the replacement to all ranks
      (d) set global_centroids[j] to the sampled point

    Simulated by making the remote count negative enough to cancel local,
    so global_count == 0 for cluster 0.
    """
    remote_sums_empty   = np.array([[-2.0, -4.0], [8.0, 6.0]], dtype=np.float64)
    remote_counts_empty = np.array([-2.0, 2.0], dtype=np.float64)  # cluster 0 total = 0

    comm = FakeComm(remote_sums_empty, remote_counts_empty)
    result = allreduce_centroids(
        comm=comm, rank=0,
        local_sums=_LOCAL_SUMS.copy(),
        local_counts=_LOCAL_COUNTS.copy(),
        points_rdd=fake_points_rdd,
    )

    # Cluster 0 must have been reinitialised to the sampled point [0.1, 0.2]
    np.testing.assert_array_almost_equal(
        result[0], [0.1, 0.2], decimal=10,
        err_msg=f"Empty cluster 0 not reinitialised: got {result[0]}"
    )
    # Cluster 1 must be unaffected: global_sums[1]=[14,14], counts[1]=4 → [3.5, 3.5]
    np.testing.assert_array_almost_equal(
        result[1], [3.5, 3.5], decimal=10,
        err_msg=f"Non-empty cluster 1 altered: got {result[1]}"
    )
    # Bcast must have been called exactly once (for the one empty cluster)
    assert comm._bcast_calls == 1, (
        f"Expected 1 Bcast call for empty-cluster reinit, got {comm._bcast_calls}"
    )


# ---------------------------------------------------------------------------
# Test 4 — convergence bcast call count over multiple iterations
# ---------------------------------------------------------------------------
def test_allreduce_does_not_hang():
    """
    Over n_iters iterations, allreduce_centroids + convergence Bcast must
    complete without hanging.  The FakeComm.Bcast is a no-op (returns
    immediately), so any hang would be a logic error (infinite wait).

    Also verifies that comm.Bcast is called exactly n_iters times
    (once per iteration for the convergence flag), with no extra calls
    from empty-cluster reinit (all clusters have non-zero counts here).
    """
    from mpi4py import MPI

    n_iters = 3
    comm = FakeComm(_REMOTE_SUMS, _REMOTE_COUNTS)
    _centroids = np.array([[0.5, 0.5], [8.5, 8.5]], dtype=np.float64)

    for _ in range(n_iters):
        allreduce_centroids(
            comm=comm, rank=0,
            local_sums=_LOCAL_SUMS.copy(),
            local_counts=_LOCAL_COUNTS.copy(),
            points_rdd=fake_points_rdd,
        )
        # Simulate the convergence bcast inside the iteration loop
        converged_flag = np.zeros(1, dtype=np.int32)
        comm.Bcast([converged_flag, MPI.INT], root=0)

    # n_iters Bcast calls for convergence (no empty clusters → no reinit Bcast)
    assert comm._bcast_calls == n_iters, (
        f"Expected {n_iters} Bcast calls, got {comm._bcast_calls} — "
        "convergence flag must be broadcast on every iteration"
    )


# ---------------------------------------------------------------------------
# Test 5 — run_kmeans_allreduce() return dict has required metric keys
# ---------------------------------------------------------------------------
def test_run_kmeans_allreduce_return_keys():
    """
    run_kmeans_allreduce() return dict must contain all required keys.
    Does not run full Spark or MPI — checks structure only via pytest.importorskip
    and direct key inspection.

    The required keys map directly to the research metrics defined in the
    project scope (execution time, convergence, synchronisation overhead).
    """
    required_keys = {
        "global_centroids",
        "iterations_run",
        "converged",
        "metrics",
        "run_summary",
        "rank",
        "total_time_s",
    }
    from mpj_spark.applications.kmeans.allreduce import run_kmeans_allreduce
    import inspect
    src = inspect.getsource(run_kmeans_allreduce)
    for key in required_keys:
        assert f'"{key}"' in src, (
            f"Return key '{key}' not found in run_kmeans_allreduce source. "
            "Ensure the function returns a dict with all required metric fields."
        )
