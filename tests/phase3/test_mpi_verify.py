# =============================================================================
# tests/phase3/test_mpi_verify.py
# Phase 3 — Issue #8 — Step 1: MPI Environment Verification
#
# PURPOSE
# -------
# Confirm that the mpi4py / OpenMPI environment is correctly set up before
# any Allreduce centroid-sync code is written.  These tests are intentionally
# low-level: they test MPI primitives only, with no PySpark dependency.
#
# HOW TO RUN
# ----------
#   mpirun --oversubscribe -n 3 python -m pytest tests/phase3/test_mpi_verify.py -v
#
# All ranks execute the full test module.  Each test function guards its
# assertions with rank checks so that rank-specific logic stays readable.
#
# WHAT IS TESTED
# --------------
#   1. COMM_WORLD size >= 2   (Allreduce is meaningless with a single rank)
#   2. Point-to-point send/recv (rank 0 <-> rank 1) — basic transport health
#   3. MPI.Barrier across all ranks  — synchronisation primitive
#   4. Allreduce SUM on a scalar     — smoke test for the primitive used in
#                                      centroid sync (comm.Allreduce)
#   5. Allreduce SUM on a 1-D numpy array — proves numpy buffer protocol works
#      with mpi4py (required for centroid matrix Allreduce in Step 4)
# =============================================================================

import numpy as np
from mpi4py import MPI

# ---------------------------------------------------------------------------
# Module-level communicator — mirrors the pattern in mpj_spark_mpi.py so that
# these tests exercise the same global comm object the production code uses.
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ---------------------------------------------------------------------------
# Test 1 — Minimum rank count
# ---------------------------------------------------------------------------
def test_mpi_size_at_least_two():
    """
    Allreduce across a single process is a no-op and would never catch
    synchronisation bugs.  Enforce >= 2 ranks so every subsequent test
    exercises real inter-process communication.
    """
    assert size >= 2, (
        f"Expected at least 2 MPI ranks, got {size}.  "
        "Re-launch with: mpirun --oversubscribe -n 3 python -m pytest ..."
    )


# ---------------------------------------------------------------------------
# Test 2 — Point-to-point send / recv (rank 0 <-> rank 1)
# ---------------------------------------------------------------------------
def test_mpi_send_recv():
    """
    Validates that rank 0 can send a message to rank 1 and rank 1 can reply.
    This is the transport primitive underlying MpiQueue in mpj_spark_mpi.py.
    """
    TAG_PING = 100
    TAG_PONG = 101

    if rank == 0:
        comm.send("ping", dest=1, tag=TAG_PING)
        reply = comm.recv(source=1, tag=TAG_PONG)
        assert reply == "pong", f"rank 0 expected 'pong', got {reply!r}"

    elif rank == 1:
        msg = comm.recv(source=0, tag=TAG_PING)
        assert msg == "ping", f"rank 1 expected 'ping', got {msg!r}"
        comm.send("pong", dest=0, tag=TAG_PONG)

    # All other ranks skip — they are not part of this exchange
    comm.Barrier()  # ensure all ranks clear this test before next one


# ---------------------------------------------------------------------------
# Test 3 — MPI.Barrier synchronisation
# ---------------------------------------------------------------------------
def test_mpi_barrier():
    """
    Confirms that comm.Barrier() does not deadlock and returns on every rank.
    The Barrier is used after each K-Means iteration to synchronise all
    drivers before the next centroid broadcast.
    """
    # Each rank records its rank number; after the barrier all should proceed
    local_val = rank
    comm.Barrier()
    # If we reach here on every rank without hanging, the barrier works
    assert local_val == rank  # value unchanged — sanity check only


# ---------------------------------------------------------------------------
# Test 4 — Allreduce SUM on a Python scalar
# ---------------------------------------------------------------------------
def test_mpi_allreduce_scalar_sum():
    """
    Smoke-test comm.allreduce() (lowercase — for Python objects) on a scalar.
    Expected result: sum of all rank indices = 0 + 1 + ... + (size-1).

    Note: this uses the object-level allreduce (pickle-based).  Step 4 of
    the K-Means implementation will use the buffer-level Allreduce (uppercase)
    with numpy arrays for performance.  Both are tested here.
    """
    local_value = rank  # each rank contributes its own rank index
    global_sum = comm.allreduce(local_value, op=MPI.SUM)
    expected = sum(range(size))  # 0+1+...+(size-1)
    assert global_sum == expected, (
        f"Allreduce scalar SUM failed: got {global_sum}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Allreduce SUM on a numpy array (buffer protocol)
# ---------------------------------------------------------------------------
def test_mpi_allreduce_numpy_sum():
    """
    Validates comm.Allreduce() (uppercase — buffer-level, zero-copy) with a
    numpy float64 array.  This is the exact call pattern that will be used
    for centroid synchronisation in Step 4:

        comm.Allreduce(local_centroids, global_centroids, op=MPI.SUM)

    Each rank contributes an array filled with its rank index as a float.
    The global sum must equal rank_index_sum * np.ones(shape).
    """
    ARRAY_LEN = 8  # simulate 8-dimensional centroid vector for 1 cluster

    local_arr = np.full(ARRAY_LEN, float(rank), dtype=np.float64)
    global_arr = np.zeros(ARRAY_LEN, dtype=np.float64)

    comm.Allreduce(local_arr, global_arr, op=MPI.SUM)

    expected_val = float(sum(range(size)))  # 0.0 + 1.0 + ... + (size-1).0
    expected_arr = np.full(ARRAY_LEN, expected_val, dtype=np.float64)

    np.testing.assert_array_almost_equal(
        global_arr,
        expected_arr,
        decimal=10,
        err_msg=(
            f"Allreduce numpy SUM failed on rank {rank}: "
            f"got {global_arr}, expected {expected_arr}"
        ),
    )
