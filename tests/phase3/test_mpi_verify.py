# =============================================================================
# tests/phase3/test_mpi_verify.py
# Phase 3 — MPI environment smoke tests
#
# SINGLE-RANK vs MULTI-RANK BEHAVIOUR
# ------------------------------------
# Tests that require >= 2 ranks are decorated with @_NEEDS_MULTI_RANK so
# that plain `pytest` (size=1) reports SKIPPED rather than FAILED.
# barrier and allreduce tests work at size=1 and always run.
#
# HOW TO RUN
#   mpirun --oversubscribe -n 3 python -m pytest tests/phase3/test_mpi_verify.py -v
# =============================================================================

from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TAG_PING = 42
TAG_PONG = 43

_NEEDS_MULTI_RANK = pytest.mark.skipif(
    size < 2,
    reason="Multi-rank MPI test — re-launch with: "
    "mpirun --oversubscribe -n 3 python -m pytest tests/phase3/test_mpi_verify.py",
)


@_NEEDS_MULTI_RANK
def test_mpi_size_at_least_two():
    """
    When this test runs (size >= 2 guaranteed by the skipif guard),
    the assertion is trivially true — the guard is the real check.
    """
    assert size >= 2, (
        f"Expected at least 2 MPI ranks, got {size}.  "
        f"Re-launch with: mpirun --oversubscribe -n 3 python -m pytest ..."
    )


@_NEEDS_MULTI_RANK
def test_mpi_send_recv():
    """
    Rank 0 sends a ping to rank 1; rank 1 sends a pong back.
    Verifies point-to-point MPI communication between two ranks.
    """
    if rank == 0:
        comm.send("ping", dest=1, tag=TAG_PING)
        pong = comm.recv(source=1, tag=TAG_PONG)
        assert pong == "pong", f"Expected 'pong', got {pong!r}"
    elif rank == 1:
        ping = comm.recv(source=0, tag=TAG_PING)
        assert ping == "ping", f"Expected 'ping', got {ping!r}"
        comm.send("pong", dest=0, tag=TAG_PONG)


def test_mpi_barrier():
    """MPI_Barrier must complete without deadlock at any rank count."""
    comm.Barrier()
    assert True


def test_mpi_allreduce_scalar_sum():
    """
    Each rank contributes its rank number; expected sum = size*(size-1)//2.
    Works correctly at size=1 (allreduce of a single value is identity).
    """
    result = comm.allreduce(rank, op=MPI.SUM)
    expected = size * (size - 1) // 2
    assert result == expected, f"Allreduce scalar: expected {expected}, got {result}"


def test_mpi_allreduce_numpy_sum():
    """
    Uppercase Allreduce over a NumPy array (buffer-like interface).
    Each rank contributes rank * ones(3); expected = sum(0..size-1) * ones(3).
    """
    local_arr = np.full(3, float(rank))
    result = np.zeros(3)
    comm.Allreduce(local_arr, result, op=MPI.SUM)
    expected_sum = float(size * (size - 1) // 2)
    np.testing.assert_allclose(
        result,
        np.full(3, expected_sum),
        err_msg=f"Allreduce numpy: expected {expected_sum}, got {result}",
    )
