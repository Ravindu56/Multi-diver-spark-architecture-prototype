"""tests/unit/test_sync_modes.py
Unit tests for mpj_spark.core.sync_modes central registry.
"""

import pytest

from mpj_spark.core.sync_modes import (
    MODE_ALLREDUCE_MPI,
    MODE_HYBRID_PS_ALLREDUCE,
    MODE_NONE,
    MODE_PS_ASYNC,
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    REGISTRY,
    get_descriptor,
    is_mpi_required,
    normalize_sync_mode,
)


def test_registry_contains_all_canonical_modes():
    expected = {
        MODE_NONE,
        MODE_PS_SYNC_FEDAVG_QUEUE,
        MODE_PS_SYNC_FEDAVG_MPI,
        MODE_ALLREDUCE_MPI,
        MODE_PS_ASYNC,
        MODE_HYBRID_PS_ALLREDUCE,
    }
    assert set(REGISTRY.keys()) == expected


def test_normalization_canonical_names():
    assert normalize_sync_mode("none") == MODE_NONE
    assert normalize_sync_mode("ps_sync_fedavg_queue") == MODE_PS_SYNC_FEDAVG_QUEUE
    assert normalize_sync_mode("ps_sync_fedavg_mpi") == MODE_PS_SYNC_FEDAVG_MPI
    assert normalize_sync_mode("allreduce_mpi") == MODE_ALLREDUCE_MPI
    assert normalize_sync_mode("ps_async") == MODE_PS_ASYNC
    assert normalize_sync_mode("hybrid_ps_allreduce") == MODE_HYBRID_PS_ALLREDUCE


def test_normalization_aliases():
    assert normalize_sync_mode("no_sync") == MODE_NONE
    assert normalize_sync_mode("queue") == MODE_PS_SYNC_FEDAVG_QUEUE
    assert normalize_sync_mode("fedavg_queue") == MODE_PS_SYNC_FEDAVG_QUEUE
    assert normalize_sync_mode("fedavg_mpi") == MODE_PS_SYNC_FEDAVG_MPI
    assert normalize_sync_mode("mpi_fedavg") == MODE_PS_SYNC_FEDAVG_MPI
    assert normalize_sync_mode("mpi") == MODE_ALLREDUCE_MPI
    assert normalize_sync_mode("allreduce") == MODE_ALLREDUCE_MPI
    assert normalize_sync_mode("async") == MODE_PS_ASYNC
    assert normalize_sync_mode("async_ps") == MODE_PS_ASYNC
    assert normalize_sync_mode("hybrid") == MODE_HYBRID_PS_ALLREDUCE
    assert normalize_sync_mode("hybrid_ps_allreduce") == MODE_HYBRID_PS_ALLREDUCE


def test_normalization_default_and_case_insensitivity():
    assert normalize_sync_mode(None) == MODE_PS_SYNC_FEDAVG_MPI
    assert normalize_sync_mode("  PS_SYNC_FEDAVG_MPI  ") == MODE_PS_SYNC_FEDAVG_MPI
    assert normalize_sync_mode("QUEUE") == MODE_PS_SYNC_FEDAVG_QUEUE
    assert normalize_sync_mode("PS_Async") == MODE_PS_ASYNC
    assert normalize_sync_mode("Hybrid_PS_AllReduce") == MODE_HYBRID_PS_ALLREDUCE


def test_normalization_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown sync_mode 'invalid_strategy'"):
        normalize_sync_mode("invalid_strategy")


def test_get_descriptor_properties():
    desc_mpi = get_descriptor("ps_sync_fedavg_mpi")
    assert desc_mpi.name == MODE_PS_SYNC_FEDAVG_MPI
    assert desc_mpi.is_periodic is True
    assert desc_mpi.requires_mpi is True

    desc_none = get_descriptor("none")
    assert desc_none.name == MODE_NONE
    assert desc_none.is_periodic is False
    assert desc_none.requires_mpi is False

    desc_async = get_descriptor("ps_async")
    assert desc_async.name == MODE_PS_ASYNC
    assert desc_async.is_periodic is True
    assert desc_async.requires_mpi is True

    desc_hybrid = get_descriptor("hybrid_ps_allreduce")
    assert desc_hybrid.name == MODE_HYBRID_PS_ALLREDUCE
    assert desc_hybrid.is_periodic is True
    assert desc_hybrid.requires_mpi is True


def test_is_mpi_required_helper():
    assert is_mpi_required("ps_sync_fedavg_mpi") is True
    assert is_mpi_required("allreduce_mpi") is True
    assert is_mpi_required("ps_async") is True
    assert is_mpi_required("hybrid_ps_allreduce") is True
    assert is_mpi_required("ps_sync_fedavg_queue") is False
    assert is_mpi_required("none") is False
