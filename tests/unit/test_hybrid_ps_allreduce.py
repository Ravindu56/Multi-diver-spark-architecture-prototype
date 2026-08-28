"""tests/unit/test_hybrid_ps_allreduce.py
Unit tests for the P3-10 hybrid PS + Allreduce modules (Issue #62).

CI-safe: no mpi4py runtime, no JVM, no Spark execution.  The root
scalar-PS loop is exercised through a scripted fake communicator; the
worker module is checked for its two-communicator contract and its
lazy-mpi4py import guarantee.
"""

import inspect

import pytest

from mpj_spark.core.hybrid_ps import (
    TAG_ALLREDUCE_DOWN,
    TAG_ALLREDUCE_UP,
    _serve,
    row_weighted_mean,
    run_logreg_hybrid_scalar_ps,
)


class _FakeComm:
    """Scripted stand-in for the COMM_WORLD channel (recv/send subset)."""

    def __init__(self, inbox):
        self._inbox = {rank: list(msgs) for rank, msgs in inbox.items()}
        self.sent = []

    def recv(self, source, tag):
        return self._inbox[source].pop(0)

    def send(self, msg, dest, tag):
        self.sent.append((dest, tag, msg))


def _scalar_msg(intercept, row_count=100, worker_round=0):
    return {
        "intercept": intercept,
        "row_count": row_count,
        "rank": 0,
        "worker_round": worker_round,
    }


class TestRowWeightedMean:
    def test_equal_weights_plain_mean(self):
        assert row_weighted_mean([1.0, 3.0], [100.0, 100.0]) == pytest.approx(2.0)

    def test_unequal_weights_bias(self):
        assert row_weighted_mean([1.0, 4.0], [300.0, 100.0]) == pytest.approx(1.75)

    def test_zero_total_falls_back_to_plain_mean(self):
        assert row_weighted_mean([2.0, 4.0], [0.0, 0.0]) == pytest.approx(3.0)

    def test_empty_is_zero(self):
        assert row_weighted_mean([], []) == 0.0


class TestScalarServe:
    def test_two_workers_two_rounds(self):
        inbox = {
            1: [_scalar_msg(0.1, 100, 0), _scalar_msg(0.3, 100, 1)],
            2: [_scalar_msg(0.3, 300, 0), _scalar_msg(0.5, 300, 1)],
        }
        comm = _FakeComm(inbox)
        result = _serve(comm, num_workers=2, num_iterations=2)

        # round 1: (100*0.1 + 300*0.3)/400 = 0.25
        # round 2: (100*0.3 + 300*0.5)/400 = 0.45
        assert result["intercept"] == pytest.approx(0.45)
        assert result["weight_vector"] is None  # dense weights ride Allreduce
        assert result["iterations_done"] == 2

        replies = [m for _dest, tag, m in comm.sent if tag == TAG_ALLREDUCE_DOWN]
        assert len(replies) == 4  # 2 workers x 2 rounds
        round1 = [m for m in replies if m["iteration"] == 0]
        assert all(m["intercept"] == pytest.approx(0.25) for m in round1)
        assert {d for d, t, _m in comm.sent if t == TAG_ALLREDUCE_DOWN} == {1, 2}

    def test_tags_match_mpi_tag_allocation(self):
        # Consistent with root_mpi.py / async_ps.py tag allocation
        assert TAG_ALLREDUCE_UP == 30
        assert TAG_ALLREDUCE_DOWN == 31


class TestWorkerModuleContract:
    def test_run_signature_exposes_both_communicators(self):
        from mpj_spark.applications.logreg.hybrid_run import run

        params = inspect.signature(run).parameters
        for required in ("partition_path", "comm", "rank", "num_workers", "root_comm"):
            assert required in params
        assert params["root_comm"].default is None

    def test_run_fails_fast_without_both_comms(self):
        from mpj_spark.applications.logreg.hybrid_run import run

        with pytest.raises(RuntimeError, match="hybrid_ps_allreduce"):
            run("x.csv", comm=None, rank=0, num_workers=2, root_comm=None)

    def test_worker_module_imports_without_libmpi(self):
        import importlib

        mod = importlib.import_module("mpj_spark.applications.logreg.hybrid_run")
        assert hasattr(mod, "run")

    def test_root_entry_signature(self):
        params = inspect.signature(run_logreg_hybrid_scalar_ps).parameters
        for required in ("comm", "num_workers", "num_iterations"):
            assert required in params


class TestDispatchContract:
    def test_hybrid_mode_registered_and_requires_mpi(self):
        from mpj_spark.core.sync_modes import (
            MODE_HYBRID_PS_ALLREDUCE,
            is_mpi_required,
            normalize_sync_mode,
        )

        assert normalize_sync_mode("hybrid") == MODE_HYBRID_PS_ALLREDUCE
        assert is_mpi_required(MODE_HYBRID_PS_ALLREDUCE) is True
