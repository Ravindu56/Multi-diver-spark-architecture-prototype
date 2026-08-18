"""tests/unit/test_async_ps.py
Unit tests for mpj_spark.core.async_ps (P3-09 asynchronous parameter server).

CI-safe: no mpi4py runtime, no JVM, no network.  The serving loop is
exercised through a scripted fake communicator implementing the
Iprobe/recv/send subset used by _serve().
"""

import numpy as np
import pytest

from mpj_spark.core.async_ps import (
    TAG_ALLREDUCE_DOWN,
    TAG_ALLREDUCE_UP,
    _serve,
    blend,
    effective_alpha,
)


class _FakeComm:
    """Scripted stand-in for the MPI communicator (Iprobe/recv/send subset)."""

    def __init__(self, inbox):
        self._inbox = list(inbox)
        self.sent = []

    def Iprobe(self, source, tag):
        return bool(self._inbox)

    def recv(self, source, tag):
        return self._inbox.pop(0)

    def send(self, msg, dest, tag):
        self.sent.append((dest, tag, msg))


def _msg(rank, worker_round, weights, intercept, base_version, row_count=100):
    return {
        "rank": rank,
        "worker_round": worker_round,
        "weights": weights,
        "intercept": intercept,
        "base_version": base_version,
        "row_count": row_count,
    }


# ---------------------------------------------------------------------------
# blend / effective_alpha
# ---------------------------------------------------------------------------


class TestBlend:
    def test_weighted_mix(self):
        w, b = blend(np.array([2.0, 0.0]), 1.0, np.array([4.0, 4.0]), 0.0, 0.25)
        np.testing.assert_allclose(w, [2.5, 1.0])
        assert b == pytest.approx(0.75)

    def test_alpha_one_copies_worker(self):
        w, b = blend(np.array([0.0, 0.0]), 0.0, np.array([3.0, -1.0]), 2.0, 1.0)
        np.testing.assert_allclose(w, [3.0, -1.0])
        assert b == pytest.approx(2.0)

    def test_alpha_zero_keeps_global(self):
        w, b = blend(np.array([1.0, 1.0]), 0.5, np.array([9.0, 9.0]), 9.0, 0.0)
        np.testing.assert_allclose(w, [1.0, 1.0])
        assert b == pytest.approx(0.5)


class TestEffectiveAlpha:
    def test_zero_staleness_returns_server_lr(self):
        assert effective_alpha(0.5, 0) == pytest.approx(0.5)

    def test_inverse_staleness_damping(self):
        assert effective_alpha(0.5, 1) == pytest.approx(0.25)
        assert effective_alpha(0.5, 3) == pytest.approx(0.125)

    def test_damping_disabled_is_constant(self):
        assert effective_alpha(0.5, 7, staleness_damping=False) == pytest.approx(0.5)

    def test_negative_staleness_clamped(self):
        assert effective_alpha(0.5, -3) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _serve — scripted two-worker scenario
# ---------------------------------------------------------------------------


class TestServe:
    def test_two_workers_three_rounds(self):
        inbox = [
            _msg(1, 0, [2.0, 0.0], 1.0, base_version=0),
            _msg(2, 0, [4.0, 4.0], 0.0, base_version=0, row_count=300),
            _msg(1, 1, [2.0, 2.0], 1.0, base_version=1),
            _msg(2, 1, [4.0, 0.0], 0.0, base_version=2, row_count=300),
            _msg(1, 2, [0.0, 2.0], 0.0, base_version=3),
            _msg(2, 2, [2.0, 2.0], 1.0, base_version=4, row_count=300),
        ]
        comm = _FakeComm(inbox)
        result = _serve(
            comm,
            num_workers=2,
            num_features=2,
            num_iterations=3,
            any_source=-1,
            server_lr=0.5,
            staleness_damping=True,
        )

        # First update bootstraps; every later update is stale by exactly 1
        # version (alpha = 0.5 / 2 = 0.25) in this interleaving.
        np.testing.assert_allclose(result["weight_vector"], [2.064453125, 1.40234375])
        assert result["intercept"] == pytest.approx(0.5927734375)
        assert result["global_version"] == 6

        # One reply per update, addressed back to the sending worker
        assert len(comm.sent) == 6
        assert [dest for dest, tag, _ in comm.sent] == [1, 2, 1, 2, 1, 2]
        assert all(tag == TAG_ALLREDUCE_DOWN for _, tag, _ in comm.sent)

        staleness = [r["staleness"] for r in result["staleness_records"]]
        assert staleness == [0, 1, 1, 1, 1, 1]
        assert result["mean_staleness"] == pytest.approx(5.0 / 6.0)
        assert result["max_staleness"] == 1

    def test_first_update_bootstraps_global_model(self):
        comm = _FakeComm([_msg(1, 0, [1.5, -2.5], 0.75, base_version=0)])
        result = _serve(
            comm, num_workers=1, num_features=2, num_iterations=1, any_source=-1
        )
        np.testing.assert_allclose(result["weight_vector"], [1.5, -2.5])
        assert result["intercept"] == pytest.approx(0.75)
        assert result["staleness_records"][0]["alpha_eff"] == pytest.approx(1.0)

    def test_damping_disabled_uses_constant_alpha(self):
        inbox = [
            _msg(1, 0, [1.0, 1.0], 0.0, base_version=0),
            _msg(2, 0, [3.0, 1.0], 1.0, base_version=0),
        ]
        comm = _FakeComm(inbox)
        result = _serve(
            comm,
            num_workers=2,
            num_features=2,
            num_iterations=1,
            any_source=-1,
            server_lr=0.5,
            staleness_damping=False,
        )
        # bootstrap -> [1,1],b=0 ; then alpha=0.5 -> [2,1],b=0.5
        np.testing.assert_allclose(result["weight_vector"], [2.0, 1.0])
        assert result["intercept"] == pytest.approx(0.5)
        assert result["staleness_records"][1]["alpha_eff"] == pytest.approx(0.5)

    def test_unexpected_rank_raises(self):
        comm = _FakeComm([_msg(5, 0, [1.0], 0.0, base_version=0)])
        with pytest.raises(ValueError, match="unexpected rank 5"):
            _serve(comm, num_workers=2, num_features=1, num_iterations=1, any_source=-1)

    def test_timeout_when_workers_silent(self):
        comm = _FakeComm([])
        with pytest.raises(TimeoutError, match="Timed out"):
            _serve(
                comm,
                num_workers=1,
                num_features=1,
                num_iterations=1,
                any_source=-1,
                timeout_s=0.05,
                poll_interval_s=0.005,
            )


# ---------------------------------------------------------------------------
# Module contract
# ---------------------------------------------------------------------------


class TestModuleContract:
    def test_run_signature_matches_sync_coordinator_contract(self):
        import inspect

        from mpj_spark.core.async_ps import run_logreg_async_ps

        params = inspect.signature(run_logreg_async_ps).parameters
        for required in ("comm", "num_workers", "num_iterations", "num_features"):
            assert required in params

    def test_tags_match_mpi_tag_allocation(self):
        # TAG_ALLREDUCE_UP=30 / TAG_ALLREDUCE_DOWN=31 per root_mpi.py
        assert TAG_ALLREDUCE_UP == 30
        assert TAG_ALLREDUCE_DOWN == 31
