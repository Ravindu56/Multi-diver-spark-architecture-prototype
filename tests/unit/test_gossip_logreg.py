"""tests/unit/test_gossip_logreg.py
Unit tests for the P3-11 decentralized gossip modules (Issue #63).

CI-safe: no mpi4py runtime, no JVM, no Spark execution.  The ring
exchange is exercised through scripted fake communicators; the worker
module is checked for its sub-comm-only contract.  The dispatch test
documents the wiring contract and is expected to fail until
scripts/apply_p3_11_wiring.py has been applied.
"""

import inspect

import numpy as np
import pytest

from mpj_spark.core.gossip_protocol import (
    TAG_GOSSIP_EXCHANGE,
    consensus_mix,
    gossip_exchange,
    ring_neighbors,
)
from mpj_spark.core.sync_modes import MODE_GOSSIP


class _FakeRingComm:
    """Scripted stand-in for the worker sub-comm: sendrecv returns the
    pre-seeded state of the requested source rank."""

    def __init__(self, states_by_rank):
        self._states = states_by_rank
        self.calls = []

    def sendrecv(self, sendobj, dest, sendtag, source, recvtag):
        assert sendtag == recvtag == TAG_GOSSIP_EXCHANGE
        self.calls.append((dest, source))
        return self._states[source]


def _state(w, intercept=0.0, row_count=100, rank=0):
    return {"weights": list(w), "intercept": intercept, "row_count": row_count, "rank": rank}


class TestRingNeighbors:
    def test_size4_fanout1_contacts_immediate_neighbours(self):
        assert ring_neighbors(0, 4, fanout=1) == [1, 3]
        assert ring_neighbors(2, 4, fanout=1) == [3, 1]

    def test_size2_collapses_to_single_peer(self):
        assert ring_neighbors(0, 2, fanout=1) == [1]
        assert ring_neighbors(1, 2, fanout=3) == [0]

    def test_size1_has_no_neighbours(self):
        assert ring_neighbors(0, 1, fanout=1) == []

    def test_fanout2_covers_distance2_peers(self):
        assert sorted(ring_neighbors(0, 5, fanout=2)) == [1, 2, 3, 4]


class TestConsensusMix:
    def test_two_party_exact_consensus(self):
        self_state = _state([1.0, 3.0], intercept=0.0, row_count=100, rank=0)
        peer = _state([3.0, 5.0], intercept=1.0, row_count=300, rank=1)
        w, b = consensus_mix(self_state, [peer])
        np.testing.assert_allclose(w, [2.5, 4.5])  # 0.25/0.75 row weighting
        assert b == pytest.approx(0.75)

    def test_both_sides_compute_identical_mix(self):
        a = _state([1.0], intercept=0.0, row_count=100, rank=0)
        b = _state([5.0], intercept=2.0, row_count=100, rank=1)
        wa, ba = consensus_mix(a, [b])
        wb, bb = consensus_mix(b, [a])
        np.testing.assert_allclose(wa, wb)  # one-round consensus on a 2-ring
        assert ba == pytest.approx(bb)

    def test_fixed_point_when_states_identical(self):
        s = _state([2.0, -1.0], intercept=0.5, row_count=100, rank=0)
        w, b = consensus_mix(s, [dict(s, rank=1), dict(s, rank=2)])
        np.testing.assert_allclose(w, [2.0, -1.0])
        assert b == pytest.approx(0.5)

    def test_zero_rows_fall_back_to_uniform(self):
        a = _state([0.0], row_count=0, rank=0)
        b = _state([4.0], row_count=0, rank=1)
        w, _ = consensus_mix(a, [b])
        np.testing.assert_allclose(w, [2.0])


class TestGossipExchange:
    def test_four_worker_ring_fanout1(self):
        states = {r: _state([float(r)], rank=r) for r in range(4)}
        comm = _FakeRingComm(states)
        received = gossip_exchange(comm, states[0], rank=0, size=4, fanout=1)
        assert len(comm.calls) == 2  # one paired sendrecv per direction
        assert {m["rank"] for m in received} == {1, 3}

    def test_two_worker_ring_dedupes_peer(self):
        states = {r: _state([float(r)], rank=r) for r in range(2)}
        comm = _FakeRingComm(states)
        received = gossip_exchange(comm, states[0], rank=0, size=2, fanout=1)
        assert len(received) == 1
        assert received[0]["rank"] == 1

    def test_single_worker_no_exchange(self):
        comm = _FakeRingComm({0: _state([0.0], rank=0)})
        assert gossip_exchange(comm, {"rank": 0}, rank=0, size=1) == []


class TestWorkerModuleContract:
    def test_run_signature_subcomm_only(self):
        from mpj_spark.applications.logreg.gossip_run import run

        params = inspect.signature(run).parameters
        for required in ("partition_path", "comm", "rank", "num_workers", "fanout", "tol"):
            assert required in params
        assert "root_comm" not in params  # decentralized: no root channel

    def test_run_fails_fast_without_subcomm(self):
        from mpj_spark.applications.logreg.gossip_run import run

        with pytest.raises(RuntimeError, match="gossip"):
            run("x.csv", comm=None, rank=0, num_workers=2)

    def test_worker_module_imports_without_libmpi(self):
        import importlib

        mod = importlib.import_module("mpj_spark.applications.logreg.gossip_run")
        assert hasattr(mod, "run")


class TestDispatchContract:
    def test_gossip_dispatch_uses_worker_subcomm_only(self, monkeypatch):
        """run_worker_core must route gossip to gossip_run.run over the
        worker sub-comm (comm), with NO root_comm requirement."""
        import mpj_spark.applications.logreg.gossip_run as gossip_run

        calls = {}

        def _fake_run(**kwargs):
            calls.update(kwargs)
            return {
                "weight_vector": [0.0] * kwargs["num_features"],
                "intercept": 0.0,
                "train_accuracy": 0.0,
                "row_count": 0,
                "iterations_done": 0,
                "partition_path": kwargs["partition_path"],
                "iter_metrics": [],
                "sync_mode": MODE_GOSSIP,
            }

        monkeypatch.setattr(gossip_run, "run", _fake_run)

        from mpj_spark.workers.worker_process import run_worker_core

        sentinel = object()
        run_worker_core(
            worker_id=0,
            partition_path="/nonexistent/partition.csv",
            spark=None,
            worker_config={
                "app": "logreg",
                "sync_mode": MODE_GOSSIP,
                "num_workers": 2,
                "results_dir": "results",
                "logreg_iter": 3,
                "logreg_features": 4,
            },
            comm=sentinel,
            root_comm=None,  # gossip must NOT need COMM_WORLD
        )

        assert calls["comm"] is sentinel  # worker sub-comm, 0-based rank
        assert calls["rank"] == 0
        assert calls["num_workers"] == 2
        assert calls["max_iter"] == 3
        assert calls["num_features"] == 4
