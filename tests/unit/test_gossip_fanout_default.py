"""fanout=None (CLI flag not given) must behave like the default ring distance."""

from __future__ import annotations

import pytest

from mpj_spark.core.gossip_protocol import (
    DEFAULT_FANOUT,
    TAG_GOSSIP_EXCHANGE,
    gossip_exchange,
    ring_neighbors,
)


class _ScriptedComm:
    def __init__(self, states_by_rank):
        self._states = states_by_rank
        self.calls = []

    def sendrecv(self, sendobj, dest, sendtag, source, recvtag):
        assert sendtag == recvtag == TAG_GOSSIP_EXCHANGE
        self.calls.append((dest, source))
        return self._states[source]


def _states(n):
    return {
        r: {"weights": [float(r)], "intercept": 0.0, "row_count": 100, "rank": r} for r in range(n)
    }


@pytest.mark.parametrize("size", [2, 3, 4, 5, 8])
def test_ring_neighbors_none_equals_default(size):
    for rank in range(size):
        assert ring_neighbors(rank, size, None) == ring_neighbors(rank, size, DEFAULT_FANOUT)


def test_gossip_exchange_none_equals_default():
    states = _states(4)
    c_none, c_default = _ScriptedComm(states), _ScriptedComm(states)
    got_none = gossip_exchange(c_none, states[0], rank=0, size=4, fanout=None)
    got_default = gossip_exchange(c_default, states[0], rank=0, size=4, fanout=DEFAULT_FANOUT)
    assert c_none.calls == c_default.calls
    assert [m["rank"] for m in got_none] == [m["rank"] for m in got_default]


def test_default_fanout_matches_signature_default():
    assert ring_neighbors(0, 5) == ring_neighbors(0, 5, None)


def test_explicit_fanout_is_still_respected():
    assert sorted(ring_neighbors(0, 5, fanout=2)) == [1, 2, 3, 4]
    assert ring_neighbors(0, 4, fanout=1) == [1, 3]


def test_single_worker_has_no_neighbours_for_none():
    assert ring_neighbors(0, 1, None) == []
    assert (
        gossip_exchange(_ScriptedComm(_states(1)), {"rank": 0}, rank=0, size=1, fanout=None) == []
    )
