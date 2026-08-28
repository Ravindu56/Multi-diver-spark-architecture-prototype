"""mpj_spark/core/gossip_protocol.py
Phase 3 - P3-11: generic decentralized gossip primitives (Issue #63).

Transport-agnostic, CI-safe core for gossip-based parameter consensus:
  - ring_neighbors():  fixed-ring neighbour selection with fanout
  - consensus_mix():   row-count-weighted diffusion of parameter states
  - gossip_exchange(): deadlock-free paired sendrecv ring exchange

No mpi4py import at module level; the exchange helper works with any
communicator implementing sendrecv().  Used by
applications/logreg/gossip_run.py on the worker-only sub-communicator.
Decentralized by design: no root coordinator, no collectives, no global
barrier - each worker syncs only with its ring neighbours per round
(D-PSGD-style diffusion).
"""

from __future__ import annotations

from typing import Any

import numpy as np

TAG_GOSSIP_EXCHANGE = 35


def ring_neighbors(rank: int, size: int, fanout: int = 1) -> list[int]:
    """Distinct ring neighbours contacted per round.

    fanout is the ring distance: fanout=1 contacts the immediate
    neighbours (rank±1), fanout=2 also contacts rank±2, and so on.
    Degenerate rings collapse gracefully: size<=1 -> no neighbours,
    size=2 -> the single peer (the ±1 offsets coincide).
    """
    if size <= 1:
        return []
    max_dist = max(1, min(fanout, size // 2))
    peers: list[int] = []
    for d in range(1, max_dist + 1):
        for off in (d, -d):
            peer = (rank + off) % size
            if peer != rank and peer not in peers:
                peers.append(peer)
    return peers


def consensus_mix(self_state: dict[str, Any], peer_states: list[dict[str, Any]]):
    """Row-count-weighted diffusion mixing of parameter states.

    Each state is {"weights": list[float], "intercept": float,
    "row_count": int}.  Static row counts act as the mixing weights
    (weighted diffusion); with equal partitions this is uniform
    averaging, and on a 2-worker ring both sides compute the identical
    mix - exact one-round consensus.  Mixing identical states is a
    fixed point (mix returns the same state).

    Returns (mixed_weights: np.ndarray, mixed_intercept: float).
    """
    states = [self_state, *peer_states]
    total = float(sum(int(s["row_count"]) for s in states))
    w_sum = np.zeros(len(self_state["weights"]), dtype=np.float64)
    b_sum = 0.0
    for s in states:
        frac = (int(s["row_count"]) / total) if total > 0 else (1.0 / len(states))
        w_sum += frac * np.asarray(s["weights"], dtype=np.float64)
        b_sum += frac * float(s["intercept"])
    return w_sum, float(b_sum)


def gossip_exchange(
    comm,
    payload: dict[str, Any],
    rank: int,
    size: int,
    fanout: int = 1,
    tag: int = TAG_GOSSIP_EXCHANGE,
) -> list[dict[str, Any]]:
    """One round of paired ring exchanges (deadlock-free sendrecv).

    For each ring distance d = 1..max_dist two paired exchanges run:
      sendrecv(dest=rank+d, source=rank-d)  -> receives the left state
      sendrecv(dest=rank-d, source=rank+d)  -> receives the right state
    All ranks execute the same sequence of paired calls, so every send
    is matched by its counterpart's recv.  Returns the distinct received
    neighbour states, deduplicated by sender sub-comm rank (matters on
    rings of size 2 or 4 where the ±d offsets coincide).
    """
    if size <= 1:
        return []
    max_dist = max(1, min(fanout, size // 2))
    received: list[dict[str, Any]] = []
    seen: set[int] = set()
    for d in range(1, max_dist + 1):
        right = (rank + d) % size
        left = (rank - d) % size
        for dest, source in ((right, left), (left, right)):
            msg = comm.sendrecv(payload, dest=dest, sendtag=tag, source=source, recvtag=tag)
            sender = msg.get("rank", source) if isinstance(msg, dict) else source
            if sender not in seen:
                seen.add(sender)
                received.append(msg)
    return received
