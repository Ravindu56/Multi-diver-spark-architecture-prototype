# ================================================================
# mpj_spark/core/gossip_aggregator.py
#
# Adaptive Gossip Protocol for distributed centroid aggregation.
#
# Research Objective 1b / 2:
#   Cross-driver parameter synchronisation via gossip-style
#   peer exchange, reducing aggregation from O(N) root-gather
#   to O(log N) rounds as worker count scales.
#
# ── Phase Boundary Note (IMPORTANT) ─────────────────────────────
# Phase 1 (this file): single-machine multiprocessing prototype.
#   Gossip "exchange" is SIMULATED in the root coordinator:
#   worker states are held in a shared in-memory dict and the
#   root iterates over pairs, merging centroids on their behalf.
#   The gossip_queue is used only as a one-shot collection channel
#   (workers push once; root drains once at Round 0).
#
# Phase 3 (mpi4py, future): each worker will participate directly.
#   The root will broadcast peer assignments; workers will call
#   MPI Send/Recv to exchange centroid vectors per round; the
#   root will only check convergence and issue stop/continue.
#   The _weighted_avg / _hungarian_align helpers are transport-
#   agnostic and will be reused unchanged in Phase 3.
# ────────────────────────────────────────────────────────────────
#
# How it works (Phase 1 simulation)
# ----------------------------------
# Round 0  — drain gossip_queue, build state table
#             { worker_id: {centres, wcss, row_count,
#                           original_row_count} }
# Round 1+ — shuffle worker list; for each worker pick F random
#             peers; merge (weighted average with Hungarian align)
#             into worker's centroid vector; row_count accumulates
#             as blend weight (original_row_count stays fixed)
# Convergence — stop when max per-worker centroid drift
#             < convergence_threshold, OR max_rounds reached
# Final merge — weighted average of all post-gossip states using
#             original_row_count (not accumulated blend weight)
#
# Adaptive fan-out
# ----------------
# Fan-out F (peers contacted per worker per round) self-tunes:
#   drift dropped > 50% vs prev round → shrink F  (less comms)
#   drift dropped < 10% or increased  → grow  F  (push harder)
# ================================================================

import time
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Any, Optional


# ── helpers ───────────────────────────────────────────────────────

def _hungarian_align(
    reference: List[List[float]],
    candidate: List[List[float]],
) -> List[List[float]]:
    """
    Re-order candidate centroids to best match reference ordering
    using the Hungarian algorithm (O(k^3), negligible for k<=50).
    """
    ref  = np.array(reference)                              # (k, d)
    cand = np.array(candidate)                              # (k, d)
    diff = ref[:, np.newaxis, :] - cand[np.newaxis, :, :]  # (k, k, d)
    cost = np.linalg.norm(diff, axis=2)                     # (k, k)
    _, col_ind = linear_sum_assignment(cost)
    return [candidate[i] for i in col_ind.tolist()]


def _weighted_avg(
    centres_a: List[List[float]], rows_a: int,
    centres_b: List[List[float]], rows_b: int,
) -> List[List[float]]:
    """
    Weighted average of two centroid sets by row counts.
    B is Hungarian-aligned to A before averaging.
    """
    aligned_b = _hungarian_align(centres_a, centres_b)
    total     = rows_a + rows_b
    merged    = (
        (rows_a / total) * np.array(centres_a) +
        (rows_b / total) * np.array(aligned_b)
    )
    return merged.tolist()


def _per_worker_drift(
    old_centres: List[List[List[float]]],
    new_centres: List[List[List[float]]],
) -> List[float]:
    """
    Compute per-worker centroid drift.

    Both old[i] and new[i] are aligned against the same reference
    (old[0]) before computing Euclidean distance, so each
    (aligned_old[i], aligned_new[i]) pair corresponds to the same
    physical centroid ordering.

    Parameters
    ----------
    old_centres : list of (k, d) centroid lists — one per worker, pre-round
    new_centres : list of (k, d) centroid lists — one per worker, post-round

    Returns
    -------
    list[float] — max centroid displacement per worker
    """
    reference = old_centres[0]   # fixed alignment target for this round
    drifts = []
    for old_w, new_w in zip(old_centres, new_centres):
        aligned_old = _hungarian_align(reference, old_w)
        aligned_new = _hungarian_align(reference, new_w)
        a = np.array(aligned_old)   # (k, d)
        b = np.array(aligned_new)   # (k, d)
        drifts.append(float(np.max(np.linalg.norm(a - b, axis=1))))
    return drifts


# ── main class ────────────────────────────────────────────────────

class GossipAggregator:
    """
    Adaptive gossip-based centroid aggregation (Phase 1 simulation).

    Parameters
    ----------
    num_workers           : total number of worker processes
    convergence_threshold : stop when max centroid drift < this
    max_rounds            : hard cap on gossip rounds
    initial_fanout        : peers contacted per worker per round
                            (adaptive after round 1)
    seed                  : RNG seed for reproducible peer selection
    verbose               : print per-round diagnostics
    """

    def __init__(
        self,
        num_workers           : int,
        convergence_threshold : float = 1e-3,
        max_rounds            : int   = 10,
        initial_fanout        : int   = 2,
        seed                  : int   = 42,
        verbose               : bool  = True,
    ):
        self.num_workers           = num_workers
        self.convergence_threshold = convergence_threshold
        self.max_rounds            = max_rounds
        self.fanout                = max(1, min(initial_fanout, num_workers - 1))
        self.rng                   = random.Random(seed)
        self.verbose               = verbose
        self.round_log: List[Dict[str, Any]] = []

    # ── public API ────────────────────────────────────────────────

    def aggregate(
        self,
        gossip_queue,
        timeout_per_worker: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Collect worker states, run adaptive gossip rounds, return
        global centroid result.

        Returns
        -------
        dict with keys:
            centres, total_wcss, total_rows, num_workers,
            rounds_run, converged, round_log, agg_time_s
        """
        t_start = time.perf_counter()

        # ── Round 0: collect ─────────────────────────────────────
        states = self._collect_states(gossip_queue, timeout_per_worker)
        if len(states) != self.num_workers:
            raise RuntimeError(
                f'[Gossip] Expected {self.num_workers} states, '
                f'got {len(states)}. Worker failure?'
            )
        self._log(f'Round 0 — collected {len(states)} worker states.')

        # ── Gossip rounds ─────────────────────────────────────────
        prev_drift = None
        converged  = False
        rounds_run = 0

        for rnd in range(1, self.max_rounds + 1):
            rounds_run = rnd

            # Snapshot centroid vectors BEFORE this round for drift calc
            old_centres = [list(states[wid]['centres'])
                           for wid in sorted(states.keys())]

            # ── Peer exchange (Phase 1: root-simulated) ───────────
            # In Phase 3 (mpi4py) this loop body becomes MPI Send/Recv
            # calls issued by each worker process directly.
            worker_ids = list(states.keys())
            self.rng.shuffle(worker_ids)

            for wid in worker_ids:
                peers = self._pick_peers(wid, worker_ids)
                for pid in peers:
                    # Merge wid ← average(wid, pid)
                    # row_count is the mutable blend weight;
                    # original_row_count is never modified (Fix 1)
                    states[wid]['centres'] = _weighted_avg(
                        states[wid]['centres'], states[wid]['row_count'],
                        states[pid]['centres'], states[pid]['row_count'],
                    )
                    states[wid]['row_count'] = (
                        states[wid]['row_count'] + states[pid]['row_count']
                    )

            # ── Drift measurement (Fix 2) ─────────────────────────
            # Both old[i] and new[i] aligned to old[0] before distance
            # so pairs correspond to the same physical centroid ordering.
            new_centres = [list(states[wid]['centres'])
                           for wid in sorted(states.keys())]
            drifts    = _per_worker_drift(old_centres, new_centres)
            max_drift = max(drifts)

            round_info = {
                'round'    : rnd,
                'fanout'   : self.fanout,
                'max_drift': round(max_drift, 6),
                'per_worker_drift': [round(d, 6) for d in drifts],
            }
            self.round_log.append(round_info)
            self._log(
                f'Round {rnd} | fanout={self.fanout} | '
                f'max_drift={max_drift:.6f} | '
                f'threshold={self.convergence_threshold}'
            )

            if max_drift < self.convergence_threshold:
                converged = True
                self._log(
                    f'Converged after round {rnd} '
                    f'(drift {max_drift:.6f} < {self.convergence_threshold}).'
                )
                break

            self.fanout = self._adapt_fanout(max_drift, prev_drift, self.fanout)
            prev_drift  = max_drift

        # ── Final global merge (Fix 3) ────────────────────────────
        # Uses original_row_count — not the accumulated blend weight —
        # to prevent double-mixing of already-merged centroid vectors.
        final = self._global_merge(states)

        agg_time = time.perf_counter() - t_start
        self._print_summary(final, rounds_run, converged, agg_time)

        return {
            'centres'    : final['centres'],
            'total_wcss' : final['total_wcss'],
            'total_rows' : final['total_rows'],
            'num_workers': self.num_workers,
            'rounds_run' : rounds_run,
            'converged'  : converged,
            'round_log'  : self.round_log,
            'agg_time_s' : round(agg_time, 4),
        }

    # ── private helpers ───────────────────────────────────────────

    def _collect_states(
        self,
        gossip_queue,
        timeout: float,
    ) -> Dict[int, Dict]:
        """
        Drain gossip_queue into a state table.
        Stores original_row_count separately from the mutable
        row_count that accumulates as blend weight during exchange.
        """
        states = {}
        for _ in range(self.num_workers):
            item = gossip_queue.get(timeout=timeout)
            wid  = item['worker_id']
            states[wid] = {
                'worker_id'          : wid,
                'centres'            : [list(c) for c in item['centres']],
                'wcss'               : item['wcss'],
                'row_count'          : item['row_count'],   # mutable blend weight
                'original_row_count' : item['row_count'],   # immutable — never written after this
            }
        return states

    def _pick_peers(
        self,
        worker_id : int,
        all_ids   : List[int],
    ) -> List[int]:
        candidates = [i for i in all_ids if i != worker_id]
        n          = min(self.fanout, len(candidates))
        return self.rng.sample(candidates, n)

    def _adapt_fanout(
        self,
        current_drift : float,
        prev_drift    : Optional[float],
        fanout        : int,
    ) -> int:
        """
        Adaptive fan-out:
          ratio < 0.5  (big improvement) → shrink F
          ratio > 0.9  (slow/no improve) → grow  F
          otherwise                      → keep  F
        Clamped to [1, num_workers-1].
        """
        if prev_drift is None or prev_drift == 0.0:
            return fanout
        ratio = current_drift / prev_drift
        if ratio < 0.5:
            new_f = max(1, fanout - 1)
        elif ratio > 0.9:
            new_f = min(self.num_workers - 1, fanout + 1)
        else:
            new_f = fanout
        if new_f != fanout:
            self._log(f'Adaptive fanout: {fanout} → {new_f}  (drift_ratio={ratio:.3f})')
        return new_f

    def _global_merge(self, states: Dict[int, Dict]) -> Dict[str, Any]:
        """
        Final weighted average of all post-gossip states.

        Uses original_row_count (Fix 3) — not the accumulated
        row_count blend weight — so that workers that participated
        in more exchanges are not artificially over-weighted.
        """
        reference  = states[sorted(states.keys())[0]]['centres']
        k          = len(reference)
        d          = len(reference[0])
        total_rows = sum(s['original_row_count'] for s in states.values())
        total_wcss = sum(s['wcss']               for s in states.values())

        global_centres = np.zeros((k, d))
        for s in states.values():
            aligned = _hungarian_align(reference, s['centres'])
            weight  = s['original_row_count'] / total_rows
            global_centres += weight * np.array(aligned)

        return {
            'centres'   : global_centres.tolist(),
            'total_rows': total_rows,
            'total_wcss': total_wcss,
        }

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f'  [Gossip] {msg}')

    def _print_summary(
        self, final: Dict, rounds_run: int,
        converged: bool, agg_time: float,
    ) -> None:
        DASH = '\u2500' * 70
        print(f'\n{DASH}')
        print('  Gossip Aggregation Summary')
        print(DASH)
        print(f'  {"Workers":<22} {self.num_workers}')
        print(f'  {"Rounds run":<22} {rounds_run}')
        print(f'  {"Converged":<22} {converged}')
        print(f'  {"Agg time":<22} {agg_time:.4f} s')
        print(f'  {"Total rows":<22} {final["total_rows"]:,}')
        print(f'  {"Total WCSS":<22} {final["total_wcss"]:.4f}')
        print(f'  {"─"*40}')
        print('  Global centres:')
        for i, c in enumerate(final['centres']):
            preview = ', '.join(f'{v:.3f}' for v in c[:4])
            more    = '...' if len(c) > 4 else ''
            print(f'    C{i}: [{preview}{more}]')
        print(f'  {"─"*40}')
        print('  Round log:')
        for r in self.round_log:
            print(
                f'    Round {r["round"]:>2} | '
                f'fanout={r["fanout"]} | '
                f'max_drift={r["max_drift"]:.6f}'
            )
        print(DASH)
