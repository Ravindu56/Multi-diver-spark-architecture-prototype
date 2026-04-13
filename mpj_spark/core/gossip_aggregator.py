# ================================================================
# mpj_spark/core/gossip_aggregator.py
#
# Adaptive Gossip Protocol for distributed centroid aggregation.
#
# Research Objective 2:
#   "Develop resource allocation strategy to handle big data in
#    a cluster" — Gossip reduces aggregation from O(N) root-gather
#    to O(log N) peer-exchange rounds, cutting coordinator bottleneck
#    at large worker counts.
#
# How it works
# ------------
# After each worker finishes local KMeans, it pushes its centroids
# and row_count into a shared multiprocessing.Queue (gossip_queue).
# The GossipAggregator runs in the ROOT process and does:
#
#   Round 0  — collect ALL initial states from gossip_queue
#   Round 1+ — pair workers, exchange & average centroids
#              (Hungarian re-alignment per Fix #5 logic)
#   Stop     — when max centroid drift < convergence_threshold
#              OR max_rounds reached
#
# Adaptive fan-out
# ----------------
# Fan-out F (how many peers each node exchanges with per round)
# is adapted each round based on observed drift:
#   - If drift drops < 10% vs previous round  → shrink F  (save comms)
#   - If drift drops < 1%  vs previous round  → converged, stop early
#   - If drift is flat or growing             → grow F    (push harder)
# This is the "adaptive" part: the protocol self-tunes communication
# intensity without a fixed schedule.
#
# Plug-in integration
# -------------------
# The aggregator is completely decoupled from Spark and workers.
# It only needs a pre-populated multiprocessing.Queue of dicts:
#   { 'worker_id': int, 'centres': list[list[float]],
#     'wcss': float, 'row_count': int }
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
    Identical to the logic in root_process.aggregate_kmeans_results().
    """
    ref  = np.array(reference)   # (k, d)
    cand = np.array(candidate)   # (k, d)
    diff = ref[:, np.newaxis, :] - cand[np.newaxis, :, :]  # (k, k, d)
    cost = np.linalg.norm(diff, axis=2)                     # (k, k)
    _, col_ind = linear_sum_assignment(cost)
    return [candidate[i] for i in col_ind.tolist()]


def _weighted_avg(
    centres_a: List[List[float]], rows_a: int,
    centres_b: List[List[float]], rows_b: int,
) -> List[List[float]]:
    """
    Weighted average of two centroid sets, weighted by row counts.
    Alignment of B to A is done via Hungarian before averaging.
    """
    aligned_b = _hungarian_align(centres_a, centres_b)
    total     = rows_a + rows_b
    a         = np.array(centres_a)
    b         = np.array(aligned_b)
    merged    = (rows_a / total) * a + (rows_b / total) * b
    return merged.tolist()


def _max_drift(
    old_centres: List[List[float]],
    new_centres: List[List[float]],
) -> float:
    """Maximum Euclidean distance between matched old/new centroids."""
    a = np.array(old_centres)   # (k, d)
    b = np.array(new_centres)   # (k, d)
    return float(np.max(np.linalg.norm(a - b, axis=1)))


# ── main class ────────────────────────────────────────────────────

class GossipAggregator:
    """
    Adaptive gossip-based centroid aggregation.

    Parameters
    ----------
    num_workers          : int   — total number of worker processes
    convergence_threshold: float — stop when max centroid drift < this
                                   (default 1e-3, ~0.001 in feature space)
    max_rounds           : int   — hard cap on gossip rounds (default 10)
    initial_fanout       : int   — peers contacted per worker per round
                                   (default 2, adaptive after round 1)
    seed                 : int   — RNG seed for reproducible peer selection
    verbose              : bool  — print per-round diagnostics
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

        # per-round diagnostics for research logging
        self.round_log: List[Dict[str, Any]] = []

    # ── public API ────────────────────────────────────────────────

    def aggregate(
        self,
        gossip_queue,          # multiprocessing.Queue already populated by workers
        timeout_per_worker: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Collect worker states from gossip_queue, then run adaptive
        gossip rounds until convergence.

        Returns
        -------
        dict:
            centres      : list[list[float]]  — final global centroids
            total_wcss   : float
            total_rows   : int
            num_workers  : int
            rounds_run   : int
            converged    : bool
            round_log    : list[dict]         — per-round drift + fanout
            agg_time_s   : float
        """
        t_start = time.perf_counter()

        # ── Step 1: Drain queue into local state table ─────────────
        states = self._collect_states(gossip_queue, timeout_per_worker)
        if len(states) != self.num_workers:
            raise RuntimeError(
                f"[Gossip] Expected {self.num_workers} worker states, "
                f"got {len(states)}. Check for worker failures."
            )

        self._log(f"Round 0 — collected {len(states)} worker states.")

        # ── Step 2: Gossip rounds ──────────────────────────────────
        prev_drift   = None
        converged    = False
        rounds_run   = 0

        for rnd in range(1, self.max_rounds + 1):
            rounds_run  = rnd
            old_centres = [list(s['centres']) for s in states.values()]

            # Pair workers and exchange
            worker_ids = list(states.keys())
            self.rng.shuffle(worker_ids)

            for wid in worker_ids:
                peers = self._pick_peers(wid, worker_ids)
                for pid in peers:
                    # Merge wid ← pid  (in-place update of wid's state)
                    merged = _weighted_avg(
                        states[wid]['centres'], states[wid]['row_count'],
                        states[pid]['centres'], states[pid]['row_count'],
                    )
                    # Weighted-combine row counts too
                    total_rows           = states[wid]['row_count'] + states[pid]['row_count']
                    states[wid]['centres']   = merged
                    states[wid]['row_count'] = total_rows

            # Measure max drift across all workers
            new_centres = [list(s['centres']) for s in states.values()]
            # Align new to old for drift measurement
            aligned_new = [new_centres[0]]
            for nc in new_centres[1:]:
                aligned_new.append(_hungarian_align(new_centres[0], nc))
            aligned_old = [old_centres[0]]
            for oc in old_centres[1:]:
                aligned_old.append(_hungarian_align(old_centres[0], oc))

            drifts    = [_max_drift(o, n) for o, n in zip(aligned_old, aligned_new)]
            max_drift = max(drifts)

            round_info = {
                'round'    : rnd,
                'fanout'   : self.fanout,
                'max_drift': round(max_drift, 6),
            }
            self.round_log.append(round_info)
            self._log(
                f"Round {rnd} | fanout={self.fanout} | "
                f"max_drift={max_drift:.6f} | "
                f"threshold={self.convergence_threshold}"
            )

            # Convergence check
            if max_drift < self.convergence_threshold:
                converged = True
                self._log(f"Converged after round {rnd} (drift {max_drift:.6f} < {self.convergence_threshold}).")
                break

            # Adaptive fan-out adjustment
            self.fanout = self._adapt_fanout(
                max_drift, prev_drift, self.fanout
            )
            prev_drift = max_drift

        # ── Step 3: Final global centroid = weighted avg of all states ─
        final = self._global_merge(states)

        t_end    = time.perf_counter()
        agg_time = t_end - t_start

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
        """Drain gossip_queue and return {worker_id: state_dict}."""
        states = {}
        for _ in range(self.num_workers):
            item = gossip_queue.get(timeout=timeout)
            wid  = item['worker_id']
            states[wid] = {
                'worker_id' : wid,
                'centres'   : [list(c) for c in item['centres']],
                'wcss'      : item['wcss'],
                'row_count' : item['row_count'],
            }
        return states

    def _pick_peers(
        self,
        worker_id : int,
        all_ids   : List[int],
    ) -> List[int]:
        """Pick `fanout` random peers for worker_id (excluding itself)."""
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
        Adaptive fan-out rule:
          - prev_drift is None (round 1)         → keep initial fanout
          - drift dropped by > 50%               → shrink fanout (save BW)
          - drift dropped by < 10% or increased  → grow fanout (push harder)
          - otherwise                            → keep same
        Fan-out is clamped to [1, num_workers-1].
        """
        if prev_drift is None or prev_drift == 0.0:
            return fanout

        ratio = current_drift / prev_drift  # <1 = improving

        if ratio < 0.5:    # big improvement — reduce comms
            new_f = max(1, fanout - 1)
        elif ratio > 0.9:  # slow / no improvement — increase comms
            new_f = min(self.num_workers - 1, fanout + 1)
        else:
            new_f = fanout

        if new_f != fanout:
            self._log(f"[Gossip] Adaptive fanout: {fanout} → {new_f} (drift_ratio={ratio:.3f})")
        return new_f

    def _global_merge(
        self,
        states: Dict[int, Dict],
    ) -> Dict[str, Any]:
        """
        Produce a single global result by doing a final weighted
        average of all worker states (post-gossip).
        Reference = Worker 0 (or lowest available id).
        """
        wids      = sorted(states.keys())
        ref_id    = wids[0]
        reference = states[ref_id]['centres']

        k         = len(reference)
        d         = len(reference[0])
        total_rows = sum(s['row_count'] for s in states.values())
        total_wcss = sum(s['wcss']      for s in states.values())

        global_centres = np.zeros((k, d))
        for s in states.values():
            aligned = _hungarian_align(reference, s['centres'])
            weight  = s['row_count'] / total_rows
            global_centres += weight * np.array(aligned)

        return {
            'centres'    : global_centres.tolist(),
            'total_rows' : total_rows,
            'total_wcss' : total_wcss,
        }

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[GossipAggregator] {msg}")

    def _print_summary(self, final, rounds_run, converged, agg_time):
        SEP = '-' * 70
        print(f'\n{SEP}')
        print(f'  Gossip Aggregation Summary')
        print(SEP)
        print(f'  Workers          : {self.num_workers}')
        print(f'  Rounds run       : {rounds_run}')
        print(f'  Converged        : {converged}')
        print(f'  Aggregation time : {agg_time:.4f} s')
        print(f'  Total rows       : {final["total_rows"]:,}')
        print(f'  Total WCSS       : {final["total_wcss"]:.4f}')
        print(f'  Global Centres:')
        for i, c in enumerate(final['centres']):
            preview = ', '.join(f'{v:.3f}' for v in c[:4])
            more    = '...' if len(c) > 4 else ''
            print(f'    C{i}: [{preview}{more}]')
        print(SEP)
        print(f'  Round log:')
        for r in self.round_log:
            print(f'    Round {r["round"]:>2} | fanout={r["fanout"]} | max_drift={r["max_drift"]:.6f}')
        print(SEP)
