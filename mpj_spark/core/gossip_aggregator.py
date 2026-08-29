# ================================================================
# mpj_spark/core/gossip_aggregator.py
#
# Adaptive Gossip Protocol for distributed centroid aggregation.
#
# Fix 1 (this commit):
#   _global_merge now accepts an optional seed_reference centroid
#   list. When provided (global seeding ON), all post-gossip worker
#   states are aligned to seed_reference instead of states[0].
#
#   Rationale: states[0] is a gossip-blended mix whose ordering is
#   arbitrary when k > true cluster count. The global seed centroids
#   (computed from a 5% stratified sample in Phase 1b) are the
#   authoritative, stable ordering. Aligning to them ensures the
#   final weighted average sums corresponding physical clusters
#   rather than phantom averaged positions.
#
#   When seed_centres=None the fallback is states[0] — unchanged
#   from previous behaviour for runs without global seeding.
# ================================================================

import random
import time
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

# ── helpers ───────────────────────────────────────────────────────


def _hungarian_align(
    reference: list[list[float]],
    candidate: list[list[float]],
) -> list[list[float]]:
    """
    Re-order candidate centroids to best match reference ordering
    using the Hungarian algorithm (O(k^3), negligible for k<=50).
    """
    ref = np.array(reference)
    cand = np.array(candidate)
    diff = ref[:, np.newaxis, :] - cand[np.newaxis, :, :]
    cost = np.linalg.norm(diff, axis=2)
    _, col_ind = linear_sum_assignment(cost)
    return [candidate[i] for i in col_ind.tolist()]


def _weighted_avg(
    centres_a: list[list[float]],
    rows_a: int,
    centres_b: list[list[float]],
    rows_b: int,
) -> list[list[float]]:
    """
    Weighted average of two centroid sets by row counts.
    B is Hungarian-aligned to A before averaging.
    """
    aligned_b = _hungarian_align(centres_a, centres_b)
    total = rows_a + rows_b
    merged = (rows_a / total) * np.array(centres_a) + (rows_b / total) * np.array(aligned_b)
    return merged.tolist()


def _per_worker_drift(
    old_centres: list[list[list[float]]],
    new_centres: list[list[list[float]]],
) -> list[float]:
    reference = old_centres[0]
    drifts = []
    for old_w, new_w in zip(old_centres, new_centres, strict=False):
        aligned_old = _hungarian_align(reference, old_w)
        aligned_new = _hungarian_align(reference, new_w)
        a = np.array(aligned_old)
        b = np.array(aligned_new)
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
    seed                  : RNG seed for reproducible peer selection
    verbose               : print per-round diagnostics
    """

    def __init__(
        self,
        num_workers: int,
        convergence_threshold: float = 1e-3,
        max_rounds: int = 10,
        initial_fanout: int = 2,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.num_workers = num_workers
        self.convergence_threshold = convergence_threshold
        self.max_rounds = max_rounds
        self.fanout = max(1, min(initial_fanout, num_workers - 1))
        self.rng = random.Random(seed)
        self.verbose = verbose
        self.round_log: list[dict[str, Any]] = []

    # ── public API ────────────────────────────────────────────────

    def aggregate(
        self,
        gossip_queue,
        timeout_per_worker: float = 60.0,
        seed_centres: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """
        Collect worker states, run adaptive gossip rounds, return
        global centroid result.

        Parameters
        ----------
        gossip_queue       : multiprocessing.Queue — workers push centroid states
        timeout_per_worker : seconds to wait per worker message
        seed_centres       : optional list[list[float]] from Phase 1b global
                             seeding. When provided, used as the alignment
                             reference in _global_merge so the final weighted
                             average combines corresponding physical clusters.
                             When None, falls back to states[0] (old behaviour).

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
                f"[Gossip] Expected {self.num_workers} states, got {len(states)}. Worker failure?"
            )
        self._log(f"Round 0 — collected {len(states)} worker states.")

        # ── Gossip rounds ─────────────────────────────────────────
        prev_drift = None
        converged = False
        rounds_run = 0

        for rnd in range(1, self.max_rounds + 1):
            rounds_run = rnd

            old_centres = [list(states[wid]["centres"]) for wid in sorted(states.keys())]

            worker_ids = list(states.keys())
            self.rng.shuffle(worker_ids)

            for wid in worker_ids:
                peers = self._pick_peers(wid, worker_ids)
                for pid in peers:
                    states[wid]["centres"] = _weighted_avg(
                        states[wid]["centres"],
                        states[wid]["row_count"],
                        states[pid]["centres"],
                        states[pid]["row_count"],
                    )
                    states[wid]["row_count"] = states[wid]["row_count"] + states[pid]["row_count"]

            new_centres = [list(states[wid]["centres"]) for wid in sorted(states.keys())]
            drifts = _per_worker_drift(old_centres, new_centres)
            max_drift = max(drifts)

            round_info = {
                "round": rnd,
                "fanout": self.fanout,
                "max_drift": round(max_drift, 6),
                "per_worker_drift": [round(d, 6) for d in drifts],
            }
            self.round_log.append(round_info)
            self._log(
                f"Round {rnd} | fanout={self.fanout} | "
                f"max_drift={max_drift:.6f} | "
                f"threshold={self.convergence_threshold}"
            )

            if max_drift < self.convergence_threshold:
                converged = True
                self._log(
                    f"Converged after round {rnd} "
                    f"(drift {max_drift:.6f} < {self.convergence_threshold})."
                )
                break

            self.fanout = self._adapt_fanout(max_drift, prev_drift, self.fanout)
            prev_drift = max_drift

        # ── Final global merge ────────────────────────────────────
        # Uses seed_centres as alignment reference when available
        # (Fix 1), otherwise falls back to states[0].
        final = self._global_merge(states, seed_reference=seed_centres)

        agg_time = time.perf_counter() - t_start
        self._print_summary(final, rounds_run, converged, agg_time)

        return {
            "centres": final["centres"],
            "total_wcss": final["total_wcss"],
            "total_rows": final["total_rows"],
            "num_workers": self.num_workers,
            "rounds_run": rounds_run,
            "converged": converged,
            "round_log": self.round_log,
            "agg_time_s": round(agg_time, 4),
        }

    # ── private helpers ───────────────────────────────────────────

    def _collect_states(
        self,
        gossip_queue,
        timeout: float,
    ) -> dict[int, dict]:
        states = {}
        for _ in range(self.num_workers):
            item = gossip_queue.get(timeout=timeout)
            wid = item["worker_id"]
            states[wid] = {
                "worker_id": wid,
                "centres": [list(c) for c in item["centres"]],
                "wcss": item["wcss"],
                "row_count": item["row_count"],
                "original_row_count": item["row_count"],
            }
        return states

    def _pick_peers(
        self,
        worker_id: int,
        all_ids: list[int],
    ) -> list[int]:
        candidates = [i for i in all_ids if i != worker_id]
        n = min(self.fanout, len(candidates))
        return self.rng.sample(candidates, n)

    def _adapt_fanout(
        self,
        current_drift: float,
        prev_drift: float | None,
        fanout: int,
    ) -> int:
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
            self._log(f"Adaptive fanout: {fanout} → {new_f}  (drift_ratio={ratio:.3f})")
        return new_f

    def _global_merge(
        self,
        states: dict[int, dict],
        seed_reference: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """
        Final weighted average of all post-gossip states.

        Alignment reference priority:
          1. seed_reference (Phase 1b global seed centroids) — authoritative
             ordering computed from a 5% stratified sample before workers fork.
             Using this prevents phantom centroids when k > true cluster count
             because every worker's sub-cluster labels are resolved against the
             same stable reference.
          2. states[0]['centres'] — fallback when global seeding is off.

        Uses original_row_count (not accumulated blend weight) so workers that
        participated in more exchanges are not artificially over-weighted.
        """
        sorted_ids = sorted(states.keys())

        # Choose alignment reference
        if seed_reference is not None:
            reference = seed_reference
            self._log(
                f"_global_merge: aligning to seed_reference "
                f"({len(reference)} centroids from Phase 1b)"
            )
        else:
            reference = states[sorted_ids[0]]["centres"]
            self._log("_global_merge: aligning to states[0] (no seed reference)")

        k = len(reference)
        d = len(reference[0])
        total_rows = sum(s["original_row_count"] for s in states.values())
        total_wcss = sum(s["wcss"] for s in states.values())

        global_centres = np.zeros((k, d))
        for s in states.values():
            aligned = _hungarian_align(reference, s["centres"])
            weight = s["original_row_count"] / total_rows
            global_centres += weight * np.array(aligned)

        return {
            "centres": global_centres.tolist(),
            "total_rows": total_rows,
            "total_wcss": total_wcss,
        }

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [Gossip] {msg}")

    def _print_summary(
        self,
        final: dict,
        rounds_run: int,
        converged: bool,
        agg_time: float,
    ) -> None:
        DASH = "\u2500" * 70
        print(f"\n{DASH}")
        print("  Gossip Aggregation Summary")
        print(DASH)
        print(f"  {'Workers':<22} {self.num_workers}")
        print(f"  {'Rounds run':<22} {rounds_run}")
        print(f"  {'Converged':<22} {converged}")
        print(f"  {'Agg time':<22} {agg_time:.4f} s")
        print(f"  {'Total rows':<22} {final['total_rows']:,}")
        print(f"  {'Total WCSS':<22} {final['total_wcss']:.4f}")
        print(f"  {'─' * 40}")
        print("  Global centres:")
        for i, c in enumerate(final["centres"]):
            preview = ", ".join(f"{v:.3f}" for v in c[:4])
            more = "..." if len(c) > 4 else ""
            print(f"    C{i}: [{preview}{more}]")
        print(f"  {'─' * 40}")
        print("  Round log:")
        for r in self.round_log:
            print(
                f"    Round {r['round']:>2} | fanout={r['fanout']} | max_drift={r['max_drift']:.6f}"
            )
        print(DASH)
