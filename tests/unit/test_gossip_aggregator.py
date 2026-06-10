# =============================================================
# tests/unit/test_gossip_aggregator.py
#
# Unit tests for mpj_spark/core/gossip_aggregator.py
# Covers Phase 2: Queue-based Allreduce simulation.
#
# Research alignment:
#   - Objective 1b: cross-driver parameter synchronization
#   - Objective 1c: correctness on iterative ML workloads
# =============================================================
from multiprocessing import Queue

import numpy as np
import pytest

from mpj_spark.core.gossip_aggregator import (
    GossipAggregator,
    _hungarian_align,
    _per_worker_drift,
    _weighted_avg,
)
from tests.unit.conftest import push_worker_state


# =============================================================
# Helper: build a fresh queue pre-loaded with worker states
# =============================================================
def _make_queue(states):
    """
    states: list of (worker_id, centres, row_count, wcss) tuples
    Returns a Queue with all states already pushed.
    """
    q = Queue()
    for worker_id, centres, row_count, wcss in states:
        push_worker_state(q, worker_id, centres, row_count, wcss)
    return q


# =============================================================
# Section 1: Helper function unit tests
# =============================================================


class TestHungarianAlign:
    """Tests for the _hungarian_align() centroid reordering helper."""

    def test_identity_alignment(self):
        """Aligned to itself — order must be unchanged."""
        ref = [[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]
        result = _hungarian_align(ref, ref)
        np.testing.assert_array_almost_equal(result, ref)

    def test_reversal_is_corrected(self):
        """
        Candidate has centroids in reversed order relative to reference.
        Hungarian algorithm must swap them back.
        """
        ref = [[0.0, 0.0], [10.0, 10.0]]
        cand = [[10.0, 10.0], [0.0, 0.0]]  # reversed
        result = _hungarian_align(ref, cand)
        np.testing.assert_array_almost_equal(result, ref)

    def test_single_centroid_passthrough(self):
        """With k=1 there is nothing to permute."""
        ref = [[3.0, 3.0]]
        cand = [[3.0, 3.0]]
        result = _hungarian_align(ref, cand)
        np.testing.assert_array_almost_equal(result, ref)

    def test_higher_dimensional_centroids(self):
        """Alignment must work for d > 2 (e.g. d=5 feature space)."""
        ref = [[1] * 5, [9] * 5]
        cand = [[9] * 5, [1] * 5]  # reversed
        result = _hungarian_align(ref, cand)
        np.testing.assert_array_almost_equal(result, ref)


class TestWeightedAvg:
    """Tests for the _weighted_avg() centroid blending helper."""

    def test_equal_weights_produce_midpoint(self):
        """
        Two workers with equal row counts → global centroid is the
        simple arithmetic mean of the two centroid sets.
        """
        a = [[0.0, 0.0], [10.0, 10.0]]
        b = [[0.0, 0.0], [10.0, 10.0]]
        result = _weighted_avg(a, 100, b, 100)
        np.testing.assert_array_almost_equal(result, a)

    def test_unequal_weights_bias_toward_larger(self):
        """
        Worker A has 3x more rows than B → result must be closer to A.
        """
        a = [[0.0], [0.0]]  # centroids at 0
        b = [[4.0], [4.0]]  # centroids at 4
        result = _weighted_avg(a, 300, b, 100)  # A has 3x weight
        # Expected: 0*0.75 + 4*0.25 = 1.0
        np.testing.assert_array_almost_equal(result, [[1.0], [1.0]])

    def test_reversed_candidate_is_aligned_before_averaging(self):
        """
        If B's centroid labels are reversed, weighted_avg must still
        produce the correct mean (not blend cross-cluster).
        """
        a = [[0.0, 0.0], [10.0, 10.0]]
        b = [[10.0, 10.0], [0.0, 0.0]]  # reversed
        result = _weighted_avg(a, 50, b, 50)
        # After alignment, both sets are identical → mean == a
        np.testing.assert_array_almost_equal(result, a)


class TestPerWorkerDrift:
    """Tests for the _per_worker_drift() convergence metric."""

    def test_zero_drift_when_unchanged(self):
        centres = [[[1.0, 2.0], [3.0, 4.0]]] * 3
        drifts = _per_worker_drift(centres, centres)
        assert all(d == pytest.approx(0.0) for d in drifts)

    def test_drift_increases_with_movement(self):
        old = [[[0.0, 0.0], [10.0, 10.0]]]
        new = [[[1.0, 0.0], [10.0, 10.0]]]  # C0 moved by 1.0
        drifts = _per_worker_drift(old, new)
        assert drifts[0] == pytest.approx(1.0)


# =============================================================
# Section 2: GossipAggregator integration tests
# =============================================================


class TestGossipAggregatorCore:
    """Core aggregation behaviour tests."""

    def test_two_workers_converge(self):
        """
        Two nearly-identical workers should converge in 1–2 rounds.
        Validates basic gossip loop termination (Obj 1b).
        """
        q = _make_queue(
            [
                (0, [[2.0, 2.0], [8.0, 8.0]], 100, 1.0),
                (1, [[2.1, 2.1], [7.9, 7.9]], 100, 1.1),
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=False)
        result = agg.aggregate(q)
        assert result["converged"] is True

    def test_total_rows_preserved(self):
        """
        Sum of per-worker row counts must equal total_rows in result.
        """
        q = _make_queue(
            [
                (0, [[1.0, 1.0]], 150, 0.5),
                (1, [[1.0, 1.0]], 250, 0.5),
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=False)
        result = agg.aggregate(q)
        assert result["total_rows"] == 400

    def test_total_wcss_is_sum_of_workers(self):
        """total_wcss must equal sum of individual worker WCSS values."""
        q = _make_queue(
            [
                (0, [[1.0, 1.0]], 100, 2.5),
                (1, [[1.0, 1.0]], 100, 3.5),
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=False)
        result = agg.aggregate(q)
        assert result["total_wcss"] == pytest.approx(6.0)

    def test_result_keys_complete(self):
        """Result dict must contain all documented keys."""
        q = _make_queue(
            [
                (0, [[1.0, 1.0]], 50, 0.5),
                (1, [[1.0, 1.0]], 50, 0.5),
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=False)
        result = agg.aggregate(q)
        expected_keys = {
            "centres",
            "total_wcss",
            "total_rows",
            "num_workers",
            "rounds_run",
            "converged",
            "round_log",
            "agg_time_s",
        }
        assert expected_keys.issubset(result.keys())

    def test_num_workers_matches_config(self):
        """num_workers in result must match GossipAggregator config."""
        q = _make_queue(
            [
                (0, [[1.0, 1.0]], 50, 0.5),
                (1, [[1.0, 1.0]], 50, 0.5),
                (2, [[1.0, 1.0]], 50, 0.5),
            ]
        )
        agg = GossipAggregator(num_workers=3, verbose=False)
        result = agg.aggregate(q)
        assert result["num_workers"] == 3

    def test_centres_shape_preserved(self):
        """
        Output centres list must have same k and d as input.
        k=3 clusters, d=4 dimensions.
        """
        k, d = 3, 4
        centres = [[float(i)] * d for i in range(k)]
        q = _make_queue(
            [
                (0, centres, 100, 1.0),
                (1, centres, 100, 1.0),
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=False)
        result = agg.aggregate(q)
        assert len(result["centres"]) == k
        assert all(len(c) == d for c in result["centres"])


class TestGossipAggregatorEdgeCases:
    """Edge case and failure mode tests."""

    def test_missing_worker_raises_runtime_error(self):
        """
        If fewer states arrive than num_workers, aggregate() must
        raise RuntimeError (timeout path).
        """
        q = _make_queue(
            [
                (0, [[1.0, 1.0]], 50, 0.5),
                # Worker 1 never pushes — simulates crash
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=False)
        with pytest.raises((RuntimeError, Exception)):
            agg.aggregate(q, timeout_per_worker=0.3)

    def test_round_log_populated(self):
        """round_log must have at least one entry after aggregation."""
        q = _make_queue(
            [
                (0, [[0.0, 0.0], [5.0, 5.0]], 100, 1.0),
                (1, [[0.1, 0.1], [4.9, 4.9]], 100, 1.0),
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=False)
        result = agg.aggregate(q)
        assert len(result["round_log"]) >= 1

    def test_round_log_schema(self):
        """Each round_log entry must contain required keys."""
        q = _make_queue(
            [
                (0, [[1.0, 1.0]], 50, 0.5),
                (1, [[1.0, 1.0]], 50, 0.5),
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=False)
        result = agg.aggregate(q)
        for entry in result["round_log"]:
            assert {"round", "fanout", "max_drift", "per_worker_drift"}.issubset(entry.keys())


class TestGossipAggregatorSeedCentres:
    """
    Tests for Fix 1: seed_centres alignment reference (Obj 1b).
    Validates that global seed centroids from Phase 1b are used
    as the authoritative alignment reference during _global_merge.
    """

    def test_seed_centres_corrects_label_reversal(self):
        """
        Worker 1 has reversed centroid labels relative to seed.
        Result C0 must be near [2,2], C1 near [8,8] after alignment.
        """
        seed = [[2.0, 2.0], [8.0, 8.0]]
        q = _make_queue(
            [
                (0, [[2.0, 2.0], [8.0, 8.0]], 100, 1.0),
                (1, [[8.0, 8.0], [2.0, 2.0]], 100, 1.0),  # reversed labels
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=False)
        result = agg.aggregate(q, seed_centres=seed)
        # C0 should be near [2,2], not near [8,8]
        assert result["centres"][0][0] < 5.0
        assert result["centres"][1][0] > 5.0

    def test_no_seed_centres_falls_back_to_states0(self):
        """
        When seed_centres=None, aggregation still completes without error.
        (Fallback to states[0] alignment — original behaviour.)
        """
        q = _make_queue(
            [
                (0, [[1.0, 1.0], [9.0, 9.0]], 100, 1.0),
                (1, [[1.0, 1.0], [9.0, 9.0]], 100, 1.0),
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=False)
        result = agg.aggregate(q, seed_centres=None)
        assert result["centres"] is not None
        assert len(result["centres"]) == 2


class TestGossipAggregatorConvergence:
    """
    Convergence property tests — validates Objective 1c:
    iterative ML algorithms converge to a shared global model state.
    """

    def test_identical_workers_converge_in_one_round(self):
        """
        Workers with identical centroid states have zero drift → must
        converge on round 1.
        """
        centres = [[3.0, 3.0], [7.0, 7.0]]
        q = _make_queue(
            [
                (0, centres, 100, 1.0),
                (1, centres, 100, 1.0),
                (2, centres, 100, 1.0),
            ]
        )
        agg = GossipAggregator(num_workers=3, convergence_threshold=1e-3, verbose=False)
        result = agg.aggregate(q)
        assert result["converged"] is True
        assert result["rounds_run"] == 1

    def test_max_rounds_cap_respected(self):
        """
        When workers are very far apart, rounds_run must not exceed
        max_rounds even if convergence is not achieved.
        """
        q = _make_queue(
            [
                (0, [[0.0, 0.0], [1.0, 1.0]], 100, 1.0),
                (1, [[1000.0, 1000.0], [1001.0, 1001.0]], 100, 1.0),
            ]
        )
        agg = GossipAggregator(
            num_workers=2,
            max_rounds=3,
            convergence_threshold=1e-10,  # impossibly tight
            verbose=False,
        )
        result = agg.aggregate(q)
        assert result["rounds_run"] <= 3

    def test_agg_time_is_positive_float(self):
        """Execution timing must be a positive number."""
        q = _make_queue(
            [
                (0, [[1.0, 1.0]], 50, 0.5),
                (1, [[1.0, 1.0]], 50, 0.5),
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=False)
        result = agg.aggregate(q)
        assert result["agg_time_s"] > 0.0


# =============================================================
# Section 7: verbose=True coverage (lines 145, 264-267, 269, 323)
# =============================================================


class TestGossipAggregatorVerbose:
    """
    Exercises the verbose=True code paths to cover the 6 lines
    that were previously unreachable with verbose=False.

    Covered lines:
      - 145  : self._log() body (the print statement inside _log)
      - 264-267, 269 : _print_summary() body
      - 323  : _log() call inside _global_merge with no seed_reference
    """

    def test_verbose_aggregate_does_not_raise(self, capsys):
        """
        Running with verbose=True must complete without error and
        emit [Gossip] diagnostic output to stdout.
        Covers line 145 (_log print) and lines 264-267, 269 (_print_summary).
        """
        q = _make_queue(
            [
                (0, [[1.0, 1.0], [9.0, 9.0]], 100, 1.0),
                (1, [[1.0, 1.0], [9.0, 9.0]], 100, 1.0),
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=True)
        result = agg.aggregate(q)
        captured = capsys.readouterr()
        assert result["converged"] is True
        assert "[Gossip]" in captured.out

    def test_verbose_prints_summary_block(self, capsys):
        """
        _print_summary() must emit the 'Gossip Aggregation Summary'
        header. Covers lines 264-267, 269 in full.
        """
        q = _make_queue(
            [
                (0, [[2.0, 2.0]], 50, 0.5),
                (1, [[2.0, 2.0]], 50, 0.5),
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=True)
        agg.aggregate(q)
        captured = capsys.readouterr()
        assert "Gossip Aggregation Summary" in captured.out
        assert "Rounds run" in captured.out
        assert "Total WCSS" in captured.out

    def test_verbose_no_seed_logs_fallback_message(self, capsys):
        """
        Line 323: _global_merge logs 'aligning to states[0]' when
        seed_centres=None and verbose=True.
        """
        q = _make_queue(
            [
                (0, [[3.0, 3.0], [7.0, 7.0]], 100, 1.0),
                (1, [[3.0, 3.0], [7.0, 7.0]], 100, 1.0),
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=True)
        agg.aggregate(q, seed_centres=None)
        captured = capsys.readouterr()
        assert "states[0]" in captured.out

    def test_verbose_with_seed_logs_seed_reference_message(self, capsys):
        """
        When seed_centres is provided with verbose=True, the log
        must mention 'seed_reference' (the other _global_merge branch).
        """
        seed = [[3.0, 3.0], [7.0, 7.0]]
        q = _make_queue(
            [
                (0, [[3.0, 3.0], [7.0, 7.0]], 100, 1.0),
                (1, [[3.0, 3.0], [7.0, 7.0]], 100, 1.0),
            ]
        )
        agg = GossipAggregator(num_workers=2, verbose=True)
        agg.aggregate(q, seed_centres=seed)
        captured = capsys.readouterr()
        assert "seed_reference" in captured.out
