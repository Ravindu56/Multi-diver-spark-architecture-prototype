# =============================================================
# tests/unit/test_root_process.py
#
# Unit tests for pure helper functions in
# mpj_spark/core/root_process.py
#
# Strategy: test only the stateless helper functions that do NOT
# spawn real subprocesses, JVMs, or Spark sessions. The heavy
# orchestration entry point (run_root) is covered by integration
# tests in a later phase.
#
# Functions under test:
#   - align_centres_hungarian()
#   - aggregate_kmeans_results()
#   - run_logreg_allreduce()
#   - aggregate_logreg_results()
#
# Research alignment:
#   - Objective 1b: cross-driver parameter synchronisation
#   - Objective 1c: correctness on iterative ML workloads
#   - Objective 2d: baseline validation helpers
# =============================================================
import pytest
import numpy as np
from multiprocessing import Queue

from mpj_spark.core.root_process import (
    align_centres_hungarian,
    aggregate_kmeans_results,
    run_logreg_allreduce,
    aggregate_logreg_results,
)


# =============================================================
# Section 1: align_centres_hungarian()
# =============================================================


class TestAlignCentresHungarian:
    """
    Tests for the Hungarian-algorithm centroid alignment helper.
    This is the correctness backbone of Phase 2 K-Means aggregation
    (Objective 1b — cross-driver label alignment before averaging).
    """

    def test_identity_reference_unchanged(self):
        """
        When reference and candidate are identical, the returned
        centroid list must match reference exactly.
        """
        ref = [[1.0, 2.0], [5.0, 6.0], [9.0, 9.0]]
        aligned, perm = align_centres_hungarian(ref, ref)
        np.testing.assert_array_almost_equal(aligned, ref)
        assert perm == [0, 1, 2]

    def test_reversed_labels_are_corrected(self):
        """
        Candidate with reversed centroid order must be reordered to
        match reference ordering after alignment.
        """
        ref = [[0.0, 0.0], [10.0, 10.0]]
        cand = [[10.0, 10.0], [0.0, 0.0]]  # reversed
        aligned, perm = align_centres_hungarian(ref, cand)
        np.testing.assert_array_almost_equal(aligned, ref)

    def test_returns_permutation_list(self):
        """Second return value must be a list of column indices."""
        ref = [[1.0, 1.0], [5.0, 5.0]]
        cand = [[5.0, 5.0], [1.0, 1.0]]
        _, perm = align_centres_hungarian(ref, cand)
        assert isinstance(perm, list)
        assert sorted(perm) == [0, 1]  # must be a valid permutation

    def test_three_clusters_correct_permutation(self):
        """3-cluster case: alignment must find the optimal permutation."""
        ref = [[0.0], [5.0], [10.0]]
        cand = [[10.0], [0.0], [5.0]]  # rotated
        aligned, _ = align_centres_hungarian(ref, cand)
        np.testing.assert_array_almost_equal(aligned, ref)

    def test_high_dimensional_centroids(self):
        """Alignment correctness must hold for d=8 feature space."""
        ref = [[float(i)] * 8 for i in range(4)]
        cand = list(reversed(ref))
        aligned, _ = align_centres_hungarian(ref, cand)
        np.testing.assert_array_almost_equal(aligned, ref)


# =============================================================
# Section 2: aggregate_kmeans_results()
# =============================================================


class TestAggregateKmeansResults:
    """
    Tests for the batch Hungarian K-Means aggregation path
    (used when use_gossip=False in run_root).
    """

    def _make_worker(self, worker_id, centres, row_count, wcss=1.0):
        return {
            "worker_id": worker_id,
            "centres": centres,
            "row_count": row_count,
            "wcss": wcss,
            "k": len(centres),
        }

    def test_result_has_required_keys(self):
        workers = [
            self._make_worker(0, [[1.0, 1.0], [9.0, 9.0]], 100),
            self._make_worker(1, [[1.0, 1.0], [9.0, 9.0]], 100),
        ]
        result = aggregate_kmeans_results(workers)
        assert {"centres", "total_wcss", "total_rows", "num_workers"}.issubset(result)

    def test_total_rows_summed(self):
        workers = [
            self._make_worker(0, [[1.0, 1.0]], 150, wcss=1.0),
            self._make_worker(1, [[1.0, 1.0]], 250, wcss=1.0),
        ]
        result = aggregate_kmeans_results(workers)
        assert result["total_rows"] == 400

    def test_total_wcss_summed(self):
        workers = [
            self._make_worker(0, [[1.0, 1.0]], 100, wcss=2.5),
            self._make_worker(1, [[1.0, 1.0]], 100, wcss=3.5),
        ]
        result = aggregate_kmeans_results(workers)
        assert result["total_wcss"] == pytest.approx(6.0)

    def test_identical_workers_produce_same_centres(self):
        """
        When both workers converge to the same centroids, the
        aggregated result must match those centroids exactly.
        """
        centres = [[2.0, 2.0], [8.0, 8.0]]
        workers = [
            self._make_worker(0, centres, 100),
            self._make_worker(1, centres, 100),
        ]
        result = aggregate_kmeans_results(workers)
        np.testing.assert_array_almost_equal(result["centres"], centres)

    def test_reversed_labels_aligned_before_averaging(self):
        """
        Worker 1 has reversed centroid labels. After Hungarian alignment
        and weighted average, C0 must be near [0,0] and C1 near [10,10].
        """
        w0 = self._make_worker(0, [[0.0, 0.0], [10.0, 10.0]], 100)
        w1 = self._make_worker(1, [[10.0, 10.0], [0.0, 0.0]], 100)  # reversed
        result = aggregate_kmeans_results([w0, w1])
        assert result["centres"][0][0] < 5.0  # C0 ≈ [0,0]
        assert result["centres"][1][0] > 5.0  # C1 ≈ [10,10]

    def test_weighted_average_biased_toward_larger_partition(self):
        """
        Worker 0 has 3x more rows than Worker 1.
        The global centroid must be 75% biased toward Worker 0.
        """
        w0 = self._make_worker(0, [[0.0]], 300)  # centroid at 0
        w1 = self._make_worker(1, [[4.0]], 100)  # centroid at 4
        result = aggregate_kmeans_results([w0, w1])
        # Expected: 0*0.75 + 4*0.25 = 1.0
        assert result["centres"][0][0] == pytest.approx(1.0)

    def test_num_workers_in_result(self):
        workers = [self._make_worker(i, [[float(i), float(i)]], 50) for i in range(3)]
        result = aggregate_kmeans_results(workers)
        assert result["num_workers"] == 3


# =============================================================
# Section 3: run_logreg_allreduce()
# =============================================================


class TestRunLogregAllreduce:
    """
    Tests for the root-side FedAvg coordinator (two-queue design).
    Workers are simulated by pushing messages directly onto the
    up_queue — no real subprocess spawning needed.

    IMPORTANT: every test creates its OWN fresh Queue pair and calls
    _simulate_workers BEFORE run_logreg_allreduce so that the
    coordinator finds all messages already waiting on up_q.
    Never share queue state between test methods.
    """

    def _simulate_workers(
        self, up_queue, num_workers, num_iterations, num_features, weights_factory=None
    ):
        """
        Push synthetic weight messages for all workers across all
        iterations onto up_queue, simulating what real worker
        processes would send.
        """
        for _ in range(num_iterations):
            for wid in range(num_workers):
                w = weights_factory(wid) if weights_factory else [1.0] * num_features
                up_queue.put(
                    {
                        "type": "weights",
                        "worker_id": wid,
                        "weights": w,
                        "intercept": 0.1 * wid,
                        "row_count": 100,
                    }
                )

    def test_result_keys_complete(self):
        up_q, down_q = Queue(), Queue()
        N, ITERS, FEAT = 2, 3, 4
        self._simulate_workers(up_q, N, ITERS, FEAT)
        result = run_logreg_allreduce(up_q, down_q, N, ITERS, FEAT)
        assert {"weight_vector", "intercept", "iterations_done"}.issubset(result)

    def test_iterations_done_matches_config(self):
        up_q, down_q = Queue(), Queue()
        N, ITERS, FEAT = 2, 5, 3
        self._simulate_workers(up_q, N, ITERS, FEAT)
        result = run_logreg_allreduce(up_q, down_q, N, ITERS, FEAT)
        assert result["iterations_done"] == ITERS

    def test_weight_vector_length_matches_features(self):
        up_q, down_q = Queue(), Queue()
        N, ITERS, FEAT = 3, 2, 6
        self._simulate_workers(up_q, N, ITERS, FEAT)
        result = run_logreg_allreduce(up_q, down_q, N, ITERS, FEAT)
        assert len(result["weight_vector"]) == FEAT

    def test_fedavg_equal_workers_produces_mean_weights(self):
        """
        Two workers with equal row counts and weights [0,0,...] and
        [2,2,...] respectively. FedAvg must produce [1,1,...] each iter.
        """
        up_q, down_q = Queue(), Queue()
        FEAT = 4
        # Worker 0: weights = [0]*FEAT, Worker 1: weights = [2]*FEAT
        for _ in range(1):  # single iteration
            up_q.put(
                {
                    "type": "weights",
                    "worker_id": 0,
                    "weights": [0.0] * FEAT,
                    "intercept": 0.0,
                    "row_count": 100,
                }
            )
            up_q.put(
                {
                    "type": "weights",
                    "worker_id": 1,
                    "weights": [2.0] * FEAT,
                    "intercept": 0.0,
                    "row_count": 100,
                }
            )
        result = run_logreg_allreduce(up_q, down_q, 2, 1, FEAT)
        np.testing.assert_array_almost_equal(result["weight_vector"], [1.0] * FEAT)

    def test_down_queue_receives_broadcast_messages(self):
        """
        After each iteration the coordinator must put exactly
        num_workers messages onto down_queue for workers to read.
        Total messages = num_workers * num_iterations.
        """
        up_q, down_q = Queue(), Queue()
        N, ITERS, FEAT = 3, 2, 4
        self._simulate_workers(up_q, N, ITERS, FEAT)
        run_logreg_allreduce(up_q, down_q, N, ITERS, FEAT)
        msg_count = 0
        while not down_q.empty():
            down_q.get_nowait()
            msg_count += 1
        assert msg_count == N * ITERS

    def test_down_queue_message_schema(self):
        """
        Each broadcast message must contain the required keys.

        Uses its OWN fresh queue pair and pre-loads worker messages
        onto up_q BEFORE calling run_logreg_allreduce — the coordinator
        blocks on up_q.get() so messages must already be present.
        Never relies on state left by any other test method.
        """
        up_q, down_q = Queue(), Queue()
        N, ITERS, FEAT = 2, 1, 3
        # Pre-load all worker messages before the coordinator runs
        self._simulate_workers(up_q, N, ITERS, FEAT)
        run_logreg_allreduce(up_q, down_q, N, ITERS, FEAT)
        # Drain all N broadcast messages and inspect the first one
        messages = []
        while not down_q.empty():
            messages.append(down_q.get_nowait())
        assert (
            len(messages) == N
        ), f"Expected {N} broadcast messages, got {len(messages)}"
        msg = messages[0]
        assert {"type", "iteration", "weights", "intercept"}.issubset(msg.keys())
        assert msg["type"] == "avg_weights"


# =============================================================
# Section 4: aggregate_logreg_results()
# =============================================================


class TestAggregateLogregResults:
    """
    Tests for the post-training LogReg result aggregation step.
    Uses a tmp_path fixture to avoid writing CSV files to the
    real project directory during tests.
    """

    def _make_worker_result(
        self, worker_id, row_count, weight_vector, intercept=0.1, train_accuracy=0.9
    ):
        return {
            "worker_id": worker_id,
            "row_count": row_count,
            "weight_vector": weight_vector,
            "intercept": intercept,
            "train_accuracy": train_accuracy,
            "iter_metrics": [],
        }

    def test_result_keys_complete(self, tmp_path):
        workers = [
            self._make_worker_result(0, 100, [1.0, 2.0]),
            self._make_worker_result(1, 100, [1.0, 2.0]),
        ]
        result = aggregate_logreg_results(workers, results_dir=str(tmp_path))
        required = {
            "weight_vector",
            "intercept",
            "avg_accuracy",
            "total_rows",
            "num_workers",
            "weight_norm",
            "agg_mode",
        }
        assert required.issubset(result.keys())

    def test_total_rows_summed(self, tmp_path):
        workers = [
            self._make_worker_result(0, 150, [1.0]),
            self._make_worker_result(1, 250, [1.0]),
        ]
        result = aggregate_logreg_results(workers, results_dir=str(tmp_path))
        assert result["total_rows"] == 400

    def test_weighted_accuracy(self, tmp_path):
        """
        Worker 0: 100 rows, accuracy=1.0
        Worker 1: 100 rows, accuracy=0.0
        Weighted average must be 0.5.
        """
        workers = [
            self._make_worker_result(0, 100, [1.0], train_accuracy=1.0),
            self._make_worker_result(1, 100, [1.0], train_accuracy=0.0),
        ]
        result = aggregate_logreg_results(workers, results_dir=str(tmp_path))
        assert result["avg_accuracy"] == pytest.approx(0.5)

    def test_allreduce_result_takes_priority(self, tmp_path):
        """
        When allreduce_result is provided, its weight_vector must be
        used in preference to the row-weighted mean.
        """
        workers = [
            self._make_worker_result(0, 100, [0.0, 0.0]),
            self._make_worker_result(1, 100, [0.0, 0.0]),
        ]
        allreduce = {
            "weight_vector": [9.9, 9.9],
            "intercept": 0.5,
        }
        result = aggregate_logreg_results(
            workers, allreduce_result=allreduce, results_dir=str(tmp_path)
        )
        np.testing.assert_array_almost_equal(result["weight_vector"], [9.9, 9.9])
        assert result["agg_mode"] == "Allreduce (FedAvg)"

    def test_no_allreduce_uses_row_weighted_mean(self, tmp_path):
        """
        Without allreduce_result, weights must be row-weighted mean.
        Worker 0: [0.0], 300 rows; Worker 1: [4.0], 100 rows.
        Expected mean: 0*0.75 + 4*0.25 = 1.0
        """
        workers = [
            self._make_worker_result(0, 300, [0.0]),
            self._make_worker_result(1, 100, [4.0]),
        ]
        result = aggregate_logreg_results(workers, results_dir=str(tmp_path))
        assert result["weight_vector"][0] == pytest.approx(1.0)
        assert result["agg_mode"] == "Row-weighted mean (no Allreduce)"

    def test_weight_norm_is_positive(self, tmp_path):
        workers = [
            self._make_worker_result(0, 100, [3.0, 4.0]),
            self._make_worker_result(1, 100, [3.0, 4.0]),
        ]
        result = aggregate_logreg_results(workers, results_dir=str(tmp_path))
        assert result["weight_norm"] == pytest.approx(5.0)  # sqrt(9+16)

    def test_csv_written_to_results_dir(self, tmp_path):
        """aggregate_logreg_results must write logreg_iter_metrics.csv."""
        workers = [
            self._make_worker_result(0, 100, [1.0, 1.0]),
        ]
        workers[0]["iter_metrics"] = [
            {
                "worker_id": 0,
                "iteration": 0,
                "iter_time_s": 0.1,
                "weight_norm": 1.4,
                "weight_delta": 0.0,
                "local_weight_norm": 1.4,
                "intercept": 0.1,
                "row_count": 100,
            }
        ]
        aggregate_logreg_results(
            workers,
            results_dir=str(tmp_path),
            run_id="test_run",
            num_workers=1,
            reg_param=0.01,
            num_features=2,
        )
        import os

        assert os.path.isfile(str(tmp_path / "logreg_iter_metrics.csv"))
