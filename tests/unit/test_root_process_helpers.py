# =============================================================
# tests/unit/test_root_process_helpers.py
#
# Unit tests for the pure helper functions in root_process.py
# that were previously uncovered:
#
#   • dynamic_partition      (wraps MPJSparkFileManager, returns paths)
#   • reassign_pass_root     (queue-based centroid correction)
#   • compute_global_seed_centres — subprocess path tested via mock
#     to avoid spinning up a real SparkSession in CI.
#
# Objectives covered: 1c (correctness), 2a (profiling helpers)
# =============================================================
import os
import math
import threading
from multiprocessing import Queue
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from mpj_spark.core.root_process import (
    dynamic_partition,
    reassign_pass_root,
    compute_global_seed_centres,
)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def small_csv(tmp_path):
    """Write a 12-line CSV to a temp file; return its path."""
    lines = [f"{i},{i*2},{i*3}\n" for i in range(12)]
    p = tmp_path / "data.csv"
    p.write_text("".join(lines), encoding="utf-8")
    return str(p)


# ─────────────────────────────────────────────────────────────────
# dynamic_partition
# ─────────────────────────────────────────────────────────────────

class TestDynamicPartitionHelper:
    """Tests for the root_process.dynamic_partition wrapper."""

    def test_returns_list_of_paths(self, small_csv, tmp_path):
        output_dir = str(tmp_path / "out")
        paths = dynamic_partition(small_csv, num_partitions=3, output_dir=output_dir)
        assert isinstance(paths, list)
        assert len(paths) == 3

    def test_all_paths_exist_on_disk(self, small_csv, tmp_path):
        output_dir = str(tmp_path / "out")
        paths = dynamic_partition(small_csv, num_partitions=2, output_dir=output_dir)
        for p in paths:
            assert os.path.isfile(p), f"Expected partition file missing: {p}"

    def test_no_lines_lost(self, small_csv, tmp_path):
        """Union of all partition lines must equal the original file."""
        output_dir = str(tmp_path / "out")
        paths = dynamic_partition(small_csv, num_partitions=4, output_dir=output_dir)

        with open(small_csv, encoding="utf-8") as f:
            original = set(f.readlines())

        recovered = set()
        for p in paths:
            with open(p, encoding="utf-8") as f:
                recovered.update(f.readlines())

        assert original == recovered

    def test_output_dir_created_automatically(self, small_csv, tmp_path):
        output_dir = str(tmp_path / "new" / "nested" / "dir")
        assert not os.path.exists(output_dir)
        dynamic_partition(small_csv, num_partitions=2, output_dir=output_dir)
        assert os.path.isdir(output_dir)

    def test_single_partition_returns_one_path(self, small_csv, tmp_path):
        output_dir = str(tmp_path / "out")
        paths = dynamic_partition(small_csv, num_partitions=1, output_dir=output_dir)
        assert len(paths) == 1

    def test_paths_are_strings(self, small_csv, tmp_path):
        output_dir = str(tmp_path / "out")
        paths = dynamic_partition(small_csv, num_partitions=2, output_dir=output_dir)
        for p in paths:
            assert isinstance(p, str)


# ─────────────────────────────────────────────────────────────────
# reassign_pass_root
# ─────────────────────────────────────────────────────────────────

def _make_stats_msg(k, dims, row_count, rng=None):
    """Build a synthetic 'stats' message as workers would send."""
    if rng is None:
        rng = np.random.default_rng(0)
    counts = rng.integers(10, 100, size=k)
    sums   = [rng.random(dims) * counts[j] for j in range(k)]
    return {
        "type"          : "stats",
        "cluster_sums"  : [s.tolist() for s in sums],
        "cluster_counts": counts.tolist(),
        "row_count"     : row_count,
    }


class TestReassignPassRoot:
    """Tests for reassign_pass_root — verifies the queue protocol
    and centroid correction arithmetic without a live Spark context."""

    def _run_reassign(
        self, gossip_centres, num_workers, worker_msgs,
    ):
        """
        Helper: puts gossip centres onto a shared queue, then runs
        reassign_pass_root in the *current* thread while a background
        thread puts the worker 'stats' replies.
        """
        q = Queue()
        k    = len(gossip_centres)
        dims = len(gossip_centres[0])

        def _simulate_workers():
            # Drain the 'reassign' broadcast messages sent by root
            for _ in range(num_workers):
                q.get(timeout=5)   # consume the 'reassign' message
            # Now push back synthetic stats
            for msg in worker_msgs:
                q.put(msg)

        t = threading.Thread(target=_simulate_workers, daemon=True)
        t.start()

        result = reassign_pass_root(
            processes_alive=[],     # not used for queue logic
            gossip_centres=gossip_centres,
            reassign_queue=q,
            num_workers=num_workers,
            k=k,
            dims=dims,
        )
        t.join(timeout=5)
        return result

    # ── basic shape / type ──────────────────────────────────────────

    def test_returns_list(self):
        k, dims = 2, 3
        centres = [[float(i)] * dims for i in range(k)]
        msgs    = [_make_stats_msg(k, dims, 50) for _ in range(2)]
        result  = self._run_reassign(centres, num_workers=2, worker_msgs=msgs)
        assert isinstance(result, list)

    def test_output_k_matches_input(self):
        k, dims = 3, 4
        centres = [[float(i)] * dims for i in range(k)]
        msgs    = [_make_stats_msg(k, dims, 100, rng=np.random.default_rng(i))
                   for i in range(2)]
        result  = self._run_reassign(centres, num_workers=2, worker_msgs=msgs)
        assert len(result) == k

    def test_each_centre_has_correct_dims(self):
        k, dims = 2, 5
        centres = [[float(i)] * dims for i in range(k)]
        msgs    = [_make_stats_msg(k, dims, 80)]
        result  = self._run_reassign(centres, num_workers=1, worker_msgs=msgs)
        for c in result:
            assert len(c) == dims

    # ── arithmetic correctness ──────────────────────────────────────

    def test_single_worker_returns_own_centroid(self):
        """With one worker, the corrected centroid = sums / counts exactly."""
        k, dims = 2, 2
        # Construct a predictable message
        sums   = [[4.0, 8.0], [6.0, 3.0]]
        counts = [2, 3]
        msg = {
            "type"          : "stats",
            "cluster_sums"  : sums,
            "cluster_counts": counts,
            "row_count"     : 5,
        }
        centres = [[0.0, 0.0], [0.0, 0.0]]   # gossip centres (ignored arithmetically)
        result  = self._run_reassign(centres, num_workers=1, worker_msgs=[msg])
        expected = [[2.0, 4.0], [2.0, 1.0]]
        for got, exp in zip(result, expected):
            np.testing.assert_allclose(got, exp, rtol=1e-6)

    def test_two_workers_averaging(self):
        """Corrected centroid = (sum_w1 + sum_w2) / (count_w1 + count_w2)."""
        k, dims = 1, 2
        msg1 = {"type": "stats", "cluster_sums": [[3.0, 6.0]],
                "cluster_counts": [3], "row_count": 3}
        msg2 = {"type": "stats", "cluster_sums": [[5.0, 4.0]],
                "cluster_counts": [2], "row_count": 2}
        centres = [[0.0, 0.0]]
        result  = self._run_reassign(centres, num_workers=2, worker_msgs=[msg1, msg2])
        # (3+5)/(3+2) = 1.6,  (6+4)/(3+2) = 2.0
        np.testing.assert_allclose(result[0], [1.6, 2.0], rtol=1e-6)

    # ── empty cluster fallback ──────────────────────────────────────

    def test_empty_cluster_falls_back_to_gossip_centre(self):
        """A cluster with count=0 must retain the original gossip centroid."""
        k, dims = 2, 2
        gossip  = [[1.0, 2.0], [9.0, 8.0]]
        msg = {
            "type"          : "stats",
            "cluster_sums"  : [[10.0, 20.0], [0.0, 0.0]],   # cluster 1 has no points
            "cluster_counts": [5, 0],
            "row_count"     : 5,
        }
        result = self._run_reassign(gossip, num_workers=1, worker_msgs=[msg])
        # cluster 0: 10/5=2.0, 20/5=4.0
        np.testing.assert_allclose(result[0], [2.0, 4.0], rtol=1e-6)
        # cluster 1: falls back to gossip centroid
        np.testing.assert_allclose(result[1], gossip[1], rtol=1e-6)


# ─────────────────────────────────────────────────────────────────
# compute_global_seed_centres — subprocess path (mocked)
# ─────────────────────────────────────────────────────────────────

class TestComputeGlobalSeedCentres:
    """
    compute_global_seed_centres launches a subprocess internally.
    We mock multiprocessing.Process and Queue to avoid spinning up
    a real SparkSession in CI while still exercising the orchestration
    logic (exit-code checking, result extraction, error propagation).
    """

    def _mock_successful_process(self, centres, monkeypatch):
        """Patch Process + Queue so the function sees a successful subprocess."""
        mock_q = MagicMock()
        mock_q.get_nowait.return_value = {"status": "ok", "centres": centres}

        mock_proc = MagicMock()
        mock_proc.exitcode = 0

        monkeypatch.setattr(
            "mpj_spark.core.root_process.Queue", lambda: mock_q
        )
        monkeypatch.setattr(
            "mpj_spark.core.root_process.Process",
            lambda **kw: mock_proc,
        )
        return mock_q, mock_proc

    def test_returns_list_of_centres(self, tmp_path, monkeypatch):
        centres = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        self._mock_successful_process(centres, monkeypatch)
        result = compute_global_seed_centres(
            input_file=str(tmp_path / "data.csv"),
            k=2, total_cores=2,
        )
        assert result == centres

    def test_returns_correct_k(self, tmp_path, monkeypatch):
        centres = [[float(i)] * 4 for i in range(5)]
        self._mock_successful_process(centres, monkeypatch)
        result = compute_global_seed_centres(
            input_file=str(tmp_path / "data.csv"),
            k=5, total_cores=2,
        )
        assert len(result) == 5

    def test_nonzero_exitcode_raises_runtime_error(self, tmp_path, monkeypatch):
        mock_q    = MagicMock()
        mock_proc = MagicMock()
        mock_proc.exitcode = 1

        monkeypatch.setattr("mpj_spark.core.root_process.Queue", lambda: mock_q)
        monkeypatch.setattr(
            "mpj_spark.core.root_process.Process", lambda **kw: mock_proc
        )

        with pytest.raises(RuntimeError, match="subprocess exited with code 1"):
            compute_global_seed_centres(
                input_file=str(tmp_path / "data.csv"),
                k=2, total_cores=2,
            )

    def test_worker_error_status_raises_runtime_error(self, tmp_path, monkeypatch):
        mock_q = MagicMock()
        mock_q.get_nowait.return_value = {
            "status": "error", "msg": "OOM during KMeans fit"
        }
        mock_proc = MagicMock()
        mock_proc.exitcode = 0

        monkeypatch.setattr("mpj_spark.core.root_process.Queue", lambda: mock_q)
        monkeypatch.setattr(
            "mpj_spark.core.root_process.Process", lambda **kw: mock_proc
        )

        with pytest.raises(RuntimeError, match="seeding worker failed"):
            compute_global_seed_centres(
                input_file=str(tmp_path / "data.csv"),
                k=2, total_cores=2,
            )
