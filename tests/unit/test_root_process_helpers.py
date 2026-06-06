# =============================================================
# tests/unit/test_root_process_helpers.py
#
# Unit tests for pure helpers in core/root_process.py:
#
#   * dynamic_partition      — wraps MPJSparkFileManager
#   * reassign_pass_root     — queue-based centroid correction
#   * compute_global_seed_centres — subprocess path, fully mocked
#
# BUG FIXES applied vs. first draft:
#   1. reassign_pass_root uses a SINGLE queue for both directions
#      (broadcast down, stats up).  Using multiprocessing.Queue
#      across threads can deadlock when the OS pipe buffer fills.
#      Fix: use queue.Queue (threading-based, no pipe limit) so
#      the helper's .put() / .get(timeout=...) calls never block.
#
#   2. compute_global_seed_centres patches — root_process.py does
#      "from multiprocessing import Queue, Process" at module top,
#      so Queue and Process are bound names IN that module's
#      namespace.  Patching "mpj_spark.core.root_process.Queue"
#      (not "multiprocessing.Queue") is correct.  The mock Process
#      must accept both positional and keyword arguments (target,
#      args, daemon) and expose .start() / .join() without blocking.
#
#   3. dynamic_partition — the real key returned by
#      MPJSparkFileManager.dynamic_partition is 'partition_path'
#      (verified in file_manager.py).  The root_process wrapper
#      already handles that key; the test just reads the returned
#      path list, so no dict-key issue here.
# =============================================================
import os
import queue   # <-- threading Queue, not multiprocessing
import threading
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from mpj_spark.core.root_process import (
    dynamic_partition,
    reassign_pass_root,
    compute_global_seed_centres,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture()
def small_csv(tmp_path):
    """Write a 12-line CSV; return its path."""
    lines = [f"{i},{i * 2},{i * 3}\n" for i in range(12)]
    p = tmp_path / "data.csv"
    p.write_text("".join(lines), encoding="utf-8")
    return str(p)


# ─────────────────────────────────────────────────────────────
# dynamic_partition
# ─────────────────────────────────────────────────────────────

class TestDynamicPartitionHelper:

    def test_returns_list_of_paths(self, small_csv, tmp_path):
        paths = dynamic_partition(small_csv, num_partitions=3,
                                  output_dir=str(tmp_path / "out"))
        assert isinstance(paths, list)
        assert len(paths) == 3

    def test_all_paths_exist_on_disk(self, small_csv, tmp_path):
        paths = dynamic_partition(small_csv, num_partitions=2,
                                  output_dir=str(tmp_path / "out"))
        for p in paths:
            assert os.path.isfile(p), f"Partition file missing: {p}"

    def test_no_lines_lost(self, small_csv, tmp_path):
        """Union of all partition lines == original file lines."""
        paths = dynamic_partition(small_csv, num_partitions=4,
                                  output_dir=str(tmp_path / "out"))
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
        paths = dynamic_partition(small_csv, num_partitions=1,
                                  output_dir=str(tmp_path / "out"))
        assert len(paths) == 1

    def test_paths_are_strings(self, small_csv, tmp_path):
        paths = dynamic_partition(small_csv, num_partitions=2,
                                  output_dir=str(tmp_path / "out"))
        for p in paths:
            assert isinstance(p, str)


# ─────────────────────────────────────────────────────────────
# reassign_pass_root
# ─────────────────────────────────────────────────────────────
#
# Protocol reminder (from root_process.py):
#   root  → puts  {'type': 'reassign', 'centres': ...}  N times  (down)
#   workers → put  {'type': 'stats', ...}                N times  (up)
#   Both directions travel through THE SAME queue object.
#
# We simulate the worker side in a daemon thread:
#   1. drain exactly num_workers 'reassign' messages
#   2. put back num_workers 'stats' replies
#
# threading.queue.Queue is used instead of multiprocessing.Queue
# so there is no OS-pipe buffer limit that could cause a deadlock
# when producer and consumer run in the same process.
# ─────────────────────────────────────────────────────────────

def _make_stats_msg(k, dims, row_count, rng=None):
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


def _run_reassign(gossip_centres, num_workers, worker_msgs):
    """
    Run reassign_pass_root with a threading.Queue so put/get never
    deadlock, regardless of message size or ordering.
    """
    q = queue.Queue()          # threading.Queue — no OS pipe
    k    = len(gossip_centres)
    dims = len(gossip_centres[0])

    def _simulate_workers():
        # drain the N 'reassign' broadcast messages root puts first
        for _ in range(num_workers):
            q.get(timeout=5)
        # push back the synthetic stats replies
        for msg in worker_msgs:
            q.put(msg)

    t = threading.Thread(target=_simulate_workers, daemon=True)
    t.start()

    result = reassign_pass_root(
        processes_alive=[],
        gossip_centres=gossip_centres,
        reassign_queue=q,
        num_workers=num_workers,
        k=k,
        dims=dims,
    )
    t.join(timeout=5)
    return result


class TestReassignPassRoot:

    def test_returns_list(self):
        k, dims = 2, 3
        centres = [[float(i)] * dims for i in range(k)]
        msgs    = [_make_stats_msg(k, dims, 50) for _ in range(2)]
        assert isinstance(_run_reassign(centres, 2, msgs), list)

    def test_output_k_matches_input(self):
        k, dims = 3, 4
        centres = [[float(i)] * dims for i in range(k)]
        msgs    = [_make_stats_msg(k, dims, 100, np.random.default_rng(i))
                   for i in range(2)]
        assert len(_run_reassign(centres, 2, msgs)) == k

    def test_each_centre_has_correct_dims(self):
        k, dims = 2, 5
        centres = [[float(i)] * dims for i in range(k)]
        result  = _run_reassign(centres, 1, [_make_stats_msg(k, dims, 80)])
        for c in result:
            assert len(c) == dims

    def test_single_worker_returns_own_centroid(self):
        """corrected centroid = sums / counts exactly."""
        k, dims = 2, 2
        sums    = [[4.0, 8.0], [6.0, 3.0]]
        counts  = [2, 3]
        msg = {
            "type"          : "stats",
            "cluster_sums"  : sums,
            "cluster_counts": counts,
            "row_count"     : 5,
        }
        result = _run_reassign([[0.0, 0.0], [0.0, 0.0]], 1, [msg])
        np.testing.assert_allclose(result[0], [2.0, 4.0], rtol=1e-6)
        np.testing.assert_allclose(result[1], [2.0, 1.0], rtol=1e-6)

    def test_two_workers_weighted_average(self):
        """(sum_w1 + sum_w2) / (count_w1 + count_w2)."""
        k, dims = 1, 2
        msg1 = {"type": "stats", "cluster_sums": [[3.0, 6.0]],
                "cluster_counts": [3], "row_count": 3}
        msg2 = {"type": "stats", "cluster_sums": [[5.0, 4.0]],
                "cluster_counts": [2], "row_count": 2}
        result = _run_reassign([[0.0, 0.0]], 2, [msg1, msg2])
        # (3+5)/(3+2)=1.6,  (6+4)/(3+2)=2.0
        np.testing.assert_allclose(result[0], [1.6, 2.0], rtol=1e-6)

    def test_empty_cluster_falls_back_to_gossip_centre(self):
        """cluster with count=0 keeps the original gossip centroid."""
        gossip = [[1.0, 2.0], [9.0, 8.0]]
        msg = {
            "type"          : "stats",
            "cluster_sums"  : [[10.0, 20.0], [0.0, 0.0]],
            "cluster_counts": [5, 0],
            "row_count"     : 5,
        }
        result = _run_reassign(gossip, 1, [msg])
        np.testing.assert_allclose(result[0], [2.0, 4.0], rtol=1e-6)
        np.testing.assert_allclose(result[1], gossip[1],  rtol=1e-6)


# ─────────────────────────────────────────────────────────────
# compute_global_seed_centres  (subprocess path — fully mocked)
# ─────────────────────────────────────────────────────────────
#
# root_process.py does:
#   from multiprocessing import Queue, Process, Event
# so Queue and Process are bound into the module namespace as
#   mpj_spark.core.root_process.Queue
#   mpj_spark.core.root_process.Process
#
# We patch those bound names directly.  The mock Process must:
#   - accept target, args, daemon as keyword args (matching the
#     call site: Process(target=..., args=..., daemon=False))
#   - expose .start() and .join() that return immediately
#   - expose .exitcode
# The mock Queue must expose .get_nowait() returning the result.
# ─────────────────────────────────────────────────────────────

MODULE = "mpj_spark.core.root_process"


def _mock_process(exitcode=0):
    p = MagicMock()
    p.exitcode = exitcode
    p.start.return_value = None
    p.join.return_value  = None
    return p


class TestComputeGlobalSeedCentres:

    def test_returns_list_of_centres(self, tmp_path):
        centres = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        mock_q   = MagicMock()
        mock_q.get_nowait.return_value = {"status": "ok", "centres": centres}
        mock_proc = _mock_process(exitcode=0)

        with patch(f"{MODULE}.Queue", return_value=mock_q), \
             patch(f"{MODULE}.Process", return_value=mock_proc):
            result = compute_global_seed_centres(
                input_file=str(tmp_path / "d.csv"), k=2, total_cores=2)

        assert result == centres

    def test_returns_correct_k(self, tmp_path):
        centres = [[float(i)] * 4 for i in range(5)]
        mock_q   = MagicMock()
        mock_q.get_nowait.return_value = {"status": "ok", "centres": centres}
        mock_proc = _mock_process(exitcode=0)

        with patch(f"{MODULE}.Queue", return_value=mock_q), \
             patch(f"{MODULE}.Process", return_value=mock_proc):
            result = compute_global_seed_centres(
                input_file=str(tmp_path / "d.csv"), k=5, total_cores=2)

        assert len(result) == 5

    def test_nonzero_exitcode_raises_runtime_error(self, tmp_path):
        mock_q    = MagicMock()
        mock_proc = _mock_process(exitcode=1)

        with patch(f"{MODULE}.Queue", return_value=mock_q), \
             patch(f"{MODULE}.Process", return_value=mock_proc), \
             pytest.raises(RuntimeError, match="subprocess exited with code 1"):
            compute_global_seed_centres(
                input_file=str(tmp_path / "d.csv"), k=2, total_cores=2)

    def test_worker_error_status_raises_runtime_error(self, tmp_path):
        mock_q = MagicMock()
        mock_q.get_nowait.return_value = {
            "status": "error", "msg": "OOM during KMeans fit"
        }
        mock_proc = _mock_process(exitcode=0)

        with patch(f"{MODULE}.Queue", return_value=mock_q), \
             patch(f"{MODULE}.Process", return_value=mock_proc), \
             pytest.raises(RuntimeError, match="seeding worker failed"):
            compute_global_seed_centres(
                input_file=str(tmp_path / "d.csv"), k=2, total_cores=2)
