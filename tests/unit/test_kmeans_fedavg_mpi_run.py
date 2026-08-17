"""tests/unit/test_kmeans_fedavg_mpi_run.py

Unit tests for mpj_spark.applications.kmeans.fedavg_mpi_run (P3-08).

Coverage:
  - _unpack_stats() defensive tuple/dict normalisation
  - _local_wcss() squared-distance computation over a fake RDD
  - Hungarian-aligned FedAvg math across permuted centroid labels
  - Mock-comm gather/bcast round structure (no real MPI, no real Spark)

All tests are CI-safe: no mpi4py runtime, no JVM, no network.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from mpj_spark.applications.kmeans.fedavg_mpi_run import (
    _local_wcss,
    _unpack_stats,
)


class _FakeRDD:
    """Minimal RDD stand-in supporting .map().sum() for _local_wcss tests."""

    def __init__(self, points):
        self._points = [np.asarray(p, dtype=np.float64) for p in points]

    def map(self, fn):
        return _FakeRDD([fn(p) for p in self._points])

    def sum(self):
        return float(sum(self._points))


# ---------------------------------------------------------------------------
# _unpack_stats
# ---------------------------------------------------------------------------


class TestUnpackStats:
    def test_two_element_tuple(self):
        sums = np.array([[1.0, 2.0], [3.0, 4.0]])
        counts = np.array([10.0, 20.0])
        s, c, w = _unpack_stats((sums, counts))
        np.testing.assert_allclose(s, sums)
        np.testing.assert_allclose(c, counts)
        assert w == 0.0  # no WCSS element -> default

    def test_three_element_tuple(self):
        sums = np.array([[1.0]])
        counts = np.array([5.0])
        s, c, w = _unpack_stats((sums, counts, 42.5))
        assert w == pytest.approx(42.5)

    def test_dict_shape(self):
        stats = {
            "cluster_sums": [[1.0, 2.0]],
            "cluster_counts": [4.0],
            "local_wcss": 7.0,
        }
        s, c, w = _unpack_stats(stats)
        np.testing.assert_allclose(s, [[1.0, 2.0]])
        np.testing.assert_allclose(c, [4.0])
        assert w == pytest.approx(7.0)

    def test_dtype_is_float64(self):
        s, c, _ = _unpack_stats((np.array([[1, 2]]), np.array([3])))
        assert s.dtype == np.float64
        assert c.dtype == np.float64


# ---------------------------------------------------------------------------
# _local_wcss
# ---------------------------------------------------------------------------


class TestLocalWcss:
    def test_points_at_centroids_give_zero(self):
        rdd = _FakeRDD([[0.0, 0.0], [10.0, 10.0]])
        centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
        assert _local_wcss(rdd, centroids) == pytest.approx(0.0, abs=1e-9)

    def test_single_point_squared_distance(self):
        rdd = _FakeRDD([[3.0, 4.0]])
        centroids = np.array([[0.0, 0.0]])
        assert _local_wcss(rdd, centroids) == pytest.approx(25.0)

    def test_nearest_centroid_selected(self):
        # Point [1.0] is closer to centroid 0; [9.0] and [10.0] to centroid 10
        rdd = _FakeRDD([[1.0], [9.0], [10.0]])
        centroids = np.array([[0.0], [10.0]])
        assert _local_wcss(rdd, centroids) == pytest.approx(1.0 + 1.0 + 0.0)


# ---------------------------------------------------------------------------
# FedAvg math with Hungarian alignment (mock-comm gather/bcast)
# ---------------------------------------------------------------------------


def _align(reference, candidate):
    """Same Hungarian alignment used by root_process.align_centres_hungarian."""
    from mpj_spark.core.root_process import align_centres_hungarian

    aligned, _perm = align_centres_hungarian(reference, candidate)
    return aligned


class TestFedavgMath:
    def test_aligned_weighted_mean_two_workers(self):
        """Worker 1's labels are permuted; alignment must restore before averaging."""
        gathered = [
            {"centroids": [[0.0, 0.0], [10.0, 10.0]], "row_count": 100},
            {"centroids": [[12.0, 12.0], [2.0, 2.0]], "row_count": 100},
        ]
        ref = gathered[0]["centroids"]
        aligned = [gathered[0]]
        for g in gathered[1:]:
            aligned.append({**g, "centroids": _align(ref, g["centroids"])})

        total_rows = sum(g["row_count"] for g in aligned)
        avg = np.zeros((2, 2))
        for g in aligned:
            avg += (g["row_count"] / total_rows) * np.asarray(g["centroids"], dtype=np.float64)

        np.testing.assert_allclose(avg, [[1.0, 1.0], [11.0, 11.0]])

    def test_uneven_row_weighting(self):
        gathered = [
            {"centroids": [[0.0]], "row_count": 300},
            {"centroids": [[4.0]], "row_count": 100},
        ]
        ref = gathered[0]["centroids"]
        aligned = [gathered[0]]
        for g in gathered[1:]:
            aligned.append({**g, "centroids": _align(ref, g["centroids"])})
        total_rows = sum(g["row_count"] for g in aligned)
        avg = np.zeros((1, 1))
        for g in aligned:
            avg += (g["row_count"] / total_rows) * np.asarray(g["centroids"], dtype=np.float64)
        np.testing.assert_allclose(avg, [[1.0]])  # 0*0.75 + 4*0.25

    def test_mock_comm_gather_bcast_round(self):
        """Simulate one FedAvg round: 2 workers gather to rank 0, bcast averaged result."""
        worker_payloads = [
            {"centroids": [[0.0, 0.0]], "row_count": 100, "local_wcss": 5.0},
            {"centroids": [[4.0, 4.0]], "row_count": 300, "local_wcss": 15.0},
        ]

        comm = MagicMock()
        comm.gather.side_effect = lambda payload, root=0: worker_payloads
        comm.bcast.side_effect = lambda payload, root=0: payload

        gathered = comm.gather({"centroids": [[0.0, 0.0]], "row_count": 100}, root=0)
        assert len(gathered) == 2

        total_rows = sum(g["row_count"] for g in gathered)
        avg = np.zeros((1, 2))
        for g in gathered:
            avg += (g["row_count"] / total_rows) * np.asarray(g["centroids"], dtype=np.float64)
        global_payload = {
            "centroids": avg.tolist(),
            "global_wcss": sum(g["local_wcss"] for g in gathered),
            "iteration": 0,
        }

        result = comm.bcast(global_payload, root=0)
        np.testing.assert_allclose(result["centroids"], [[3.0, 3.0]])
        assert result["global_wcss"] == pytest.approx(20.0)
        assert comm.gather.call_count == 1
        assert comm.bcast.call_count == 1


# ---------------------------------------------------------------------------
# Module contract
# ---------------------------------------------------------------------------


class TestModuleContract:
    def test_run_signature_accepts_sync_mode(self):
        import inspect

        from mpj_spark.applications.kmeans.fedavg_mpi_run import run_kmeans_fedavg_mpi

        params = inspect.signature(run_kmeans_fedavg_mpi).parameters
        assert "sync_mode" in params
        assert params["sync_mode"].default == "ps_sync_fedavg_mpi"

    def test_sync_mode_constant_matches_registry(self):
        from mpj_spark.applications.kmeans.fedavg_mpi_run import run_kmeans_fedavg_mpi
        from mpj_spark.core.sync_modes import MODE_PS_SYNC_FEDAVG_MPI

        import inspect

        assert (
            inspect.signature(run_kmeans_fedavg_mpi).parameters["sync_mode"].default
            == MODE_PS_SYNC_FEDAVG_MPI
        )
