import numpy as np

from mpj_spark.applications import kmeans, logreg
from mpj_spark.applications.kmeans.driver import run_kmeans_driver
from mpj_spark.applications.kmeans.local_iteration import load_partition_rdd
from mpj_spark.applications.logreg.driver import run_logreg_driver


def test_application_package_exposes_subpackages():
    assert kmeans is not None
    assert logreg is not None


def test_load_partition_rdd_skips_header_and_blank_lines(spark, tmp_path):
    partition_path = tmp_path / "partition.csv"
    partition_path.write_text(
        "f0,f1\n1.0,2.0\n3.0,4.0\n\n",
        encoding="utf-8",
    )

    points_rdd = load_partition_rdd(spark, str(partition_path))
    rows = points_rdd.collect()

    assert len(rows) == 2
    np.testing.assert_allclose(rows[0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(rows[1], np.array([3.0, 4.0]))


def test_kmeans_driver_accepts_worker_style_kwargs(monkeypatch):
    def fake_allreduce(**kwargs):
        return {
            "global_centroids": [[1.0, 2.0]],
            "run_summary": {"final_wcss": 3.5},
        }

    monkeypatch.setattr(
        "mpj_spark.applications.kmeans.allreduce.run_kmeans_allreduce",
        fake_allreduce,
    )

    # Create a mock comm object to trigger the allreduce path
    class MockComm:
        def bcast(self, val, root=0):
            return val

    result = run_kmeans_driver(
        dataset_path="/tmp/data.csv",
        k=1,
        max_iter=4,
        worker_id=2,
        num_workers=4,
        comm=MockComm(),  # Pass mock comm to trigger allreduce path
    )

    assert result["centres"] == [[1.0, 2.0]]
    assert result["wcss"] == 3.5


def test_logreg_driver_accepts_worker_style_kwargs(monkeypatch):
    def fake_allreduce(**kwargs):
        return {"weights": [0.1, 0.2], "intercept": 0.3}

    monkeypatch.setattr(
        "mpj_spark.applications.logreg.allreduce.run_logreg_allreduce",
        fake_allreduce,
    )

    # Create a mock comm object to trigger the allreduce path
    class MockComm:
        def bcast(self, val, root=0):
            return val

    result = run_logreg_driver(
        dataset_path="/tmp/data.csv",
        max_iter=5,
        worker_id=1,
        num_workers=3,
        comm=MockComm(),  # Pass mock comm to trigger allreduce path
    )

    assert result["weights"] == [0.1, 0.2]
    assert result["intercept"] == 0.3
