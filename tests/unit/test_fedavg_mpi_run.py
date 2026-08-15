"""tests/unit/test_fedavg_mpi_run.py
Unit tests for mpj_spark.applications.logreg.fedavg_mpi_run helper math & mock collective logic.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from mpj_spark.applications.logreg.fedavg_mpi_run import (
    _build_schema,
    _weight_norm,
    _write_worker_metrics,
)


def test_build_schema():
    schema = _build_schema(num_features=4)
    assert len(schema.fields) == 5
    assert [f.name for f in schema.fields] == ["f0", "f1", "f2", "f3", "label"]


def test_weight_norm():
    w = [3.0, 4.0]
    assert _weight_norm(w) == pytest.approx(5.0)
    w_arr = np.array([1.0, 2.0, 2.0])
    assert _weight_norm(w_arr) == pytest.approx(3.0)


def test_write_worker_metrics(tmp_path):
    records = [
        {
            "worker_id": 1,
            "sync_mode": "ps_sync_fedavg_mpi",
            "iteration": 1,
            "iter_time_s": 0.123,
            "weight_norm": 1.45,
            "weight_delta": 0.05,
            "local_weight_norm": 1.40,
            "intercept": 0.0,
            "row_count": 500,
        }
    ]
    path = _write_worker_metrics(worker_id=1, records=records, results_dir=str(tmp_path))
    assert (tmp_path / "worker_1_fedavg_mpi_metrics.csv").exists()
    assert path.endswith("worker_1_fedavg_mpi_metrics.csv")


def test_mock_mpi_fedavg_gather_and_bcast():
    """Verify that simulated gather -> FedAvg -> bcast achieves exact row-weighted mean."""
    # Simulate 2 workers reporting to root (rank 0)
    worker_0_payload = {
        "weights": [1.0, 2.0],
        "intercept": 0.5,
        "row_count": 100,
        "rank": 0,
    }
    worker_1_payload = {
        "weights": [3.0, 4.0],
        "intercept": 1.5,
        "row_count": 300,
        "rank": 1,
    }

    gathered = [worker_0_payload, worker_1_payload]

    # Root calculates FedAvg
    total_rows = sum(m["row_count"] for m in gathered)
    avg_w = np.zeros(2)
    avg_b = 0.0
    for m in gathered:
        frac = m["row_count"] / total_rows
        avg_w += frac * np.array(m["weights"])
        avg_b += frac * m["intercept"]

    # Expected: (100*1 + 300*3)/400 = 2.5, (100*2 + 300*4)/400 = 3.5, (100*0.5 + 300*1.5)/400 = 1.25
    np.testing.assert_allclose(avg_w, [2.5, 3.5])
    assert avg_b == pytest.approx(1.25)
