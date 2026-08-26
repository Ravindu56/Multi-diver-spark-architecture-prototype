"""tests/unit/test_merged_iter_metrics_extras.py
Regression test for the P3-09 E2E failure (2026-08-26):

    ValueError: dict contains fields not in fieldnames:
    'global_version', 'staleness'

ps_async worker iter_metrics carry two per-mode extra fields that the
fixed merged-schema fieldnames in _write_merged_iter_metrics() do not
declare.  The writer must ignore per-mode extras (they are persisted in
results/worker_N_async_ps_metrics.csv and
results/logreg_async_ps_staleness.csv) instead of raising, so the
merged logreg_iter_metrics.csv keeps a stable cross-mode schema for
the P3-12 benchmark (#64) and the P6-01 characterization dataset.
"""

import csv
import os

from mpj_spark.core.root_process import _write_merged_iter_metrics


def test_merged_iter_metrics_tolerates_ps_async_extra_fields(tmp_path):
    worker_results = [
        {
            "row_count": 100,
            "train_accuracy": 0.6,
            "weight_vector": [0.1, 0.2],
            "intercept": 0.0,
            "iter_metrics": [
                {
                    "worker_id": 1,
                    "sync_mode": "ps_async",
                    "iteration": 1,
                    "iter_time_s": 0.5,
                    "weight_norm": 0.2236,
                    "weight_delta": 0.2236,
                    "local_weight_norm": 0.2236,
                    "intercept": 0.0,
                    "row_count": 100,
                    "staleness": 1,
                    "global_version": 2,
                }
            ],
        }
    ]

    out_path, n_written = _write_merged_iter_metrics(
        worker_results,
        str(tmp_path),
        "20260826T000000Z",
        2,
        0.01,
        2,
        sync_mode="ps_async",
    )

    assert n_written == 1
    assert os.path.exists(out_path)
    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert "staleness" not in (reader.fieldnames or [])
    assert "global_version" not in (reader.fieldnames or [])
    assert len(rows) == 1
    assert rows[0]["sync_mode"] == "ps_async"
