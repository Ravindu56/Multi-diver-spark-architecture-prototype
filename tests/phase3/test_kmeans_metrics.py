# =============================================================================
# tests/phase3/test_kmeans_metrics.py
# Phase 3 — Issue #8 — Step 6: Metrics Collection Unit Tests
#
# WHAT IS TESTED
# --------------
#   1.  record_iteration() stores a dict with exactly the 6 required fields
#   2.  record_iteration() rounds values to expected precision
#   3.  sync_overhead_pct() arithmetic: sync_time / iter_time * 100
#   4.  sync_overhead_pct() returns 0.0 when iter_time_s == 0 (guard)
#   5.  convergence_rate() returns centroid_shift series in order
#   6.  wcss_series() returns global_wcss series in order
#   7.  summary_table() rows contain sync_overhead_pct field
#   8.  to_csv() writes correct header and correct row count
#   9.  to_json() has run / iterations / derived keys with correct lengths
#  10.  aggregate_across_ranks() mean computation across 2 synthetic CSVs
#
# HOW TO RUN  (no MPI needed)
# ----------------------------
#   python -m pytest tests/phase3/test_kmeans_metrics.py -v
# =============================================================================

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from mpj_spark.applications.kmeans.metrics import _ITER_FIELDS, KMeansMetricsCollector


# ---------------------------------------------------------------------------
# Shared helper: build a collector with N pre-recorded iterations
# ---------------------------------------------------------------------------
def _make_collector(
    n_iters: int = 3, rank: int = 0, tmp_dir: str = "/tmp"
) -> KMeansMetricsCollector:
    collector = KMeansMetricsCollector(rank=rank, output_dir=tmp_dir)
    for i in range(1, n_iters + 1):
        collector.record_iteration(
            iteration=i,
            spark_time_s=float(i) * 0.5,
            sync_time_s=float(i) * 0.1,
            iter_time_s=float(i) * 0.6,
            centroid_shift=1.0 / i,
            global_wcss=100.0 / i,
        )
    return collector


# ---------------------------------------------------------------------------
# Test 1 — record_iteration stores dict with exactly 6 required fields
# ---------------------------------------------------------------------------
def test_record_iteration_fields():
    with tempfile.TemporaryDirectory() as tmp:
        collector = KMeansMetricsCollector(rank=0, output_dir=tmp)
        collector.record_iteration(
            iteration=1,
            spark_time_s=0.5,
            sync_time_s=0.1,
            iter_time_s=0.6,
            centroid_shift=0.05,
            global_wcss=1234.56,
        )
        assert len(collector._iterations) == 1
        row = collector._iterations[0]
        missing = set(_ITER_FIELDS) - set(row.keys())
        assert not missing, f"Missing fields in iteration row: {missing}"
        extra = set(row.keys()) - set(_ITER_FIELDS)
        assert not extra, f"Unexpected extra fields in iteration row: {extra}"


# ---------------------------------------------------------------------------
# Test 2 — record_iteration rounds to expected precision
# ---------------------------------------------------------------------------
def test_record_iteration_rounding():
    with tempfile.TemporaryDirectory() as tmp:
        collector = KMeansMetricsCollector(rank=0, output_dir=tmp)
        collector.record_iteration(
            iteration=1,
            spark_time_s=1.123456789,  # should round to 6dp → 1.123457
            sync_time_s=0.111111111,  # → 0.111111
            iter_time_s=1.234567890,  # → 1.234568
            centroid_shift=0.123456789,  # should round to 8dp → 0.12345679
            global_wcss=9876.54321,  # should round to 4dp → 9876.5432
        )
        row = collector._iterations[0]
        assert row["spark_time_s"] == round(1.123456789, 6)
        assert row["sync_time_s"] == round(0.111111111, 6)
        assert row["iter_time_s"] == round(1.234567890, 6)
        assert row["centroid_shift"] == round(0.123456789, 8)
        assert row["global_wcss"] == round(9876.54321, 4)


# ---------------------------------------------------------------------------
# Test 3 — sync_overhead_pct arithmetic
# ---------------------------------------------------------------------------
def test_sync_overhead_pct_arithmetic():
    with tempfile.TemporaryDirectory() as tmp:
        collector = _make_collector(n_iters=2, tmp_dir=tmp)
        overheads = collector.sync_overhead_pct()
        assert len(overheads) == 2
        # iter 1: sync=0.1, iter=0.6 → 0.1/0.6*100 = 16.6667
        # iter 2: sync=0.2, iter=1.2 → 0.2/1.2*100 = 16.6667
        # (values are rounded; compare to 4dp precision)
        for overhead in overheads:
            assert (
                abs(overhead - round(0.1 / 0.6 * 100.0, 4)) < 1e-3
            ), f"Unexpected sync overhead: {overhead}"


# ---------------------------------------------------------------------------
# Test 4 — sync_overhead_pct guard: returns 0.0 when iter_time_s == 0
# ---------------------------------------------------------------------------
def test_sync_overhead_pct_zero_iter_time_guard():
    with tempfile.TemporaryDirectory() as tmp:
        collector = KMeansMetricsCollector(rank=0, output_dir=tmp)
        collector.record_iteration(
            iteration=1,
            spark_time_s=0.0,
            sync_time_s=0.0,
            iter_time_s=0.0,  # edge case: zero division guard
            centroid_shift=0.0,
            global_wcss=0.0,
        )
        overheads = collector.sync_overhead_pct()
        assert overheads == [0.0], f"Expected [0.0] when iter_time_s == 0, got {overheads}"


# ---------------------------------------------------------------------------
# Test 5 — convergence_rate returns centroid_shift series in order
# ---------------------------------------------------------------------------
def test_convergence_rate_series():
    with tempfile.TemporaryDirectory() as tmp:
        collector = _make_collector(n_iters=3, tmp_dir=tmp)
        rate = collector.convergence_rate()
        assert len(rate) == 3
        # shifts were recorded as 1/i for i in [1, 2, 3]
        expected = [round(1.0 / i, 8) for i in range(1, 4)]
        for actual, exp in zip(rate, expected, strict=False):
            assert (
                abs(actual - exp) < 1e-10
            ), f"Convergence rate mismatch: got {actual}, expected {exp}"


# ---------------------------------------------------------------------------
# Test 6 — wcss_series returns global_wcss in order
# ---------------------------------------------------------------------------
def test_wcss_series():
    with tempfile.TemporaryDirectory() as tmp:
        collector = _make_collector(n_iters=3, tmp_dir=tmp)
        wcss = collector.wcss_series()
        assert len(wcss) == 3
        expected = [round(100.0 / i, 4) for i in range(1, 4)]
        for actual, exp in zip(wcss, expected, strict=False):
            assert abs(actual - exp) < 1e-6, f"WCSS series mismatch: got {actual}, expected {exp}"


# ---------------------------------------------------------------------------
# Test 7 — summary_table enriches rows with sync_overhead_pct
# ---------------------------------------------------------------------------
def test_summary_table_has_overhead_field():
    with tempfile.TemporaryDirectory() as tmp:
        collector = _make_collector(n_iters=2, tmp_dir=tmp)
        table = collector.summary_table()
        assert len(table) == 2
        for row in table:
            assert (
                "sync_overhead_pct" in row
            ), f"summary_table row missing 'sync_overhead_pct': {row.keys()}"
            assert isinstance(row["sync_overhead_pct"], float)


# ---------------------------------------------------------------------------
# Test 8 — to_csv writes correct header and row count
# ---------------------------------------------------------------------------
def test_to_csv_header_and_row_count():
    with tempfile.TemporaryDirectory() as tmp:
        n_iters = 4
        collector = _make_collector(n_iters=n_iters, rank=0, tmp_dir=tmp)
        path = collector.to_csv()

        assert path.exists(), f"CSV file not written: {path}"

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == n_iters, f"Expected {n_iters} rows, got {len(rows)}"
        expected_headers = set(_ITER_FIELDS) | {"sync_overhead_pct"}
        assert set(reader.fieldnames) == expected_headers, (
            f"CSV header mismatch: got {set(reader.fieldnames)}, " f"expected {expected_headers}"
        )


# ---------------------------------------------------------------------------
# Test 9 — to_json has run / iterations / derived keys with correct lengths
# ---------------------------------------------------------------------------
def test_to_json_structure():
    with tempfile.TemporaryDirectory() as tmp:
        n_iters = 3
        collector = _make_collector(n_iters=n_iters, rank=0, tmp_dir=tmp)
        collector.record_run(
            total_time_s=5.0,
            iterations_run=n_iters,
            converged=False,
            k=3,
            dataset_size=1000,
            num_ranks=2,
        )
        path = collector.to_json()

        assert path.exists(), f"JSON file not written: {path}"

        with open(path) as f:
            payload = json.load(f)

        assert "run" in payload, "Missing 'run' key in JSON output"
        assert "iterations" in payload, "Missing 'iterations' key in JSON output"
        assert "derived" in payload, "Missing 'derived' key in JSON output"

        assert len(payload["iterations"]) == n_iters
        assert len(payload["derived"]["sync_overhead_pct"]) == n_iters
        assert len(payload["derived"]["convergence_rate"]) == n_iters
        assert len(payload["derived"]["wcss_series"]) == n_iters


# ---------------------------------------------------------------------------
# Test 10 — aggregate_across_ranks mean computation
# ---------------------------------------------------------------------------
def test_aggregate_across_ranks_mean():
    """
    Write 2 synthetic per-rank CSVs with known values, call
    aggregate_across_ranks, and verify the mean of sync_overhead_pct
    across both ranks equals the analytically correct value.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Rank 0: sync_overhead_pct = 10.0, 20.0
        # Rank 1: sync_overhead_pct = 30.0, 40.0
        # Expected means: [20.0, 30.0]
        fieldnames = _ITER_FIELDS + ["sync_overhead_pct"]

        for rank_id, overheads in [(0, [10.0, 20.0]), (1, [30.0, 40.0])]:
            path = Path(tmp) / f"kmeans_metrics_rank{rank_id}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i, overhead in enumerate(overheads, start=1):
                    writer.writerow(
                        {
                            "iteration": i,
                            "spark_time_s": 0.5,
                            "sync_time_s": 0.1,
                            "iter_time_s": 0.6,
                            "centroid_shift": 0.1,
                            "global_wcss": 100.0,
                            "sync_overhead_pct": overhead,
                        }
                    )

        agg_path = KMeansMetricsCollector.aggregate_across_ranks(
            output_dir=tmp,
            num_ranks=2,
        )

        assert agg_path.exists(), f"Aggregated CSV not written: {agg_path}"

        rows = list(csv.DictReader(open(agg_path)))
        assert len(rows) == 2, f"Expected 2 aggregated rows, got {len(rows)}"

        assert (
            abs(float(rows[0]["sync_overhead_pct_mean"]) - 20.0) < 1e-4
        ), f"Iter 1 mean mismatch: {rows[0]['sync_overhead_pct_mean']}"
        assert (
            abs(float(rows[1]["sync_overhead_pct_mean"]) - 30.0) < 1e-4
        ), f"Iter 2 mean mismatch: {rows[1]['sync_overhead_pct_mean']}"
