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
import os
import tempfile
from pathlib import Path

import pytest

from mpj_spark.applications.kmeans.metrics import KMeansMetricsCollector, _ITER_FIELDS


# ---------------------------------------------------------------------------
# Helper: build a collector with N synthetic iterations
# ---------------------------------------------------------------------------
def _make_collector(rank: int = 0, n_iters: int = 5, tmp_dir: str = ".") -> KMeansMetricsCollector:
    c = KMeansMetricsCollector(rank=rank, output_dir=tmp_dir)
    for i in range(1, n_iters + 1):
        c.record_iteration(
            iteration=i,
            spark_time_s=0.10 * i,
            sync_time_s=0.02 * i,
            iter_time_s=0.15 * i,
            centroid_shift=1.0 / (i * 10),
            global_wcss=100.0 - i * 5.0,
        )
    return c


# ---------------------------------------------------------------------------
# Test 1 — record_iteration stores exactly the 6 required fields
# ---------------------------------------------------------------------------
def test_record_iteration_fields():
    with tempfile.TemporaryDirectory() as tmp:
        c = KMeansMetricsCollector(rank=0, output_dir=tmp)
        c.record_iteration(
            iteration=1, spark_time_s=0.1, sync_time_s=0.02,
            iter_time_s=0.15, centroid_shift=0.5, global_wcss=200.0,
        )
        assert len(c._iterations) == 1
        row = c._iterations[0]
        for field in _ITER_FIELDS:
            assert field in row, f"Missing field '{field}' in iteration row"


# ---------------------------------------------------------------------------
# Test 2 — record_iteration rounds values correctly
# ---------------------------------------------------------------------------
def test_record_iteration_rounding():
    with tempfile.TemporaryDirectory() as tmp:
        c = KMeansMetricsCollector(rank=0, output_dir=tmp)
        c.record_iteration(
            iteration=1,
            spark_time_s=0.123456789,   # should round to 6dp
            sync_time_s=0.987654321,    # should round to 6dp
            iter_time_s=1.111111111,    # should round to 6dp
            centroid_shift=0.123456789, # should round to 8dp
            global_wcss=99.99999,       # should round to 4dp
        )
        row = c._iterations[0]
        assert row["spark_time_s"]   == round(0.123456789, 6)
        assert row["sync_time_s"]    == round(0.987654321, 6)
        assert row["iter_time_s"]    == round(1.111111111, 6)
        assert row["centroid_shift"] == round(0.123456789, 8)
        assert row["global_wcss"]    == round(99.99999,    4)


# ---------------------------------------------------------------------------
# Test 3 — sync_overhead_pct arithmetic
# ---------------------------------------------------------------------------
def test_sync_overhead_pct_arithmetic():
    """
    For iteration i: sync=0.02*i, iter=0.15*i
    overhead = (0.02*i / 0.15*i) * 100 = (0.02/0.15)*100 ≈ 13.3333%
    Should be constant across iterations (ratio of constants * i / constants * i).
    """
    with tempfile.TemporaryDirectory() as tmp:
        c = _make_collector(tmp_dir=tmp, n_iters=4)
        overheads = c.sync_overhead_pct()
        expected = round(0.02 / 0.15 * 100, 4)
        for pct in overheads:
            assert abs(pct - expected) < 0.01, (
                f"Expected overhead ≈ {expected:.4f}%, got {pct:.4f}%"
            )


# ---------------------------------------------------------------------------
# Test 4 — sync_overhead_pct returns 0.0 when iter_time_s == 0
# ---------------------------------------------------------------------------
def test_sync_overhead_pct_zero_division_guard():
    with tempfile.TemporaryDirectory() as tmp:
        c = KMeansMetricsCollector(rank=0, output_dir=tmp)
        c.record_iteration(
            iteration=1, spark_time_s=0.0, sync_time_s=0.0,
            iter_time_s=0.0, centroid_shift=0.0, global_wcss=0.0,
        )
        result = c.sync_overhead_pct()
        assert result == [0.0], f"Expected [0.0] for zero iter_time_s, got {result}"


# ---------------------------------------------------------------------------
# Test 5 — convergence_rate() returns centroid_shift series in order
# ---------------------------------------------------------------------------
def test_convergence_rate_series():
    with tempfile.TemporaryDirectory() as tmp:
        c = _make_collector(tmp_dir=tmp, n_iters=5)
        rates = c.convergence_rate()
        expected = [round(1.0 / (i * 10), 8) for i in range(1, 6)]
        assert rates == expected, f"convergence_rate mismatch: {rates} != {expected}"


# ---------------------------------------------------------------------------
# Test 6 — wcss_series() returns global_wcss series in order
# ---------------------------------------------------------------------------
def test_wcss_series_order():
    with tempfile.TemporaryDirectory() as tmp:
        c = _make_collector(tmp_dir=tmp, n_iters=3)
        series = c.wcss_series()
        # global_wcss = 100 - i*5 for i in [1,2,3] → [95.0, 90.0, 85.0]
        assert series == [95.0, 90.0, 85.0], f"wcss_series mismatch: {series}"


# ---------------------------------------------------------------------------
# Test 7 — summary_table() rows contain sync_overhead_pct field
# ---------------------------------------------------------------------------
def test_summary_table_has_overhead_field():
    with tempfile.TemporaryDirectory() as tmp:
        c = _make_collector(tmp_dir=tmp, n_iters=3)
        table = c.summary_table()
        assert len(table) == 3
        for row in table:
            assert "sync_overhead_pct" in row, (
                f"summary_table row missing 'sync_overhead_pct': {row.keys()}"
            )


# ---------------------------------------------------------------------------
# Test 8 — to_csv() writes correct header and correct row count
# ---------------------------------------------------------------------------
def test_to_csv_header_and_row_count():
    with tempfile.TemporaryDirectory() as tmp:
        n = 4
        c = _make_collector(rank=0, tmp_dir=tmp, n_iters=n)
        path = c.to_csv()
        assert path.exists(), f"CSV file not created at {path}"

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        expected_fields = _ITER_FIELDS + ["sync_overhead_pct"]
        assert list(reader.fieldnames) == expected_fields, (
            f"CSV headers mismatch: {reader.fieldnames} != {expected_fields}"
        )
        assert len(rows) == n, f"Expected {n} data rows, got {len(rows)}"


# ---------------------------------------------------------------------------
# Test 9 — to_json() structure has run / iterations / derived keys
# ---------------------------------------------------------------------------
def test_to_json_structure():
    with tempfile.TemporaryDirectory() as tmp:
        n = 3
        c = _make_collector(rank=0, tmp_dir=tmp, n_iters=n)
        c.record_run(
            total_time_s=1.5, iterations_run=n,
            converged=True, k=3, dataset_size=1000, num_ranks=3,
        )
        path = c.to_json()
        assert path.exists()

        with open(path) as f:
            payload = json.load(f)

        assert "run"        in payload, "JSON missing 'run' key"
        assert "iterations" in payload, "JSON missing 'iterations' key"
        assert "derived"    in payload, "JSON missing 'derived' key"
        assert len(payload["iterations"]) == n
        assert len(payload["derived"]["sync_overhead_pct"]) == n
        assert len(payload["derived"]["convergence_rate"])  == n
        assert len(payload["derived"]["wcss_series"])       == n
        assert payload["run"]["converged"] is True
        assert payload["run"]["throughput"] == round(1000 / 1.5, 2)


# ---------------------------------------------------------------------------
# Test 10 — aggregate_across_ranks() mean computation
# ---------------------------------------------------------------------------
def test_aggregate_across_ranks_mean():
    """
    Write 2 synthetic per-rank CSVs (rank 0 and rank 1) with known values,
    run aggregate_across_ranks(), verify the mean sync_overhead_pct.

    Rank 0, iter 1: sync=0.02, iter=0.10  → overhead = 20.0%
    Rank 1, iter 1: sync=0.04, iter=0.10  → overhead = 40.0%
    Mean = 30.0%
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Write rank 0 CSV manually
        for rank_id, sync in [(0, 0.02), (1, 0.04)]:
            path = Path(tmp) / f"kmeans_metrics_rank{rank_id}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=_ITER_FIELDS + ["sync_overhead_pct"],
                )
                writer.writeheader()
                writer.writerow({
                    "iteration"      : 1,
                    "spark_time_s"   : 0.08,
                    "sync_time_s"    : sync,
                    "iter_time_s"    : 0.10,
                    "centroid_shift" : 0.5,
                    "global_wcss"    : 150.0,
                    "sync_overhead_pct": round(sync / 0.10 * 100, 4),
                })

        agg_path = KMeansMetricsCollector.aggregate_across_ranks(
            output_dir=tmp, num_ranks=2
        )
        assert agg_path.exists()

        with open(agg_path, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        mean_overhead = float(rows[0]["sync_overhead_pct_mean"])
        assert abs(mean_overhead - 30.0) < 0.01, (
            f"Expected mean sync_overhead_pct ≈ 30.0%, got {mean_overhead:.4f}%"
        )
