# tests/logreg/test_metrics.py
#
# Unit tests for:
#   mpj_spark.applications.logreg.metrics.LogRegMetricsCollector
#
# Strategy:
#   All file I/O uses pytest's tmp_path fixture so nothing is written
#   into the working directory.  No MPI or Spark is required.

import csv
import json
import os

import pytest

from mpj_spark.applications.logreg.metrics import LogRegMetricsCollector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_collector(tmp_path, rank=0):
    """Return a fresh collector writing to tmp_path."""
    return LogRegMetricsCollector(rank=rank, output_dir=str(tmp_path))


def _add_n_epochs(collector, n):
    """Append n synthetic epoch records to the collector."""
    for i in range(1, n + 1):
        collector.record_epoch(
            epoch=i,
            spark_time_s=0.1 * i,
            sync_time_s=0.05 * i,
            epoch_time_s=0.15 * i,
            grad_norm=1.0 / i,
            global_loss=0.693 - 0.01 * i,
            weight_norm=0.5 * i,
        )


# ---------------------------------------------------------------------------
# record_epoch
# ---------------------------------------------------------------------------


class TestRecordEpoch:
    def test_appends_correct_keys(self, tmp_path):
        """Each record must contain all seven EPOCH_FIELDS."""
        c = _make_collector(tmp_path)
        c.record_epoch(
            epoch=1,
            spark_time_s=0.1,
            sync_time_s=0.05,
            epoch_time_s=0.15,
            grad_norm=0.5,
            global_loss=0.693,
            weight_norm=0.3,
        )

        assert len(c._epochs) == 1
        record = c._epochs[0]
        for field in LogRegMetricsCollector.EPOCH_FIELDS:
            assert field in record, f"Missing field: {field}"

    def test_epoch_number_stored_correctly(self, tmp_path):
        c = _make_collector(tmp_path)
        c.record_epoch(
            epoch=5,
            spark_time_s=0.2,
            sync_time_s=0.1,
            epoch_time_s=0.3,
            grad_norm=0.1,
            global_loss=0.5,
            weight_norm=0.2,
        )
        assert c._epochs[0]["epoch"] == 5

    def test_multiple_epochs_accumulate(self, tmp_path):
        c = _make_collector(tmp_path)
        _add_n_epochs(c, 5)
        assert len(c._epochs) == 5

    def test_values_are_rounded(self, tmp_path):
        """
        spark_time_s is rounded to 6 dp; grad_norm/global_loss/weight_norm
        are rounded to 8 dp.
        """
        c = _make_collector(tmp_path)
        c.record_epoch(
            epoch=1,
            spark_time_s=0.123456789,
            sync_time_s=0.0,
            epoch_time_s=0.0,
            grad_norm=1.123456789123,
            global_loss=0.5,
            weight_norm=0.0,
        )
        assert c._epochs[0]["spark_time_s"] == round(0.123456789, 6)
        assert c._epochs[0]["grad_norm"] == round(1.123456789123, 8)


# ---------------------------------------------------------------------------
# record_run
# ---------------------------------------------------------------------------


class TestRecordRun:
    def test_stores_all_run_fields(self, tmp_path):
        c = _make_collector(tmp_path, rank=2)
        c.record_run(
            total_time_s=12.34,
            epochs_run=10,
            converged=True,
            dataset_size=50000,
            num_ranks=3,
            learning_rate=0.01,
            tol=1e-4,
        )

        run = c._run
        assert run["rank"] == 2
        assert run["epochs_run"] == 10
        assert run["converged"] is True
        assert run["dataset_size"] == 50000
        assert run["num_ranks"] == 3
        assert run["learning_rate"] == pytest.approx(0.01)
        assert run["tol"] == pytest.approx(1e-4)

    def test_total_time_rounded_to_4dp(self, tmp_path):
        c = _make_collector(tmp_path)
        c.record_run(
            total_time_s=3.141592653,
            epochs_run=1,
            converged=False,
            dataset_size=100,
            num_ranks=1,
            learning_rate=0.1,
            tol=0.01,
        )
        assert c._run["total_time_s"] == round(3.141592653, 4)


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------


class TestToCsv:
    def test_csv_file_created(self, tmp_path):
        c = _make_collector(tmp_path, rank=1)
        _add_n_epochs(c, 3)
        path = c.to_csv()
        assert os.path.exists(path)

    def test_csv_filename_contains_rank(self, tmp_path):
        c = _make_collector(tmp_path, rank=2)
        _add_n_epochs(c, 1)
        path = c.to_csv()
        assert "rank2" in os.path.basename(path)

    def test_csv_has_correct_header(self, tmp_path):
        c = _make_collector(tmp_path)
        _add_n_epochs(c, 1)
        path = c.to_csv()
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert list(reader.fieldnames) == LogRegMetricsCollector.EPOCH_FIELDS

    def test_csv_row_count_matches_epochs(self, tmp_path):
        n = 7
        c = _make_collector(tmp_path)
        _add_n_epochs(c, n)
        path = c.to_csv()
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == n

    def test_csv_values_match_recorded_data(self, tmp_path):
        c = _make_collector(tmp_path)
        c.record_epoch(
            epoch=1,
            spark_time_s=0.5,
            sync_time_s=0.25,
            epoch_time_s=0.75,
            grad_norm=0.333,
            global_loss=0.6,
            weight_norm=1.0,
        )
        path = c.to_csv()
        with open(path, newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        assert row["epoch"] == "1"
        assert float(row["spark_time_s"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------


class TestToJson:
    def test_json_file_created(self, tmp_path):
        c = _make_collector(tmp_path)
        c.record_run(
            total_time_s=1.0,
            epochs_run=3,
            converged=False,
            dataset_size=100,
            num_ranks=1,
            learning_rate=0.01,
            tol=1e-4,
        )
        path = c.to_json()
        assert os.path.exists(path)

    def test_json_is_valid_and_matches_run(self, tmp_path):
        c = _make_collector(tmp_path, rank=0)
        c.record_run(
            total_time_s=5.0,
            epochs_run=10,
            converged=True,
            dataset_size=200,
            num_ranks=3,
            learning_rate=0.05,
            tol=1e-3,
        )
        path = c.to_json()
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["converged"] is True
        assert data["epochs_run"] == 10
        assert data["rank"] == 0


# ---------------------------------------------------------------------------
# summary_table
# ---------------------------------------------------------------------------


class TestSummaryTable:
    def test_returns_list_of_dicts(self, tmp_path):
        c = _make_collector(tmp_path)
        _add_n_epochs(c, 3)
        table = c.summary_table()
        assert isinstance(table, list)
        assert len(table) == 3
        assert all(isinstance(row, dict) for row in table)

    def test_returns_copy_not_reference(self, tmp_path):
        """Mutating the returned list must not affect the internal _epochs."""
        c = _make_collector(tmp_path)
        _add_n_epochs(c, 2)
        table = c.summary_table()
        table.clear()
        assert len(c._epochs) == 2


# ---------------------------------------------------------------------------
# aggregate_across_ranks
# ---------------------------------------------------------------------------


class TestAggregateAcrossRanks:
    def test_aggregated_csv_contains_all_rank_rows(self, tmp_path):
        """Two ranks, 3 epochs each → aggregated CSV must have 6 data rows."""
        for rank in range(2):
            c = _make_collector(tmp_path, rank=rank)
            _add_n_epochs(c, 3)
            c.to_csv()

        out = LogRegMetricsCollector.aggregate_across_ranks(output_dir=str(tmp_path), num_ranks=2)
        assert out is not None
        with open(out, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 6

    def test_aggregated_csv_has_rank_column(self, tmp_path):
        """The aggregated CSV must include a 'rank' column."""
        c = _make_collector(tmp_path, rank=0)
        _add_n_epochs(c, 1)
        c.to_csv()

        out = LogRegMetricsCollector.aggregate_across_ranks(output_dir=str(tmp_path), num_ranks=1)
        with open(out, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert "rank" in reader.fieldnames

    def test_returns_none_when_no_rank_files(self, tmp_path):
        """If no rank CSV files exist, aggregate must return None gracefully."""
        result = LogRegMetricsCollector.aggregate_across_ranks(
            output_dir=str(tmp_path), num_ranks=3
        )
        assert result is None
