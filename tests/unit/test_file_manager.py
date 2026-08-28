# =============================================================
# tests/unit/test_file_manager.py
#
# Unit tests for mpj_spark/core/file_manager.py (MPJSparkFileManager)
# Covers Phase 1: dynamic data partitioning and streaming I/O.
#
# Research alignment:
#   - Objective 1a: multi-driver dataset partitioning
#   - Objective 1c: correctness of data distribution to workers
# =============================================================
import os

import pytest

from mpj_spark.core.file_manager import MPJSparkFileManager

# =============================================================
# Fixtures
# =============================================================


@pytest.fixture
def file_manager(tmp_path):
    """MPJSparkFileManager pointing at a fresh temp directory."""
    return MPJSparkFileManager(shared_storage_path=str(tmp_path))


def _write_lines(tmp_path, lines, filename="input.txt"):
    """
    Write a list of strings as newline-separated lines to a temp file.
    Returns the absolute path.
    """
    p = tmp_path / filename
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


# =============================================================
# Section 1: _count_lines() static method
# =============================================================


class TestCountLines:
    """Tests for the O(1)-memory binary line counter."""

    def test_count_exact(self, tmp_path):
        path = _write_lines(tmp_path, [f"line {i}" for i in range(50)])
        assert MPJSparkFileManager._count_lines(path) == 50

    def test_count_single_line(self, tmp_path):
        path = _write_lines(tmp_path, ["only one line"])
        assert MPJSparkFileManager._count_lines(path) == 1

    def test_count_large_file(self, tmp_path):
        """10 000-line file — validates chunk-based counting."""
        path = _write_lines(tmp_path, [f"word{i} word{i + 1}" for i in range(10_000)])
        assert MPJSparkFileManager._count_lines(path) == 10_000

    def test_count_file_without_trailing_newline(self, tmp_path):
        """
        Lines 46-48: the 'if last byte != newline: count += 1' branch.
        A file written WITHOUT a trailing newline must still count
        its last line. This covers the branch that was previously
        uncovered.
        """
        p = tmp_path / "no_newline.txt"
        # Write 3 lines with NO trailing newline
        p.write_bytes(b"line_one\nline_two\nline_three")
        assert MPJSparkFileManager._count_lines(str(p)) == 3

    def test_count_single_byte_no_newline(self, tmp_path):
        """
        Line 65: fh.seek(-1, 2) is executed even on a 1-byte file.
        A single character with no newline must count as 1 line.
        """
        p = tmp_path / "one_byte.txt"
        p.write_bytes(b"x")  # 1 byte, no newline
        assert MPJSparkFileManager._count_lines(str(p)) == 1

    def test_count_file_with_trailing_newline_unchanged(self, tmp_path):
        """
        Regression guard: files WITH a trailing newline must not
        get an extra line counted (the else branch must not fire).
        """
        p = tmp_path / "trailing.txt"
        p.write_bytes(b"alpha\nbeta\n")  # ends WITH \n
        assert MPJSparkFileManager._count_lines(str(p)) == 2


# =============================================================
# Section 2: dynamic_partition() — partition count & metadata
# =============================================================


class TestDynamicPartitionCount:
    """Validate partition count and metadata structure."""

    def test_correct_number_of_partitions(self, file_manager, tmp_path):
        """N workers must produce exactly N partition metadata entries."""
        path = _write_lines(tmp_path, [f"line {i}" for i in range(100)])
        result = file_manager.dynamic_partition(path, num_workers=4)
        assert len(result) == 4

    def test_metadata_keys_present(self, file_manager, tmp_path):
        """Each metadata dict must contain all required keys."""
        path = _write_lines(tmp_path, [f"line {i}" for i in range(20)])
        result = file_manager.dynamic_partition(path, num_workers=2)
        required = {
            "partition_id",
            "partition_path",
            "num_lines",
            "start_line",
            "end_line",
            "file_size_bytes",
        }
        for meta in result:
            assert required.issubset(meta.keys())

    def test_partition_ids_are_sequential(self, file_manager, tmp_path):
        """partition_id values must be 0, 1, 2, ... N-1."""
        path = _write_lines(tmp_path, [f"x {i}" for i in range(30)])
        result = file_manager.dynamic_partition(path, num_workers=3)
        ids = [m["partition_id"] for m in result]
        assert ids == list(range(3))

    def test_partition_files_exist_on_disk(self, file_manager, tmp_path):
        """Each partition_path in metadata must be a real file."""
        path = _write_lines(tmp_path, [f"word {i}" for i in range(40)])
        result = file_manager.dynamic_partition(path, num_workers=4)
        for meta in result:
            assert os.path.isfile(meta["partition_path"]), (
                f"Partition file not found: {meta['partition_path']}"
            )

    def test_single_worker_gets_all_lines(self, file_manager, tmp_path):
        """With num_workers=1, one partition must contain all lines."""
        lines = [f"line {i}" for i in range(25)]
        path = _write_lines(tmp_path, lines)
        result = file_manager.dynamic_partition(path, num_workers=1)
        assert result[0]["num_lines"] == 25


# =============================================================
# Section 3: dynamic_partition() — data completeness
# =============================================================


class TestDynamicPartitionCompleteness:
    """
    Validate that partitioning is lossless and non-duplicating.
    These tests read back partition files from disk and verify
    that the full content is recovered exactly.
    """

    def _read_partition_lines(self, meta):
        """Read all non-empty lines from a partition file."""
        with open(meta["partition_path"], encoding="utf-8") as fh:
            return [ln.rstrip("\n") for ln in fh if ln.strip()]

    def test_no_lines_lost(self, file_manager, tmp_path):
        """
        Total lines across all partitions must equal input line count.
        Validates lossless round-robin assignment.
        """
        lines = [f"token_{i}" for i in range(100)]
        path = _write_lines(tmp_path, lines)
        result = file_manager.dynamic_partition(path, num_workers=4)
        total = sum(m["num_lines"] for m in result)
        assert total == 100

    def test_no_lines_duplicated(self, file_manager, tmp_path):
        """
        Each line must appear in exactly one partition.
        Validates that round-robin assignment does not overlap.
        """
        lines = [f"unique_line_{i}" for i in range(80)]
        path = _write_lines(tmp_path, lines)
        result = file_manager.dynamic_partition(path, num_workers=4)
        all_lines = []
        for meta in result:
            all_lines.extend(self._read_partition_lines(meta))
        assert len(all_lines) == len(set(all_lines)), "Duplicate lines found across partitions"

    def test_uneven_split_covered(self, file_manager, tmp_path):
        """
        97 lines into 4 workers (not evenly divisible).
        Total across partitions must still be 97 — no line dropped.
        """
        lines = [f"item_{i}" for i in range(97)]
        path = _write_lines(tmp_path, lines)
        result = file_manager.dynamic_partition(path, num_workers=4)
        total = sum(m["num_lines"] for m in result)
        assert total == 97

    def test_full_content_recoverable(self, file_manager, tmp_path):
        """
        Reconstructed content from all partitions must match
        original input (regardless of line order within partitions).
        Round-robin interleaving means line order changes, so we
        compare sorted sets.
        """
        lines = [f"word_{i}" for i in range(60)]
        path = _write_lines(tmp_path, lines)
        result = file_manager.dynamic_partition(path, num_workers=3)
        recovered = []
        for meta in result:
            recovered.extend(self._read_partition_lines(meta))
        assert sorted(recovered) == sorted(lines)


# =============================================================
# Section 4: cleanup()
# =============================================================


class TestCleanup:
    """Validate that cleanup() removes all partition files."""

    def test_cleanup_removes_partition_files(self, file_manager, tmp_path):
        path = _write_lines(tmp_path, [f"line {i}" for i in range(20)])
        result = file_manager.dynamic_partition(path, num_workers=2)
        file_manager.cleanup()
        for meta in result:
            assert not os.path.isfile(meta["partition_path"]), (
                f"Partition file still exists after cleanup: {meta['partition_path']}"
            )

    def test_cleanup_recreates_partitions_dir(self, file_manager):
        """After cleanup, the partitions directory must still exist (empty)."""
        file_manager.cleanup()
        assert os.path.isdir(file_manager.partitions_dir)
