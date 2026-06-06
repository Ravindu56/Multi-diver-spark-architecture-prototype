# =============================================================
# tests/unit/test_file_manager_edge.py
#
# Targets the 3 uncovered lines in core/file_manager.py:
#   Line 46-48 — _count_lines: edge case where the file is
#   empty (zero bytes).  The existing tests cover non-empty
#   files with and without trailing newlines; this suite
#   closes the remaining gap.
# =============================================================
import os
import tempfile
import pytest

from mpj_spark.core.file_manager import MPJSparkFileManager


class TestCountLinesEdgeCases:
    """
    Lines 46-48 of file_manager.py guard the fh.seek(-1, 2) call
    against a zero-byte file (seeking to offset -1 from EOF raises
    OSError when the file is empty).  These tests exercise that path.
    """

    def test_empty_file_returns_zero(self, tmp_path):
        """An empty file has 0 lines — no OSError should be raised."""
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        assert MPJSparkFileManager._count_lines(str(f)) == 0

    def test_empty_file_does_not_raise(self, tmp_path):
        """_count_lines on a zero-byte file must not raise any exception."""
        f = tmp_path / "zero.txt"
        f.write_bytes(b"")
        try:
            MPJSparkFileManager._count_lines(str(f))
        except Exception as exc:
            pytest.fail(f"_count_lines raised unexpectedly: {exc}")

    def test_single_newline_is_one_line(self, tmp_path):
        """A file containing only '\\n' counts as 1 line."""
        f = tmp_path / "newline_only.txt"
        f.write_bytes(b"\n")
        assert MPJSparkFileManager._count_lines(str(f)) == 1

    def test_dynamic_partition_on_empty_file_raises_or_returns_empty(self, tmp_path):
        """
        Partitioning an empty input file with 2 workers should either
        return a list of 2 empty partition files OR raise a descriptive
        error — it must NOT produce an unhandled OSError from _count_lines.
        """
        empty_input = tmp_path / "empty_input.txt"
        empty_input.write_bytes(b"")

        storage = tmp_path / "shared"
        manager = MPJSparkFileManager(shared_storage_path=str(storage))

        try:
            result = manager.dynamic_partition(str(empty_input), num_workers=2)
            # If it succeeds, each partition should report 0 lines
            for meta in result:
                assert meta["num_lines"] == 0
        except (ValueError, ZeroDivisionError):
            # Acceptable: the implementation may reject a zero-line file
            pass
        except OSError as exc:
            pytest.fail(
                f"dynamic_partition raised an unexpected OSError on empty input: {exc}"
            )

    def test_count_lines_on_large_binary_boundary(self, tmp_path):
        """
        A file whose size is exactly 1 MB (the chunk size used inside
        _count_lines) must be handled correctly without off-by-one errors.
        """
        chunk = 1 << 20  # 1 MB
        # Fill with 'A' except for two embedded newlines and no trailing newline
        content = b"A" * (chunk // 2 - 1) + b"\n" + b"B" * (chunk // 2 - 1) + b"\n" + b"C"
        f = tmp_path / "boundary.txt"
        f.write_bytes(content)
        # 2 embedded newlines + 1 line without trailing newline = 3 lines
        assert MPJSparkFileManager._count_lines(str(f)) == 3
