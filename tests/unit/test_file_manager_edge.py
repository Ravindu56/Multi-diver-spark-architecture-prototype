# =============================================================
# tests/unit/test_file_manager_edge.py
#
# Targets the 3 uncovered lines in core/file_manager.py:
#   Lines 46-48 — _count_lines: the fh.seek(-1, 2) block that
#   reads the last byte to check for a trailing newline.  An
#   empty file (0 bytes) causes OSError on that seek; these
#   tests verify the guard works correctly.
# =============================================================
import pytest

from mpj_spark.core.file_manager import MPJSparkFileManager


class TestCountLinesEdgeCases:
    def test_empty_file_returns_zero(self, tmp_path):
        """A zero-byte file has 0 lines — no OSError."""
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

    def test_dynamic_partition_on_empty_file_raises_or_empty(self, tmp_path):
        """
        Partitioning a zero-line file must either raise a descriptive
        error (ValueError/ZeroDivisionError) or return empty partitions.
        It must NOT propagate a raw OSError from _count_lines.
        """
        empty_input = tmp_path / "empty_input.txt"
        empty_input.write_bytes(b"")

        storage = tmp_path / "shared"
        manager = MPJSparkFileManager(shared_storage_path=str(storage))

        try:
            result = manager.dynamic_partition(str(empty_input), num_workers=2)
            for meta in result:
                assert meta["num_lines"] == 0
        except (ValueError, ZeroDivisionError):
            pass  # acceptable — implementation may reject zero-line input
        except OSError as exc:
            pytest.fail(
                f"dynamic_partition raised unexpected OSError on empty input: {exc}"
            )

    def test_count_lines_chunk_boundary(self, tmp_path):
        """
        File with content spanning the 1 MB chunk boundary must be
        counted without off-by-one errors.
        """
        chunk = 1 << 20  # 1 MB
        # 2 embedded newlines + 1 line without trailing newline = 3 lines
        content = (
            b"A" * (chunk // 2 - 1) + b"\n" + b"B" * (chunk // 2 - 1) + b"\n" + b"C"
        )
        f = tmp_path / "boundary.txt"
        f.write_bytes(content)
        assert MPJSparkFileManager._count_lines(str(f)) == 3
