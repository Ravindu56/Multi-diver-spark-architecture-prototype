# =============================================================
# tests/unit/test_spark_session.py
#
# Unit tests for workers/spark_session.py — covers:
#   Lines 28-39  : get_total_ram_mb() fallback chain
#                  (psutil present, psutil absent + /proc/meminfo,
#                   both absent → default 8192 MB)
#   Lines 53-124 : build_spark_session() RAM/CPU allocation
#                  logic — mocked so no real JVM is started.
#
# All SparkSession construction is intercepted via monkeypatching;
# we test the *config values* that would be passed, not Spark itself.
# =============================================================
import sys
import types
from unittest.mock import MagicMock, patch, call

import pytest


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _make_spark_builder_mock():
    """
    Return a mock that chains .appName().master().config()...getOrCreate()
    so build_spark_session() can call the full builder chain without
    a running JVM.
    """
    session   = MagicMock(name="SparkSession")
    builder   = MagicMock(name="builder")
    # Every builder method returns the builder itself (fluent API)
    builder.appName.return_value   = builder
    builder.master.return_value    = builder
    builder.config.return_value    = builder
    builder.getOrCreate.return_value = session
    session.builder = builder
    session.sparkContext = MagicMock()
    return session, builder


# ─────────────────────────────────────────────────────────────────
# get_total_ram_mb
# ─────────────────────────────────────────────────────────────────

class TestGetTotalRamMb:
    """Tests for the three-tier RAM detection fallback."""

    def test_uses_psutil_when_available(self, monkeypatch):
        """When psutil is importable, return its virtual_memory().total."""
        fake_psutil          = types.ModuleType("psutil")
        fake_vm              = MagicMock()
        fake_vm.total        = 16 * 1024 * 1024 * 1024   # 16 GB
        fake_psutil.virtual_memory = lambda: fake_vm

        # Inject fake psutil before importing the module under test
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        # Re-import to pick up the patched sys.modules
        import importlib
        import mpj_spark.workers.spark_session as ss
        importlib.reload(ss)

        result = ss.get_total_ram_mb()
        assert result == 16 * 1024   # 16384 MB

    def test_fallback_to_proc_meminfo(self, monkeypatch, tmp_path):
        """When psutil is absent, fall back to /proc/meminfo."""
        import importlib

        # Remove psutil from sys.modules so ImportError is triggered
        monkeypatch.setitem(sys.modules, "psutil", None)

        # Write a synthetic /proc/meminfo
        fake_meminfo = tmp_path / "meminfo"
        fake_meminfo.write_text(
            "MemTotal:       8192000 kB\nMemFree: 4096000 kB\n",
            encoding="utf-8",
        )

        import mpj_spark.workers.spark_session as ss
        importlib.reload(ss)

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__ = lambda s: iter(
                ["MemTotal:       8192000 kB\n", "MemFree: 4096000 kB\n"]
            )
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            result = ss.get_total_ram_mb()

        # 8192000 kB // 1024 = 8000 MB
        assert result == 8000

    def test_default_8192_when_both_absent(self, monkeypatch):
        """When psutil is absent AND /proc/meminfo raises, return 8192."""
        import importlib

        monkeypatch.setitem(sys.modules, "psutil", None)

        import mpj_spark.workers.spark_session as ss
        importlib.reload(ss)

        with patch("builtins.open", side_effect=OSError("no meminfo")):
            result = ss.get_total_ram_mb()

        assert result == 8192

    def test_returns_integer(self):
        """Return type is always int regardless of path taken."""
        from mpj_spark.workers.spark_session import get_total_ram_mb
        result = get_total_ram_mb()
        assert isinstance(result, int)

    def test_returns_positive_value(self):
        """Return value is always > 0."""
        from mpj_spark.workers.spark_session import get_total_ram_mb
        assert get_total_ram_mb() > 0


# ─────────────────────────────────────────────────────────────────
# build_spark_session — RAM allocation paths
# ─────────────────────────────────────────────────────────────────

class TestBuildSparkSessionRamAllocation:
    """
    All tests mock SparkSession.builder so no JVM is launched.
    We inspect the 'spark.driver.memory' config call to verify
    the correct heap_mb was computed.
    """

    @pytest.fixture(autouse=True)
    def _patch_spark(self, monkeypatch):
        """Intercept SparkSession import inside build_spark_session."""
        self.session, self.builder = _make_spark_builder_mock()
        spark_module = types.ModuleType("pyspark.sql")
        spark_module.SparkSession = MagicMock()
        spark_module.SparkSession.builder = self.builder
        monkeypatch.setitem(sys.modules, "pyspark",      types.ModuleType("pyspark"))
        monkeypatch.setitem(sys.modules, "pyspark.sql",  spark_module)

    def _driver_memory_from_calls(self):
        """Extract the 'spark.driver.memory' value from builder.config calls."""
        for c in self.builder.config.call_args_list:
            args = c[0]
            if args and args[0] == "spark.driver.memory":
                return args[1]   # e.g. '512m'
        return None

    def test_driver_memory_override_used_directly(self, monkeypatch):
        """driver_memory_mb=1024 must produce 'spark.driver.memory' == '1024m'."""
        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", driver_memory_mb=1024)
        assert self._driver_memory_from_calls() == "1024m"

    def test_per_worker_allocation_divides_ram(self, monkeypatch):
        """
        When num_workers=2, memory_fraction=1.0, heap_mb should equal
        get_total_ram_mb() // 2 (clamped to >= 512).
        """
        from mpj_spark.workers import spark_session as ss
        total_ram = 4096
        monkeypatch.setattr(ss, "get_total_ram_mb", lambda: total_ram)

        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", num_workers=2, memory_fraction=1.0)

        expected_heap = max(512, total_ram // 2)
        assert self._driver_memory_from_calls() == f"{expected_heap}m"

    def test_memory_floor_is_512mb(self, monkeypatch):
        """
        With a very low total RAM and many workers, heap_mb must not
        go below 512 MB.
        """
        from mpj_spark.workers import spark_session as ss
        monkeypatch.setattr(ss, "get_total_ram_mb", lambda: 512)

        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", num_workers=100, memory_fraction=0.75)

        assert self._driver_memory_from_calls() == "512m"

    def test_no_workers_uses_fraction_of_total(self, monkeypatch):
        """
        Without num_workers, heap = total_ram * memory_fraction (no division).
        """
        from mpj_spark.workers import spark_session as ss
        monkeypatch.setattr(ss, "get_total_ram_mb", lambda: 8192)

        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", memory_fraction=0.5)

        assert self._driver_memory_from_calls() == "4096m"


# ─────────────────────────────────────────────────────────────────
# build_spark_session — CPU allocation paths
# ─────────────────────────────────────────────────────────────────

class TestBuildSparkSessionCpuAllocation:
    """Verify that master() receives the correct local[N] string."""

    @pytest.fixture(autouse=True)
    def _patch_spark(self, monkeypatch):
        self.session, self.builder = _make_spark_builder_mock()
        spark_module = types.ModuleType("pyspark.sql")
        spark_module.SparkSession = MagicMock()
        spark_module.SparkSession.builder = self.builder
        monkeypatch.setitem(sys.modules, "pyspark",     types.ModuleType("pyspark"))
        monkeypatch.setitem(sys.modules, "pyspark.sql", spark_module)

    def test_cores_override_sets_local_master(self, monkeypatch):
        """cores_override=4 → master='local[4]'."""
        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", cores_override=4)
        self.builder.master.assert_called_once_with("local[4]")

    def test_default_cores_from_config(self, monkeypatch):
        """
        Without cores_override, master uses TOTAL_CORES from config.
        """
        import mpj_spark.workers.spark_session as ss
        monkeypatch.setattr(
            "mpj_spark.workers.spark_session",
            "TOTAL_CORES",
            None,   # will be overridden below via config patch
        )
        from mpj_spark.workers.spark_session import build_spark_session
        with patch("mpj_spark.workers.spark_session.TOTAL_CORES", 8):
            build_spark_session("TestApp")
        self.builder.master.assert_called_once_with("local[8]")


# ─────────────────────────────────────────────────────────────────
# build_spark_session — config keys emitted
# ─────────────────────────────────────────────────────────────────

class TestBuildSparkSessionConfigKeys:
    """Verify that the full set of expected config keys is passed."""

    EXPECTED_KEYS = {
        "spark.driver.memory",
        "spark.executor.memory",
        "spark.driver.maxResultSize",
        "spark.memory.fraction",
        "spark.memory.storageFraction",
        "spark.sql.shuffle.partitions",
        "spark.default.parallelism",
        "spark.serializer",
        "spark.kryoserializer.buffer.max",
        "spark.driver.extraJavaOptions",
        "spark.executor.extraJavaOptions",
        "spark.eventLog.gcMetrics.youngGenerationGarbageCollectors",
        "spark.eventLog.gcMetrics.oldGenerationGarbageCollectors",
    }

    @pytest.fixture(autouse=True)
    def _patch_spark(self, monkeypatch):
        self.session, self.builder = _make_spark_builder_mock()
        spark_module = types.ModuleType("pyspark.sql")
        spark_module.SparkSession = MagicMock()
        spark_module.SparkSession.builder = self.builder
        monkeypatch.setitem(sys.modules, "pyspark",     types.ModuleType("pyspark"))
        monkeypatch.setitem(sys.modules, "pyspark.sql", spark_module)

    def test_all_expected_config_keys_are_set(self):
        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", cores_override=2, driver_memory_mb=512)

        called_keys = {
            c[0][0] for c in self.builder.config.call_args_list
        }
        for key in self.EXPECTED_KEYS:
            assert key in called_keys, f"Missing config key: {key}"

    def test_blas_flags_in_java_options(self):
        """Verify native BLAS system property is present in extraJavaOptions."""
        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", cores_override=2, driver_memory_mb=512)

        jvm_val = None
        for c in self.builder.config.call_args_list:
            if c[0][0] == "spark.driver.extraJavaOptions":
                jvm_val = c[0][1]
                break

        assert jvm_val is not None
        assert "NativeSystemBLAS" in jvm_val

    def test_gc_flags_in_java_options(self):
        """Verify G1GC flag is present in extraJavaOptions."""
        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", cores_override=2, driver_memory_mb=512)

        jvm_val = None
        for c in self.builder.config.call_args_list:
            if c[0][0] == "spark.driver.extraJavaOptions":
                jvm_val = c[0][1]
                break

        assert jvm_val is not None
        assert "UseG1GC" in jvm_val

    def test_returns_spark_session_object(self):
        """build_spark_session must return the object from getOrCreate()."""
        from mpj_spark.workers.spark_session import build_spark_session
        result = build_spark_session("TestApp", cores_override=1, driver_memory_mb=512)
        assert result is self.builder.getOrCreate.return_value
