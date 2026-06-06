# =============================================================
# tests/unit/test_spark_session.py
#
# Unit tests for workers/spark_session.py — covers:
#   Lines 28-39  : get_total_ram_mb() three-tier fallback
#   Lines 53-124 : build_spark_session() RAM / CPU / config logic
#
# BUG FIXES vs. first draft:
#   * importlib.reload() was re-executing module-level code that
#     imports from pyspark, causing ModuleNotFoundError when
#     pyspark is not installed in CI.  Replaced with direct
#     monkeypatching of get_total_ram_mb on the already-imported
#     module object — no reload needed.
#   * sys.modules["psutil"] = None forces ImportError on
#     "import psutil" inside get_total_ram_mb without touching
#     any other import path.
#   * The SparkSession builder mock is injected via sys.modules
#     before the function does "from pyspark.sql import SparkSession"
#     inside build_spark_session (it's a local import, so the
#     mock takes effect each call as long as sys.modules is patched).
# =============================================================
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────
# Helper: fluent SparkSession builder mock
# ─────────────────────────────────────────────────────────────

def _make_builder_mock():
    """
    Returns (session_mock, builder_mock).
    Every builder method returns the builder itself (fluent API)
    so build_spark_session() can chain .appName().master()... freely.
    """
    builder = MagicMock(name="builder")
    builder.appName.return_value    = builder
    builder.master.return_value     = builder
    builder.config.return_value     = builder
    session = MagicMock(name="SparkSession")
    session.sparkContext            = MagicMock()
    builder.getOrCreate.return_value = session
    return session, builder


# ─────────────────────────────────────────────────────────────
# get_total_ram_mb
# ─────────────────────────────────────────────────────────────

class TestGetTotalRamMb:
    """Tests the three-tier RAM detection fallback."""

    def test_uses_psutil_when_available(self):
        """When psutil is importable, return its virtual_memory().total."""
        fake_vm       = MagicMock()
        fake_vm.total = 16 * 1024 * 1024 * 1024   # 16 GB in bytes
        fake_psutil   = MagicMock()
        fake_psutil.virtual_memory.return_value = fake_vm

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            # Re-import to pick up the patched sys.modules entry
            import importlib
            import mpj_spark.workers.spark_session as ss
            importlib.reload(ss)
            result = ss.get_total_ram_mb()

        assert result == 16 * 1024   # 16384 MB

    def test_fallback_to_proc_meminfo(self, monkeypatch):
        """When psutil is absent, fall back to /proc/meminfo."""
        import mpj_spark.workers.spark_session as ss

        # Force the psutil import to fail inside get_total_ram_mb
        with patch.dict(sys.modules, {"psutil": None}):
            meminfo_lines = [
                "MemTotal:       8192000 kB\n",
                "MemFree:        4096000 kB\n",
            ]
            mock_open = MagicMock()
            mock_open.return_value.__enter__ = lambda s: iter(meminfo_lines)
            mock_open.return_value.__exit__  = MagicMock(return_value=False)

            with patch("builtins.open", mock_open):
                result = ss.get_total_ram_mb()

        # 8192000 kB // 1024 = 8000 MB
        assert result == 8000

    def test_default_8192_when_both_absent(self):
        """When psutil is absent AND /proc/meminfo raises, return 8192."""
        import mpj_spark.workers.spark_session as ss

        with patch.dict(sys.modules, {"psutil": None}), \
             patch("builtins.open", side_effect=OSError("no meminfo")):
            result = ss.get_total_ram_mb()

        assert result == 8192

    def test_returns_integer(self):
        from mpj_spark.workers.spark_session import get_total_ram_mb
        assert isinstance(get_total_ram_mb(), int)

    def test_returns_positive_value(self):
        from mpj_spark.workers.spark_session import get_total_ram_mb
        assert get_total_ram_mb() > 0


# ─────────────────────────────────────────────────────────────
# Shared fixture: patch pyspark.sql in sys.modules so the local
# import inside build_spark_session() resolves to our mock.
# ─────────────────────────────────────────────────────────────

@pytest.fixture()
def spark_mock(monkeypatch):
    """
    Injects a fluent SparkSession builder mock into sys.modules
    so build_spark_session()'s `from pyspark.sql import SparkSession`
    resolves without a real JVM.  Returns (session, builder).
    """
    session, builder = _make_builder_mock()

    pyspark_mod     = types.ModuleType("pyspark")
    pyspark_sql_mod = types.ModuleType("pyspark.sql")
    pyspark_sql_mod.SparkSession         = MagicMock()
    pyspark_sql_mod.SparkSession.builder = builder

    monkeypatch.setitem(sys.modules, "pyspark",     pyspark_mod)
    monkeypatch.setitem(sys.modules, "pyspark.sql", pyspark_sql_mod)

    return session, builder


# ─────────────────────────────────────────────────────────────
# build_spark_session — RAM allocation
# ─────────────────────────────────────────────────────────────

class TestBuildSparkSessionRamAllocation:

    def _driver_memory(self, builder):
        for c in builder.config.call_args_list:
            if c[0][0] == "spark.driver.memory":
                return c[0][1]
        return None

    def test_driver_memory_override(self, spark_mock):
        """driver_memory_mb=1024 must produce 'spark.driver.memory' == '1024m'."""
        _, builder = spark_mock
        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", driver_memory_mb=1024)
        assert self._driver_memory(builder) == "1024m"

    def test_per_worker_divides_ram(self, spark_mock, monkeypatch):
        """
        num_workers=2, memory_fraction=1.0 → heap = total_ram // 2.
        """
        _, builder = spark_mock
        import mpj_spark.workers.spark_session as ss
        monkeypatch.setattr(ss, "get_total_ram_mb", lambda: 4096)

        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", num_workers=2, memory_fraction=1.0)
        assert self._driver_memory(builder) == f"{max(512, 4096 // 2)}m"

    def test_memory_floor_is_512mb(self, spark_mock, monkeypatch):
        """heap_mb must never drop below 512 MB regardless of workers."""
        _, builder = spark_mock
        import mpj_spark.workers.spark_session as ss
        monkeypatch.setattr(ss, "get_total_ram_mb", lambda: 512)

        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", num_workers=100, memory_fraction=0.75)
        assert self._driver_memory(builder) == "512m"

    def test_no_workers_uses_fraction_of_total(self, spark_mock, monkeypatch):
        """Without num_workers, heap = total_ram * memory_fraction."""
        _, builder = spark_mock
        import mpj_spark.workers.spark_session as ss
        monkeypatch.setattr(ss, "get_total_ram_mb", lambda: 8192)

        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", memory_fraction=0.5)
        assert self._driver_memory(builder) == "4096m"


# ─────────────────────────────────────────────────────────────
# build_spark_session — CPU allocation
# ─────────────────────────────────────────────────────────────

class TestBuildSparkSessionCpuAllocation:

    def test_cores_override_sets_local_master(self, spark_mock):
        """cores_override=4 → master('local[4]')."""
        _, builder = spark_mock
        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", cores_override=4)
        builder.master.assert_called_once_with("local[4]")

    def test_default_cores_from_config(self, spark_mock):
        """Without cores_override, master uses TOTAL_CORES from config."""
        _, builder = spark_mock
        from mpj_spark.workers.spark_session import build_spark_session
        with patch("mpj_spark.workers.spark_session.TOTAL_CORES", 8):
            build_spark_session("TestApp")
        builder.master.assert_called_once_with("local[8]")


# ─────────────────────────────────────────────────────────────
# build_spark_session — config keys emitted
# ─────────────────────────────────────────────────────────────

class TestBuildSparkSessionConfigKeys:

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

    def test_all_expected_config_keys_set(self, spark_mock):
        _, builder = spark_mock
        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", cores_override=2, driver_memory_mb=512)
        called_keys = {c[0][0] for c in builder.config.call_args_list}
        for key in self.EXPECTED_KEYS:
            assert key in called_keys, f"Missing config key: {key}"

    def test_blas_flags_in_java_options(self, spark_mock):
        _, builder = spark_mock
        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", cores_override=2, driver_memory_mb=512)
        jvm_val = next(
            (c[0][1] for c in builder.config.call_args_list
             if c[0][0] == "spark.driver.extraJavaOptions"), None
        )
        assert jvm_val is not None
        assert "NativeSystemBLAS" in jvm_val

    def test_gc_flags_in_java_options(self, spark_mock):
        _, builder = spark_mock
        from mpj_spark.workers.spark_session import build_spark_session
        build_spark_session("TestApp", cores_override=2, driver_memory_mb=512)
        jvm_val = next(
            (c[0][1] for c in builder.config.call_args_list
             if c[0][0] == "spark.driver.extraJavaOptions"), None
        )
        assert jvm_val is not None
        assert "UseG1GC" in jvm_val

    def test_returns_spark_session_object(self, spark_mock):
        session, builder = spark_mock
        from mpj_spark.workers.spark_session import build_spark_session
        result = build_spark_session("TestApp", cores_override=1, driver_memory_mb=512)
        assert result is builder.getOrCreate.return_value
