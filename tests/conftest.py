# tests/conftest.py
# Shared pytest fixtures for the Phase 3 logreg test suite.
#
# spark  — module-scoped local[1] SparkSession.
#           Scope is "module" so the JVM starts once per test file,
#           not once per test function.  This keeps the suite fast
#           (~3 s total) while still allowing real RDD operations.
#           Tests that do NOT need Spark should NOT request this fixture
#           so the JVM is never started on machines without Java/PySpark.

import pytest


@pytest.fixture(scope="module")
def spark():
    """Return a local[1] SparkSession; stop it after the test module."""
    pytest.importorskip("pyspark", reason="pyspark not installed")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .appName("logreg-unit-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "512m")
        .getOrCreate()
    )
    yield session
    session.stop()
