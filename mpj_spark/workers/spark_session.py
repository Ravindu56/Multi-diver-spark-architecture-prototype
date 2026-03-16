# ============================================================
# workers/spark_session.py
# SparkSession factory — single place to tune all Spark configs
# ============================================================
from mpj_spark.config import (
    SPARK_MASTER, SPARK_UI_ENABLED, SPARK_DRIVER_HOST,
    SPARK_SHUFFLE_PARTITIONS, SPARK_DEFAULT_PARALLELISM,
    JAVA_SECURITY_OPT,
)


def build_spark_session(app_name: str, cores_per_worker: int = None):
    """
    Build and return an isolated SparkSession for an MPJ worker.

    Parameters
    ----------
    app_name        : str  — unique application name (used as Spark app name)
    cores_per_worker: int  — override parallelism; defaults to config value
    """
    from pyspark.sql import SparkSession

    parallelism = str(cores_per_worker) if cores_per_worker else SPARK_DEFAULT_PARALLELISM

    spark = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName(app_name)
        .config('spark.ui.enabled',                   SPARK_UI_ENABLED)
        .config('spark.driver.host',                  SPARK_DRIVER_HOST)
        .config('spark.sql.shuffle.partitions',       SPARK_SHUFFLE_PARTITIONS)
        .config('spark.default.parallelism',          parallelism)
        .config('spark.driver.extraJavaOptions',      JAVA_SECURITY_OPT)
        .config('spark.executor.extraJavaOptions',    JAVA_SECURITY_OPT)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel('ERROR')
    return spark
