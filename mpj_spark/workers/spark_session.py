# ============================================================
# workers/spark_session.py
# SparkSession factory — single place to tune all Spark configs
# ============================================================
# Thread-budget fairness
# ----------------------
# Each MPJ worker is given  cores = TOTAL_CORES // num_workers  threads
# via  local[N].  The baseline is constrained to the same N so that
# every entity in the comparison has an identical CPU budget.
# This mirrors HPC behaviour: each cluster node has a fixed core count.
# ============================================================
from mpj_spark.config import (
    SPARK_UI_ENABLED, SPARK_DRIVER_HOST,
    SPARK_SHUFFLE_PARTITIONS, SPARK_DEFAULT_PARALLELISM,
    JAVA_SECURITY_OPT, TOTAL_CORES,
)


def build_spark_session(app_name: str,
                        num_workers: int = 1,
                        cores_override: int = None):
    """
    Build and return an isolated SparkSession for an MPJ worker.

    Thread-budget formula
    ---------------------
    cores_per_entity = max(1, TOTAL_CORES // num_workers)

    This gives each worker (and the baseline when num_workers=1) an
    equal share of the machine's logical cores so the comparison is
    fair on a single node.

    Parameters
    ----------
    app_name      : str — unique Spark application name
    num_workers   : int — total number of MPJ workers in this run;
                          used to compute cores_per_entity.
                          Pass 1 for the baseline (gets all cores / 1).
    cores_override: int — bypass the formula and use this exact count
                          (useful for manual experiments).
    """
    from pyspark.sql import SparkSession

    if cores_override is not None:
        cores = max(1, cores_override)
    else:
        cores = max(1, TOTAL_CORES // num_workers)

    master = f'local[{cores}]'

    spark = (
        SparkSession.builder
        .master(master)
        .appName(app_name)
        .config('spark.ui.enabled',                   SPARK_UI_ENABLED)
        .config('spark.driver.host',                  SPARK_DRIVER_HOST)
        .config('spark.sql.shuffle.partitions',       SPARK_SHUFFLE_PARTITIONS)
        .config('spark.default.parallelism',          str(cores))
        .config('spark.driver.extraJavaOptions',      JAVA_SECURITY_OPT)
        .config('spark.executor.extraJavaOptions',    JAVA_SECURITY_OPT)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel('ERROR')
    return spark
