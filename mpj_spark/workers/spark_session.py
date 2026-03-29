# ================================================================
# mpj_spark/workers/spark_session.py
#
# SparkSession factory with proportional memory isolation.
#
# Memory allocation strategy:
#   JVM heap per worker = TOTAL_RAM_MB * MEMORY_FRACTION / num_workers
#   (mirrors HPC node resource partitioning from the reference paper)
# ================================================================
import os
import math


def get_total_ram_mb() -> int:
    """Return total system RAM in MB."""
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024 * 1024))
    except ImportError:
        # Fallback: read from /proc/meminfo on Linux
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        return int(line.split()[1]) // 1024
        except Exception:
            pass
    return 8192  # safe default: 8 GB


def build_spark_session(
    app_name:         str,
    cores_override:   int  = None,
    num_workers:      int  = None,
    memory_fraction:  float = 0.75,
    driver_memory_mb: int  = None,
):
    """
    Build a SparkSession with proportional CPU and RAM allocation.

    Memory Isolation Strategy
    -------------------------
    Total usable RAM = TOTAL_RAM * memory_fraction
    Per-worker heap  = Total usable RAM / num_workers

    This ensures N workers collectively use at most memory_fraction
    of total system RAM, mirroring HPC node resource partitioning.

    Parameters
    ----------
    app_name         : str   — Spark application name
    cores_override   : int   — number of CPU threads (local[N])
    num_workers      : int   — total worker count (used for RAM division)
    memory_fraction  : float — fraction of total RAM to use (default 0.75)
    driver_memory_mb : int   — explicit heap override in MB (optional)
    """
    from pyspark.sql import SparkSession
    from mpj_spark.config import TOTAL_CORES

    # ── CPU allocation ────────────────────────────────────────────────
    cores = cores_override if cores_override else TOTAL_CORES

    # ── RAM allocation ────────────────────────────────────────────────
    if driver_memory_mb is not None:
        heap_mb = driver_memory_mb
    elif num_workers is not None and num_workers > 0:
        total_ram_mb   = get_total_ram_mb()
        usable_ram_mb  = int(total_ram_mb * memory_fraction)
        heap_mb        = max(512, usable_ram_mb // num_workers)
    else:
        # Single caller (baseline) — use memory_fraction of total RAM
        total_ram_mb   = get_total_ram_mb()
        heap_mb        = int(total_ram_mb * memory_fraction)

    heap_str = f'{heap_mb}m'

    print(f'[SparkSession] {app_name}: local[{cores}]  '
          f'heap={heap_mb} MB  '
          f'(system RAM: {get_total_ram_mb()} MB)')

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(f'local[{cores}]')
        .config('spark.driver.memory',            heap_str)
        .config('spark.executor.memory',          heap_str)
        .config('spark.driver.maxResultSize',     f'{max(256, heap_mb // 4)}m')
        .config('spark.memory.fraction',          '0.8')
        .config('spark.memory.storageFraction',   '0.3')
        .config('spark.sql.shuffle.partitions',   str(cores * 2))
        .config('spark.default.parallelism',      str(cores * 2))
        .config('spark.serializer',               'org.apache.spark.serializer.KryoSerializer')
        .config('spark.kryoserializer.buffer.max','512m')
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel('WARN')
    return spark