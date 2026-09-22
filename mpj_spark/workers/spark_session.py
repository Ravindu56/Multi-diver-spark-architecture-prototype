# ================================================================
# mpj_spark/workers/spark_session.py
#
# SparkSession factory with proportional memory isolation.
#
# Memory allocation strategy (precedence, highest first):
#   1. SPARK_DRIVER_MEMORY env var (e.g. "400m") — set by Swarm
#      overlays to cap co-located ranks (P5-04 cells B/C).
#   2. driver_memory_mb argument — explicit caller override.
#   3. Proportional: TOTAL_RAM_MB * MEMORY_FRACTION / num_workers
#      (mirrors HPC node resource partitioning from the reference paper).
#
# CPU allocation strategy (precedence, highest first):
#   1. SLOTS_OVERRIDE env var — set by Swarm overlays to give each
#      co-located rank a deterministic, physical-core-aware share
#      instead of the full VM vCPU count (P5-04 cells B/C).
#   2. cores_override argument — explicit caller override.
#   3. TOTAL_CORES (multiprocessing.cpu_count(), i.e. nproc) — Phase 4
#      single-host default, unchanged when SLOTS_OVERRIDE is unset.
#
# FIXES APPLIED:
#   FIX 3a — Add JVM extraJavaOptions to load native BLAS (OpenBLAS/MKL).
#             Eliminates: "Failed to load implementation from VectorBLAS"
#             Eliminates: all K-Means centroid update math running in
#             slow pure-JVM mode instead of native SIMD instructions.
#
#   FIX 3b — Configure G1GC explicitly so Spark can report GC metrics
#             and avoid the GarbageCollectionMetrics WARN in logs.
#             G1GC is better suited than default GC for large heap workloads.
#
#   FIX 5  — Honor SPARK_DRIVER_MEMORY env var. P5-04 (2026-09-16)
#             showed the Swarm overlay value was silently ignored, so
#             co-located ranks sized heap from the proportional formula
#             and could oversubscribe the VM at Cell C density.
#
#   FIX 6  — Honor SLOTS_OVERRIDE env var. Issue #82 point 4: the
#             Phase-4 entrypoint/config path derives per-rank core
#             budget from TOTAL_CORES (nproc), which is identical for
#             every co-located container on a VM — 4 ranks sharing one
#             2-vCPU VM would each request local[2], not local[~0.5].
#
# Pre-requisite (run once on your machine):
#   sudo apt-get install -y libopenblas-dev
# ================================================================

import os

# Hoist TOTAL_CORES to module level so tests can patch
# 'mpj_spark.workers.spark_session.TOTAL_CORES' directly.
from mpj_spark.config import TOTAL_CORES


def get_total_ram_mb() -> int:
    """Return total system RAM in MB."""
    try:
        import psutil

        return int(psutil.virtual_memory().total / (1024 * 1024))
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
        except Exception:
            pass
    return 8192


def _parse_memory_mb(value: str) -> int:
    """Parse a Spark-style memory string ("400m", "1g", "512") to MB.

    Bare numbers are interpreted as megabytes. Raises ValueError on
    anything unparseable so a misconfigured overlay fails loudly at
    session build instead of silently falling through to the formula.
    """
    v = value.strip().lower()
    try:
        if v.endswith("g"):
            return int(float(v[:-1]) * 1024)
        if v.endswith("m"):
            return int(float(v[:-1]))
        if v.endswith("k"):
            return max(1, int(float(v[:-1]) / 1024))
        return int(float(v))
    except ValueError:
        raise ValueError(
            f"Unparseable SPARK_DRIVER_MEMORY value: {value!r} "
            "(expected forms: '400m', '1g', or bare MB integer)"
        )


def _resolve_cores(cores_override: int = None) -> tuple:
    """Resolve the per-rank Spark core budget and its provenance.

    Precedence: SLOTS_OVERRIDE env > cores_override arg > TOTAL_CORES
    (nproc-derived) formula. See FIX 6 above for why this matters at
    Cell B/C co-location density.
    """
    env_slots = os.environ.get("SLOTS_OVERRIDE", "").strip()
    if env_slots:
        try:
            return max(1, int(env_slots)), f"env:{env_slots}"
        except ValueError:
            raise ValueError(
                f"Unparseable SLOTS_OVERRIDE value: {env_slots!r} "
                "(expected an integer core count)"
            )
    if cores_override:
        return max(1, cores_override), f"arg:{cores_override}"
    return TOTAL_CORES, "nproc(TOTAL_CORES)"


def build_spark_session(
    app_name: str,
    cores_override: int = None,
    num_workers: int = None,
    memory_fraction: float = 0.75,
    driver_memory_mb: int = None,
):
    """
    Build a SparkSession with proportional CPU and RAM allocation,
    native BLAS, and G1GC configured for low-latency GC pauses.
    """
    from pyspark.sql import SparkSession

    # ── CPU allocation ─────────────────────────────────────────────────────
    cores, cores_source = _resolve_cores(cores_override)

    # ── RAM allocation ─────────────────────────────────────────────────────
    env_mem = os.environ.get("SPARK_DRIVER_MEMORY", "").strip()
    if env_mem:
        heap_mb = _parse_memory_mb(env_mem)
        heap_source = f"env:{env_mem}"
    elif driver_memory_mb is not None:
        heap_mb = driver_memory_mb
        heap_source = f"arg:{driver_memory_mb}m"
    elif num_workers is not None and num_workers > 0:
        total_ram_mb = get_total_ram_mb()
        usable_ram_mb = int(total_ram_mb * memory_fraction)
        heap_mb = max(512, usable_ram_mb // num_workers)
        heap_source = "proportional"
    else:
        total_ram_mb = get_total_ram_mb()
        heap_mb = int(total_ram_mb * memory_fraction)
        heap_source = "proportional"

    heap_str = f"{heap_mb}m"

    # ── FIX 3a: Native BLAS JVM flags ──────────────────────────
    # Forces netlib-java to use the system's native OpenBLAS/MKL library.
    # Without this, all matrix operations in K-Means fall back to
    # pure-JVM F2J BLAS which is ~3-5x slower.
    blas_flags = (
        "-Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.NativeSystemBLAS "
        "-Dcom.github.fommil.netlib.LAPACK=com.github.fommil.netlib.NativeSystemLAPACK "
        "-Dcom.github.fommil.netlib.ARPACK=com.github.fommil.netlib.NativeSystemARPACK"
    )

    # ── FIX 3b: G1GC flags ──────────────────────────────────────
    # Configures G1GC explicitly so Spark's GarbageCollectionMetrics
    # can report young/old gen GC events. Removes the WARN:
    # "To enable non-built-in garbage collector(s) List(G1 Concurrent GC)..."
    gc_flags = (
        "-XX:+UseG1GC "
        "-XX:G1HeapRegionSize=16m "
        "-XX:InitiatingHeapOccupancyPercent=35 "
        "-XX:+G1UseAdaptiveIHOP"
    )

    jvm_options = f"{blas_flags} {gc_flags}"

    print(
        f"[SparkSession] {app_name}: local[{cores}] ({cores_source})  "
        f"heap={heap_mb} MB ({heap_source})  "
        f"(system RAM: {get_total_ram_mb()} MB)"
    )

    spark = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", heap_str)
        .config("spark.executor.memory", heap_str)
        .config("spark.driver.maxResultSize", f"{max(256, heap_mb // 4)}m")
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.3")
        .config("spark.sql.shuffle.partitions", str(cores * 2))
        .config("spark.default.parallelism", str(cores * 2))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        # FIX 3a + 3b: native BLAS + G1GC
        .config("spark.driver.extraJavaOptions", jvm_options)
        .config("spark.executor.extraJavaOptions", jvm_options)
        # FIX 3b: register G1GC with Spark's GC metrics collector
        .config(
            "spark.eventLog.gcMetrics.youngGenerationGarbageCollectors",
            "G1 Young Generation",
        )
        .config("spark.eventLog.gcMetrics.oldGenerationGarbageCollectors", "G1 Old Gen")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark
