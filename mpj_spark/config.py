# ============================================================
# config.py — Central configuration for MPJ-SPARK prototype
# ============================================================
import os
import multiprocessing

# --- Shared Storage ---
SHARED_STORAGE_PATH = os.environ.get("MPJ_SHARED_STORAGE", "./shared_storage")
PARTITIONS_DIR = os.path.join(SHARED_STORAGE_PATH, "partitions")

# --- Machine CPU budget ---
# Total logical cores on this machine. Each worker AND the baseline
# will be constrained to  cores_per_entity = TOTAL_CORES // num_workers
# so every entity gets the same thread budget — fair single-machine
# comparison that mirrors an HPC cluster where each node has its own
# dedicated core allocation.
TOTAL_CORES = multiprocessing.cpu_count()

# --- Spark Defaults ---
# SPARK_MASTER is intentionally left as local[*] here; the actual
# per-run master string (local[N]) is computed dynamically in
# build_spark_session() and run_baseline() using TOTAL_CORES.
SPARK_MASTER = "local[*]"  # fallback only
SPARK_UI_ENABLED = "false"
SPARK_DRIVER_HOST = "localhost"
SPARK_SHUFFLE_PARTITIONS = "4"
SPARK_DEFAULT_PARALLELISM = "4"
JAVA_SECURITY_OPT = "-Djava.security.manager=allow"

# --- Dataset Generator ---
DEFAULT_DATASET_PATH = "./test_dataset.txt"
DEFAULT_DATASET_SIZE_MB = 50

os.environ["HADOOP_HOME"] = "/dev/null"

# --- Benchmark ---
DEFAULT_NUM_WORKERS = 4

# ── Data directory ─────────────────────────────────────────────────────
# All generated datasets and partition files are stored here.
# Path is relative to project root (where main.py lives).
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)

# ── Spark defaults ─────────────────────────────────────────────────────
SPARK_LOG_LEVEL = "ERROR"  # suppress verbose Spark INFO/WARN output

# ── Dataset paths (shared across all MPI ranks via NFS/shared_storage) ──
#
# These constants are the single source of truth for where dataset files
# live.  Both the CLI runners (allreduce.py __main__) and the standalone
# scripts/generate_datasets.py use these values so paths never diverge.
#
# Override at runtime via environment variables:
#   MPJ_KMEANS_DATA  → KMEANS_DATASET_PATH
#   MPJ_LOGREG_DATA  → LOGREG_DATASET_PATH
# ─────────────────────────────────────────────────────────────────────
KMEANS_DATASET_PATH = os.environ.get(
    "MPJ_KMEANS_DATA",
    os.path.join(SHARED_STORAGE_PATH, "kmeans_data.csv"),
)
LOGREG_DATASET_PATH = os.environ.get(
    "MPJ_LOGREG_DATA",
    os.path.join(SHARED_STORAGE_PATH, "logreg_data.csv"),
)

# Default generation sizes — 50 MB gives ~3 meaningful shards at 3 MPI ranks.
# Increase for Phase 5 (multi-node) benchmarks.
KMEANS_DATASET_SIZE_MB = int(os.environ.get("MPJ_KMEANS_SIZE_MB", "50"))
LOGREG_DATASET_SIZE_MB = int(os.environ.get("MPJ_LOGREG_SIZE_MB", "50"))

# Number of feature columns for the classification dataset (logreg).
# Must match the value used when the file was generated.
LOGREG_NUM_FEATURES = int(os.environ.get("MPJ_LOGREG_FEATURES", "10"))

# Number of clusters K for the numeric dataset (kmeans).
KMEANS_NUM_FEATURES = int(os.environ.get("MPJ_KMEANS_FEATURES", "10"))
