# ============================================================
# config.py — Central configuration for MPJ-SPARK prototype
# ============================================================
import os
import multiprocessing

# --- Shared Storage ---
SHARED_STORAGE_PATH = os.environ.get('MPJ_SHARED_STORAGE', './shared_storage')
PARTITIONS_DIR      = os.path.join(SHARED_STORAGE_PATH, 'partitions')

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
SPARK_MASTER              = 'local[*]'   # fallback only
SPARK_UI_ENABLED          = 'false'
SPARK_DRIVER_HOST         = 'localhost'
SPARK_SHUFFLE_PARTITIONS  = '4'
SPARK_DEFAULT_PARALLELISM = '4'
JAVA_SECURITY_OPT         = '-Djava.security.manager=allow'

# --- Dataset Generator ---
DEFAULT_DATASET_PATH    = './test_dataset.txt'
DEFAULT_DATASET_SIZE_MB = 50

os.environ['HADOOP_HOME'] = '/dev/null'

# --- Benchmark ---
DEFAULT_NUM_WORKERS = 4
