# ============================================================
# config.py — Central configuration for MPJ-SPARK prototype
# ============================================================
import os

# --- Shared Storage ---
SHARED_STORAGE_PATH = os.environ.get('MPJ_SHARED_STORAGE', './shared_storage')
PARTITIONS_DIR      = os.path.join(SHARED_STORAGE_PATH, 'partitions')

# --- Spark Defaults ---
SPARK_MASTER            = 'local[*]'
SPARK_UI_ENABLED        = 'false'
SPARK_DRIVER_HOST       = 'localhost'
SPARK_SHUFFLE_PARTITIONS = '4'
SPARK_DEFAULT_PARALLELISM = '4'
JAVA_SECURITY_OPT       = '-Djava.security.manager=allow'

# --- Dataset Generator ---
DEFAULT_DATASET_PATH    = './test_dataset.txt'
DEFAULT_DATASET_SIZE_MB = 50

os.environ['HADOOP_HOME'] = '/dev/null'

# --- Benchmark ---
DEFAULT_NUM_WORKERS = 4
