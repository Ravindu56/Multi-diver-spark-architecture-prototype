#!/usr/bin/env bash
# =============================================================================
# validate_p4_06_kmeans.sh
# P4-06: Validate K-Means Allreduce convergence in Docker cluster (Obj 1c)
# =============================================================================
set -euo pipefail

NP="${NP:-3}"
K="${K:-3}"
KMEANS_ITER="${KMEANS_ITER:-20}"
INPUT_KMEANS="${INPUT_KMEANS:-/data/input/kmeans_data.csv}"
RESULTS_DIR="${RESULTS_DIR:-/data/results/p4_06}"
CONTAINER="mpi-root"
HOSTFILE="/etc/mpi/hostfile"
CENTROID_TOL="${CENTROID_TOL:-1e-4}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

log_pass() { echo -e "${GREEN}[PASS]${NC} $*"; PASS=$((PASS + 1)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $*"; FAIL=$((FAIL + 1)); }
log_info() { echo -e "${YELLOW}[INFO]${NC} $*"; }

echo "========================================================"
echo "P4-06 Validation: K-Means Convergence in Docker Cluster"
echo "K=${K}  max_iter=${KMEANS_ITER}  np=${NP}  tol=${CENTROID_TOL}"
echo "Timestamp: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================"

log_info "T1: Checking cluster containers..."
for cname in mpi-root mpi-worker-1 mpi-worker-2; do
    status=$(docker inspect --format='{{.State.Status}}' "$cname" 2>/dev/null || echo "missing")
    if [ "$status" = "running" ]; then
        log_pass "$cname running"
    else
        log_fail "$cname status=$status"
    fi
done

log_info "T2: Checking or generating K-Means dataset..."
file_lines=$(docker exec "$CONTAINER" \
    sh -c "[ -s '$INPUT_KMEANS' ] && wc -l < '$INPUT_KMEANS' || echo 0")

if [ "$file_lines" -gt 1 ]; then
    log_pass "K-Means input exists: $INPUT_KMEANS (${file_lines} rows)"
else
    docker exec "$CONTAINER" python3 - <<PY
import csv
import os
import random

random.seed(42)
path = "${INPUT_KMEANS}"
os.makedirs(os.path.dirname(path), exist_ok=True)

centers = [(0.0, 0.0), (8.0, 8.0), (-8.0, 8.0)]
with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    for cx, cy in centers:
        for _ in range(1000):
            writer.writerow([
                round(random.gauss(cx, 0.8), 6),
                round(random.gauss(cy, 0.8), 6),
            ])
print(f"Generated 3000 deterministic samples at {path}")
PY
    log_pass "Generated deterministic K-Means dataset"
fi

log_info "T3: Validating MPI hostfile..."
hostfile_lines=$(docker exec "$CONTAINER" \
    sh -c "wc -l < '$HOSTFILE'" 2>/dev/null || echo 0)

if [ "$hostfile_lines" -ge "$NP" ]; then
    log_pass "Hostfile has ${hostfile_lines} entries"
else
    log_fail "Hostfile has ${hostfile_lines} entries; expected at least ${NP}"
fi

log_info "T4: Importing K-Means execution paths..."
if docker exec -i "$CONTAINER" python3 - <<'PY'
from mpj_spark.applications.kmeans.allreduce import run_kmeans_allreduce
from mpj_spark.applications.kmeans.driver import run_kmeans_driver
print("KMeans imports resolved")
PY
then
    log_pass "K-Means Allreduce and driver imports resolved"
else
    log_fail "K-Means import preflight failed"
fi

docker exec "$CONTAINER" sh -c \
    "rm -rf '$RESULTS_DIR' && mkdir -p '$RESULTS_DIR/baseline' '$RESULTS_DIR/multidriver'"

log_info "T5: Running K-Means multi-driver MPI workload..."
start_ts=$(date +%s)
if docker exec "$CONTAINER" bash -c "
    set -o pipefail
    cd /app
    mpirun --hostfile '$HOSTFILE' -np '$NP' \
      --mca btl_tcp_if_include eth0 \
      python3 mpj_spark_mpi.py \
        --app kmeans \
        --input '$INPUT_KMEANS' \
        --kmeans-k '$K' \
        --kmeans-iter '$KMEANS_ITER' \
        --global-seed \
        --results-dir '$RESULTS_DIR/multidriver' \
      2>&1 | tee '$RESULTS_DIR/multidriver_stdout.log'
"
then
    end_ts=$(date +%s)
    log_pass "MPI K-Means completed in $((end_ts - start_ts))s"
else
    log_fail "MPI K-Means command failed"
fi

log_info "T6: Checking multi-driver convergence evidence..."
run_log="${RESULTS_DIR}/multidriver_stdout.log"
if docker exec "$CONTAINER" sh -c \
    "[ -s '$run_log' ] && grep -Eqi 'converg|iteration|centroid|allreduce|global' '$run_log'"
then
    log_pass "Iteration, centroid, or synchronization evidence found"
else
    log_fail "No K-Means iteration/convergence evidence found in workload log"
fi

log_info "T7: Checking multi-driver results artifact..."
multi_json=$(docker exec "$CONTAINER" sh -c \
    "find '$RESULTS_DIR/multidriver' -maxdepth 1 -type f -name '*.json' | head -n 1")

if [ -n "$multi_json" ]; then
    log_pass "Multi-driver JSON result found: $multi_json"
else
    log_fail "No multi-driver JSON result found"
fi

log_info "T8: Running single-driver baseline..."
if docker exec "$CONTAINER" bash -c "
    cd /app
    python3 mpj_spark_mpi.py \
      --app kmeans \
      --input '$INPUT_KMEANS' \
      --kmeans-k '$K' \
      --kmeans-iter '$KMEANS_ITER' \
      --global-seed \
      --workers 1 \
      --results-dir '$RESULTS_DIR/baseline' \
      2>&1 | tee '$RESULTS_DIR/baseline_stdout.log'
"
then
    log_pass "Single-driver K-Means baseline completed"
else
    log_fail "Single-driver baseline command failed"
fi

log_info "T9: Comparing final centroids..."
if docker exec "$CONTAINER" python3 - "$RESULTS_DIR" "$CENTROID_TOL" <<'PY'
import glob
import json
import math
import sys

results_dir, tolerance = sys.argv[1], float(sys.argv[2])

def load_centroids(path_glob):
    files = sorted(glob.glob(path_glob))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        data = json.load(f)
    return data.get("final_centroids", data.get("centroids"))

base = load_centroids(f"{results_dir}/baseline/*.json")
multi = load_centroids(f"{results_dir}/multidriver/*.json")

if base is None or multi is None:
    print(f"CENTROID_DATA_MISSING baseline={base is not None} multidriver={multi is not None}")
    raise SystemExit(2)

def numeric_vector(c):
    if isinstance(c, dict):
        return list(c.values())
    return list(c)

base = sorted(base, key=lambda c: numeric_vector(c)[0])
multi = sorted(multi, key=lambda c: numeric_vector(c)[0])

if len(base) != len(multi):
    print(f"CENTROID_COUNT_MISMATCH baseline={len(base)} multidriver={len(multi)}")
    raise SystemExit(1)

maximum_distance = 0.0
for b, m in zip(base, multi):
    bv, mv = numeric_vector(b), numeric_vector(m)
    if len(bv) != len(mv):
        print("CENTROID_DIMENSION_MISMATCH")
        raise SystemExit(1)
    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(bv, mv)))
    maximum_distance = max(maximum_distance, distance)

print(f"max_centroid_L2_distance={maximum_distance:.8f} tolerance={tolerance}")
raise SystemExit(0 if maximum_distance <= tolerance else 1)
PY
then
    log_pass "Final centroids match within tolerance ${CENTROID_TOL}"
else
    centroid_exit=$?
    if [ "$centroid_exit" -eq 2 ]; then
        log_fail "Centroid data missing; inspect result artifacts"
    else
        log_fail "Centroid mismatch exceeds tolerance ${CENTROID_TOL}"
    fi
fi

echo
echo "========================================================"
echo "P4-06 Results: PASS=${PASS} FAIL=${FAIL}"
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}ALL TESTS PASSED — P4-06 ACCEPTANCE CRITERIA MET${NC}"
    exit 0
fi

echo -e "${RED}${FAIL} TEST(S) FAILED — review output above${NC}"
exit 1
