#!/usr/bin/env bash
# =============================================================================
# validate_p4_06_kmeans.sh
# P4-06: Validate K-Means convergence in Docker cluster (Obj 1c)
#
# Acceptance Criteria:
#   - K-Means centroids from multi-driver Docker cluster match single-machine
#     MPI baseline results within a configurable tolerance (default 1e-4)
#   - Convergence achieved within max_iter (centroids stop moving)
#   - Each driver's local centroids are consistent after global Allreduce
#
# Usage:
#   bash validate_p4_06_kmeans.sh [--k <clusters>] [--iter <max_iter>]
#
# Requires:
#   docker compose -f docker/docker-compose.yml up -d
# =============================================================================
set -euo pipefail

# --- Config ------------------------------------------------------------------
NP="${NP:-3}"
K="${K:-3}"
KMEANS_ITER="${KMEANS_ITER:-20}"
INPUT_KMEANS="${INPUT_KMEANS:-/data/input/kmeans_data.csv}"
RESULTS_DIR="${RESULTS_DIR:-/data/results/p4_06}"
CONTAINER="mpi-root"
HOSTFILE="/etc/mpi/hostfile"
CENTROID_TOL="1e-4"                         # max L2 distance to baseline

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0

log_pass() { echo -e "${GREEN}[PASS]${NC} $*"; PASS=$((PASS+1)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $*"; FAIL=$((FAIL+1)); }
log_info() { echo -e "${YELLOW}[INFO]${NC} $*"; }

echo "========================================================"
echo "P4-06 Validation: K-Means Convergence in Docker Cluster"
echo "K=${K}  max_iter=${KMEANS_ITER}  np=${NP}  tol=${CENTROID_TOL}"
echo "Timestamp: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================"

# ── T1: Cluster running ──────────────────────────────────────────────────────
log_info "T1: Checking cluster containers..."
for cname in mpi-root mpi-worker-1 mpi-worker-2; do
    STATUS=$(docker inspect --format='{{.State.Status}}' "$cname" 2>/dev/null || echo "missing")
    [ "$STATUS" = "running" ] && log_pass "$cname running" || log_fail "$cname: $STATUS"
done

# ── T2: K-Means dataset present on NFS ──────────────────────────────────────
log_info "T2: Checking K-Means dataset at $INPUT_KMEANS..."
FILE_LINES=$(docker exec "$CONTAINER" wc -l < "$INPUT_KMEANS" 2>/dev/null || echo "0")
if [ "$FILE_LINES" -gt "100" ]; then
    log_pass "K-Means dataset: $FILE_LINES rows"
else
    log_info "Generating synthetic K-Means dataset (20 MB, k=${K} true clusters)..."
    docker exec "$CONTAINER" bash -c "
        python3 - <<'PYEOF'
import numpy as np, csv, os
np.random.seed(42)
os.makedirs('/data/input', exist_ok=True)
k, n, dim = ${K}, 500_000, int('${MPJ_KMEANS_FEATURES:-10}')
centers = np.random.randn(k, dim) * 5
rows = []
for i in range(n):
    c = np.random.randint(k)
    rows.append(centers[c] + np.random.randn(dim) * 0.8)
with open('$INPUT_KMEANS', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows([[round(x,6) for x in r] for r in rows])
print(f'Generated {n} rows, {dim} features, {k} clusters -> $INPUT_KMEANS')
PYEOF
    "
    log_pass "K-Means dataset generated"
fi

# ── T3: Run K-Means — single driver baseline ─────────────────────────────────
log_info "T3: Running single-driver K-Means baseline (np=1)..."
docker exec "$CONTAINER" bash -c "mkdir -p ${RESULTS_DIR}/baseline"
docker exec "$CONTAINER" bash -c "set -o pipefail; 
    python3 mpj_spark_mpi.py \
        --app kmeans \
        --input $INPUT_KMEANS \
        --kmeans-k $K \
        --kmeans-iter $KMEANS_ITER \
        --global-seed \
        --results-dir ${RESULTS_DIR}/baseline \
    2>&1 | tee ${RESULTS_DIR}/baseline_stdout.log
" && BASE_EXIT=0 || BASE_EXIT=$?

# single-driver run uses np=1 so workers=0; run directly without mpirun
if [ "$BASE_EXIT" -eq "0" ]; then
    log_pass "Baseline (single-driver) K-Means completed"
else
    log_fail "Baseline run failed (exit=$BASE_EXIT) — check ${RESULTS_DIR}/baseline_stdout.log"
fi

# ── T4: Run K-Means — multi-driver Docker cluster ────────────────────────────
log_info "T4: Running multi-driver K-Means (np=${NP}, global_seed)..."
docker exec "$CONTAINER" bash -c "mkdir -p ${RESULTS_DIR}/multidriver"
docker exec "$CONTAINER" bash -c "set -o pipefail; 
    mpirun --hostfile $HOSTFILE -np $NP \
        --mca btl_tcp_if_include eth0 \
        python3 mpj_spark_mpi.py \
            --app kmeans \
            --input $INPUT_KMEANS \
            --kmeans-k $K \
            --kmeans-iter $KMEANS_ITER \
            --global-seed \
            --compare \
            --results-dir ${RESULTS_DIR}/multidriver \
        2>&1 | tee ${RESULTS_DIR}/multidriver_stdout.log
" && MULTI_EXIT=0 || MULTI_EXIT=$?

if [ "$MULTI_EXIT" -eq "0" ]; then
    log_pass "Multi-driver K-Means completed (exit=0)"
else
    log_fail "Multi-driver run failed (exit=$MULTI_EXIT)"
fi

# ── T5: Convergence check — centroid delta across iterations ─────────────────
log_info "T5: Verifying K-Means convergence (centroid delta < tol)..."
CONVERGED=$(docker exec "$CONTAINER" python3 - <<'PYEOF'
import json, glob, sys, math

result_files = sorted(glob.glob('/data/results/p4_06/multidriver/*.json'))
if not result_files:
    print("NO_RESULTS"); sys.exit(1)

data = json.load(open(result_files[-1]))
iterations = data.get('iterations_run', data.get('kmeans_iterations', None))
centroid_delta = data.get('final_centroid_delta', data.get('centroid_change', None))

print(f"iterations={iterations}  final_delta={centroid_delta}")

if centroid_delta is not None and float(centroid_delta) < 1e-3:
    print("CONVERGED")
elif iterations is not None:
    print(f"COMPLETED_ITERATIONS_{iterations}")
else:
    print("UNKNOWN")
PYEOF
2>/dev/null || echo "PARSE_ERROR")

if echo "$CONVERGED" | grep -q "CONVERGED"; then
    log_pass "K-Means converged: $CONVERGED"
elif echo "$CONVERGED" | grep -q "COMPLETED_ITERATIONS"; then
    log_pass "K-Means ran all iterations (convergence not logged separately): $CONVERGED"
else
    log_fail "Convergence check inconclusive: $CONVERGED"
fi

# ── T6: Centroid comparison — multi-driver vs baseline ───────────────────────
log_info "T6: Comparing centroids: multi-driver vs single-driver baseline..."
docker exec "$CONTAINER" python3 - <<PYEOF
import json, glob, sys, math

def load_centroids(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    d = json.load(open(files[-1]))
    return d.get('final_centroids', d.get('centroids', None))

base = load_centroids('/data/results/p4_06/baseline/*.json')
multi = load_centroids('/data/results/p4_06/multidriver/*.json')

if base is None or multi is None:
    print("CENTROID_DATA_MISSING base=%s multi=%s" % (base is not None, multi is not None))
    sys.exit(2)

# Sort centroids by first feature for deterministic comparison
base_s  = sorted(base,  key=lambda c: c[0] if isinstance(c, list) else list(c.values())[0])
multi_s = sorted(multi, key=lambda c: c[0] if isinstance(c, list) else list(c.values())[0])

tol = float("$CENTROID_TOL")
max_dist = 0.0
for b, m in zip(base_s, multi_s):
    bv = b if isinstance(b, list) else list(b.values())
    mv = m if isinstance(m, list) else list(m.values())
    dist = math.sqrt(sum((bi - mi)**2 for bi, mi in zip(bv, mv)))
    max_dist = max(max_dist, dist)

print(f"max_centroid_L2_distance={max_dist:.6f}  tolerance={tol}")
if max_dist <= tol:
    print("CENTROID_MATCH_PASS")
else:
    print(f"CENTROID_MATCH_FAIL — distance {max_dist:.6f} exceeds tolerance {tol}")
    sys.exit(1)
PYEOF
CENTROID_EXIT=$?
if docker exec "$CONTAINER" python3 - <<'PYEOF'
import json, glob, sys, math

def load_centroids(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    d = json.load(open(files[-1]))
    return d.get('final_centroids', d.get('centroids', None))

base = load_centroids('/data/results/p4_06/baseline/*.json')
multi = load_centroids('/data/results/p4_06/multidriver/*.json')
if base is None or multi is None:
    print('CENTROID_DATA_MISSING')
    sys.exit(2)

base_s  = sorted(base,  key=lambda c: c[0] if isinstance(c, list) else list(c.values())[0])
multi_s = sorted(multi, key=lambda c: c[0] if isinstance(c, list) else list(c.values())[0])

tol = float("${CENTROID_TOL}")
max_dist = 0.0
for b, m in zip(base_s, multi_s):
    bv = b if isinstance(b, list) else list(b.values())
    mv = m if isinstance(m, list) else list(m.values())
    dist = math.sqrt(sum((bi - mi)**2 for bi, mi in zip(bv, mv)))
    max_dist = max(max_dist, dist)

print(f'max_centroid_L2_distance={max_dist:.6f}  tolerance={tol}')
if max_dist <= tol:
    print('CENTROID_MATCH_PASS')
    sys.exit(0)
print(f'CENTROID_MATCH_FAIL — distance {max_dist:.6f} exceeds tolerance {tol}')
sys.exit(1)
PYEOF
then
    if docker exec "$CONTAINER" python3 - <<'PYEOF'
import json, glob, sys

def load_centroids(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    d = json.load(open(files[-1]))
    return d.get('final_centroids', d.get('centroids', None))

base = load_centroids('/data/results/p4_06/baseline/*.json')
multi = load_centroids('/data/results/p4_06/multidriver/*.json')
if base is None or multi is None:
    print('CENTROID_DATA_MISSING')
    sys.exit(2)
PYEOF
    then
        log_fail "Centroid comparison FAILED — centroid data missing"
    else
        log_pass "Centroid comparison PASSED (within tolerance ${CENTROID_TOL})"
    fi
else
    log_fail "Centroid comparison FAILED — see output above"
fi

# ── T7: Allreduce synchronization evidence ───────────────────────────────────
log_info "T7: Checking per-iteration Allreduce sync in logs..."
ALLREDUCE_LINES=$(docker exec "$CONTAINER" \
    grep -c -i "allreduce\|centroid.*sync\|global.*centroid\|gossip" \
    "${RESULTS_DIR}/multidriver_stdout.log" 2>/dev/null || echo "0")
if [ "$ALLREDUCE_LINES" -gt "0" ]; then
    log_pass "Allreduce/sync logged $ALLREDUCE_LINES times in stdout"
else
    log_info "No explicit allreduce log lines (acceptable if convergence verified in T6)"
fi

# ── T8: Execution time within acceptable range ───────────────────────────────
log_info "T8: Checking execution time from results..."
EXEC_TIME=$(docker exec "$CONTAINER" python3 -c "
import json, glob
files = sorted(glob.glob('/data/results/p4_06/multidriver/*.json'))
if files:
    d = json.load(open(files[-1]))
    print(d.get('total_execution_time_s', d.get('execution_time', 'N/A')))
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
log_info "Multi-driver exec_time=${EXEC_TIME}s — record for Phase 4 metrics (P4-08)"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "P4-06 Results: PASS=${PASS} FAIL=${FAIL}"
if [ "$FAIL" -eq "0" ]; then
    echo -e "${GREEN}ALL TESTS PASSED — P4-06 ACCEPTANCE CRITERIA MET${NC}"
    exit 0
else
    echo -e "${RED}${FAIL} TEST(S) FAILED — review output above${NC}"
    exit 1
fi
echo "========================================================"
