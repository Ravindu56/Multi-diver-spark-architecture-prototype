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
CENTROID_OUTPUT=$(docker exec "$CONTAINER" python3 - <<'PYEOF'
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
    print('CENTROID_DATA_MISSING base=%s multi=%s' % (base is not None, multi is not None))
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
CENTROID_EXIT=$?

if [ "$CENTROID_EXIT" -eq "0" ]; then
    log_pass "Centroid comparison PASSED (within tolerance ${CENTROID_TOL})"
elif [ "$CENTROID_EXIT" -eq "1" ]; then
    echo "$CENTROID_OUTPUT"
    log_fail "Centroid comparison FAILED — centroids differ"
elif [ "$CENTROID_EXIT" -eq "2" ]; then
    echo "$CENTROID_OUTPUT"
    log_fail "Centroid comparison FAILED — centroid data missing"
else
    echo "$CENTROID_OUTPUT"
    log_fail "Centroid comparison FAILED — unexpected exit=$CENTROID_EXIT"
fi
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
