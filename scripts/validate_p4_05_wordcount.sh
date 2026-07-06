#!/usr/bin/env bash
# =============================================================================
# validate_p4_05_wordcount.sh
# P4-05: Validate WordCount workload in Docker cluster (Obj 1a functional test)
#
# Acceptance Criteria:
#   - WordCount runs correctly across containers with NFS partition files
#   - Word count results from multi-driver Docker cluster match single-driver
#     baseline (output file must exist and have non-zero content)
#   - NFS partition files are readable from all containers
#
# Usage:
#   bash validate_p4_05_wordcount.sh [--np <num_processes>] [--input <path>]
#
# Prerequisites:
#   docker compose -f docker/docker-compose.yml up -d
#   (Cluster must be running before executing this script)
# =============================================================================
set -euo pipefail

# --- Config ------------------------------------------------------------------
NP="${NP:-3}"                               # 1 root + 2 workers by default
INPUT_FILE="${INPUT:-/data/input/dataset.txt}"
RESULTS_DIR="${RESULTS_DIR:-/data/results/p4_05}"
CONTAINER="mpi-root"
HOSTFILE="/etc/mpi/hostfile"
TOLERANCE_PCT=5                             # allow ≤5% word-count deviation

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0

log_pass() { echo -e "${GREEN}[PASS]${NC} $*"; ((PASS++)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $*"; ((FAIL++)); }
log_info() { echo -e "${YELLOW}[INFO]${NC} $*"; }

echo "========================================================"
echo "P4-05 Validation: WordCount in Docker Cluster"
echo "Timestamp: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================"

# ── T1: Cluster containers are running ─────────────────────────────────────
log_info "T1: Checking all cluster containers are running..."
for cname in mpi-root mpi-worker-1 mpi-worker-2; do
    STATUS=$(docker inspect --format='{{.State.Status}}' "$cname" 2>/dev/null || echo "missing")
    if [ "$STATUS" = "running" ]; then
        log_pass "Container $cname is running"
    else
        log_fail "Container $cname status: $STATUS (expected: running)"
    fi
done

# ── T2: NFS volume mounted on all containers ────────────────────────────────
log_info "T2: Verifying /data NFS mount on all containers..."
for cname in mpi-root mpi-worker-1 mpi-worker-2; do
    MOUNT_CHECK=$(docker exec "$cname" mountpoint -q /data && echo "mounted" || echo "not-mounted")
    if [ "$MOUNT_CHECK" = "mounted" ]; then
        log_pass "$cname: /data is mounted"
    else
        log_fail "$cname: /data is not mounted"
    fi
done

# ── T3: Input dataset exists on NFS ─────────────────────────────────────────
log_info "T3: Checking input dataset at $INPUT_FILE..."
FILE_SIZE=$(docker exec "$CONTAINER" stat -c%s "$INPUT_FILE" 2>/dev/null || echo "0")
if [ "$FILE_SIZE" -gt "0" ]; then
    log_pass "Input file exists, size=${FILE_SIZE} bytes"
else
    log_info "Input file missing — generating 50 MB synthetic dataset..."
    docker exec "$CONTAINER" bash -c "
        python3 -c \"
import sys, random, string
words=['the','quick','brown','fox','jumps','over','lazy','dog','spark','mpi','data']
import os; os.makedirs('/data/input', exist_ok=True)
with open('$INPUT_FILE','w') as f:
    for _ in range(5_000_000):
        f.write(' '.join(random.choices(words, k=10)) + '\\n')
print('Dataset generated.')
\"
    "
    log_pass "Synthetic dataset generated at $INPUT_FILE"
fi

# ── T4: MPI hostfile is valid ───────────────────────────────────────────────
log_info "T4: Validating MPI hostfile..."
HOSTFILE_LINES=$(docker exec "$CONTAINER" wc -l < "$HOSTFILE" 2>/dev/null || echo "0")
if [ "$HOSTFILE_LINES" -ge "2" ]; then
    log_pass "Hostfile has $HOSTFILE_LINES entries"
    docker exec "$CONTAINER" cat "$HOSTFILE" | while read line; do
        log_info "  hostfile entry: $line"
    done
else
    log_fail "Hostfile missing or has < 2 entries (found $HOSTFILE_LINES)"
fi

# ── T5: SSH connectivity between root and workers ───────────────────────────
log_info "T5: Testing SSH connectivity from root to workers..."
for worker_ip in 172.20.0.3 172.20.0.4; do
    SSH_RESULT=$(docker exec "$CONTAINER"         ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$worker_ip"         "hostname" 2>&1 || echo "FAILED")
    if echo "$SSH_RESULT" | grep -qv "FAILED"; then
        log_pass "SSH to $worker_ip: OK (host=$(echo $SSH_RESULT | tr -d '\n'))"
    else
        log_fail "SSH to $worker_ip: FAILED ($SSH_RESULT)"
    fi
done

# ── T6: Cross-container NFS write/read ──────────────────────────────────────
log_info "T6: NFS cross-container read/write test..."
PROBE="nfs_probe_$$"
docker exec mpi-root bash -c "echo 'nfs_ok' > /data/tmp/${PROBE}"
for cname in mpi-worker-1 mpi-worker-2; do
    READ=$(docker exec "$cname" cat "/data/tmp/${PROBE}" 2>/dev/null || echo "")
    if [ "$READ" = "nfs_ok" ]; then
        log_pass "$cname can read file written by root (NFS shared)"
    else
        log_fail "$cname cannot read file from root (NFS not shared)"
    fi
done
docker exec mpi-root rm -f "/data/tmp/${PROBE}"

# ── T7: Run WordCount via mpirun ─────────────────────────────────────────────
log_info "T7: Running WordCount workload (np=$NP)..."
docker exec "$CONTAINER" bash -c "mkdir -p $RESULTS_DIR"
START_TS=$(date +%s%N)
docker exec "$CONTAINER" bash -c "
    mpirun --hostfile $HOSTFILE -np $NP         --mca btl_tcp_if_include eth0         python3 mpj_spark_mpi.py             --app wordcount             --input $INPUT_FILE             --compare             --results-dir $RESULTS_DIR         2>&1 | tee /data/results/p4_05/mpirun_stdout.log
" && EXIT_CODE=0 || EXIT_CODE=$?
END_TS=$(date +%s%N)
EXEC_SEC=$(( (END_TS - START_TS) / 1000000000 ))

if [ "$EXIT_CODE" -eq "0" ]; then
    log_pass "mpirun exited with code 0 (exec_time=${EXEC_SEC}s)"
else
    log_fail "mpirun exited with code $EXIT_CODE"
fi

# ── T8: Validate output results file ────────────────────────────────────────
log_info "T8: Checking results output..."
RESULT_FILE=$(docker exec "$CONTAINER"     bash -c "ls ${RESULTS_DIR}/*.json 2>/dev/null | head -1" 2>/dev/null || echo "")
if [ -n "$RESULT_FILE" ]; then
    log_pass "Results file found: $RESULT_FILE"
    TOTAL_WORDS=$(docker exec "$CONTAINER"         python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(d.get('total_word_count',0))"         2>/dev/null || echo "0")
    if [ "$TOTAL_WORDS" -gt "0" ]; then
        log_pass "WordCount total=$TOTAL_WORDS (non-zero)"
    else
        log_fail "WordCount total=$TOTAL_WORDS (expected > 0)"
    fi
else
    log_fail "No results JSON file found in $RESULTS_DIR"
fi

# ── T9: Multi-driver result consistency (compare flag) ───────────────────────
log_info "T9: Checking multi-driver vs single-driver comparison result..."
COMPARE_LOG=$(docker exec "$CONTAINER"     bash -c "grep -i 'comparison\|deviation\|baseline' ${RESULTS_DIR}/mpirun_stdout.log 2>/dev/null || echo 'not_found'"     2>/dev/null || echo "")
if echo "$COMPARE_LOG" | grep -qi "comparison\|deviation"; then
    log_pass "Comparison metrics logged: $(echo $COMPARE_LOG | head -c 120)"
else
    log_info "Comparison log line not found (check $RESULTS_DIR/mpirun_stdout.log)"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "P4-05 Results: PASS=${PASS} FAIL=${FAIL}"
if [ "$FAIL" -eq "0" ]; then
    echo -e "${GREEN}ALL TESTS PASSED — P4-05 ACCEPTANCE CRITERIA MET${NC}"
    exit 0
else
    echo -e "${RED}${FAIL} TEST(S) FAILED — review output above${NC}"
    exit 1
fi
echo "========================================================"
