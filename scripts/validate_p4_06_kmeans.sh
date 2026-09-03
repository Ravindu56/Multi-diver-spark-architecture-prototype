#!/usr/bin/env bash
# P4-06: Docker MPI K-Means parity validation.

set -euo pipefail

NP="${NP:-3}"
K="${K:-3}"
KMEANS_ITER="${KMEANS_ITER:-20}"
KMEANS_TOL="${KMEANS_TOL:-0.02}"

CONTAINER="${CONTAINER:-mpi-root}"
HOSTFILE="${HOSTFILE:-/etc/mpi/hostfile}"

RUN_ID="${RUN_ID:-p4_06}"
REPORT_DIR="${REPORT_DIR:-/data/results/${RUN_ID}/parity}"
METRICS_DIR="${METRICS_DIR:-/data/metrics/${RUN_ID}}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

log_pass() {
    echo -e "${GREEN}PASS${NC} $*"
    PASS=$((PASS + 1))
}

log_fail() {
    echo -e "${RED}FAIL${NC} $*"
    FAIL=$((FAIL + 1))
}

log_info() {
    echo -e "${YELLOW}INFO${NC} $*"
}

run_in_root() {
    docker exec "${CONTAINER}" bash -lc "$1"
}

container_running() {
    docker inspect --format '{{.State.Status}}' "$1" 2>/dev/null | grep -qx "running"
}

parity_status_pass() {
    local report="$1"
    local workload="$2"

    run_in_root "python3 - '${report}' '${workload}' <<'PY'
import csv
import sys

report_path = sys.argv[1]
target_workload = sys.argv[2].strip().lower()

try:
    with open(report_path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
except (FileNotFoundError, OSError):
    raise SystemExit(1)

matches = [
    row for row in rows
    if row.get('workload', '').strip().lower() == target_workload
]

if not matches:
    raise SystemExit(1)

if all(row.get('status', '').strip().upper() == 'PASS' for row in matches):
    raise SystemExit(0)

raise SystemExit(1)
PY"
}

echo
echo "P4-06 K-Means parity validation"
echo "MPI ranks=${NP}, k=${K}, max_iter=${KMEANS_ITER}, WCSS tolerance=${KMEANS_TOL}"
echo

for cname in mpi-root mpi-worker-1 mpi-worker-2; do
    if container_running "${cname}"; then
        log_pass "${cname} is running"
    else
        log_fail "${cname} is not running"
    fi
done

if run_in_root "test -s /data/input/kmeans_data.csv"; then
    log_pass "K-Means input exists on shared data volume"
else
    log_fail "K-Means input is missing: /data/input/kmeans_data.csv"
fi

if run_in_root "test \$(grep -cvE '^[[:space:]]*(#|$)' '${HOSTFILE}') -ge ${NP}"; then
    log_pass "MPI hostfile supports ${NP} ranks"
else
    log_fail "MPI hostfile has insufficient usable entries"
fi

run_in_root "rm -rf '${REPORT_DIR}' '${METRICS_DIR}' && mkdir -p '${REPORT_DIR}' '${METRICS_DIR}'"

log_info "Running MPI K-Means parity validation"

if run_in_root "
    cd /app
    export PYTHONPATH=/app:\${PYTHONPATH:-}

    mpirun \
      --oversubscribe \
      --bind-to none \
      --mca hwloc_base_binding_policy none \
      --hostfile '${HOSTFILE}' \
      -np '${NP}' \
      -x PYTHONPATH \
      python3 /app/scripts/validate_parity.py \
        --skip-logreg \
        --k '${K}' \
        --max-iter '${KMEANS_ITER}' \
        --kmeans-tol '${KMEANS_TOL}' \
        --report-dir '${REPORT_DIR}' \
        --metrics-dir '${METRICS_DIR}' \
      2>&1 | tee '${REPORT_DIR}/kmeans_stdout.log'
"; then
    log_pass "K-Means MPI parity command completed"
else
    log_fail "K-Means MPI parity command failed"
fi

PARITY_CSV="${REPORT_DIR}/parity_report.csv"

if run_in_root "test -s '${PARITY_CSV}'"; then
    log_pass "Parity CSV was produced"
else
    log_fail "Parity CSV is missing: ${PARITY_CSV}"
fi

if parity_status_pass "${PARITY_CSV}" "kmeans"; then
    log_pass "K-Means WCSS parity passed according to parity_report.csv"
else
    log_fail "K-Means WCSS parity did not pass according to parity_report.csv"
fi

if run_in_root "find '${METRICS_DIR}' -type f -iname '*kmeans*.csv' -size +0c | grep -q ."; then
    log_pass "Per-rank K-Means metrics were written"
else
    log_fail "Per-rank K-Means metrics are missing"
fi

echo
echo "P4-06 Results: PASS=${PASS}, FAIL=${FAIL}"

if [[ "${FAIL}" -eq 0 ]]; then
    echo -e "${GREEN}ALL TESTS PASSED — P4-06 ACCEPTANCE CRITERIA MET${NC}"
    exit 0
fi

echo -e "${RED}P4-06 VALIDATION FAILED — REVIEW OUTPUT ABOVE${NC}"
exit 1