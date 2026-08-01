#!/usr/bin/env bash
# =============================================================================
# validate_p4_06_kmeans.sh
# P4-06 — K-Means Docker/MPI data-integrity validation
# Criterion: baseline vs multi-driver WCSS relative delta
# =============================================================================
set -euo pipefail

NP="${NP:-3}"
K="${K:-3}"
KMEANS_ITER="${KMEANS_ITER:-20}"
KMEANS_TOL="${KMEANS_TOL:-0.02}"
CONTAINER="${CONTAINER:-mpi-root}"
HOSTFILE="${HOSTFILE:-/etc/mpi/hostfile}"
RUN_ID="${RUN_ID:-p4_06}"
REPORT_DIR="${REPORT_DIR:-/data/results/${RUN_ID}_parity}"
METRICS_DIR="${METRICS_DIR:-/data/metrics/${RUN_ID}}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

log_pass() { echo -e "${GREEN}[PASS]${NC} $*"; PASS=$((PASS + 1)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $*"; FAIL=$((FAIL + 1)); }
log_info() { echo -e "${YELLOW}[INFO]${NC} $*"; }

run_in_root() {
    docker exec "${CONTAINER}" bash -lc "$1"
}

echo "========================================================"
echo "P4-06: K-Means parity validation"
echo "MPI ranks=${NP}, k=${K}, max_iter=${KMEANS_ITER}, WCSS tol=${KMEANS_TOL}"
echo "========================================================"

for cname in mpi-root mpi-worker-1 mpi-worker-2; do
    status="$(docker inspect --format='{{.State.Status}}' "${cname}" 2>/dev/null || echo missing)"
    if [[ "${status}" == "running" ]]; then
        log_pass "${cname} is running"
    else
        log_fail "${cname} status=${status}"
    fi
done

if run_in_root "test -s /data/input/kmeans_data.csv"; then
    log_pass "K-Means input exists on shared /data volume"
else
    log_info "Generating K-Means input on shared volume"
    if run_in_root "cd /app && python scripts/generate_datasets.py --kmeans-only"; then
        log_pass "K-Means input generated"
    else
        log_fail "K-Means input generation failed"
    fi
fi

if run_in_root "test \$(grep -cvE '^[[:space:]]*(#|$)' '${HOSTFILE}') -ge ${NP}"; then
    log_pass "MPI hostfile supports ${NP} ranks"
else
    log_fail "MPI hostfile has insufficient usable entries"
fi

run_in_root "rm -rf '${REPORT_DIR}' '${METRICS_DIR}' && mkdir -p '${REPORT_DIR}' '${METRICS_DIR}'"

log_info "Running MPI K-Means parity validation"
if run_in_root "
    cd /app &&
    export PYTHONPATH=/app:\${PYTHONPATH:-} &&
    mpirun --oversubscribe \
      --bind-to none \
      --mca hwloc_base_binding_policy none \
      --hostfile '${HOSTFILE}' \
      -np '${NP}' \
      -x PYTHONPATH \
      python scripts/validate_parity.py \
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

if run_in_root "test -s '${REPORT_DIR}/parity_report.csv'"; then
    log_pass "Parity CSV was produced"
else
    log_fail "Parity CSV is missing"
fi

if run_in_root "awk -F, 'NR > 1 && \$2 == \"kmeans\" && \$NF == \"PASS\" { found=1 } END { exit !found }' '${REPORT_DIR}/parity_report.csv'"; then
    log_pass "K-Means WCSS parity passed"
else
    log_fail "K-Means WCSS parity did not pass"
fi

if run_in_root "ls '${METRICS_DIR}'/kmeans_metrics_rank*.csv >/dev/null 2>&1"; then
    log_pass "Per-rank K-Means metrics were written"
else
    log_fail "Per-rank K-Means metrics are missing"
fi

echo "========================================================"
echo "P4-06 results: PASS=${PASS}, FAIL=${FAIL}"
[[ "${FAIL}" -eq 0 ]] && exit 0
exit 1