#!/usr/bin/env bash
# =============================================================================
# validate_p4_07_logreg.sh
# P4-07 — Logistic Regression Docker/MPI data-integrity validation
# Criterion: baseline vs MPI held-out accuracy delta
# =============================================================================
set -euo pipefail

NP="${NP:-3}"
MAX_ITER="${MAX_ITER:-20}"
LOGREG_TOL="${LOGREG_TOL:-0.03}"
ACCURACY_SAMPLE="${ACCURACY_SAMPLE:-1000}"
CONTAINER="${CONTAINER:-mpi-root}"
HOSTFILE="${HOSTFILE:-/etc/mpi/hostfile}"
RUN_ID="${RUN_ID:-p4_07}"
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
echo "P4-07: Logistic Regression parity validation"
echo "MPI ranks=${NP}, max_iter=${MAX_ITER}, accuracy tol=${LOGREG_TOL}"
echo "========================================================"

for cname in mpi-root mpi-worker-1 mpi-worker-2; do
    status="$(docker inspect --format='{{.State.Status}}' "${cname}" 2>/dev/null || echo missing)"
    if [[ "${status}" == "running" ]]; then
        log_pass "${cname} is running"
    else
        log_fail "${cname} status=${status}"
    fi
done

if run_in_root "test -s /data/input/logreg_data.csv"; then
    log_pass "LogReg input exists on shared /data volume"
else
    log_info "Generating LogReg input on shared volume"
    if run_in_root "cd /app && python scripts/generate_datasets.py --logreg-only"; then
        log_pass "LogReg input generated"
    else
        log_fail "LogReg input generation failed"
    fi
fi

run_in_root "rm -rf '${REPORT_DIR}' '${METRICS_DIR}' && mkdir -p '${REPORT_DIR}' '${METRICS_DIR}'"

log_info "Running MPI LogReg parity validation"
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
        --skip-kmeans \
        --max-iter '${MAX_ITER}' \
        --logreg-tol '${LOGREG_TOL}' \
        --accuracy-sample '${ACCURACY_SAMPLE}' \
        --report-dir '${REPORT_DIR}' \
        --metrics-dir '${METRICS_DIR}' \
      2>&1 | tee '${REPORT_DIR}/logreg_stdout.log'
"; then
    log_pass "LogReg MPI parity command completed"
else
    log_fail "LogReg MPI parity command failed"
fi

if run_in_root "test -s '${REPORT_DIR}/parity_report.csv'"; then
    log_pass "Parity CSV was produced"
else
    log_fail "Parity CSV is missing"
fi

if run_in_root "awk -F, 'NR > 1 && \$2 == \"logreg\" && \$NF == \"PASS\" { found=1 } END { exit !found }' '${REPORT_DIR}/parity_report.csv'"; then
    log_pass "LogReg held-out accuracy parity passed"
else
    log_fail "LogReg accuracy parity did not pass"
fi

if run_in_root "ls '${METRICS_DIR}'/logreg_rank*_epochs.csv >/dev/null 2>&1"; then
    log_pass "Per-rank LogReg metrics were written"
else
    log_fail "Per-rank LogReg metrics are missing"
fi

echo "========================================================"
echo "P4-07 results: PASS=${PASS}, FAIL=${FAIL}"
[[ "${FAIL}" -eq 0 ]] && exit 0
exit 1