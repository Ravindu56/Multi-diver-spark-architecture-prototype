#!/usr/bin/env bash
# P4-05: Validate containerized MPI WordCount and baseline comparison.

set -euo pipefail

NP="${NP:-3}"
ROOT_CONTAINER="${ROOT_CONTAINER:-mpi-root}"
INPUT_FILE="${INPUT_FILE:-/data/input/dataset.txt}"
RESULTS_DIR="${RESULTS_DIR:-/data/results/p4_05}"
HOSTFILE="${HOSTFILE:-/etc/mpi/hostfile}"

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

container_running() {
    docker inspect --format '{{.State.Status}}' "$1" 2>/dev/null | grep -qx "running"
}

echo
echo "P4-05 Validation: WordCount in Docker MPI Cluster"
echo "Timestamp: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "MPI ranks: ${NP}"
echo

log_info "T1: Checking cluster containers"
for container in mpi-root mpi-worker-1 mpi-worker-2; do
    if container_running "${container}"; then
        log_pass "Container ${container} is running"
    else
        log_fail "Container ${container} is not running"
    fi
done

log_info "T2: Checking NFS/shared data mount"
for container in mpi-root mpi-worker-1 mpi-worker-2; do
    if docker exec "${container}" mountpoint -q /data; then
        log_pass "${container}: /data is mounted"
    else
        log_fail "${container}: /data is not mounted"
    fi
done

log_info "T3: Checking WordCount input"
INPUT_SIZE="$(docker exec "${ROOT_CONTAINER}" sh -lc "stat -c '%s' '${INPUT_FILE}' 2>/dev/null || echo 0")"
if [[ "${INPUT_SIZE}" -gt 0 ]]; then
    log_pass "Input exists: ${INPUT_FILE} (${INPUT_SIZE} bytes)"
else
    log_fail "Input file missing or empty: ${INPUT_FILE}"
fi

log_info "T4: Checking MPI hostfile"
HOST_LINES="$(docker exec "${ROOT_CONTAINER}" sh -lc "grep -Ev '^[[:space:]]*(#|$)' '${HOSTFILE}' | wc -l")"
if [[ "${HOST_LINES}" -ge "${NP}" ]]; then
    log_pass "Hostfile contains ${HOST_LINES} active entries"
    docker exec "${ROOT_CONTAINER}" sh -lc "grep -Ev '^[[:space:]]*(#|$)' '${HOSTFILE}'" \
        | while IFS= read -r line; do
            log_info "Hostfile: ${line}"
        done
else
    log_fail "Hostfile has ${HOST_LINES} entries; expected at least ${NP}"
fi

log_info "T5: Testing SSH root-to-worker connectivity"
for worker in mpi-worker-1 mpi-worker-2; do
    if docker exec "${ROOT_CONTAINER}" sh -lc \
        "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 '${worker}' hostname" \
        >/dev/null 2>&1; then
        log_pass "SSH to ${worker} succeeded"
    else
        log_fail "SSH to ${worker} failed"
    fi
done

log_info "T6: Testing cross-container NFS read/write"
PROBE="/data/tmp/p4_05_nfs_probe.txt"
docker exec "${ROOT_CONTAINER}" sh -lc \
    "mkdir -p /data/tmp && printf 'p4-05-nfs-ok\n' > '${PROBE}'"

for worker in mpi-worker-1 mpi-worker-2; do
    if docker exec "${worker}" sh -lc "test \"\$(cat '${PROBE}' 2>/dev/null)\" = 'p4-05-nfs-ok'"; then
        log_pass "${worker} read root-written NFS probe"
    else
        log_fail "${worker} could not read root-written NFS probe"
    fi
done
docker exec "${ROOT_CONTAINER}" rm -f "${PROBE}"

log_info "T7: Running MPI WordCount and baseline comparison"
START_NS="$(date +%s%N)"

set +e
docker exec "${ROOT_CONTAINER}" bash -lc "
    set -o pipefail
    cd /app
    export PYTHONPATH=/app:\${PYTHONPATH:-}
    mkdir -p '${RESULTS_DIR}'

    mpirun \
      --oversubscribe \
      --bind-to none \
      --mca hwloc_base_binding_policy none \
      --hostfile '${HOSTFILE}' \
      -np '${NP}' \
      -x PYTHONPATH \
      python3 /app/mpj_spark_mpi.py \
        --app wordcount \
        --input '${INPUT_FILE}' \
        --compare \
        --results-dir '${RESULTS_DIR}' \
      2>&1 | tee '${RESULTS_DIR}/mpirun_stdout.log'
"
MPI_EXIT=$?
set -e

END_NS="$(date +%s%N)"
EXEC_MS=$(( (END_NS - START_NS) / 1000000 ))

if [[ "${MPI_EXIT}" -eq 0 ]]; then
    log_pass "WordCount MPI command completed successfully (${EXEC_MS} ms)"
else
    log_fail "WordCount MPI command failed with exit code ${MPI_EXIT}"
fi



RUN_LOG="${RESULTS_DIR}/mpirun_stdout.log"

log_info "T8: Checking workload evidence"
if docker exec "${ROOT_CONTAINER}" sh -lc "test -s '${RUN_LOG}'"; then
    log_pass "Workload log exists: ${RUN_LOG}"
else
    log_fail "Workload log is missing or empty: ${RUN_LOG}"
fi

if docker exec "${ROOT_CONTAINER}" grep -q "All 2 workers completed" "${RUN_LOG}"; then
    log_pass "Both MPI worker processes completed"
else
    log_fail "Worker completion marker not found"
fi

if docker exec "${ROOT_CONTAINER}" grep -Eq \
    "Unique words[[:space:]]*:?[[:space:]]*[1-9][0-9,]*" \
    "${RUN_LOG}"; then
    log_pass "Single-driver baseline reported a non-zero unique-word count"
else
    log_fail "Baseline unique-word count not found"
fi

log_info "T9: Checking comparison output"
if docker exec "${ROOT_CONTAINER}" grep -q "Multi-Driver vs Baseline" "${RUN_LOG}"; then
    log_pass "Multi-driver versus baseline comparison table produced"
else
    log_fail "Comparison table was not found"
fi

echo
echo "P4-05 Results: PASS=${PASS}, FAIL=${FAIL}"

if [[ "${FAIL}" -eq 0 ]]; then
    echo -e "${GREEN}ALL TESTS PASSED — P4-05 ACCEPTANCE CRITERIA MET${NC}"
    exit 0
fi


echo -e "${RED}P4-05 VALIDATION FAILED — REVIEW OUTPUT ABOVE${NC}"
exit 1