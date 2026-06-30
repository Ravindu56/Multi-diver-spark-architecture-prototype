#!/usr/bin/env bash
# =============================================================================
# run_tests.sh
# =============================================================================
# Full Objective 2a test sweep for MPJ-Spark Multi-Driver logreg + kmeans.
#
# Produces the 3x3 convergence matrix needed by the Objective 2b
# LSTM/regression predictor:
#
#   Axis 1 (reg_param / difficulty): 0.01, 0.1, 1.0
#   Axis 2 (workers):                1,    2,   3
#
# Also runs:
#   - Baseline single-driver (--compare) for each worker config
#   - WordCount smoke test
#   - K-Means sweep
#
# Usage:
#   chmod +x run_tests.sh
#   ./run_tests.sh              # full sweep (~60 min on single machine)
#   ./run_tests.sh smoke        # quick 5-iter sanity check only (~5 min)
#   ./run_tests.sh logreg       # logreg sweep only
#   ./run_tests.sh kmeans       # kmeans sweep only
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
DATA_DIR="${SCRIPT_DIR}/shared_storage"
LOGREG_DATA="${DATA_DIR}/logreg_data.csv"
KMEANS_DATA="${DATA_DIR}/kmeans_data.csv"
WORDCOUNT_DATA="${DATA_DIR}/wordcount_data.txt"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[run_tests] $(date '+%H:%M:%S')  $*"; }
sep() { echo "─────────────────────────────────────────────────────────────────"; }

check_datasets() {
    local missing=0
    for f in "${LOGREG_DATA}" "${KMEANS_DATA}" "${WORDCOUNT_DATA}"; do
        if [[ ! -f "$f" ]]; then
            log "MISSING: $f"
            missing=1
        fi
    done
    if [[ $missing -eq 1 ]]; then
        log "Run 'python generate_data.py' first to create all datasets."
        exit 1
    fi
    log "All datasets present."
}

run_main() {
    # run_main <label> <extra args...>
    local label="$1"; shift
    local logfile="${LOG_DIR}/${label}.log"
    log "START  ${label}"
    python main.py "$@" 2>&1 | tee "${logfile}"
    log "END    ${label}  (log: ${logfile})"
    sep
}

# ── Mode selection ────────────────────────────────────────────────────────────
MODE="${1:-full}"

# =============================================================================
# SMOKE TEST — quick sanity check, 5 iterations only
# =============================================================================
if [[ "${MODE}" == "smoke" ]]; then
    log "=== SMOKE TEST (5 iters, 2 workers) ==="
    check_datasets

    run_main "smoke_wordcount" \
        --app wordcount --workers 2 \
        --input "${WORDCOUNT_DATA}"

    run_main "smoke_logreg" \
        --app logreg --workers 2 \
        --input "${LOGREG_DATA}" \
        --logreg-iter 5 --logreg-reg-param 0.1 --logreg-features 10

    run_main "smoke_kmeans" \
        --app kmeans --workers 2 \
        --input "${KMEANS_DATA}" \
        --kmeans-iter 5 --kmeans-k 5 --kmeans-features 8

    log "Smoke test complete."
    exit 0
fi

# =============================================================================
# PRE-FLIGHT
# =============================================================================
check_datasets

# =============================================================================
# WORDCOUNT — baseline smoke
# =============================================================================
if [[ "${MODE}" == "full" ]]; then
    sep
    log "=== WORDCOUNT BASELINE ==="

    run_main "wordcount_w1" \
        --app wordcount --workers 1 \
        --input "${WORDCOUNT_DATA}"

    run_main "wordcount_w3" \
        --app wordcount --workers 3 \
        --input "${WORDCOUNT_DATA}"
fi

# =============================================================================
# LOGISTIC REGRESSION SWEEP
# =============================================================================
if [[ "${MODE}" == "full" || "${MODE}" == "logreg" ]]; then
    sep
    log "=== LOGREG SWEEP: 3x3 matrix (reg_param x workers) ==="
    ITERS=30
    FEATURES=10

    # ── Baseline: single-driver, no allreduce (--compare adds baseline col) ──
    for REG in 0.01 0.1 1.0; do
        run_main "logreg_baseline_reg${REG}" \
            --app logreg --workers 1 --compare \
            --input "${LOGREG_DATA}" \
            --logreg-iter "${ITERS}" \
            --logreg-reg-param "${REG}" \
            --logreg-features "${FEATURES}"
    done

    # ── Multi-driver sweep ──────────────────────────────────────────────────
    for WORKERS in 1 2 3; do
        for REG in 0.01 0.1 1.0; do
            run_main "logreg_w${WORKERS}_reg${REG}" \
                --app logreg --workers "${WORKERS}" \
                --input "${LOGREG_DATA}" \
                --logreg-iter "${ITERS}" \
                --logreg-reg-param "${REG}" \
                --logreg-features "${FEATURES}"
        done
    done

    log "LogReg sweep complete. Results: ${RESULTS_DIR}/logreg_iter_metrics.csv"
fi

# =============================================================================
# KMEANS SWEEP
# =============================================================================
if [[ "${MODE}" == "full" || "${MODE}" == "kmeans" ]]; then
    sep
    log "=== KMEANS SWEEP: workers x iter-count ==="
    FEATURES=8
    K=5

    # ── Baseline ────────────────────────────────────────────────────────────
    for ITERS in 10 20 30; do
        run_main "kmeans_baseline_iter${ITERS}" \
            --app kmeans --workers 1 --compare \
            --input "${KMEANS_DATA}" \
            --kmeans-iter "${ITERS}" \
            --kmeans-k "${K}" \
            --kmeans-features "${FEATURES}"
    done

    # ── Multi-driver sweep ──────────────────────────────────────────────────
    for WORKERS in 1 2 3; do
        for ITERS in 10 20 30; do
            run_main "kmeans_w${WORKERS}_iter${ITERS}" \
                --app kmeans --workers "${WORKERS}" \
                --input "${KMEANS_DATA}" \
                --kmeans-iter "${ITERS}" \
                --kmeans-k "${K}" \
                --kmeans-features "${FEATURES}"
        done
    done

    log "KMeans sweep complete."
fi

sep
log "All tests complete. Results under: ${RESULTS_DIR}/"
log "Logs under: ${LOG_DIR}/"
