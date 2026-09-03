#!/usr/bin/env bash
set -euo pipefail

NP="${NP:-3}"
RUN_ID="${RUN_ID:-p4_full}"
CONTAINER="${CONTAINER:-mpi-root}"
HOSTFILE="${HOSTFILE:-/etc/mpi/hostfile}"
REPORT_DIR="/data/results/${RUN_ID}_parity"
METRICS_DIR="/data/metrics/${RUN_ID}"
TIMING_DIR="/data/results/${RUN_ID}_timing"

docker exec "${CONTAINER}" bash -lc "
  set -euo pipefail
  cd /app

  export PYTHONPATH=/app:\${PYTHONPATH:-}
  export MPI_NUM_RANKS=${NP}

  python scripts/generate_datasets.py

  rm -rf '${REPORT_DIR}' '${METRICS_DIR}' '${TIMING_DIR}'
  mkdir -p '${REPORT_DIR}' '${METRICS_DIR}' '${TIMING_DIR}'

  mpirun --oversubscribe \
    --bind-to none \
    --mca hwloc_base_binding_policy none \
    --hostfile '${HOSTFILE}' \
    -np '${NP}' \
    -x PYTHONPATH \
    -x MPI_NUM_RANKS \
    python scripts/validate_parity.py \
      --k 3 \
      --max-iter 20 \
      --kmeans-tol 0.02 \
      --logreg-tol 0.03 \
      --report-dir '${REPORT_DIR}' \
      --metrics-dir '${METRICS_DIR}'

  python scripts/timing_analysis.py \
    --metrics-dir '${METRICS_DIR}' \
    --out-dir '${TIMING_DIR}' \
    --num-ranks '${NP}'
"