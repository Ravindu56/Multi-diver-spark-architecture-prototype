#!/usr/bin/env bash
# P4-09: Docker cluster lifecycle and validation launcher.
#
# Usage:
#   ./scripts/run_docker.sh up
#   ./scripts/run_docker.sh build
#   ./scripts/run_docker.sh status
#   ./scripts/run_docker.sh logs
#   ./scripts/run_docker.sh shell
#   ./scripts/run_docker.sh down
#   ./scripts/run_docker.sh validate-p4-05
#   ./scripts/run_docker.sh validate-p4-06
#   ./scripts/run_docker.sh validate-p4-07
#   ./scripts/run_docker.sh benchmark-p4-08

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."\ && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker/docker-compose.yml"
ROOT_CONTAINER="${MPI_ROOT_CONTAINER:-mpi-root}"
MPI_RANKS="${MPI_NUM_RANKS:-3}"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is required but was not found in PATH." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: Docker Compose v2 is required." >&2
    exit 1
fi

compose() {
    docker compose -f "${COMPOSE_FILE}" "$@"
}

require_running_cluster() {
    if ! docker inspect -f '{{.State.Running}}' "${ROOT_CONTAINER}" 2>/dev/null | grep -qx "true"; then
        echo "ERROR: ${ROOT_CONTAINER} is not running."
        echo "Run: ./scripts/run_docker.sh up"
        exit 1
    fi
}

container_exec() {
    require_running_cluster
    docker exec "${ROOT_CONTAINER}" bash -lc "$1"
}

build_cluster() {
    echo "Building Docker images..."
    compose build
}

start_cluster() {
    build_cluster
    echo "Starting Docker MPI cluster..."
    compose up -d

    echo "Waiting for containers..."
    for _ in $(seq 1 30); do
        if docker inspect -f '{{.State.Running}}' "${ROOT_CONTAINER}" 2>/dev/null | grep -qx "true"; then
            break
        fi
        sleep 1
    done

    require_running_cluster
    echo "Cluster started successfully."
    compose ps
}

status_cluster() {
    compose ps
    echo
    docker network inspect mpj-spark-net \
        --format 'Network: {{.Name}}; containers: {{len .Containers}}' \
        2>/dev/null || true
}

validate_p4_05() {
    container_exec "
        cd /app
        export PYTHONPATH=/app:\${PYTHONPATH:-}
        bash scripts/validate_p4_05_wordcount.sh
    "
}

validate_p4_06() {
    container_exec "
        cd /app
        export PYTHONPATH=/app:\${PYTHONPATH:-}
        bash scripts/validate_p4_06_kmeans.sh
    "
}

validate_p4_07() {
    container_exec "
        cd /app
        export PYTHONPATH=/app:\${PYTHONPATH:-}
        bash scripts/validate_p4_07_logreg.sh
    "
}

benchmark_p4_08() {
    container_exec "
        cd /app
        export PYTHONPATH=/app:\${PYTHONPATH:-}
        mkdir -p /data/metrics/p4_08 /data/results/p4_08_sync

        cp -f /data/metrics/p4_06/*.csv /data/metrics/p4_08/ 2>/dev/null || true
        cp -f /data/metrics/p4_07/*.csv /data/metrics/p4_08/ 2>/dev/null || true

        mpirun \\
            --oversubscribe \\
            --bind-to none \\
            --mca hwloc_base_binding_policy none \\
            --hostfile /etc/mpi/hostfile \\
            -np ${MPI_RANKS} \\
            -x PYTHONPATH \\
            python3 scripts/sync_overhead_benchmark.py \\
                --ranks ${MPI_RANKS} \\
                --kmeans-k 3 \\
                --kmeans-iter 20 \\
                --logreg-epochs 20 \\
                --baseline-workers 2 \\
                --baseline-cores 0 \\
                --metrics-dir /data/metrics/p4_08 \\
                --results-dir /data/results/p4_08_sync \\
                --data-dir /data/input

        echo
        echo 'P4-08 result CSV:'
        cat /data/results/p4_08_sync/sync_overhead_benchmark.csv
    "
}

show_logs() {
    compose logs --tail=200 "$@"
}

open_shell() {
    require_running_cluster
    docker exec -it "${ROOT_CONTAINER}" bash
}

usage() {
    cat <<'EOF'
Usage: ./scripts/run_docker.sh <command>

Commands:
  build             Build the Docker image
  up                Build and start the three-container MPI cluster
  down              Stop and remove containers and project network
  status            Show Docker Compose service and network status
  logs [service]    Show recent Compose logs
  shell             Open an interactive shell in mpi-root
  validate-p4-05    Run Docker WordCount acceptance validation
  validate-p4-06    Run Docker K-Means acceptance validation
  validate-p4-07    Run Docker Logistic Regression acceptance validation
  benchmark-p4-08   Run synchronization-overhead benchmark
EOF
}

COMMAND="${1:-}"

case "${COMMAND}" in
    build) build_cluster ;;
    up) start_cluster ;;
    down) compose down --remove-orphans ;;
    status) status_cluster ;;
    logs) shift; show_logs "$@" ;;
    shell) open_shell ;;
    validate-p4-05) validate_p4_05 ;;
    validate-p4-06) validate_p4_06 ;;
    validate-p4-07) validate_p4_07 ;;
    benchmark-p4-08) benchmark_p4_08 ;;
    -h|--help|help|"") usage ;;
    *)
        echo "ERROR: Unknown command: ${COMMAND}" >&2
        usage >&2
        exit 2
        ;;
esac
