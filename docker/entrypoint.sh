#!/usr/bin/env bash
# =============================================================================
# docker/entrypoint.sh  —  Phase 4 Container Entrypoint
# MPJ-SPARK Multi-Driver Architecture
# University of Jaffna — EC6070 — 2022/E/033 & 2022/E/090
#
# PURPOSE
# -------
# This script is the ENTRYPOINT for every container in the cluster.
# It is rank-agnostic: the same image runs on root and all workers.
# Rank identity comes from the MPI_RANK env var injected by docker-compose.
#
# WHAT IT DOES (in order)
# -----------------------
#   1. Start the SSH daemon (required by mpirun to spawn remote processes)
#   2. Generate /etc/mpi/hostfile listing all MPI ranks by hostname
#   3. Wait until all peer containers are SSH-reachable (readiness gate)
#   4. Rank 0 (root): stay alive as interactive shell or run CMD
#      Rank 1+ (workers): stay alive via sleep; mpirun drives execution
#
# HOW MPIRUN WORKS IN THIS SETUP
# --------------------------------
# From the root container:
#   mpirun --hostfile /etc/mpi/hostfile -np <1+N> python3 mpj_spark_mpi.py [args]
#
# mpirun SSH-connects to each worker container and runs the same Python
# command.  mpj_spark_mpi.py reads MPI.COMM_WORLD.Get_rank() at startup
# and dispatches to root_main() (rank 0) or worker_main() (rank 1+).
#
# ENVIRONMENT VARIABLES (set in docker-compose.yml)
# --------------------------------------------------
#   MPI_RANK   : this container's MPI rank (0 = root, 1+ = worker)
#   MPI_SIZE   : total number of MPI processes (1 root + N workers)
#
# =============================================================================
set -euo pipefail

export PYTHONPATH="/app${PYTHONPATH:+:${PYTHONPATH}}"
export MPI_NUM_RANKS="${MPI_NUM_RANKS:-3}"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[ENTRYPOINT rank=${MPI_RANK:-?}]${NC} $*"; }
warn()  { echo -e "${YELLOW}[ENTRYPOINT rank=${MPI_RANK:-?}]${NC} $*"; }
step()  { echo -e "${CYAN}[ENTRYPOINT rank=${MPI_RANK:-?}]${NC} $*"; }

# ---------------------------------------------------------------------------
# Validate required env vars
# ---------------------------------------------------------------------------
: "${MPI_RANK:?MPI_RANK must be set in docker-compose.yml}"
: "${MPI_SIZE:?MPI_SIZE must be set in docker-compose.yml}"

NUM_WORKERS=$(( MPI_SIZE - 1 ))

info "Container starting  rank=${MPI_RANK}  size=${MPI_SIZE}  workers=${NUM_WORKERS}"

# ---------------------------------------------------------------------------
# STEP 1 — Start SSH daemon
# Required so mpirun on rank 0 can SSH into this container and launch
# the Python MPI process for each worker rank.
# ---------------------------------------------------------------------------
step "Starting SSH daemon..."
mkdir -p /var/run/sshd
/usr/sbin/sshd
info "sshd started (PID: $(pgrep sshd | head -1))"

# ---------------------------------------------------------------------------
# STEP 2 — Generate MPI hostfile
# /etc/mpi/hostfile is read by mpirun --hostfile on rank 0.
# Format: one hostname per line, one slot per host.
#
# Hostnames follow the docker-compose.yml naming convention:
#   mpi-root        (rank 0)
#   mpi-worker-1    (rank 1)
#   mpi-worker-2    (rank 2)
#   ...             (up to NUM_WORKERS)
#
# The hostfile is generated on ALL containers so mpirun can be invoked
# from any rank (useful for debugging).
# ---------------------------------------------------------------------------
step "Generating MPI hostfile at /etc/mpi/hostfile..."
mkdir -p /etc/mpi

{
    echo "mpi-root slots=1"
    for i in $(seq 1 "${NUM_WORKERS}"); do
        echo "mpi-worker-${i} slots=1"
    done
} > /etc/mpi/hostfile

info "Hostfile contents:"
cat /etc/mpi/hostfile

# ---------------------------------------------------------------------------
# STEP 3 — SSH readiness gate
# Wait until all peer containers are SSH-reachable before proceeding.
# This prevents mpirun from failing with "connection refused" on fast starts.
#
# Only rank 0 (root) needs to wait for all workers; workers only need
# to confirm their own sshd is up (already done in step 1).
# ---------------------------------------------------------------------------
if [[ "${MPI_RANK}" -eq 0 ]]; then
    step "Rank 0: waiting for all worker containers to become SSH-ready..."

    MAX_RETRIES=30
    RETRY_INTERVAL=2

    for i in $(seq 1 "${NUM_WORKERS}"); do
        WORKER_HOST="mpi-worker-${i}"
        info "  Checking SSH reachability: ${WORKER_HOST}"

        for attempt in $(seq 1 "${MAX_RETRIES}"); do
            if ssh -o ConnectTimeout=2 \
                   -o StrictHostKeyChecking=no \
                   -o UserKnownHostsFile=/dev/null \
                   -o BatchMode=yes \
                   "${WORKER_HOST}" "echo ssh-ok" &>/dev/null; then
                info "  ${WORKER_HOST} is SSH-ready (attempt ${attempt})"
                break
            fi

            if [[ "${attempt}" -eq "${MAX_RETRIES}" ]]; then
                echo "ERROR: ${WORKER_HOST} did not become SSH-ready after ${MAX_RETRIES} attempts."
                exit 1
            fi

            warn "  ${WORKER_HOST} not ready yet — retrying in ${RETRY_INTERVAL}s (${attempt}/${MAX_RETRIES})..."
            sleep "${RETRY_INTERVAL}"
        done
    done

    info "All ${NUM_WORKERS} worker(s) are SSH-ready."
fi

# ---------------------------------------------------------------------------
# STEP 4 — Rank-based dispatch
# ---------------------------------------------------------------------------
if [[ "${MPI_RANK}" -eq 0 ]]; then
    # ---- ROOT ---------------------------------------------------------------
    # Rank 0 is the interactive entry point. It stays alive as a bash shell
    # so the user (or run_docker.sh / CI) can invoke mpirun manually:
    #
    #   docker exec -it mpi-root bash
    #   mpirun --hostfile /etc/mpi/hostfile -np 3 \
    #       python3 mpj_spark_mpi.py --app wordcount \
    #       --input /data/input/dataset.txt --compare
    #
    # If CMD args are passed (e.g., via docker-compose command: [...]),
    # execute them directly instead of dropping to bash.
    # -------------------------------------------------------------------------
    info "Rank 0 (root) ready. Dropping to interactive shell."
    info "Run workloads with:"
    info "  mpirun --hostfile /etc/mpi/hostfile -np ${MPI_SIZE} \\"
    info "      python3 /app/mpj_spark_mpi.py --app <wordcount|kmeans|logreg> \\"
    info "      --input /data/input/dataset.txt --compare"
    info ""

    if [[ $# -gt 0 ]]; then
        # CMD passed — execute directly (useful for automated tests P4-05 to P4-08)
        info "Executing CMD: $*"
        exec "$@"
    else
        # Interactive mode
        exec /bin/bash
    fi

else
    # ---- WORKERS ------------------------------------------------------------
    # Rank 1+ containers act as passive MPI workers. They do not run the
    # Python process directly — mpirun on rank 0 SSHes in and spawns:
    #   python3 /app/mpj_spark_mpi.py [same args as rank 0]
    # mpj_spark_mpi.py reads comm.Get_rank() and calls worker_main().
    #
    # Workers stay alive via a sleep loop so sshd remains running and
    # mpirun can connect at any time.
    # -------------------------------------------------------------------------
    info "Rank ${MPI_RANK} (worker) ready. Waiting for mpirun from root..."
    info "sshd is running — mpirun will SSH in to launch worker_main()."

    # Keep container alive; sshd handles incoming mpirun connections
    exec sleep infinity
fi
