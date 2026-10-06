#!/usr/bin/env bash
# preflight_check.sh -- collects environment verification info for the
# resource-profiling experiment (Phase 6 prep).
# Usage:  bash preflight_check.sh [repo_dir]
# Output: preflight_report_<hostname>_<timestamp>.txt  (send me this file)

set -u
REPO="${1:-$PWD}"
REPORT="preflight_report_$(hostname)_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee "$REPORT") 2>&1

echo "================================================================"
echo " PREFLIGHT CHECK  $(date -Is)"
echo "================================================================"

section() { echo; echo "------------------------------------------------------------"; echo " $1"; echo "------------------------------------------------------------"; }

############################
# 1. LAPTOP HARDWARE / OS
############################
section "1. CPU model and cores"
lscpu | head -20

section "2. RAM"
free -h

section "3. Ubuntu / OS version"
lsb_release -a 2>/dev/null || cat /etc/os-release

section "4. Free disk space"
df -h ~

############################
# 2. DOCKER + CGROUP
############################
section "5. Docker version"
docker version --format 'server={{.Server.Version}} client={{.Client.Version}}' 2>&1

section "6. cgroup driver/version (need: cgroupfs/systemd, v2)"
docker info 2>/dev/null | grep -i cgroup || echo "docker info failed (daemon up?)"
CGRP=$(docker info 2>/dev/null | grep -i "cgroup version" | awk '{print $NF}')
if [ "$CGRP" = "v2" ]; then
    echo ">>> RESULT: cgroup v2  OK (memory/cpu stats per container will work)"
else
    echo ">>> RESULT: NOT cgroup v2 ($CGRP)  -> per-container memory stats may be limited"
fi

############################
# 3. PYTHON / MPI (host mode)
############################
section "7. Python + mpi4py + pyspark versions"
cd "$REPO" || { echo "repo dir not found: $REPO"; exit 1; }
python --version 2>&1
python - <<'PY' 2>&1
for m in ("mpi4py", "pyspark", "psutil", "numpy"):
    try:
        mod = __import__(m)
        print(f"{m} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"{m} MISSING ({e})")
PY

section "8. mpirun"
which mpirun && mpirun --version | head -3

############################
# 4. PHASE 3 SMOKE TEST (host mpirun, direct)
############################
section "9. Phase 3 smoke test: mpirun -np 3 kmeans --generate 50"
T0=$(date +%s)
if timeout 900 mpirun --oversubscribe -np 3 \
        python -m mpj_spark.core.main_mpi --app kmeans --generate 50 ; then
    echo ">>> RESULT: Phase 3 mpirun works  (took $(( $(date +%s) - T0 )) s)"
else
    echo ">>> RESULT: Phase 3 mpirun FAILED (exit $?)"
fi

############################
# 5. PHASE 4 DOCKER SETUP
############################
section "10. docker compose up (Phase 4)"
cd "$REPO/docker" 2>/dev/null || { echo "no docker/ dir under $REPO"; cd "$REPO"; }

if docker compose ps 2>/dev/null | grep -q .; then
    echo "--- containers before ---"
    docker compose ps
fi

T0=$(date +%s)
if docker compose up -d --build ; then
    sleep 10
    echo "--- containers after up ---"
    docker compose ps 2>&1
    for c in mpi-root mpi-worker-1 mpi-worker-2; do
        if docker ps --format '{{.Names}}' | grep -qx "$c"; then
            echo ">>> RESULT: container $c RUNNING  OK"
        else
            echo ">>> RESULT: container $c NOT RUNNING"
        fi
    done

    section "11. job inside containers (rank-0 runs mpirun across the 3 nodes)"
    if docker exec mpi-root bash -lc \
        "mpirun --oversubscribe -np 3 --host mpi-root,mpi-worker-1,mpi-worker-2 \
         python -m mpj_spark.core.main_mpi --app kmeans --generate 50" ; then
        echo ">>> RESULT: Phase 4 container job works  (took $(( $(date +%s) - T0 )) s total incl. build)"
    else
        echo ">>> RESULT: container job FAILED — check entrypoint.sh / ssh setup between containers"
    fi
else
    echo ">>> RESULT: docker compose up FAILED"
fi

section "12. Nvidia/LVM misc (dmesg OOM recent? useful if a run dies)"
dmesg 2>/dev/null | grep -i -E 'oom|killed process' | tail -5 || echo "(no OOM evidence / no permission)"

echo
echo "================================================================"
echo " Done. Report saved to: $PWD/$REPORT"
echo " Send this file back. Estimated wall time of this script: 10-20 min."
echo "================================================================"
