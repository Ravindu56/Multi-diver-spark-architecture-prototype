#!/usr/bin/env bash
# =============================================================================
# P5-03: render, deploy and validate the Swarm multi-driver stack (issue #28).
# Idempotent - safe to re-run; redeploys and re-validates.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

SSH_USER="${SSH_USER:-ubuntu}"
NODE_IPS="${NODE_IPS:-192.168.122.224 192.168.122.27 192.168.122.7}"

# --- env: Phase-4 defaults first, Swarm overrides second -------------------
set -a
# shellcheck disable=SC1091
source docker/.env
# shellcheck disable=SC1091
source docker/swarm/swarm.env
set +a
STACK="${STACK_NAME:-mpj}"
MANAGER="$MANAGER_IP"

echo "[stack] manager: $MANAGER | stack: $STACK"
echo "[stack] MPI_SIZE=$MPI_SIZE  NUM_WORKERS=${NUM_WORKERS:-2}"

# --- precondition 1: overlay network exists on the manager ------------------
ssh -o ConnectTimeout=10 "$SSH_USER@$MANAGER" \
  "docker network inspect mpj-net >/dev/null 2>&1" \
  || { echo "[stack] ERROR: mpj-net missing on manager - run scripts/bootstrap_swarm.sh first"; exit 1; }
echo "[stack] mpj-net present on manager"

# --- precondition 2: identical image on every node --------------------------
HOST_ID="$(docker image inspect -f '{{.Id}}' mpj-spark:latest 2>/dev/null || true)"
[[ -n "$HOST_ID" ]] || { echo "[stack] ERROR: mpj-spark:latest missing on host - run scripts/build_and_distribute_image.sh"; exit 1; }
for ip in $NODE_IPS; do
  rid="$(ssh "$SSH_USER@$ip" "docker image inspect -f '{{.Id}}' mpj-spark:latest 2>/dev/null" || true)"
  [[ "$rid" == "$HOST_ID" ]] \
    || { echo "[stack] ERROR: $ip has stale/missing image (${rid:-none}) - run scripts/build_and_distribute_image.sh"; exit 1; }
  echo "[stack] $ip: image OK"
done

# --- render (shell env interpolation) and copy to the manager ---------------
RENDERED="$(mktemp)"
docker compose -f docker/swarm/stack-mpj-spark.yml config --quiet > "$RENDERED" \
  || docker compose -f docker/swarm/stack-mpj-spark.yml config > "$RENDERED"
scp -q "$RENDERED" "$SSH_USER@$MANAGER:/tmp/stack-mpj-spark.rendered.yml"
rm -f "$RENDERED"
echo "[stack] stack file rendered and staged on manager"

# --- deploy -----------------------------------------------------------------
ssh "$SSH_USER@$MANAGER" "docker stack deploy -c /tmp/stack-mpj-spark.rendered.yml $STACK"

# --- wait for convergence (max ~3 min/service) ------------------------------
for svc in mpi-root mpi-worker-1 mpi-worker-2; do
  for i in $(seq 1 36); do
    rep="$(ssh "$SSH_USER@$MANAGER" \
      "docker service ls --filter name=${STACK}_${svc} --format '{{.Replicas}}'")"
    if [[ "$rep" == "1/1" ]]; then
      echo "[stack]   ${STACK}_${svc}: $rep"
      break
    fi
    if [[ $i -eq 36 ]]; then
      echo "[stack] ERROR: ${STACK}_${svc} stuck at ${rep:-not-created}"
      ssh "$SSH_USER@$MANAGER" "docker service ps --no-trunc ${STACK}_${svc}"
      exit 1
    fi
    sleep 5
  done
done

# --- acceptance check 1: placement -------------------------------------------
expected=("mpi-root:swarm-manager" "mpi-worker-1:swarm-worker1" "mpi-worker-2:swarm-worker2")
for pair in "${expected[@]}"; do
  svc="${pair%%:*}"; want="${pair##*:}"
  node="$(ssh "$SSH_USER@$MANAGER" \
    "docker service ps ${STACK}_${svc} --filter desired-state=running --format '{{.Node}}' | head -1")"
  echo "[stack]   ${svc} -> ${node:-none} (expected ${want})"
  [[ "$node" == "$want" ]] || { echo "[stack] ERROR: placement mismatch for $svc"; exit 1; }
done

# --- acceptance check 2: /data mount + overlay DNS + NFS round-trip ---------
ROOT_CID="$(ssh "$SSH_USER@$MANAGER" \
  "docker ps -q --filter name=${STACK}_mpi-root --filter status=running")"
[[ -n "$ROOT_CID" ]] || { echo "[stack] ERROR: no running mpi-root container on manager"; exit 1; }

echo "[stack] checking /data, DNS aliases and NFS propagation via ${STACK}_mpi-root"
ssh "$SSH_USER@$MANAGER" "docker exec $ROOT_CID sh -c '
  ls /data >/dev/null || exit 1
  getent hosts mpi-root >/dev/null || exit 1
  getent hosts mpi-worker-1 >/dev/null || exit 1
  getent hosts mpi-worker-2 >/dev/null || exit 1
  date +%s > /data/.p5-03-probe
  sync
'"
for ip in $NODE_IPS; do
  ssh "$SSH_USER@$ip" "test -f /srv/mpj-share/.p5-03-probe" \
    || { echo "[stack] ERROR: NFS probe not visible on $ip"; exit 1; }
  echo "[stack]   $ip: sees container-written probe at /srv/mpj-share"
done
ssh "$SSH_USER@$MANAGER" "docker exec $ROOT_CID rm -f /data/.p5-03-probe"

echo "[stack] docker stack services ${STACK}:"
ssh "$SSH_USER@$MANAGER" "docker stack services $STACK"
echo "[stack] P5-03 acceptance criteria met - issue #28"
