#!/usr/bin/env bash
# =============================================================================
# P5-03 / P5-04: render, deploy and validate the Swarm multi-driver stack.
# Idempotent - safe to re-run; redeploys and re-validates.
#
# Usage (P5-04 experimental cells, see docs/p5-04-experimental-design.md):
#   ./scripts/bootstrap_stack.sh        # Cell A: dedicated 3-rank (base only)
#   ./scripts/bootstrap_stack.sh 4w     # Cell B: 5-rank (base + 4w overlay)
#   ./scripts/bootstrap_stack.sh 8w     # Cell C: 9-rank (base + 8w overlay)
#
# The overlay files add the co-located ranks (SPARK_DRIVER_MEMORY=400m) and
# override MPI_SIZE per service, so no manual env exports are required.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# --- topology selection (P5-04 cells) ----------------------------------------
OVERLAY="${1:-}"
case "$OVERLAY" in
  "")
    CELL="A | dedicated 3-rank"
    COMPOSE_FILES=(docker/swarm/stack-mpj-spark.yml)
    SERVICES=(mpi-root mpi-worker-1 mpi-worker-2)
    EXPECTED_PLACEMENT=(
      "mpi-root:swarm-manager"
      "mpi-worker-1:swarm-worker1"
      "mpi-worker-2:swarm-worker2"
    )
    ;;
  4w)
    CELL="B | 5-rank (2 co-located pairs)"
    COMPOSE_FILES=(docker/swarm/stack-mpj-spark.yml docker/swarm/stack-mpj-spark.4w.yml)
    SERVICES=(mpi-root mpi-worker-1 mpi-worker-2 mpi-worker-3 mpi-worker-4)
    EXPECTED_PLACEMENT=(
      "mpi-root:swarm-manager"
      "mpi-worker-1:swarm-worker1"
      "mpi-worker-2:swarm-worker2"
      "mpi-worker-3:swarm-worker1"
      "mpi-worker-4:swarm-worker2"
    )
    ;;
  8w)
    CELL="C | 9-rank (4 ranks per worker VM)"
    COMPOSE_FILES=(docker/swarm/stack-mpj-spark.yml docker/swarm/stack-mpj-spark.8w.yml)
    SERVICES=(mpi-root mpi-worker-1 mpi-worker-2 mpi-worker-3 mpi-worker-4
              mpi-worker-5 mpi-worker-6 mpi-worker-7 mpi-worker-8)
    EXPECTED_PLACEMENT=(
      "mpi-root:swarm-manager"
      "mpi-worker-1:swarm-worker1"
      "mpi-worker-2:swarm-worker2"
      "mpi-worker-3:swarm-worker1"
      "mpi-worker-4:swarm-worker1"
      "mpi-worker-5:swarm-worker1"
      "mpi-worker-6:swarm-worker2"
      "mpi-worker-7:swarm-worker2"
      "mpi-worker-8:swarm-worker2"
    )
    ;;
  *)
    echo "usage: $0 [4w|8w]" >&2
    exit 2
    ;;
esac

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

COMPOSE_ARGS=()
for f in "${COMPOSE_FILES[@]}"; do COMPOSE_ARGS+=(-f "$f"); done

echo "[stack] manager: $MANAGER | stack: $STACK | cell: $CELL"
echo "[stack] MPI_SIZE=$MPI_SIZE  NUM_WORKERS=${NUM_WORKERS:-2}  overlay=${OVERLAY:-none}"

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
# NOTE: `docker compose config --quiet` VALIDATES ONLY and prints nothing;
# piping it to a file produced an empty stack ('top-level object must be a
# mapping'). Use plain `config` and sanity-check the rendered YAML.
RENDERED="$(mktemp)"
docker compose "${COMPOSE_ARGS[@]}" config > "$RENDERED"
grep -q "^services:" "$RENDERED" || {
  echo "[stack] ERROR: rendered stack is empty - compose config produced no services"
  exit 1
}
# `docker compose config` renders published ports as quoted strings, which
# `docker stack deploy` rejects ('published must be a integer'). Unquote
# them, and drop the compose-only top-level `name:` field.
sed -i -E 's/(published: ?)"([0-9]+)"/\1\2/g' "$RENDERED"
sed -i -E '/^name: /d' "$RENDERED"
scp -q "$RENDERED" "$SSH_USER@$MANAGER:/tmp/stack-mpj-spark.rendered.yml"
rm -f "$RENDERED"
echo "[stack] stack file rendered and staged on manager"

# --- deploy -----------------------------------------------------------------
ssh "$SSH_USER@$MANAGER" "docker stack deploy -c /tmp/stack-mpj-spark.rendered.yml $STACK"

# --- wait for convergence (max ~3 min/service) ------------------------------
for svc in "${SERVICES[@]}"; do
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
for pair in "${EXPECTED_PLACEMENT[@]}"; do
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

ALIAS_CHECKS=""
for svc in "${SERVICES[@]}"; do
  ALIAS_CHECKS+="  getent hosts ${svc} >/dev/null || exit 1"$'\n'
done

echo "[stack] checking /data, DNS aliases and NFS propagation via ${STACK}_mpi-root"
ssh "$SSH_USER@$MANAGER" "docker exec $ROOT_CID sh -c '
  ls /data >/dev/null || exit 1
${ALIAS_CHECKS}  date +%s > /data/.p5-03-probe
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
echo "[stack] P5-03/P5-04 acceptance criteria met - issues #28 / #82 (cell: $CELL)"
