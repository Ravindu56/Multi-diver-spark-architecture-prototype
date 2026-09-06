#!/usr/bin/env bash
# bootstrap_swarm.sh - P5-01 (#26): initialize Docker Swarm on the manager VM,
# join all worker VMs, and create the attachable overlay network used by the
# multi-driver stack deploy in P5-03 (#28).
#
# Usage:
#   ./scripts/bootstrap_swarm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INVENTORY_FILE="${INVENTORY_FILE:-$SCRIPT_DIR/.swarm-inventory}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
OVERLAY_NET="${OVERLAY_NET:-mpj-net}"

log() { printf '[bootstrap] %s\n' "$*"; }
die() { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f $INVENTORY_FILE ]] || die "inventory not found: $INVENTORY_FILE (run provision_swarm_vms.sh first)"

mapfile -t NODES < "$INVENTORY_FILE"
[[ ${#NODES[@]} -ge 2 ]] || die "inventory needs at least a manager and one worker"

read -r MGR_NAME MGR_IP <<< "${NODES[0]}"
log "manager: $MGR_NAME ($MGR_IP)"

ssh_vm() { # ssh_vm <ip> <remote command...>
  local ip="$1"; shift
  ssh $SSH_OPTS "$SSH_USER@$ip" "$@"
}

swarm_state() {
  ssh_vm "$1" 'docker info --format "{{.Swarm.LocalNodeState}}" 2>/dev/null' || echo unknown
}

# ---- 1. swarm init on manager (idempotent) ----
if [[ $(swarm_state "$MGR_IP") == active ]]; then
  log "swarm already active on manager - skipping init"
else
  log "docker swarm init --advertise-addr $MGR_IP"
  ssh_vm "$MGR_IP" "docker swarm init --advertise-addr $MGR_IP"
fi

# ---- 2. join workers ----
TOKEN="$(ssh_vm "$MGR_IP" 'docker swarm join-token -q worker')"
[[ -n $TOKEN ]] || die "could not read worker join token from manager"

for line in "${NODES[@]:1}"; do
  read -r name ip <<< "$line"
  if [[ $(swarm_state "$ip") == active ]]; then
    log "$name already in swarm - skipping"
    continue
  fi
  log "joining $name ($ip)"
  ssh_vm "$ip" "docker swarm join --token $TOKEN $MGR_IP:2377"
done

# ---- 3. overlay network (attachable for P5-03 stack deploy) ----
if ssh_vm "$MGR_IP" "docker network ls --format '{{.Name}}'" | grep -qx "$OVERLAY_NET"; then
  log "overlay network '$OVERLAY_NET' already exists"
else
  log "creating overlay network '$OVERLAY_NET' (--attachable)"
  ssh_vm "$MGR_IP" "docker network create --driver overlay --attachable $OVERLAY_NET"
fi

# ---- 4. validation against issue #26 acceptance criteria ----
log "docker node ls:"
ssh_vm "$MGR_IP" 'docker node ls'

ready="$(ssh_vm "$MGR_IP" "docker node ls --format '{{.Status.State}} {{.Availability}}'" \
        | grep -c 'Ready Active' || true)"
expected="${#NODES[@]}"
driver="$(ssh_vm "$MGR_IP" "docker network inspect $OVERLAY_NET --format '{{.Driver}}/attachable={{.Attachable}}'")"

echo
log "acceptance check (issue #26):"
printf '  nodes Ready+Active : %s/%s\n' "$ready" "$expected"
printf '  overlay network    : %s (%s)\n' "$OVERLAY_NET" "$driver"

if [[ $ready -eq $expected && $driver == "overlay/attachable=true" ]]; then
  log "P5-01 acceptance criteria met"
else
  die "acceptance check failed - inspect 'docker node ls' output above"
fi
