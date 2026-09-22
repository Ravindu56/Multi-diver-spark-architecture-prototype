#!/usr/bin/env bash
# stage_datasets.sh - P5-04-prep (#82): generate the K-Means scaling datasets
# (100 MB / 500 MB / 1 GB by default) and verify they are byte-identical
# across every Swarm node before starting the P5-04 (#29) scaling grid.
#
# Datasets are generated ONCE, inside the running mpi-root container,
# directly onto the NFS-backed /data/input mount (docker/nfs-setup.sh,
# P5-02 #27) -- every other node already sees the same bytes through the
# NFS mount, not a copy. The per-node checksum loop below is a correctness
# check on that assumption (confirms the NFS mount is actually consistent),
# not a distribution step.
#
# Idempotent: generate_datasets.py skips a size if a file already at >= 80%
# of the target size exists at that path; pass FORCE=1 to regenerate anyway.
#
# Usage:
#   ./scripts/stage_datasets.sh                  # sizes: 100 500 1024 (MB)
#   SIZES_MB="100 500" ./scripts/stage_datasets.sh
#   FORCE=1 ./scripts/stage_datasets.sh           # regenerate even if present
#
# Prerequisites:
#   - scripts/.swarm-inventory exists (provision_swarm_vms.sh)
#   - the stack is deployed and mpi-root is running (bootstrap_stack.sh)
set -euo pipefail

SIZES_MB="${SIZES_MB:-100 500 1024}"
STACK="${STACK_NAME:-mpj}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INVENTORY_FILE="${INVENTORY_FILE:-$SCRIPT_DIR/.swarm-inventory}"
CONTAINER_INPUT_DIR="${CONTAINER_INPUT_DIR:-/data/input}"     # inside mpi-root
HOST_SHARE_PATH="${HOST_SHARE_PATH:-/srv/mpj-share}"          # NFS export root, every VM
FORCE="${FORCE:-0}"

log() { printf '[stage] %s\n' "$*"; }
die() { printf '[stage] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f $INVENTORY_FILE ]] || die "inventory not found: $INVENTORY_FILE (run provision_swarm_vms.sh first)"
mapfile -t NODES < "$INVENTORY_FILE"
[[ ${#NODES[@]} -ge 1 ]] || die "inventory is empty"

read -r MGR_NAME MGR_IP <<< "${NODES[0]}"
log "manager: $MGR_NAME ($MGR_IP)  |  nodes in inventory: ${#NODES[@]}"

ssh_vm() { # ssh_vm <ip> <remote command...>
    local ip="$1"; shift
    ssh $SSH_OPTS "$SSH_USER@$ip" "$@"
}

ROOT_CID="$(ssh_vm "$MGR_IP" "docker ps -q --filter name=${STACK}_mpi-root --filter status=running")"
[[ -n $ROOT_CID ]] || die "no running ${STACK}_mpi-root container on $MGR_IP - deploy the stack first (bootstrap_stack.sh)"
log "mpi-root container: $ROOT_CID"

FORCE_FLAG=""
[[ $FORCE == 1 ]] && FORCE_FLAG="--force"

# ---- 1. generate each requested size on the NFS-shared mount ---------------
FILES=()
for mb in $SIZES_MB; do
    fname="kmeans_data_${mb}mb.csv"
    FILES+=("$fname")
    log "==> ${fname}: generating ${mb} MB via mpi-root -> ${CONTAINER_INPUT_DIR}/${fname}"
    ssh_vm "$MGR_IP" "docker exec -e MPJ_KMEANS_DATA=${CONTAINER_INPUT_DIR}/${fname} \
        $ROOT_CID python3 scripts/generate_datasets.py --kmeans-only --size-mb ${mb} ${FORCE_FLAG}" \
        | sed 's/^/[stage]   /'
done

# ---- 2. verify byte-identical content across every node in the inventory ---
echo
log "verifying byte-identical content across all ${#NODES[@]} node(s) at ${HOST_SHARE_PATH}/input/ ..."

overall_fail=0
for fname in "${FILES[@]}"; do
    ref_sum=""
    file_fail=0
    for line in "${NODES[@]}"; do
        read -r name ip <<< "$line"
        path="${HOST_SHARE_PATH}/input/${fname}"
        sum="$(ssh_vm "$ip" "md5sum '$path' 2>/dev/null | awk '{print \$1}'" || true)"
        if [[ -z $sum ]]; then
            log "  FAIL  $name ($ip): $fname not found at $path"
            file_fail=1
            continue
        fi
        if [[ -z $ref_sum ]]; then
            ref_sum="$sum"
        elif [[ $sum != "$ref_sum" ]]; then
            log "  FAIL  $name ($ip): md5 $sum != reference $ref_sum for $fname"
            file_fail=1
        fi
    done
    if [[ $file_fail -eq 0 ]]; then
        size_bytes="$(ssh_vm "$MGR_IP" "stat -c %s '${HOST_SHARE_PATH}/input/${fname}' 2>/dev/null" || echo '?')"
        log "  OK    ${fname}  (md5=${ref_sum}, ${size_bytes} bytes, identical on all ${#NODES[@]} nodes)"
    else
        overall_fail=1
    fi
done

echo
if [[ $overall_fail -eq 0 ]]; then
    log "all datasets byte-identical across nodes - #82 dataset acceptance criterion met"
else
    die "one or more datasets failed the byte-identical check across nodes (see FAIL lines above)"
fi
