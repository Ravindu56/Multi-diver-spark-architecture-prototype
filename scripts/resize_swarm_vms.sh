#!/usr/bin/env bash
# resize_swarm_vms.sh - P5-01-prep (#82): resize existing P5-01 Swarm VMs to the
# low-RAM profile documented in docs/p5-01-low-ram-host-tuning.md.
#
# provision_swarm_vms.sh skips memory changes on VMs that already exist, so a
# separate resize step is required to shrink already-provisioned domains.
# This script shuts each VM down, sets persistent maxmem/mem via virsh, brings
# it back up, and verifies guest-visible RAM over SSH.
#
# Usage:
#   ./scripts/resize_swarm_vms.sh                 # MEM_MB=2560 (desktop-host profile)
#   MEM_MB=3072 ./scripts/resize_swarm_vms.sh      # research (headless) profile
#
# Prerequisites:
#   - scripts/.swarm-inventory exists (from provision_swarm_vms.sh)
#   - VMs are reachable over SSH before/after the resize
set -euo pipefail

VM_PREFIX="${VM_PREFIX:-swarm}"
MEM_MB="${MEM_MB:-2560}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
VIRSH="${VIRSH:-virsh --connect qemu:///system}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INVENTORY_FILE="${INVENTORY_FILE:-$SCRIPT_DIR/.swarm-inventory}"
SHUTDOWN_WAIT_SECS="${SHUTDOWN_WAIT_SECS:-60}"
BOOT_WAIT_SECS="${BOOT_WAIT_SECS:-90}"

log() { printf '[resize] %s\n' "$*"; }
die() { printf '[resize] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f $INVENTORY_FILE ]] || die "inventory not found: $INVENTORY_FILE (run provision_swarm_vms.sh first)"
mapfile -t NODES < "$INVENTORY_FILE"
[[ ${#NODES[@]} -ge 1 ]] || die "inventory is empty"

# Sizing rule from docs/p5-01-low-ram-host-tuning.md: total VM RAM must stay
# at or below (available host RAM - 2 GB). Warn (not block) if it's exceeded.
avail_mb="$(free -m | awk '/Mem:/{print $7}')"
total_vm_mb=$(( MEM_MB * ${#NODES[@]} ))
if (( total_vm_mb > avail_mb - 2048 )); then
    log "WARNING: ${#NODES[@]} VMs * ${MEM_MB} MB = ${total_vm_mb} MB exceeds (available ${avail_mb} MB - 2048 MB headroom)."
    log "Run ./scripts/host-mode.sh research first, or lower MEM_MB."
fi

ssh_vm() { # ssh_vm <ip> <remote command...>
    local ip="$1"; shift
    ssh $SSH_OPTS "$SSH_USER@$ip" "$@"
}

wait_for_shutoff() {
    local name="$1" elapsed=0
    while [[ "$($VIRSH domstate "$name")" != "shut off" && $elapsed -lt $SHUTDOWN_WAIT_SECS ]]; do
        sleep 5; elapsed=$((elapsed + 5))
    done
    [[ "$($VIRSH domstate "$name")" == "shut off" ]] || die "$name did not shut down within ${SHUTDOWN_WAIT_SECS}s"
}

wait_for_ssh() {
    local ip="$1" elapsed=0
    while ! ssh_vm "$ip" true 2>/dev/null; do
        sleep 5; elapsed=$((elapsed + 5))
        [[ $elapsed -lt $BOOT_WAIT_SECS ]] || die "no SSH on $ip within ${BOOT_WAIT_SECS}s after resize"
    done
}

for line in "${NODES[@]}"; do
    read -r name ip <<< "$line"
    log "==> $name ($ip): resizing to ${MEM_MB} MB"

    cur_mb=$(( $($VIRSH dommaxmem "$name") / 1024 ))
    if [[ $cur_mb -eq $MEM_MB ]]; then
        log "$name already configured for ${MEM_MB} MB - skipping"
        continue
    fi

    log "shutting down $name (currently ${cur_mb} MB)"
    $VIRSH shutdown "$name" || true
    wait_for_shutoff "$name"

    # maxmem must be set before mem; both persisted to the domain's XML (--config)
    $VIRSH setmaxmem "$name" "${MEM_MB}Mib" --config
    $VIRSH setmem "$name" "${MEM_MB}Mib" --config

    log "starting $name"
    $VIRSH start "$name"
    wait_for_ssh "$ip"

    guest_mb="$(ssh_vm "$ip" "free -m | awk '/Mem:/{print \$2}'")"
    log "$name up - guest-reported RAM: ${guest_mb} MB (target ${MEM_MB} MB)"
done

echo
log "resize complete. Verify swarm health before rerunning P5-04:"
log "  ssh $SSH_USER@<manager-ip> docker node ls"
log "  free -g   # on the host, confirm headroom per docs/p5-01-low-ram-host-tuning.md"
