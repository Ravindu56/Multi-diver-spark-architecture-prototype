#!/usr/bin/env bash
# teardown_swarm_vms.sh - destroy the P5-01 Swarm VMs, their disks and the
# inventory file created by provision_swarm_vms.sh.
#
# Usage:
#   ./scripts/teardown_swarm_vms.sh
set -euo pipefail

VM_PREFIX="${VM_PREFIX:-swarm}"
WORKER_COUNT="${WORKER_COUNT:-2}"
IMAGE_DIR="${IMAGE_DIR:-/var/lib/libvirt/images/${VM_PREFIX}-vms}"
VIRSH="${VIRSH:-virsh --connect qemu:///system}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INVENTORY_FILE="${INVENTORY_FILE:-$SCRIPT_DIR/.swarm-inventory}"

NODES=("${VM_PREFIX}-manager")
for i in $(seq 1 "$WORKER_COUNT"); do NODES+=("${VM_PREFIX}-worker${i}"); done

for name in "${NODES[@]}"; do
  if $VIRSH dominfo "$name" >/dev/null 2>&1; then
    echo "[teardown] destroying $name"
    $VIRSH destroy "$name" 2>/dev/null || true
    $VIRSH undefine "$name" --remove-all-storage 2>/dev/null \
      || sudo virsh --connect qemu:///system undefine "$name" --remove-all-storage
  fi
  sudo rm -f "${IMAGE_DIR}/${name}.qcow2" "${IMAGE_DIR}/${name}-seed.iso"
done

rm -f "$INVENTORY_FILE"
echo "[teardown] done"
