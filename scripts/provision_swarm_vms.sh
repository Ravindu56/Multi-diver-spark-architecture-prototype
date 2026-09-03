#!/usr/bin/env bash
# provision_swarm_vms.sh - P5-01 (#26): provision a manager + N worker KVM VMs
# on a single physical host for the multi-node Docker Swarm cluster.
#
# Each VM is an Ubuntu 24.04 cloud image with Docker preinstalled via
# cloud-init. IPs are discovered from libvirt DHCP leases and written to an
# inventory file consumed by bootstrap_swarm.sh.
#
# Usage:
#   ./scripts/provision_swarm_vms.sh
#   WORKER_COUNT=1 MEM_MB=3072 ./scripts/provision_swarm_vms.sh
#
# Prerequisites (host):
#   sudo apt install qemu-kvm libvirt-daemon-system virtinst cloud-image-utils wget
#   sudo usermod -aG libvirt,kvm "$USER"   # then log out and back in
set -euo pipefail

# ---- configuration (override via environment) --------------------------------
VM_PREFIX="${VM_PREFIX:-swarm}"
WORKER_COUNT="${WORKER_COUNT:-2}"
VCPUS="${VCPUS:-2}"
MEM_MB="${MEM_MB:-4096}"
DISK_GB="${DISK_GB:-20}"
OS_VARIANT="${OS_VARIANT:-ubuntu24.04}"   # use ubuntu22.04 on older osinfo-db
BASE_IMAGE_URL="${BASE_IMAGE_URL:-https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img}"
IMAGE_DIR="${IMAGE_DIR:-/var/lib/libvirt/images/${VM_PREFIX}-vms}"
BASE_IMAGE="${IMAGE_DIR}/$(basename "$BASE_IMAGE_URL")"
NETWORK="${NETWORK:-default}"
SSH_PUB_KEY_PATH="${SSH_PUB_KEY_PATH:-$HOME/.ssh/id_rsa.pub}"
SSH_USER="${SSH_USER:-ubuntu}"
INVENTORY_FILE="${INVENTORY_FILE:-$(cd "$(dirname "$0")" && pwd)/.swarm-inventory}"
VIRSH="${VIRSH:-virsh --connect qemu:///system}"
IP_WAIT_SECS="${IP_WAIT_SECS:-180}"

log() { printf '[provision] %s\n' "$*"; }
die() { printf '[provision] ERROR: %s\n' "$*" >&2; exit 1; }

# ---- preflight ---------------------------------------------------------------
for cmd in virsh virt-install qemu-img cloud-localds ssh-keygen; do
  command -v "$cmd" >/dev/null || die "missing dependency: $cmd"
done
[[ -f $SSH_PUB_KEY_PATH ]] || ssh-keygen -t ed25519 -N '' -f "${SSH_PUB_KEY_PATH%.pub}"
SSH_PUB_KEY="$(cat "$SSH_PUB_KEY_PATH")"

$VIRSH net-info "$NETWORK" >/dev/null 2>&1 || die "libvirt network '$NETWORK' not found"
if ! $VIRSH net-info "$NETWORK" | grep -q 'Active:.*yes'; then
  log "starting libvirt network '$NETWORK'"
  $VIRSH net-start "$NETWORK"
fi

sudo install -d -m 0755 "$IMAGE_DIR"
if [[ ! -f $BASE_IMAGE ]]; then
  log "downloading base image: $BASE_IMAGE_URL"
  sudo wget -q --show-progress -O "$BASE_IMAGE" "$BASE_IMAGE_URL"
fi

# ---- per-VM creation ----------------------------------------------------------
NODES=("${VM_PREFIX}-manager")
for i in $(seq 1 "$WORKER_COUNT"); do NODES+=("${VM_PREFIX}-worker${i}"); done

: > "$INVENTORY_FILE"
for name in "${NODES[@]}"; do
  disk="${IMAGE_DIR}/${name}.qcow2"
  seed="${IMAGE_DIR}/${name}-seed.iso"

  if $VIRSH dominfo "$name" >/dev/null 2>&1; then
    log "VM '$name' already exists - skipping creation"
  else
    log "creating VM '$name' (${VCPUS} vCPU, ${MEM_MB} MB, ${DISK_GB} GB)"
    workdir="$(mktemp -d)"
    cat > "$workdir/user-data" <<EOF
#cloud-config
hostname: ${name}
fqdn: ${name}.local
manage_etc_hosts: true
package_update: true
packages:
  - docker.io
  - nfs-common
users:
  - name: ${SSH_USER}
    sudo: "ALL=(ALL) NOPASSWD:ALL"
    shell: /bin/bash
    ssh_authorized_keys:
      - ${SSH_PUB_KEY}
runcmd:
  - systemctl enable --now docker
  - usermod -aG docker ${SSH_USER}
EOF
    cat > "$workdir/meta-data" <<EOF
instance-id: ${name}
local-hostname: ${name}
EOF
    sudo qemu-img create -f qcow2 -F qcow2 -b "$BASE_IMAGE" "$disk" "${DISK_GB}G" >/dev/null
    cloud-localds "$workdir/seed.iso" "$workdir/user-data" "$workdir/meta-data"
    sudo mv "$workdir/seed.iso" "$seed"
    rm -rf "$workdir"

    sudo virt-install --connect qemu:///system \
      --name "$name" \
      --memory "$MEM_MB" --vcpus "$VCPUS" \
      --disk "path=$disk,format=qcow2,bus=virtio" \
      --disk "path=$seed,device=cdrom" \
      --os-variant "$OS_VARIANT" \
      --network "network=$NETWORK,model=virtio" \
      --graphics none --import --noautoconsole --quiet
  fi

  # ---- wait for DHCP lease IP ----
  ip=""
  elapsed=0
  while [[ -z $ip && $elapsed -lt $IP_WAIT_SECS ]]; do
    ip="$($VIRSH domifaddr "$name" --source lease 2>/dev/null \
         | awk '/ipv4/{split($4,a,"/"); print a[1]; exit}')"
    if [[ -z $ip ]]; then
      sleep 5
      elapsed=$((elapsed + 5))
    fi
  done
  [[ -n $ip ]] || die "no DHCP lease for '$name' after ${IP_WAIT_SECS}s"
  log "$name -> $ip"
  printf '%s %s\n' "$name" "$ip" >> "$INVENTORY_FILE"
done

log "inventory written to $INVENTORY_FILE"
cat "$INVENTORY_FILE"
log "next step: ./scripts/bootstrap_swarm.sh"
