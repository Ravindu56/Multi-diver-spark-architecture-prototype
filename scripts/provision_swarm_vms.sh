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
#   sudo apt install qemu-kvm libvirt-daemon-system virtinst cloud-image-utils wget dnsmasq-base
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

command -v dnsmasq >/dev/null 2>&1 \
  || log "WARNING: dnsmasq not found - install dnsmasq-base or net-start will fail"

# Distinguish "cannot connect to libvirt" from "network undefined": both make
# net-info exit non-zero, but only the second is auto-fixable.
if ! $VIRSH list >/dev/null 2>&1; then
  die "cannot connect to qemu:///system - run: sudo usermod -aG libvirt,kvm \$USER, then re-login"
fi

if ! $VIRSH net-info "$NETWORK" >/dev/null 2>&1; then
  log "libvirt network '$NETWORK' not defined - defining it now"
  if [[ $NETWORK == default && -f /usr/share/libvirt/networks/default.xml ]]; then
    sudo virsh --connect qemu:///system net-define /usr/share/libvirt/networks/default.xml
  else
    netxml="$(mktemp)"
    cat > "$netxml" <<EOF
<network>
  <name>${NETWORK}</name>
  <forward mode='nat'/>
  <bridge name='virbr0' stp='on' delay='0'/>
  <ip address='192.168.122.1' netmask='255.255.255.0'>
    <dhcp>
      <range start='192.168.122.2' end='192.168.122.254'/>
    </dhcp>
  </ip>
</network>
EOF
    sudo virsh --connect qemu:///system net-define "$netxml"
    rm -f "$netxml"
  fi
  $VIRSH net-autostart "$NETWORK" >/dev/null 2>&1 || true
fi

# Network gate: poll instead of trusting a single one-shot probe. Ubuntu's
# libvirtd is socket-activated with a 120 s idle timeout, so the daemon may be
# mid-startup when this script is the waking client, and its autostart engine
# may be bringing the network up asynchronously - both report as "inactive" to
# a single net-info probe. A helper function with command substitution is used
# rather than an if-pipeline so `pipefail` + `grep -q` early-exit cannot be
# misread as probe failure.
net_active() {
  local out
  out="$($VIRSH net-info "$NETWORK" 2>/dev/null)" || return 1
  [[ $out =~ Active:[[:space:]]*yes ]]
}

if ! net_active; then
  log "libvirt network '$NETWORK' not active - attempting start"
  if ! $VIRSH net-start "$NETWORK"; then
    log "net-start failed - polling state (autostart race or slow daemon wake)"
  fi
  started=1
  for _ in $(seq 1 15); do
    if net_active; then
      started=0
      break
    fi
    sleep 2
  done
  [[ $started -eq 0 ]] || die "network '$NETWORK' stayed inactive for 30s - check manually: sudo virsh net-start $NETWORK (stale dnsmasq: sudo pkill -f 'dnsmasq.*libvirt/dnsmasq/default.conf')"
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
