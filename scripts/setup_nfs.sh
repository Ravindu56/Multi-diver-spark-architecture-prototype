#!/usr/bin/env bash
# setup_nfs.sh - P5-02 (#27): configure NFS shared storage across the Swarm VMs.
#
# The manager VM (first inventory row) becomes the NFS server, exporting
# EXPORT_DIR to the libvirt VM subnet. Every node - including the manager -
# mounts that export at MOUNT_DIR over /etc/fstab, so all Swarm nodes see the
# identical filesystem path (the cloud-native equivalent of Lustre shared
# storage from the MPJ-Spark reference architecture).
#
# Acceptance check (issue #27): every node writes a probe file and must see
# the probes of ALL nodes through the shared mount - shared dataset
# accessible from every Swarm node.
#
# Usage:
#   ./scripts/setup_nfs.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INVENTORY_FILE="${INVENTORY_FILE:-$SCRIPT_DIR/.swarm-inventory}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
EXPORT_DIR="${EXPORT_DIR:-/srv/mpj-share}"
MOUNT_DIR="${MOUNT_DIR:-/srv/mpj-share}"   # same path on every node (uniform container bind-mount for P5-03)
NFS_OPTS="${NFS_OPTS:-rw,sync,no_subtree_check,no_root_squash}"
MOUNT_OPTS="${MOUNT_OPTS:-rw,nfsvers=3,soft,timeo=100,retrans=2,_netdev}"
MOUNT_RETRIES="${MOUNT_RETRIES:-4}"

log() { printf '[nfs] %s\n' "$*"; }
die() { printf '[nfs] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f $INVENTORY_FILE ]] || die "inventory not found: $INVENTORY_FILE (run provision_swarm_vms.sh first)"
mapfile -t NODES < "$INVENTORY_FILE"
[[ ${#NODES[@]} -ge 2 ]] || die "inventory needs at least a manager and one worker"

read -r MGR_NAME MGR_IP <<< "${NODES[0]}"
SUBNET="${MGR_IP%.*}.0/24"
log "manager: $MGR_NAME ($MGR_IP) - export '$EXPORT_DIR' to $SUBNET"

ssh_vm() { # ssh_vm <ip> <remote command...>
  local ip="$1"; shift
  ssh $SSH_OPTS "$SSH_USER@$ip" "$@"
}

# ---- 1. NFS server on the manager ----
log "configuring NFS server on $MGR_NAME"
ssh_vm "$MGR_IP" 'sudo bash -s' <<REMOTE
set -euo pipefail
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nfs-kernel-server
mkdir -p $EXPORT_DIR/input $EXPORT_DIR/output $EXPORT_DIR/results $EXPORT_DIR/partitions
chmod -R 0777 $EXPORT_DIR
# The manager mounts its own export in step 2 for path uniformity. A directory
# that is itself an NFS mount cannot be re-exported (exportfs: "requires
# fsid="), so drop any existing self-mount BEFORE touching the export table.
mountpoint -q $EXPORT_DIR && umount -f -l $EXPORT_DIR || true
if ! grep -qsF "$EXPORT_DIR " /etc/exports; then
  echo "$EXPORT_DIR  $SUBNET($NFS_OPTS)" >> /etc/exports
fi
exportfs -ra
systemctl enable nfs-kernel-server >/dev/null
systemctl restart nfs-kernel-server
REMOTE

# ---- 2. fstab mount on every node (uniform path for P5-03 bind mounts) ----
for line in "${NODES[@]}"; do
  read -r name ip <<< "$line"
  log "mounting share on $name ($ip) at $MOUNT_DIR"
  ssh_vm "$ip" 'sudo bash -s' <<REMOTE
set -euo pipefail
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nfs-common >/dev/null 2>&1 || true
mkdir -p $MOUNT_DIR
if ! grep -qs "$MGR_IP:$EXPORT_DIR" /etc/fstab; then
  echo "$MGR_IP:$EXPORT_DIR  $MOUNT_DIR  nfs  $MOUNT_OPTS  0 0" >> /etc/fstab
fi
mountpoint -q $MOUNT_DIR && umount -f -l $MOUNT_DIR || true
mounted=0
for attempt in $(seq 1 $MOUNT_RETRIES); do
  if mount $MOUNT_DIR; then
    mounted=1
    break
  fi
  echo "[nfs:remote] mount attempt $attempt failed on $name; client view of server exports:"
  showmount -e $MGR_IP || true
  sleep 3
done
[ "${mounted}" -eq 1 ] || { echo "[nfs:remote] ERROR: mount failed on $name after $MOUNT_RETRIES attempts" >&2; exit 1; }
REMOTE
done

# ---- 3. acceptance check (issue #27) ----
log "acceptance check: every node writes a probe, every node must see all"
for line in "${NODES[@]}"; do
  read -r name ip <<< "$line"
  ssh_vm "$ip" "echo '$name' | sudo tee $MOUNT_DIR/.nfs_probe_$name >/dev/null"
done
sleep 2   # attribute-cache settle on soft mounts

fail=0
for line in "${NODES[@]}"; do
  read -r name ip <<< "$line"
  seen="$(ssh_vm "$ip" "ls $MOUNT_DIR/.nfs_probe_* 2>/dev/null | wc -l")"
  printf '  %-15s sees %s/%s probe files\n' "$name" "$seen" "${#NODES[@]}"
  [[ $seen -eq ${#NODES[@]} ]] || fail=1
done

exports="$(ssh_vm "$MGR_IP" 'sudo showmount -e localhost 2>/dev/null || exportfs -v')"
printf '  exports on %s:\n%s\n' "$MGR_NAME" "$exports"

# cleanup probes
for line in "${NODES[@]}"; do
  read -r name ip <<< "$line"
  ssh_vm "$ip" "sudo rm -f $MOUNT_DIR/.nfs_probe_*" || true
done

if [[ $fail -eq 0 ]]; then
  log "P5-02 acceptance criteria met - shared volume accessible from all Swarm nodes"
else
  die "not all nodes see the shared volume - check /etc/exports on $MGR_NAME and mount output above"
fi
