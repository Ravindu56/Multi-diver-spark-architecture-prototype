#!/usr/bin/env bash
# =============================================================================
# docker/nfs-setup.sh  —  Phase 4 NFS Host Setup Script
# MPJ-SPARK Multi-Driver Architecture
# University of Jaffna — EC6070 — 2022/E/033 & 2022/E/090
#
# PURPOSE
# -------
# Automates NFS server setup on the HOST machine (the single node running
# all Docker containers in Phase 4). Creates and exports the shared
# directory that all MPI-rank containers mount as /data.
#
# This is the functional equivalent of Lustre shared storage used in the
# Saleh et al. (2025) MPJ-SPARK paper, adapted for a Docker environment.
#
# USAGE
# -----
#   sudo bash docker/nfs-setup.sh
#
# WHAT IT DOES
# ------------
#   1. Installs nfs-kernel-server (NFS server daemon)
#   2. Creates the shared export directory: /data/mpj-spark-shared
#      with subdirs: input/, output/, results/, partitions/
#   3. Adds the export rule to /etc/exports for Docker bridge subnet
#   4. Restarts and enables the NFS server
#   5. Verifies the export is active
#
# ARCHITECTURE NOTE
# -----------------
# In Phase 4 (single-node Docker cluster), NFS server and all containers
# run on the SAME physical host. The Docker bridge network (172.20.0.0/24
# defined in docker-compose.yml P4-03) is what containers use to mount
# from the host's NFS server via the host's docker0 or bridge IP.
#
# In Phase 5 (multi-node Docker Swarm), this script runs on the designated
# NFS server node; all other Swarm nodes mount from that node's IP.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SHARED_DIR="/data/mpj-spark-shared"
DOCKER_SUBNET="172.20.0.0/24"      # must match docker-compose.yml network subnet
NFS_OPTS="rw,sync,no_subtree_check,no_root_squash"

# ---------------------------------------------------------------------------
# Colours for output
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Must run as root
# ---------------------------------------------------------------------------
[[ "$(id -u)" -eq 0 ]] || error "Run this script with sudo: sudo bash $0"

# ---------------------------------------------------------------------------
# 1. Install NFS server
# ---------------------------------------------------------------------------
info "Installing nfs-kernel-server..."
apt-get update -qq
apt-get install -y --no-install-recommends nfs-kernel-server
info "nfs-kernel-server installed."

# ---------------------------------------------------------------------------
# 2. Create shared export directory structure
# ---------------------------------------------------------------------------
info "Creating shared directory structure at ${SHARED_DIR}..."
mkdir -p \
    "${SHARED_DIR}/input" \
    "${SHARED_DIR}/output" \
    "${SHARED_DIR}/results" \
    "${SHARED_DIR}/partitions"

# Set permissive permissions so all container UIDs can read/write
chmod -R 777 "${SHARED_DIR}"
info "Directory structure created and permissions set."

# ---------------------------------------------------------------------------
# 3. Add NFS export rule
#    no_root_squash: allows root inside containers to write to the share
#    sync:           writes are committed to disk before ACK (data safety)
# ---------------------------------------------------------------------------
EXPORT_LINE="${SHARED_DIR}  ${DOCKER_SUBNET}(${NFS_OPTS})"

if grep -qsF "${SHARED_DIR}" /etc/exports; then
    warn "Export entry for ${SHARED_DIR} already exists in /etc/exports — skipping."
else
    info "Adding export rule to /etc/exports..."
    echo "${EXPORT_LINE}" >> /etc/exports
    info "Export rule added: ${EXPORT_LINE}"
fi

# ---------------------------------------------------------------------------
# 4. Restart and enable NFS server
# ---------------------------------------------------------------------------
info "Restarting NFS server..."
exportfs -ra
systemctl restart nfs-kernel-server
systemctl enable nfs-kernel-server
info "NFS server restarted and enabled."

# ---------------------------------------------------------------------------
# 5. Verify export is active
# ---------------------------------------------------------------------------
info "Verifying NFS exports..."
showmount -e localhost

info "=================================================================="
info " NFS setup complete."
info "  Shared dir : ${SHARED_DIR}"
info "  Exported to: ${DOCKER_SUBNET}"
info "  Options    : ${NFS_OPTS}"
info ""
info " Next step: run docker-compose.yml (P4-03) which defines the"
info " named volume that containers mount at /data."
info "=================================================================="
