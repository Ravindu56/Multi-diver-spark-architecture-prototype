# Phase 4 — NFS Shared Volume Setup (P4-02)

> **Objective 1a** — Functional equivalent of Lustre shared storage from  
> Saleh et al. (2025) MPJ-SPARK, adapted for a Docker single-node cluster.

---

## Architecture

In the MPJ-SPARK paper, all MPI ranks share a **Lustre parallel filesystem** so
every Spark driver can directly read its assigned dataset partition without MPI
data movement. In Phase 4, we replicate this with an **NFS-backed Docker
named volume** shared across all containers on the Docker bridge network.

```
HOST MACHINE
├── /data/mpj-spark-shared/          ← NFS export root
│   ├── input/                       ← raw dataset files
│   ├── partitions/                  ← per-worker partition files (written by rank 0)
│   ├── output/                      ← aggregated results (written by rank 0)
│   └── results/                     ← metrics CSVs (CPU, memory, exec time)
│
├── [NFS Server]  exports → 172.20.0.0/24
│
├── [Container: mpi-root  172.20.0.2]  mounts /data → NFS share
├── [Container: mpi-worker-1  172.20.0.3]  mounts /data → NFS share
└── [Container: mpi-worker-2  172.20.0.4]  mounts /data → NFS share
```

All containers read partition files directly from `/data/partitions/` —
only partition **metadata** (file path, byte range) travels over MPI.
This mirrors the Lustre data-plane / MPI control-plane separation in MPJ-SPARK.

---

## Step-by-Step Setup

### Step 1 — Run the NFS setup script on the host

```bash
sudo bash docker/nfs-setup.sh
```

This will:
- Install `nfs-kernel-server`
- Create `/data/mpj-spark-shared/{input,output,results,partitions}`
- Add the export rule to `/etc/exports` for subnet `172.20.0.0/24`
- Restart and enable the NFS daemon
- Print active exports via `showmount -e localhost`

### Step 2 — Find the Docker bridge gateway IP

After running `docker-compose up` (P4-03), the host's Docker bridge IP
is the NFS server address that containers use to mount:

```bash
# Get the host gateway IP on the mpj-spark-net bridge
docker network inspect mpj-spark-net --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}'
# Typically: 172.20.0.1
```

Set this in your environment before launching:

```bash
export NFS_SERVER_IP=172.20.0.1   # host bridge gateway
```

### Step 3 — Verify NFS mount inside a container

```bash
# Start a test container and verify the NFS volume is mounted and writable
docker run --rm \
    --network mpj-spark-net \
    -v mpj-spark-data:/data \
    mpj-spark:latest \
    bash -c "echo 'nfs-ok' > /data/nfs_test.txt && cat /data/nfs_test.txt"

# Expected output: nfs-ok
```

### Step 4 — Copy dataset into the shared volume

```bash
# Generate a synthetic dataset and place it into the NFS input directory
python3 generate_data.py --size-mb 100 --output /data/mpj-spark-shared/input/dataset.txt

# Or copy an existing dataset
cp ./test_dataset.txt /data/mpj-spark-shared/input/
```

---

## Docker Named Volume Definition

The named volume is declared in `docker/docker-compose.yml` (P4-03).
The Docker volume uses the `local` driver with NFS mount options:

```yaml
volumes:
  mpj-spark-data:
    driver: local
    driver_opts:
      type: nfs
      o: "addr=${NFS_SERVER_IP},rw,nfsvers=4,soft"
      device: ":/data/mpj-spark-shared"
```

All services in `docker-compose.yml` mount this volume at `/data`:

```yaml
services:
  mpi-root:
    volumes:
      - mpj-spark-data:/data
  mpi-worker-1:
    volumes:
      - mpj-spark-data:/data
```

---

## NFS Export Options Explained

| Option | Meaning |
|---|---|
| `rw` | Read-write access for all containers |
| `sync` | Write to disk before ACK — prevents data loss on crash |
| `no_subtree_check` | Improves reliability when exporting subdirectories |
| `no_root_squash` | Allows container root (UID 0) to write — needed since containers run as root |

---

## Acceptance Criteria Checklist (P4-02)

- [ ] `showmount -e localhost` shows `/data/mpj-spark-shared` exported to `172.20.0.0/24`
- [ ] `docker-compose` volume mounts NFS path accessible by all containers
- [ ] Root container can write a file to `/data/partitions/` and worker containers can read it
- [ ] All containers see the same `/data` contents simultaneously

---

## Troubleshooting

**Mount fails with `access denied`**  
Check `/etc/exports` subnet matches the Docker network subnet in `docker-compose.yml`.
Run `exportfs -ra && systemctl restart nfs-kernel-server` after any `/etc/exports` change.

**`nfs-kernel-server` fails to start inside WSL2**  
WSL2 does not support NFS server natively. Use a bind mount instead:
```yaml
volumes:
  mpj-spark-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/mpj-spark-shared
```

**`Operation not permitted` writing to `/data` inside container**  
Ensure `no_root_squash` is set in `/etc/exports` and the host directory has `chmod 777`.
