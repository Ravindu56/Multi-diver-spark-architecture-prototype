# P5-02: NFS Shared Volume across Swarm Nodes

**Issue:** #27 - **Phase:** 5 - **Objectives:** 1a, 1c
**Status:** implemented via `scripts/setup_nfs.sh`; run after bootstrap_swarm.sh

## 1. What this is

The functional equivalent of the Lustre shared filesystem in the MPJ-Spark
reference architecture (Saleh et al. 2025), realized for the containerized
cloud deployment: one NFS export, mounted at the same path on every node.
Dataset partitioning metadata travels over MPI; the data itself is read
directly from the shared path - the same data-plane/control-plane separation
as the HPC reference.

```
swarm-manager (192.168.122.224)
├── NFS server, exports /srv/mpj-share -> 192.168.122.0/24
│   ├── input/       raw datasets (generate_datasets.py targets live here)
│   ├── partitions/  per-worker partition files written by rank 0
│   ├── output/      aggregated results
│   └── results/     metrics CSVs (Obj 2a profiling data)
│
└── mounted at /srv/mpj-share on swarm-manager, swarm-worker1, swarm-worker2
    (manager mounts its own export for path uniformity)
```

## 2. Design decisions

- **VM-level fstab mounts, not Docker `type: nfs` volumes.** In Swarm, tasks
  can be rescheduled to any node; a guest-level mount guarantees the path
  exists on every candidate node before any container starts. P5-03 stack
  services will `bind` `/srv/mpj-share` into containers per service.
- **Manager mounts its own export.** Keeps `/srv/mpj-share` valid on all
  three nodes, so the root coordinator and workers use identical paths and
  the P5-03 stack file needs no per-node conditionals.
- **`nfsvers=3` + `no_root_squash`.** The Phase 4 lesson: nfsv4 idmapd/UID
  mapping broke container (UID 0) writes on the dev setup; v3 +
  no_root_squash sidesteps this in the controlled lab network.
- Subnet is derived from the manager's inventory IP - no hardcoded CIDR.

## 3. Usage

```bash
./scripts/provision_swarm_vms.sh   # P5-01
./scripts/bootstrap_swarm.sh       # P5-01
./scripts/setup_nfs.sh             # P5-02 (this)
```

Overrides: `EXPORT_DIR`, `MOUNT_DIR`, `NFS_OPTS`, `MOUNT_OPTS`, `SSH_USER`.

## 4. Acceptance criteria mapping (issue #27)

- [ ] "Shared dataset accessible from all Swarm nodes" - verified by the
  scripted probe exchange: each of the 3 nodes writes a file into the share
  and every node must observe all 3 files. Any visibility gap fails the
  script non-zero.

## 5. Troubleshooting

- `mount.nfs: Connection timed out` on a worker -> the manager's export does
  not cover the VM subnet; check `sudo exportfs -v` on the manager and that
  the worker IP falls inside the exported CIDR.
- Stale file handle after a VM reboot -> `sudo umount -f -l /srv/mpj-share`
  then `sudo mount /srv/mpj-share` (the script already performs a forced
  lazy unmount before remounting, and `_netdev` defers boot-time mounting
  until the network is up).
- Writers see `Permission denied` -> confirm `no_root_squash` is present in
  `/etc/exports` on the manager and re-run `sudo exportfs -ra`.

## References

[1] Docker Inc., "Docker volumes: NFS shares," docs.docker.com.
[2] Saleh et al., "MPJ-SPARK Integration-Based Technique to Enhance Big Data
    Analytics in High Performance Computing Environments," IEEE Access, 2025.
[3] Ubuntu manpages, "exports(5) - NFS server export table."
