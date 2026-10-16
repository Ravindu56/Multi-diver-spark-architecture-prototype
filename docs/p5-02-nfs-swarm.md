# P5-02: NFS Shared Volume across Swarm Nodes

**Issue:** #27 - **Phase:** 5 - **Objectives:** 1a, 1c
**Status:** implemented via `scripts/setup_nfs.sh`; validated 2026-09-12

## 1. What this is

The functional equivalent of the Lustre shared filesystem in the MPJ-Spark
reference architecture (Saleh et al. 2025), realized for the containerized
cloud deployment: one NFS export on the manager, mounted by the workers at
the same absolute path. Dataset partitioning metadata travels over MPI; the
data itself is read directly from the shared path - the same
data-plane/control-plane separation as the HPC reference.

```
swarm-manager (192.168.122.224)
├── NFS server, exports /srv/mpj-share -> 192.168.122.0/24
│   ├── input/       raw datasets (generate_datasets.py targets live here)
│   ├── partitions/  per-worker partition files written by rank 0
│   ├── output/      aggregated results
│   └── results/     metrics CSVs (Obj 2a profiling data)
│   (manager uses /srv/mpj-share NATIVELY - it is the export directory)
│
swarm-worker1 / swarm-worker2
└── NFS mounts of 192.168.122.224:/srv/mpj-share at /srv/mpj-share
```

All three nodes therefore see the identical path `/srv/mpj-share`; the
manager reaches it directly and workers over NFS, with no per-node conditionals
in the P5-03 stack file.

## 2. Design decisions

- **VM-level fstab mounts, not Docker `type: nfs` volumes.** In Swarm, tasks
  can be rescheduled to any node; a guest-level mount guarantees the path
  exists on every candidate node before any container starts. P5-03 stack
  services `bind` `/srv/mpj-share` into containers.
- **The manager does NOT self-mount.** An earlier iteration had the manager
  NFS-mount its own export for uniformity. Two hard failures invalidated
  that idea: a directory that is an NFS mount cannot be re-exported
  (`exportfs: requires fsid=`), and restarting `nfs-kernel-server` while the
  self-mount unit exists triggers a systemd "Transaction order is cyclic"
  error. The native export directory provides the same uniform path with
  neither failure mode.
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

Overrides: `EXPORT_DIR`, `MOUNT_DIR`, `NFS_OPTS`, `MOUNT_OPTS`,
`MOUNT_RETRIES`, `SSH_USER`.

## 4. Acceptance criteria mapping (issue #27)

- [x] "Shared dataset accessible from all Swarm nodes" - validated 2026-09-12
  by the scripted probe exchange in setup_nfs.sh: each of the 3 nodes wrote a
  probe file and every node observed all 3 (`swarm-manager/worker1/worker2`
  each reported 3/3). Export table: `/srv/mpj-share  192.168.122.0/24`.

## 5. Troubleshooting

- `exportfs: /srv/mpj-share requires fsid= for NFS export` -> a stale NFS
  self-mount of the export path was active when `exportfs -ra` ran; an
  NFS-mounted filesystem cannot be re-exported. Fix:
  `sudo umount -f -l /srv/mpj-share && sudo exportfs -ra`.
- `Failed to restart nfs-kernel-server.service: Transaction order is cyclic`
  -> same root cause at the systemd level: the manager's own NFS mount unit
  and the server unit form a dependency loop. Remove the self-mount and its
  fstab line: `sudo sed -i "\#/srv/mpj-share#d" /etc/fstab`, then
  `sudo systemctl daemon-reload`. The current script never self-mounts, so
  this should not recur.
- `setup_nfs.sh: changed: unbound variable` (or `mounted: unbound`) -> the
  ssh heredocs are unquoted, so the LOCAL shell expands their variables and
  `set -u` rejects remote-only names. Any new remote-only variable must be
  written as `\${name}` in the heredoc.
- `mount.nfs: access denied by server while mounting` from a worker -> the
  client IP is outside the export CIDR or mountd holds a stale table; the
  script prints the worker-side `showmount -e <manager>` per failed attempt
  and retries up to 4 times.
- `mount.nfs: Connection timed out` on a worker -> verify `sudo exportfs -v`
  on the manager and that the worker IP falls inside the exported CIDR.
- Writers see `Permission denied` -> confirm `no_root_squash` is present in
  `/etc/exports` on the manager and re-run `sudo exportfs -ra`.

## References

[1] Docker Inc., "Docker volumes: NFS shares," docs.docker.com.
[2] Saleh et al., "MPJ-SPARK Integration-Based Technique to Enhance Big Data
    Analytics in High Performance Computing Environments," IEEE Access, 2025.
[3] Ubuntu manpages, "exports(5) - NFS server export table."
