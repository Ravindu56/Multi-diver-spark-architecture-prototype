# P5-01: Multi-Node Docker Swarm via Host VMs

**Issue:** #26 - **Phase:** 5 - **Objectives:** 1a, 1c
**Topology decision (2026-09-04):** KVM/libvirt VMs on a single physical host -
1 manager + 2 workers joined over the libvirt virtual network.

## 1. Why host VMs

| Option | Real OS isolation | Independent CPU/RAM | Cost | Verdict |
|---|---|---|---|---|
| Single-node swarm | no | no | 0 | stack-file validation only |
| Docker-in-Docker | no (shared kernel) | partial | 0 | smoke tests only |
| **KVM/VirtualBox VMs** | yes | yes | 0 | **chosen** |
| CloudLab / cloud VMs | yes | yes | free (credits) | final-scale validation (future) |

VMs give genuine OS and resource separation, so MPI traffic crosses a real
virtual NIC and per-node CPU/memory limits are meaningful - a requirement for
the synchronization-overhead measurements in P5-06 (#31). Docker-in-Docker
shares the host kernel and pollutes contention metrics; it remains a
smoke-test option only.

## 2. Resource plan

| VM | Role | vCPU | RAM | Disk |
|---|---|---|---|---|
| swarm-manager | Swarm manager, root coordinator, NFS server (P5-02) | 2 | 4 GB | 20 GB |
| swarm-worker1 | Spark driver container | 2 | 4 GB | 20 GB |
| swarm-worker2 | Spark driver container | 2 | 4 GB | 20 GB |

Host budget: >= 16 GB RAM and >= 60 GB free disk recommended. All values are
overridable via environment variables (`WORKER_COUNT`, `VCPUS`, `MEM_MB`,
`DISK_GB`, `VM_PREFIX`).

## 3. Prerequisites (Ubuntu host)

```bash
sudo apt update
sudo apt install -y qemu-kvm libvirt-daemon-system virtinst cloud-image-utils wget
sudo usermod -aG libvirt,kvm "$USER"   # log out and back in
```

Docker is installed inside each VM by cloud-init (`docker.io`). `nfs-common`
is preinstalled on all nodes so P5-02 (#27) only has to configure the NFS
server on the manager.

## 4. Quick start

```bash
./scripts/provision_swarm_vms.sh    # create VMs, writes scripts/.swarm-inventory
./scripts/bootstrap_swarm.sh        # swarm init + join + mpj-net overlay
./scripts/teardown_swarm_vms.sh     # when done
```

Expected end state: `docker node ls` (on the manager) shows 3 nodes with
Status=Ready and Availability=Active; `docker network inspect mpj-net` reports
driver `overlay` with `Attachable=true`. `bootstrap_swarm.sh` re-checks both
and exits non-zero on failure.

## 5. Acceptance criteria mapping (issue #26)

- [ ] `docker swarm init` + worker join on cluster nodes - bootstrap_swarm.sh, steps 1-2
- [ ] Overlay network configured - `mpj-net`, attachable, step 3
- [ ] Both verified automatically by the validation step (step 4), which fails the script otherwise

## 6. Troubleshooting

- `virt-install: unknown OS variant` -> re-run with `OS_VARIANT=ubuntu22.04`
- No DHCP lease after 180 s -> check `virsh net-dhcp-leases default`; cloud-init
  may still be running: `ssh ubuntu@<ip> cloud-init status --wait`
- Permission denied on `virsh` -> ensure libvirt/kvm group membership, or set
  `VIRSH="sudo virsh --connect qemu:///system"`

## 7. Carry-over notes for later Phase 5 issues

- **P5-02 (#27):** install `nfs-kernel-server` on the manager, export
  `/srv/mpj-share` to the worker subnet; adapt the logic from
  `docker/nfs-setup.sh` (currently single-host).
- **P5-03 (#28):** generate the MPI hostfile with physical cores only
  (e.g. hwloc `p:1:h1-10.phys`), not `nproc` - lesson carried from Phase 4.
- **P5-04/05 (#29, #30):** fix the gossip default (`--gossip-fanout` defaults
  to None and crashes) before running gossip at 4-8 workers:
  `fanout = fanout or 1`.
- `scripts/.swarm-inventory` is machine-local; add it to `.gitignore` on first
  use.

## References

[1] Docker Inc., "Run Docker Engine in swarm mode," docs.docker.com.
[2] Docker Inc., "Getting started with swarm mode: tutorial," docs.docker.com.
[3] CloudLab, "Bare-metal cloud infrastructure for research," cloudlab.us.
