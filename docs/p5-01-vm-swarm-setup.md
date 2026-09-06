# P5-01 — KVM VM Swarm Setup (Low-RAM Host Tuning)

Host constraint: a single 14 GB RAM machine running the full 3-node Docker
Swarm cluster (1 manager + 2 workers) as KVM VMs. This document captures the
host-side tuning that makes the P5-01 topology fit safely, without converting
the machine to a server install.

## Sizing rule

Total VM RAM must stay at or below (available RAM - 2 GB). QEMU adds ~5-10%
overhead per VM on top of its configured RAM.

| Host mode           | Usable for VMs | Per-VM RAM | 3-node total | Spark driver memory |
|---------------------|----------------|------------|--------------|---------------------|
| Desktop (GUI on)    | ~9 GB          | 2560 MB    | 7.5 GB       | 2g                  |
| Research (headless) | ~12 GB         | 3072 MB    | 9 GB         | 2.5g                |

Provision VMs with fixed memory (`currentMemory` == max memory, no balloon
shrink) so guest-visible RAM is stable and the Objective 2a CPU/memory
profiling dataset stays comparable across runs.

```bash
MEM_MB=${MEM_MB:-2560}   # use 3072 in research mode
virt-install --name node$i --memory ${MEM_MB} ...
```

## KSM page deduplication

The three VMs run identical Ubuntu cloud images, so Kernel Samepage Merging
typically reclaims 1-2 GB of host RAM by merging duplicate pages. KSM only
shrinks host-side footprint; guest-reported memory stays exact, so per-driver
profiling metrics are not distorted.

Runtime enable:

```bash
echo 1 | sudo tee /sys/kernel/mm/ksm/run
echo 10000 | sudo tee /sys/kernel/mm/ksm/pages_to_scan
```

Persistent enable — `/etc/systemd/system/ksm-tune.service`:

```ini
[Unit]
Description=Enable KSM for KVM page dedup
After=libvirtd.service

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo 1 > /sys/kernel/mm/ksm/run'
ExecStart=/bin/sh -c 'echo 10000 > /sys/kernel/mm/ksm/pages_to_scan'

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ksm-tune.service
```

## Session toggle (no reinstall)

The desktop environment, not the package set, is what costs ~4 GB. Toggle it
per session with `scripts/host-mode.sh`:

```bash
./scripts/host-mode.sh research   # GUI off for the experiment run
./scripts/host-mode.sh desktop    # GUI back afterwards
./scripts/host-mode.sh status     # current mode + headroom
```

Work inside `tmux` on the TTY during research mode; a reboot always restores
the graphical desktop. Do not purge `ubuntu-desktop` or convert via `tasksel` —
the toggle yields the same headroom without losing the research workflow.

## OOM safety

- Keep swap enabled; it is the host's OOM insurance.
- Watch `free -g` during runs; if available drops under 1 GB, gracefully stop
  one VM rather than letting the kernel OOM killer pick a victim.
- Do not rely on virtio balloon overcommit for measurement runs — it perturbs
  the memory-utilization metrics collected for Objective 2a.
