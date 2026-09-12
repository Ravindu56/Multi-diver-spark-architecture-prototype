# P5-03 — Multi-Driver Stack on Docker Swarm

Maps to GitHub issue **#28** (Phase 5 — Obj 1a, 1c) — **✅ COMPLETED 2026-09-12.**

## Acceptance evidence

Validated on the 3-node KVM Swarm (swarm-manager Leader + swarm-worker1/2,
Docker 29.1.3), image `mpj-spark:latest` (`sha256:14e2098…`) distributed to
all three VM daemons:

- [x] **docker-compose adapted to docker stack** — `docker/swarm/stack-mpj-spark.yml`
      rendered via `docker compose config` and deployed with
      `docker stack deploy`; all three services converged to `1/1`.
- [x] **Service placement constraints set** — verified node-by-node with
      `docker service ps`: `mpi-root → swarm-manager` (role constraint),
      `mpi-worker-1 → swarm-worker1`, `mpi-worker-2 → swarm-worker2`
      (hostname constraints).
- [x] **Shared-storage round-trip** — probe written by the `mpj_mpi-root`
      container at `/data` visible at `/srv/mpj-share` on all three VM hosts
      (container → bind mount → NFS export → remote nodes).
- [x] **Phase-4 DNS continuity** — `mpi-root` / `mpi-worker-1` /
      `mpi-worker-2` resolve from inside stack containers via network aliases
      on `mpj-net`; entrypoint hostfile logic unchanged.

Issues #28 closed as completed with the full evidence comment.

## What was added

| File | Purpose |
|---|---|
| `docker/swarm/stack-mpj-spark.yml` | Swarm stack: one service per MPI rank, placement-pinned, NFS bind mounts, `mpj-net` overlay |
| `docker/swarm/swarm.env` | Swarm-only tunables (`docker stack deploy` reads no `.env` files) |
| `scripts/build_and_distribute_image.sh` | Registry-less image distribution via `docker save \| ssh → docker load` |
| `scripts/bootstrap_stack.sh` | Idempotent render + deploy + acceptance validation |

Phase-4 files under `docker/` are **not modified** — the single-host compose
stack remains the Obj 2d-i baseline.

## Design

- **Service-per-rank, not replicas.** MPJ semantics require each rank to be
  an identifiable, separately placed container running its own Spark driver.
  `mpi-root` is constrained to `node.role == manager`; `mpi-worker-1/2` are
  pinned by `node.hostname` to `swarm-worker1/2`.
- **DNS continuity with Phase 4.** Each service carries a network alias
  (`mpi-root`, `mpi-worker-1`, `mpi-worker-2`) on `mpj-net`, so the
  entrypoint's hostfile generation (`/etc/mpi/hostfile` from `MPI_SIZE`)
  resolves exactly the same names as on the Phase-4 bridge network.
- **Shared storage.** Every service bind-mounts `/srv/mpj-share`→`/data`.
  Because P5-02 mounts that path from the NFS export on every node, rank
  containers physically on different VMs see one shared filesystem — the
  Lustre-equivalent guarantee of the adapted architecture, now at the
  container level in a cloud-native deployment.
- **Rank-0 liveness.** Stack services have no TTY; the Phase-4 entrypoint
  execs CMD args when present, so `command: ["sleep", "infinity"]` holds
  rank 0 up. Workloads are launched via `docker exec` from the manager,
  the same operator workflow as Phase 4.
- **Distribution without a registry.** Each VM runs its own daemon, so the
  image is built once on the host and pushed to all three daemons via
  `docker save | ssh → docker load`, keyed by image-ID comparison.

## Quick start (host, repo root)

```bash
# 0. VMs up and healthy (P5-01, P5-02) - re-runs are no-ops:
./scripts/bootstrap_swarm.sh
./scripts/setup_nfs.sh

# 1. build (once) + distribute the image to the three VM daemons
chmod +x scripts/build_and_distribute_image.sh scripts/bootstrap_stack.sh
./scripts/build_and_distribute_image.sh

# 2. deploy + validate
./scripts/bootstrap_stack.sh
```

Successful run ends with `P5-03 acceptance criteria met - issue #28`.

## Acceptance criteria mapping (#28)

| Criterion | Where verified |
|---|---|
| docker-compose adapted to docker stack | `docker/swarm/stack-mpj-spark.yml` rendered + deployed by `bootstrap_stack.sh` |
| service placement constraints set | constraints in the stack file; verified node-by-node via `docker service ps` |

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `docker stack deploy` fails: "top-level object must be a mapping" | rendered file empty — caused by `compose config --quiet` (validation-only); fixed in `c9ec345` |
| `docker stack deploy` fails: "published must be a integer" | `compose config` emits quoted ports — unquoted by sed in render step; fixed in `dcdb02d` |
| Service stuck `0/1`; `service ps` shows "no suitable node" | placement constraint mismatch — check `docker node ls --format '{{.Hostname}}'` against the stack file |
| Service stuck `0/1`; "No such image" | VM daemon lacks the image — run `scripts/build_and_distribute_image.sh` |
| `bootstrap_stack.sh` fails image pre-check on one node | stale image on that node only — `NODE_IPS=<ip> ./scripts/build_and_distribute_image.sh` |
| Workers reach `1/1` then mpirun can't resolve peers | DNS aliases — from root container: `getent hosts mpi-worker-1` |

## Deferred (tracked separately)

- **Physical-core hostfile slots** (Phase-4 carry-over): entrypoint still
  derives slots from `nproc` inside the container = VM vCPU count. Harden
  for measurement runs under P5-04.
- Dropped the Phase-4 host-published SSH port (22) on rank 0 — under Swarm,
  mpirun traverses the overlay directly.
