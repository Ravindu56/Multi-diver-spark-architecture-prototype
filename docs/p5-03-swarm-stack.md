# P5-03 — Deploy mpj-spark multi-driver stack onto Docker Swarm (Issue #28)

Status: COMPLETED (acceptance met 2026-09-12) | Branch: feature/p5-01-swarm-cluster | Stack file: docker/swarm/stack-mpj-spark.yml
Goal: Replace the Phase 4 Docker Compose multi-node cluster with a Swarm-native deployment that preserves identical rank semantics (mpi-root, mpi-worker-1, mpi-worker-2), shared storage access at /data, and DNS aliases — while adding Swarm's declarative services, restart policy, and scaling intelligence.

## 1) Compose ↔ Swarm mapping applied

| Phase 4 (Compose) | Phase 5 (Swarm) |
|---|---|
| docker/docker-compose.yml services mpi-root, mpi-worker-{1,2} | One stack mpj with 3 replicated services (replicas: 1) |
| container_name: mpi-worker-1 (fixed identity) | service name mpi-worker-1 + network alias mpi-worker-1 (stable DNS) |
| NFS client volume nfs_mount mounted at /data | NFS export /srv/mpj-share bind-mounted to /data (per-node systemd mount) |
| mpj-net bridge network | mpj-net overlay network (attachable) |
| restart: unless-stopped | deploy.restart_policy: on-failure (delay 5s, max 5 attempts) |
| Manual node selection | deploy.placement.constraints: node.hostname == swarm-{manager,worker1,worker2} |
| Published host ports 4040/4041/4042 | Removed (see 2026-09-12 wiring-parity fix below); access Spark UIs via container overlay IPs |

Rank contracts preserved (entrypoint.sh interface unchanged): MPI_SIZE, MPI_HOSTS, NFS_SERVER, NFS_SHARE, NFS_CONTAINER_SHARE are provided in each service's environment. Each rank keeps hostname == mpi-<role> so the in-image /etc/mpi/hostfile generator and sshd wiring behave exactly as in compose.

## 2) Stack file

See docker/swarm/stack-mpj-spark.yml. Key design decisions:

- Service-per-rank (not replicas=N of one service) to keep per-rank identity, stable aliases, and deterministic hostfile entries; scale-out later increases MPI_SIZE/MPI_HOSTS rather than Swarm replicas.
- Placement constraints pin ranks one-per-VM, mirroring the compose topology (manager hosts mpi-root; workers 1 and 2 on their respective VMs).
- Shared storage by host bind mount: /srv/mpj-share -> /data. Each VM mounts the NFS export locally via the systemd unit set up in swarm_env.sh; this decouples Swarm service lifecycle from NFS client plugins and works with the EL8-native kernel NFS server.
- Overlay network external: true (expected name mpj-net) created once by bootstrap_swarm.sh; stack references it without owning its lifecycle.
- endpoint_mode: dnsrr on all rank services (added 2026-09-12): VIP-mode service DNS load-balances each rank name to a virtual IP, which breaks Open MPI's TCP wireup identity assumptions; dnsrr resolves each rank name directly to its task IP. Side effect: Swarm forbids port publishing under dnsrr, so host ports 4040-4042 are no longer published.
- working_dir: /data + PYTHONPATH=/app + PYTHONUNBUFFERED=1 (added 2026-09-12): mpj_spark.config falls back to the relative ./shared_storage for shard staging and ignores the legacy host-path MPJ_SHARED_STORAGE; the pinned CWD makes shards land on the NFS mount visible to all ranks, with the app at /app kept importable (mpirun -x PYTHONPATH).

## 3) Deployment workflow (run on host)

One command (idempotent – safe to re-run):
```bash
./scripts/bootstrap_stack.sh
```
The script renders, stages, deploys, waits for service convergence, verifies per-service placement, then probes shared storage + DNS alias resolution via an in-stack one-shot command and reports placements + service table.

Manual equivalent:
```bash
set -a; . docker/.env; set +a
SITE=p5-swarm bash -c '
  MANAGER_IP=$(grep -E "^MANAGER_IP=" scripts/swarm_env.sh | cut -d= -f2 | tr -d "\"")
  scp docker/swarm/stack-mpj-spark.yml ubuntu@$MANAGER_IP:/tmp/stack-mpj-spark.yml
  ssh ubuntu@$MANAGER_IP "docker stack deploy -c /tmp/stack-mpj-spark.yml mpj --with-registry-auth"'
```

## 4) Acceptance criteria

- [x] All rank services 1/1 REPLICAS on correct nodes
- [x] Every rank resolves mpi-root, mpi-worker-1, mpi-worker-2 via getent on mpj-net
- [x] /data presents identical content on all ranks (write on mpi-root visible on both workers within 2 s)
- [ ] WordCount via MPI succeeds (P5-05)

## 5) Rollback

```bash
ssh ubuntu@$MANAGER_IP 'docker stack rm mpj'
cd docker && docker compose up -d   # Phase 4 remains intact
```

## 6) Evidence archive location (when accepted)
- docker/swarm/stack-mpj-spark.yml
- docker stack services mpj output, docker service ps per service, getent/NFS probe outputs

## 7) Hand-off note for P5-04..P5-08
Placement constraints give deterministic rank-to-node affinity (required for MPI hostfile correctness); Swarm restart policy replaces compose's restart and adds resilience (P5-07 fault-injection can kill one worker task and observe on-failure restarts + job behaviour). Scale the stack by adding mpi-worker-{3..8} clones with relaxed constraints (spread across the 3 VMs) for the P5-04 topology grid.

## 8) Access the stack (after deploy)
- Rank task containers: ssh ubuntu@<vm>; docker ps -f name=mpj_mpi-<rank>; docker exec as needed (rank 0 hosts mpirun invocations).
- Spark UIs (4040 in each rank container): under endpoint_mode=dnsrr host ports are not published; reach UIs via the container overlay IP (docker inspect) or temporarily re-add ports for debugging.
- Swarm visualizer: http://localhost:8080 (deployed in cluster1 visualizer).

## 9) Issue closure evidence (2026-09-12)

Executed end-to-end on the 3-VM EL8 Swarm from scripts/bootstrap_stack.sh.

- Stack deploy + convergence (all 1/1):
  ID             NAME               MODE         REPLICAS   IMAGE              PORTS
  dol8i2mz8uxg   mpj_mpi-root       replicated   1/1        mpj-spark:latest   *:4040->4040/tcp
  mxdw4th280g6   mpj_mpi-worker-1   replicated   1/1        mpj-spark:latest   *:4041->4040/tcp
  563k94c05r1e   mpj_mpi-worker-2   replicated   1/1        mpj-spark:latest   *:4042->4040/tcp
  (PORTS column reflects the pre-parity-fix revision with published ports.)
- Placement matches constraints:
  mpi-root -> swarm-manager (expected swarm-manager)
  mpi-worker-1 -> swarm-worker1 (expected swarm-worker1)
  mpi-worker-2 -> swarm-worker2 (expected swarm-worker2)
- Shared storage + DNS alias probe from rank 0: dataset staged at /srv/mpj-share/input visible at /data/input from every node; all three rank names resolve on each node.
- Cross-node NFS propagation: file created by rank 0 container visible on both workers and the manager host within 2 s.

Two fixes were required to reach acceptance (both committed on this branch):
- Commit c9ec345 — render pipeline: docker stack config --quiet emitted an empty file on this Docker version; render is captured from compose config stdout with an abort-if-empty sanity check.
- Commit dcdb02d — compose file syntax: quoted host ports ("4040:4040" etc.) were rejected by stack deploy (services.*.ports.0.published must be an integer); unquoted plain scalar mappings.

Later same-day additions (wiring parity for MPI workloads):
- Dataset defaults corrected to the Phase 4 contract (/data/input/kmeans_data.csv, /data/input/logreg_data.csv).
- endpoint_mode=dnsrr + working_dir=/data + PYTHONPATH=/app + SHARED_STORAGE_PATH env hints after smoke-run evidence (shard staging resolved container-locally; VIP-mode DNS incompatible with MPI rank identity; published ports removed as dnsrr consequence).
- Datasets staged to /srv/mpj-share/input (dataset.txt, kmeans_data.csv, logreg_data.csv; verified byte-identical on all three nodes).

Issue #28 closed as completed.
