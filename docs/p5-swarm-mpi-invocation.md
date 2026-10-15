# P5 — Canonical MPI Invocation on the Swarm Stack

**Status:** validated 2026-09-12 · **Scope:** all `mpirun` runs against the `mpj` stack (`docker/swarm/stack-mpj-spark.yml`) · Refs #82, `docs/p5-03-swarm-stack.md`

---

## 1. Why pinning is mandatory on Swarm

Every Swarm task container is multi-homed: it holds an address on the `mpj-net` overlay **and** on the host-local `docker_gwbridge`. The gwbridge subnet is identical on every host (NAT-style, not routable between hosts). Unpinned Open MPI advertises *all* interfaces to peers; remote ranks then attempt BTL connections over gwbridge addresses that can never succeed. The run launches and each rank starts, but the first collective hangs indefinitely.

Deterministic fix: pin both the OOB (out-of-band) and BTL (byte-transfer layer) to the overlay CIDR — `10.0.1.0/24` on `mpj-net`.

Verify the CIDR on the manager (do not hardcode blindly — re-derive if the network is recreated):

```bash
docker network inspect mpj-net -f '{{range .IPAM.Config}}{{.Subnet}}{{end}}'
```

## 2. Canonical invocation

Run from the host, via `docker exec` into the root task:

```bash
ssh ubuntu@<MANAGER_IP> '
ROOT=$(docker ps -q -f name=mpj_mpi-root | head -1)
if [ -z "$ROOT" ]; then echo "NO ROOT CONTAINER — stack not deployed?"; exit 1; fi
docker exec $ROOT bash -lc "
  mpirun -n 3 --hostfile /etc/mpi/hostfile \
    -x PYTHONPATH -x PYTHONUNBUFFERED \
    -mca pml ob1 -mca btl tcp,self \
    -mca oob_tcp_if_include 10.0.1.0/24 -mca btl_tcp_if_include 10.0.1.0/24 \
    -mca plm_rsh_args \"-p 22\" \
    python -m <module> [args...]
"'
```

Component rationale:

| Flag | Why it is required |
|---|---|
| `-x PYTHONPATH -x PYTHONUNBUFFERED` | Remote ranks are spawned over ssh with a bare login env; docker-level env vars are **not** inherited. Without this, ranks 1+ fail with `ModuleNotFoundError: No module named 'mpj_spark'`. |
| `-mca pml ob1 -mca btl tcp,self` | Force the TCP transport (no shared-memory/Infiniband assumptions inside containers). |
| `-mca oob_tcp_if_include 10.0.1.0/24`<br>`-mca btl_tcp_if_include 10.0.1.0/24` | Restrict control-plane and data-plane wiring to the routable overlay only. **The fix for the 2026-09-12 hang.** |
| `-mca plm_rsh_args "-p 22"` | Containers run sshd on port 22; make it explicit. |

The stack already sets `working_dir: /data` and `PYTHONPATH=/app`, so no `cd` or local exports are needed inside the exec.

## 3. Worked examples

### K-Means (Allreduce, Obj 1b/1c)

```bash
ssh ubuntu@<MANAGER_IP> '
ROOT=$(docker ps -q -f name=mpj_mpi-root | head -1)
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p /srv/mpj-share/results/p5-smoke
docker exec $ROOT bash -lc "
  mpirun -n 3 --hostfile /etc/mpi/hostfile \
    -x PYTHONPATH -x PYTHONUNBUFFERED \
    -mca pml ob1 -mca btl tcp,self \
    -mca oob_tcp_if_include 10.0.1.0/24 -mca btl_tcp_if_include 10.0.1.0/24 \
    -mca plm_rsh_args \"-p 22\" \
    python -m mpj_spark.applications.kmeans.allreduce \
      --input /data/input/kmeans_data.csv --k 5 --max-iter 20 \
      --output /data/results/p5-smoke/kmeans-$TS \
    2>&1 | tee /data/results/p5-smoke/kmeans-$TS.log
"'
```

Validated 2026-09-12: 3 ranks × 540k rows, 20 iterations, rank-identical WCSS/shift per iteration, wall 51.37 s, metrics on NFS (`/srv/mpj-share/results/p5-smoke/`). Evidence: #82.

### WordCount (batch, Obj 1c)

Same flags; module/entrypoint per `scripts/validate_p4_05_wordcount.sh` (the `wordcount` module is library-only — do not invoke it with `python -m` directly).

## 4. Hang signature & isolation ladder

**Signature:** the job banner and partitioning logs appear, then silence; `service logs` show repeated Open MPI warnings — *"accepted a TCP connection ... cannot find a corresponding process entry for that peer"* / *"server accept cannot find guid"*. Remote rank processes may exist but the first Barrier/Allreduce never completes.

Diagnose top-down; the first failing step names the layer:

1. **Bare launch** — `mpirun -n 3 --hostfile /etc/mpi/hostfile -mca plm_rsh_args "-p 22" hostname` must print all three hostnames. Hangs here → ssh/orted layer.
2. **Fabric test** — mpi4py Barrier with the canonical flags (rank prints + "barrier crossed" from all ranks). Hangs here → BTL/OOB wiring; re-check `if_include` CIDRs match the current overlay subnet.
3. **Full app** — only after 1 and 2 are green.

Also rule out stale processes: `docker service update --force` recreates tasks and wipes zombie `orted` daemons from crashed runs.

## 5. Known tradeoffs

- `endpoint_mode: dnsrr` (required for stable per-task rank identity) disables port publishing: the 4040–4042 Spark UI mappings exist only under compose. On Swarm, reach Spark UIs via container overlay IPs.
- Published-port removal means no host-side access to in-container dashboards; use log files on NFS instead.
