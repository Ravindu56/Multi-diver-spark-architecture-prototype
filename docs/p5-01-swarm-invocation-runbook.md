# P5-01 MPI Invocation Runbook (Swarm)

All `mpirun` invocations execute **inside the `mpj_mpi-root` container**
on `swarm-manager`. Replacement rules only — everything else stays identical
to Phase 4.

## Pre-flight check (one-liner)

```bash
ssh ubuntu@192.168.122.224 \
  'docker stack services mpj --format "{{.Name}} {{.Replicas}}"'
```
All three (or nine, if extended) services must show `1/1` before proceeding.

## Canonical K-Means smoke run (validation, 2026-09-12 evidence)

```bash
ssh ubuntu@192.168.122.224 '
ROOT=$(docker ps -q -f name=mpj_mpi-root | head -1)
if [ -z "$ROOT" ]; then echo "NO ROOT"; exit 1; fi
mkdir -p /srv/mpj-share/results/p5-smoke/kmeans-$(date +%Y%m%d-%H%M%S)
docker exec $ROOT bash -lc '
  mpirun \
    --oversubscribe --bind-to none -mca hwloc_base_binding_policy none \
    --hostfile /etc/mpi/hostfile -np 3 \
    -x PYTHONPATH -x PYTHONUNBUFFERED \
    -mca pml ob1 -mca btl tcp,self \
    -mca oob_tcp_if_include 10.0.1.0/24 -mca btl_tcp_if_include 10.0.1.0/24 \
    -mca plm_rsh_args "-p 22" \
    python3 -m mpj_spark.applications.kmeans.allreduce \
      --input /data/input/kmeans_data.csv \
      --k 5 --max-iter 20 \
      --output /data/results/p5-smoke/kmeans-$(date +%Y%m%d-%H%M%S)
'
```

## Interface pinning — why it's mandatory

On multi-homed Swarm nodes (10.0.1.0/24 MPI wire + 192.168.122.0/24 NAT/host),
Open MPI may auto-select the 192.168.x interface for both OOB (out-of-band
control) and BTL (MPI data) paths, producing hang-on-wireup symptoms.

The runbook canonical command always pins:

```
-mca oob_tcp_if_include 10.0.1.0/24 -mca btl_tcp_if_include 10.0.1.0/24
```

## Co-located (8-worker) topology — NEW

Enable the extended topology by exporting `MPI_SIZE=9` before `bootstrap_stack.sh`:

| Service | rank | placement | heap |
|---------|------|-----------|------|
| mpi-root | 0 | swarm-manager (exclusive) | default |
| mpi-worker-1 | 1 | swarm-worker1 (exclusive) | default |
| mpi-worker-2 | 2 | swarm-worker2 (exclusive) | default |
| mpi-worker-3 | 3 | swarm-worker1 (co-located) | 400m |
| mpi-worker-4 | 4 | swarm-worker1 (co-located) | 400m |
| mpi-worker-5 | 5 | swarm-worker1 (co-located) | 400m |
| mpi-worker-6 | 6 | swarm-worker1 (co-located) | 400m |
| mpi-worker-7 | 7 | swarm-worker2 (co-located) | 400m |
| mpi-worker-8 | 8 | swarm-worker2 (co-located) | 400m |

The 8-worker form of the K-Means run:

```bash
ssh ubuntu@192.168.122.224 '
ROOT=$(docker ps -q -f name=mpj_mpi-root | head -1)
docker exec $ROOT bash -lc '
  mpirun --hostfile /etc/mpi/hostfile -np 9 \
    -x PYTHONPATH -x PYTHONUNBUFFERED \
    -mca pml ob1 -mca btl tcp,self \
    -mca oob_tcp_if_include 10.0.1.0/24 -mca btl_tcp_if_include 10.0.1.0/24 \
    -mca plm_rsh_args "-p 22" \
    python3 -m mpj_spark.applications.kmeans.allreduce \
      --input /data/input/kmeans_data.csv \
      --k 5 --max-iter 20 --cores 1 \
      --output /data/results/p5-smoke/kmeans-8w-$(date +%Y%m%d-%H%M%S)
'
```

Note `--cores 1` keeps the Spark thread budget (`local[N]`) deterministic
independent of VM vCPU count.

## WordCount driver (unchanged driver, Swarm-mapped flags)

```bash
ssh ubuntu@192.168.122.224 '
ROOT=$(docker ps -q -f name=mpj_mpi-root | head -1)
docker exec $ROOT bash -lc '
  mpirun --hostfile /etc/mpi/hostfile -np 3 \
    -x PYTHONPATH -x PYTHONUNBUFFERED \
    -mca pml ob1 -mca btl tcp,self \
    -mca oob_tcp_if_include 10.0.1.0/24 -mca btl_tcp_if_include 10.0.1.0/24 \
    -mca plm_rsh_args "-p 22" \
    python3 /app/mpj_spark_mpi.py \
      --app wordcount --input /data/input/dataset.txt \
      --compare --results-dir /data/results/p5-smoke/wordcount-$(date +%Y%m%d-%H%M%S)
'
```
