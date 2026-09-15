# Phase 5-04 Experimental Design — Co-located Scaling Study

## Research Question

**RQ1:** Under co-located multi-rank placement on shared VM cores, how does Spark-worker throughput and per-iteration convergence time vary as MPI rank count scales from 3 → 9, with total vCPU held constant?

## Experimental Cells

| Cell | Ranks | Topology | Total vCPU | Heap/Rank | Purpose |
|------|-------|----------|------------|-----------|---------|
| **A** | 3 | 1 root + 2 workers (1/VM) | 3 VMs × 2 vCPU = 6 | default | Reference; comparable to Phase 4 |
| **B** | 5 | 1 root + 4 workers (1/VM + 2/VM) | 3 VMs × 2 vCPU = 6 | 400m (co-located) | Co-location penalty |
| **C** | 9 | 1 root + 8 workers (1/VM + 4/VM) | 3 VMs × 2 vCPU = 6 | 400m (co-located) | Scaling limit under contention |

## Isolating Variables

The critical design constraint: increasing rank count **and** co-location simultaneously confounds two effects. To isolate them:

1. **Total vCPU held constant** — no VM resizing between cells. All cells run on the same 3-VM cluster with 2 vCPU per VM.
2. **Heap explicitly capped** — co-located ranks use `SPARK_DRIVER_MEMORY=400m`; dedicated ranks use default (~1.7 GB on 4 GB VMs).
3. **Workload fixed** — same dataset (`kmeans_data.csv`, 81 MB), same `--max-iter 20`, same `--k 5`, same `--cores` override for deterministic Spark `local[N]` budget.
4. **Metrics collected per-iteration** — convergence time per rank, cross-driver Allreduce latency, aggregate throughput (MB/s).

## Deployment Commands

```bash
# Cell A — dedicated (default)
unset MPI_SIZE
docker stack deploy -c docker/swarm/stack-mpj-spark.yml mpj

# Cell B — 5-rank (4 workers)
export MPI_SIZE=5
docker stack deploy -c docker/swarm/stack-mpj-spark.yml -c docker/swarm/stack-mpj-spark.4w.yml mpj

# Cell C — 9-rank (8 workers)
export MPI_SIZE=9
docker stack deploy -c docker/swarm/stack-mpj-spark.yml -c docker/swarm/stack-mpj-spark.8w.yml mpj
```

## Measurement Invocation Matrix

All runs execute inside `mpj_mpi-root`. Example for Cell C (9 ranks):

```bash
mpirun --hostfile /etc/mpi/hostfile -np 9 \
  -x PYTHONPATH -x PYTHONUNBUFFERED \
  -mca pml ob1 -mca btl tcp,self \
  -mca oob_tcp_if_include 10.0.1.0/24 -mca btl_tcp_if_include 10.0.1.0/24 \
  -mca plm_rsh_args "-p 22" \
  python3 -m mpj_spark.applications.kmeans.allreduce \
    --input /data/input/kmeans_data.csv \
    --k 5 --max-iter 20 --cores 1 \
    --output /data/results/p5-04/cell-c-kmeans-$(date +%Y%m%d-%H%M%S)
```

## Expected Findings

- **Hypothesis 1:** Throughput scales sub-linearly from A → B → C due to core contention on co-located ranks.
- **Hypothesis 2:** Convergence (iterations to tolerance) remains rank-invariant; only wall-clock time per iteration increases.
- **Decision point:** If C's throughput < 1.5× A's, the 2560 MB RAM downsize is deferred (OOM risk too high).

## Open Questions

1. Does the co-location penalty distribute uniformly across co-located ranks (workers 3-8 all equally slowed), or does the NFS bottleneck create a straggler hierarchy?
2. Should we add a **Cell B′** at 2560 MB worker VMs to directly test the lower-RAM constraint before committing to the downsize?

---
*Draft 2026-09-15 — to be refined after initial Cell A/B/C runs complete.*
