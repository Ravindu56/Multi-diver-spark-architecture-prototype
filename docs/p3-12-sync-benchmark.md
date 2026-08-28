# P3-12 — Sync-Strategy Benchmark Protocol (Issue #64)

Controlled comparison of all LogReg synchronization strategies on an
identical workload, capturing the project's metrics: Execution Time,
CPU/Memory Utilization, Synchronization Overhead, and Convergence Rate.

## Fixed configuration

| Parameter | Value |
|---|---|
| Dataset | `shared_storage/logreg_data.csv` (540 K rows) |
| Features | 10 |
| Rounds (`--logreg-iter`) | 10 |
| Local epochs per round | 5 |
| `reg_param` | 0.01 |
| Worker counts | 2 and 4 |

## Modes under test

| Mode | Transport | Sync structure |
|---|---|---|
| `none` (M1) | — | No sync; post-hoc merge |
| `ps_sync_fedavg_queue` (M2) | MPI P2P via adapters | Sync root PS, per-round barrier |
| `ps_sync_fedavg_mpi` | mpi4py gather/bcast | Sync collectives, per-round barrier |
| `allreduce_mpi` (M3) | mpi4py Allreduce | Sync collective (standalone module entry; enable with `--include-m3` after verifying its CLI) |
| `ps_async` (P3-09) | mpi4py P2P | Async root PS, no barrier, staleness-damped |
| `hybrid_ps_allreduce` (P3-10) | Allreduce + P2P | Dense via collective, scalars via PS |
| `gossip` (P3-11) | mpi4py P2P ring | Decentralized; neighbour-only exchange |

## Arms

- **homogeneous** — even core split per worker (default)
- **throttled** — an OpenMPI rankfile pins one worker rank to a reduced
  slot set (default rank 1 -> 2 cores).  The worker's Spark `local[N]`
  budget is unchanged, so the pinned worker is genuinely over-subscribed;
  this induces the heterogeneous resource condition of research gap (iii)
  without any application-code changes.

## Procedure

```bash
# 1. one-time CLI flag for the fanout=1 partial-consensus arm
python scripts/apply_p3_12_wiring.py

# 2. dry-run the plan first
python scripts/run_sync_benchmark.py --dry-run --gossip-fanout 1

# 3. full benchmark (both arms)
python scripts/run_sync_benchmark.py \
  --input ./shared_storage/logreg_data.csv \
  --workers 2 4 --arms homogeneous throttled \
  --logreg-iter 10 --logreg-features 10 --gossip-fanout 1

# 4. analysis
python scripts/analyze_sync_benchmark.py
```

Outputs land in `results/benchmark/`: `manifest.csv` (one row per run),
`analysis/summary.csv`, `analysis/convergence.csv`, `analysis/report.md`,
and `analysis/plots/` (when matplotlib is installed).

## Metric evidence map

| Project metric | Source |
|---|---|
| Execution time | manifest `wall_clock_s` / `elapsed_s` |
| Synchronization overhead | `sync_channel_time_s` (hybrid: allreduce+ps; gossip: gossip_time_s; M3: spark/sync log splits); staleness CSV for `ps_async` |
| Convergence rate | `convergence.csv` (mean/min/max |w| per round across workers) |
| CPU/Memory utilization | per-run worker metrics CSVs + Spark UI / host monitoring during runs |

## Results

### Arm: homogeneous

| Mode | Workers | Wall-clock (s) | Accuracy | Final \|w\| | Sync channel (s) | Mean staleness |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

### Arm: throttled

| Mode | Workers | Wall-clock (s) | Accuracy | Final \|w\| | Sync channel (s) | Mean staleness |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

## Findings (fill after analysis)

- Homogeneous arm: ...
- Throttled arm (rankfile-pinned worker): ...
- Cross-reference vs literature (Chen et al. straggler effect; Xie et al.
  FedAsync staleness damping; Lian et al. AD-PSGD under heterogeneity): ...
