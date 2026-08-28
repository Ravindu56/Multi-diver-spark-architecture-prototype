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
- **throttled** — one worker rank (default rank 1 = worker 0) is pinned
  to cores 0–1 via a generated per-rank `taskset` wrapper
  (`--throttle-method taskset`, the default: raw `sched_setaffinity`, no
  hwloc dependency).  `--throttle-method rankfile` selects OpenMPI
  rankfile slot binding for hosts with a healthy hwloc topology — on the
  lab host hwloc reports "invalid topology information", its slot
  inventory is degraded, and rankfiles beyond slot 15 were rejected at
  launch (np=3 passed, np=5 failed instantly), which motivated the
  taskset default.  The throttled worker's Spark `local[N]` budget is
  unchanged, so the pinned rank is genuinely over-subscribed; this
  induces the heterogeneous resource condition of research gap (iii)
  without application-code changes.

## Procedure

```bash
# 1. verify gates
ruff check . && pytest tests/unit/ -v

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

Fixed config: 540 K rows, 10 features, 10 rounds, reg_param = 0.01.
Duplicate cells from the smoke + full sweeps are averaged below.

### Arm: homogeneous

| Mode | Workers | Wall-clock (s) | Accuracy | Final \|w\| | Sync channel (s) | Mean staleness |
|---|---|---|---|---|---|---|
| none (M1) | 2 | 31.0 | 0.6307 | 0.3228 | — | — |
| none (M1) | 4 | 43.8 | 0.6308 | 0.3228 | — | — |
| ps_sync_fedavg_queue | 2 | 34.4 | 0.6308 | 0.3228 | — | — |
| ps_sync_fedavg_queue | 4 | 44.8 | 0.6308 | 0.3228 | — | — |
| ps_sync_fedavg_mpi | 2 | 34.8 | 0.6308 | 0.3228 | — | — |
| ps_sync_fedavg_mpi | 4 | 44.4 | 0.6308 | 0.3228 | — | — |
| ps_async | 2 | 35.8 | 0.6308 | 0.3228 | — | 0.95 (max 1) |
| ps_async | 4 | 44.2 | 0.6308 | 0.3228 | — | 2.85 (max 4) |
| hybrid_ps_allreduce | 2 | 34.6 | 0.6308 | 0.3228 | 0.41 | — |
| hybrid_ps_allreduce | 4 | 46.7 | 0.6308 | 0.3228 | 1.54 | — |
| gossip | 2 | 36.4 | 0.6308 | 0.3228 | 0.58 | — |
| gossip | 4 | 43.6 | 0.6308 | 0.3228 | 1.09 | — |

### Arm: throttled

| Mode | Workers | Wall-clock (s) | Accuracy | Final \|w\| | Sync channel (s) | Mean staleness |
|---|---|---|---|---|---|---|
| none (M1) | 2 | 48.8 | 0.6307 | 0.3228 | — | — |
| ps_sync_fedavg_queue | 2 | 43.4 | 0.6308 | 0.3228 | — | — |
| ps_sync_fedavg_mpi | 2 | 45.7 | 0.6308 | 0.3228 | — | — |
| ps_async | 2 | 43.0 | 0.6307 | 0.3223 | — | 0.85 (max 2) |
| hybrid_ps_allreduce | 2 | 43.4 | 0.6308 | 0.3228 | 1.66 | — |
| gossip | 2 | 44.3 | 0.6308 | 0.3228 | 2.40 | — |
| none (M1) | 4 | 46.8 | 0.6308 | 0.3228 | — | — |
| ps_sync_fedavg_queue | 4 | 50.7 | 0.6308 | 0.3228 | — | — |
| ps_sync_fedavg_mpi | 4 | 50.3 | 0.6308 | 0.3228 | — | — |
| ps_async | 4 | 51.3 | 0.6308 | 0.3206 | — | 2.45 (max 9) |
| hybrid_ps_allreduce | 4 | 52.0 | 0.6308 | 0.3228 | 5.84 | — |
| gossip | 4 | 51.1 | 0.6308 | 0.3228 | 10.65 | — |

## Findings

- Homogeneous arm: all six modes converge to the identical global model
  (|w| = 0.3228, accuracy 0.6307–0.6308) — convergence is transport-
  invariant.  Only ps_async shows a per-round trajectory (staleness-
  damped mixing; mean staleness 0.95 at w2 → 2.85 at w4); synchronous
  modes sit on the fixed point from round 1.
- Throttled arm (worker 0 pinned to 2 cores via taskset): at 2 workers
  (relative throttle 11→2 cores) wall-clock inflates +20–58% — M1 worst
  (+57.5%: no sync absorption; the job ends with the slowest worker),
  ps_async best (+20.2%: staleness absorbs the straggler).  At 4 workers
  the relative throttle is weaker (6→2 cores) and inflation compresses
  to +7–17%.
- Instrumented sync-channel cost amplifies sharply under heterogeneity:
  hybrid 0.41 → 1.66 s/worker (w2, ×4.0) and 1.54 → 5.84 (w4, ×3.8);
  gossip 0.58 → 2.40 (w2, ×4.1) and 1.09 → 10.65 (w4, ×9.8) — the pinned
  worker's neighbours pay the paired-exchange wait, chaining around the
  ring (Chen et al.'s straggler effect, measured per channel).
- Staleness cost made concrete: ps_async's final model deviates from the
  consensus baseline only under heterogeneity at scale (thr w4:
  |w| = 0.3206, −0.68% vs baseline, max staleness 9; accuracy unchanged,
  0.6308).  Synchronous modes pay wall-clock; async pays model freshness
  (Xie et al. FedAsync; Lian et al. AD-PSGD).
- Caveats: single-machine, compute-bound, convex workload (round-1 fixed
  point); the two throttled cells differ in relative throttle intensity
  (11→2 vs 6→2 cores); M3 excluded pending --include-m3 CLI
  verification; homogeneous duplicate cells from the smoke+full sweeps
  averaged.
