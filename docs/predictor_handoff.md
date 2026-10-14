# Predictor handoff (Objectives 2b and 2c)

What the resource predictor can train and validate on, what to avoid, and what the
allocator expects back. Collected 14 October 2026 on one host (22 CPUs, about 15 GB RAM),
headless, with Open MPI core binding disabled. See `docs/measurement_notes.md` for the
collection caveats and `results/*manifest*.txt` for the exact commits.

## Data inventory

| Set | File | Size | Purpose |
|---|---|---|---|
| A | `results/doe_kmeans.csv` | 250 driver rows, 80 runs | K-Means demand at the 4-core reference allocation |
| B | `results/doe_logreg.csv` | 224 driver rows, 80 runs | LogReg demand at the 4-core reference allocation |
| D | `results/doe_rightsize.csv` | 312 rows, 112 runs, 12 workloads | Core sweep (K-Means 1-4, LogReg 1-8 cores), source of the knee labels |
| D labels | `results/rightsize_labels.csv` | 12 workloads x 2 tolerances | `rec_cpus`, `rec_mem_mb` at 5% and 10% of best wall time |
| E1 | `results/doe_holdout.csv` (set E1) | 22 runs | Holdout sizes: 75 and 125 MB (interior), 250-400 MB (extrapolation) |
| E2 | `results/doe_holdout.csv` (set E2) | 12 runs | Third repetition at each knee, noise ceiling |
| E3 | `results/doe_holdout.csv` (set E3) | 18 runs | VOID, exclude (see caveats) |

All runs exited 0. One row per driver rank plus one root row; use `role == driver`.
L3 and L6 have no 6- and 8-core cells (skipped by the 12 GB memory guard). Anything under
`results/pre_*` was collected before the MPI binding fix and must not be used.

## Columns

- Features: `app`, `size_mb`, `workers`, `iters`, `features`, `k` (K-Means), `cores`.
  Engineered: `shard_mb = size_mb / workers`, `shard_mb x iters`, `shard_mb x features`.
- Targets: `cpu_p95_pct / 100` (cores), `rss_p95_mb` (use max over drivers for memory).
  `cpu_avg_pct`, `cpu_peak_pct`, `rss_peak_mb` and `wall_s` are also available.
- Group keys for splits: `design_id` (Sets A, B) and `workload_id` (Sets D, E).

## Caveats

1. Every run is dominated by about 15-17 s of fixed start-up (estimated 14.8 s K-Means,
   16.7 s LogReg from the intercept of wall time against shard x iterations). Compute time
   is wall time minus start-up. Wall-time knees are therefore loose.
2. The compute-time knee is identifiable for 8 of 12 workloads. K1, K4, L1 and L4 have
   only about 1.0-1.3 s of compute, which is the size of the noise.
3. CPU is capped by thread limits (Spark cores plus numeric-library threads), not by
   cgroups, so it is a soft cap: average CPU can exceed the nominal cores.
4. CPU and memory use grow with the cores given (L5: average CPU 1.97 cores at 1 core, 7.59
   at 8; RSS 2155 MB to 4500 MB). Demand at the 4-core reference is not intrinsic demand.
   Predict as a function of cores, or predict the knee labels.
5. Memory limits were not enforced: a 900 MB allocation survived a 300 MB cap under
   `systemd-run --user`. E3 is void, no memory cliff was measured, and `rec_mem_mb`
   (125% of RSS p95 or 110% of RSS peak, rounded up to 64 MB) is observation-based and
   unvalidated. Test memory limits in Docker (`--memory`) at the allocator integration stage.
6. `pss_*` columns are empty in every row. Use RSS.
7. One host, one job at a time, no contention. Cores above 8 and sizes above 200 MB are
   barely covered; extrapolation rests on 6 E1 runs.

## Validation protocol

- Split by group (`design_id` or `workload_id`), never by row, so repetitions and drivers
  of one configuration stay on the same side.
- Score on E1 for interpolation (75 and 125 MB) and extrapolation (250-400 MB) separately.
- Compare R2 with the noise ceiling from replicate runs (Set D plus E2): about 0.99 for
  `cpu_p95_cores`, 0.97 for `rss_p95_mb`, 0.99 for wall time. The backlog target is
  memory R2 of at least 0.90.
- Reference numbers from an independent cross-check on the same files (ridge, random
  forest, gradient boosting; grouped 5-fold CV on A+B+D): `cpu_p95_cores` R2 0.67 and
  `rss_p95_mb` R2 0.87 for gradient boosting. E1 interior R2 is 0.85-0.96 for CPU and
  0.81-0.89 for memory when trained on A+B; extrapolation is 0.76-0.85 for CPU and 0.77-0.82
  for memory. Use these only to detect leakage or a broken split, not as targets.

## Allocator interface

Input: `app`, `size_mb`, `workers`, `iters`, `features`, `k`.
Output: `cores_per_driver` and `mem_mb_per_driver`.

Candidate rule from the 8 identifiable workloads (compute-time knee at 5%): about 33-50 MB
per driver needs 2 cores and 100 MB or more needs 3, with exceptions (L6 needs 1). Memory
is the RSS-derived `rec_mem_mb`. This comes from 12 workloads, so treat it as a heuristic
to be validated, not a fitted model.
