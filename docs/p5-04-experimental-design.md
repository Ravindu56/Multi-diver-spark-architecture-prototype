# Phase 5-04 Experimental Design — Resource-Constrained Co-location Study

Updated: 2026-10-15. This revision supersedes the 2026-09-15 draft.

## Scope decision

Required Phase 5 experiments use two and four non-root workers across three KVM VMs. Eight-worker execution is excluded from the required evaluation because the available laptop testbed has limited shared resources and preliminary execution was unstable. Existing eight-worker deployment files and failure evidence are retained for historical and optional future use; their presence does not make eight workers an acceptance requirement.

See [the design decision record](p5-four-worker-design-decision.md). This changes the experimental range, not the primary research objectives, and does not mark any experimental issue complete.

## Research question

Within a fixed-capacity, three-VM Docker Swarm testbed, how does increasing non-root worker-process count from two to four affect execution time, resource utilization, synchronization waiting, throughput, and model-state consistency?

This is a co-location/process-density study, not physical-host scale-out.

## Testbed and terminology

- Host: ASUS Vivobook Pro 15 Creator Q533MJ running Ubuntu 24.04 LTS.
- Three KVM VM nodes share one laptop's CPU, memory subsystem, disk, and virtual network.
- Recorded VM profile: two vCPUs and approximately 2.4 GB guest RAM per VM; record actual configuration with every run.
- A worker means a non-root MPI/Spark process, not a separate VM. MPI rank count includes the root.
- Record whether rank 0 also processes a data shard for each application; do not infer compute participation solely from non-root worker count.

## Required experimental cells

| Cell | VM nodes | Non-root workers | MPI ranks | Placement |
|---|---:|---:|---:|---|
| A | 3 | 2 | 3 | Root on manager; one worker on each worker VM |
| B | 3 | 4 | 5 | Root on manager; two workers on each worker VM |

Cell C (eight non-root workers / nine MPI ranks) is excluded from required runs and completion criteria. No repair or successful timing of Cell C is required to close Phase 5.

## Workloads and dataset matrix

- K-Means: workers {2, 4} x dataset sizes {100 MB, 500 MB, 1 GB}: six candidate configurations, before repetitions.
- Logistic regression: workers {2, 4}, with documented dataset and feature-count variations.
- WordCount: deployment and correctness validation. Verify remote-rank compute participation before interpreting any WordCount worker-scaling result.
- Pilot dataset feasibility separately. Four-worker success at 100 MB does not establish feasibility at 500 MB or 1 GB.
- Record dataset checksums, actual bytes, row counts, generation seeds, and feature counts.
- Record every infeasible configuration and its failure evidence. Any dataset reduction requires an explicit documented scope revision; do not silently drop failed cases.

## VM RAM decision (pilot gate before the measured grid)

VM memory is chosen once, from pilot runs, before any measured repetition, and is then held fixed for every cell and workload. The rule below is fixed before pilot data exists. Do not change it after seeing the results.

1. Run pilots and the measured grid in the same host mode: research (headless, `./scripts/host-mode.sh research`). Stage the 500 MB and 1 GB K-Means inputs on the NFS share first (`scripts/stage_datasets.sh`).
2. On the current 2560 MB per-VM profile, pilot K-Means at 500 MB and 1 GB in Cell B first (two JVMs per worker VM, the densest cell), then in Cell A. Use one rep per configuration and skip LogReg:
   ```bash
   ./scripts/bootstrap_stack.sh 4w
   REPS=1 KM_SIZES="500 1024" LR_FEATS= ./scripts/p5_run_cell.sh B
   ./scripts/bootstrap_stack.sh
   REPS=1 KM_SIZES="500 1024" LR_FEATS= ./scripts/p5_run_cell.sh A
   ```
3. Check each pilot directory with `python scripts/p5_pilot_check.py ~/p5-runs/<dir> --csv <dir>/pilot_check.csv`. A pilot passes when it exits 0, starts no new Swarm tasks during the run, and keeps memory "full" pressure stall at or below 5% of the sampled interval on every node. A node with no PSI samples makes the pilot INCOMPLETE, not PASS. The 5% threshold is a proposal for the team to confirm before the pilots run.
4. If every pilot passes, keep 2560 MB. If any pilot fails or is incomplete, resize all three VMs to 3072 MB (`MEM_MB=3072 ./scripts/resize_swarm_vms.sh`, the research-mode profile in `p5-01-low-ram-host-tuning.md`) and repeat steps 2 and 3.
5. If a configuration still fails at 3072 MB, record it as infeasible, with its pilot logs and checker output, under the documented-exclusion rule above. Do not go above 3072 MB per VM: with QEMU overhead and the 2 GB host margin, three VMs at that size reach the sizing limit in `p5-01-low-ram-host-tuning.md`.
6. Record the chosen VM size, the pilot directories and the checker CSVs with the Phase 5 results. Pilots are not measured samples.

## Resource controls and reproducibility

1. Keep VM topology and provisioned vCPUs unchanged across the worker-count comparison. The current profile exposes six virtual CPUs in total; these are not six independent physical cores.
2. Record per-driver Spark cores, Java heap, container CPU/memory limits, MPI placement, and binding. State whether total assigned application CPU budget is fixed or increases with worker count. Fixed VM capacity alone does not imply matched application CPU budgets.
3. Use explicit SLOTS_OVERRIDE and SPARK_DRIVER_MEMORY settings and forward them through mpirun. Record the effective Spark local[N] setting and heap from logs. The recorded co-located run used 480m; the older 400m recommendation is withdrawn because it was rejected by Spark. Heap is not total process memory.
4. Keep datasets, seeds, initialization, algorithm settings, iteration budgets, and stopping rules matched across cells. Record both fixed-budget results and whether convergence tolerance was reached. Do not assume convergence iterations are rank-invariant.
5. Collect at least three successful measured repetitions per feasible retained configuration; report individual runs and a summary with variability. Distinguish smoke tests from measured repetitions.
6. Record background workload and cache policy. Avoid other substantial interactive workloads during measured runs.

## Deployment

```bash
./scripts/bootstrap_stack.sh        # Cell A: root + two workers
./scripts/bootstrap_stack.sh 4w     # Cell B: root + four workers
```

The existing 8w option, if retained by the deployment scripts, is outside the required evaluation. This documentation revision does not remove runtime support or modify scripts.

Use the canonical MPI invocation in [the Swarm invocation notes](p5-swarm-mpi-invocation.md), selecting three ranks for Cell A or five ranks for Cell B, forwarding resource settings, and confirming the overlay CIDR against the deployed network.

## Measurement and validity checks

- Capture end-to-end time, setup/load time, processing/iteration time, per-rank and aggregate CPU/memory behavior, and convergence traces.
- Define aggregate throughput as total processed rows divided by end-to-end time. Logged local-rank throughput is not aggregate throughput.
- Sample both host and guest resource behavior, including available memory, swap activity, CPU contention, I/O waiting, and service restarts. Use ssh -n for background remote samplers.
- Record Linux pressure information where available. A host RAM-use percentage alone does not establish memory exhaustion. `scripts/p5_run_cell.sh` samples `/proc/pressure/{cpu,memory,io}` once per second on the host and every VM (`psi-*.log`, alongside `vmstat-*.log`), and takes `free -m` snapshots before and after each run (`mem-before.txt`, `mem-after.txt`). Use the cumulative `total=` stall counters, last sample minus first, for per-run stall time.
- Separate barrier/straggler waiting from collective-call timing where instrumented. Collective duration can still include arrival imbalance; do not label existing sync_time as pure network communication.
- Reject an unstable run as a timing sample, retain its failure log, and report the exclusion reason.
- Verify synchronized model state across participating ranks and compare final objective values under matched settings as correctness checks.

## Existing evidence — not a completed grid

Issue #82 and the 2026-10-06 carry-over comment on #29 record a completed Cell B K-Means smoke run on the 100 MB input: 1,116,141 rows, 20 iterations, 115.32 seconds total, and rank-identical centroid-shift/WCSS values. It lacked resource sampling and must be repeated for the measured grid. Convergence tolerance was not reached in that recorded run.

The eight-worker deployment converged, but workload execution experienced worker membership flapping, task replacement, and MPI termination. No OOM kill was recorded. The cause was not established; do not describe it as confirmed memory exhaustion or an architectural scalability limit.

## Phase 4 and Phase 6 connections

Compare Phase 5 against Phase 4 with matched workload and explicit resource budgets. Carry the two/four-worker range into Phase 6 characterization, fixed-allocation multi-driver baselines, and allocator evaluation. Keep baseline synchronization policy explicit and do not mix static synchronized and no-synchronization conditions. Apply the corrected MPI binding protocol documented in measurement_notes.md before reusing older timings.

## Threats to validity

All VM nodes share one physical laptop. Virtual CPU scheduling, memory pressure, shared storage, virtual networking, and thermal behavior may affect measurements. Results describe this fixed-capacity virtualized testbed and do not establish absolute performance or scale-out behavior across independent cloud hosts. Larger physical-host and eight-worker validation remain future work.
