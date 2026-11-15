# P5-05 (#30) LogReg grid — checker output, 2026-11-15

Output of `scripts/p5_pilot_check.py <run dir> --all-reps --csv ...` for the LogReg grid
(`REPS=4 KM_SIZES= LR_FEATS="10 50 200" bash scripts/p5_run_cell.sh A|B`). Rep 0 of each
feature count is the pilot; reps 1-3 are measured samples. Every run passed: rc 0, no new
Swarm tasks, memory "full" PSI stall at or below 5%.

| File | Cell | Run directory (on the testbed host) |
|---|---|---|
| `cellA_pilot_check_all.csv` | A: root + 2 workers, 2 cores / 960m per rank | `~/p5-runs/20261115-192531-cellA` |
| `cellB_pilot_check_all.csv` | B: root + 4 workers, 1 core / 480m per rank | `~/p5-runs/20261115-195346-cellB` |

Setup: commit f663536 (image `mpj-spark:p5-campain-1-9-gf663536`), 3072 MB per VM (the VM
RAM gate outcome in `docs/p5-04-experimental-design.md`), research host mode, vCPUs pinned
with `scripts/pin_swarm_vms.sh apply`. Inputs: `logreg_f{10,50,200}_100mb.csv`, checksums in
`results/p5-04/ram-gate/datasets-20261115.txt`. Epochs 30 max, lr 0.01, tol 1e-4.

These CSVs hold the gate criteria only. Timing and convergence values come from the
per-rank `logreg_rank*_epochs.csv` / `logreg_rank*_run.json` files in each run directory.
`sync_time_s` there covers MPI calls only and `loss_time_s` is the Spark loss pass (the
timing fix in f663536; see `docs/measurement_notes.md`, caveat 9). All three VMs share one
laptop: these are co-location results, not multi-host scale-out or networked sync cost.
