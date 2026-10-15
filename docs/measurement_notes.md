# Measurement notes for the v2 data collection

These notes travel with the data produced by `scripts/run_collection.sh`. The exact
code that produced a result is the commit hash in `results/collection_manifest.txt`.

## What is measured

- **Predictor training data** (`results/doe_kmeans.csv`, `results/doe_logreg.csv`):
  per-driver CPU and RSS of K-Means and LogReg runs at one fixed reference allocation
  (4 cores and a fixed heap per driver), default sync mode (FedAvg over MPI).
- **Sync-mode evaluation** (`results/sync_eval_kmeans_v2`, `results/sync_eval_logreg_v2`):
  the four sync modes at 2 and 4 workers, 2 cores per driver. K-Means uses a fixed
  round budget; LogReg uses time to a target loss.
- **Fair single-driver baseline** (`results/baseline_*_v2`, `scripts/baseline_eval.py`):
  the same framework code, algorithm and kernels with ONE driver (np=2) and the same
  total cores as the multi-driver run it is compared with (W drivers x c cores becomes
  1 driver x W*c cores), same data, same round budget. `scripts/compare_baseline.py`
  matches runs on workers, cores per driver and rounds. The stock MLlib baselines behind
  `main_mpi --compare` use a different algorithm and objective and are only a reference.

## Known caveats

1. **Uneven partitions for the baseline.** The loader uses `sparkContext.textFile` with
   no minimum partition count. Measured with `local[4]` on the 100 MB file, the
   single-driver baseline gets 4 partitions of 345937, 345921, 345922 and 43245 rows,
   while a 50 MB half file with `local[2]` gets two balanced partitions of 257744 and
   257729 rows. The baseline's slowest task therefore carries about 1.3x the rows of a
   multi-driver task, which can flatter the multi-driver result by up to about that
   factor. It was not corrected so that the kernels match the predictor data. Report
   speedups against this baseline as an upper bound.
2. **Open MPI core binding (fixed).** With only 2 ranks Open MPI binds each rank to one
   core (measured: ranks saw 2 of 22 CPUs). Every single-driver run was therefore
   confined to one physical core regardless of `--cores`. The pre-fix pilot showed a 2.5x
   round speedup that fell to about 1.1x after the fix. The collection exports
   `OMPI_MCA_hwloc_base_binding_policy=none` and a probe aborts the run unless every
   rank sees all CPUs. All data collected before the fix is archived under
   `results/pre_opt_*` and must not be used.
3. **Short, start-up dominated runs.** About 17 s of a 24 s run is JVM and partition
   start-up. Prefer round-level times and peak or p95 statistics over whole-run averages.
4. **WCSS comparability.** FedAvg and hybrid return the WCSS of one shared model.
   Gossip returns a sum over per-worker models, so a lower value is not better
   clustering. Async is approximate. FedAvg can stop early while the other modes always
   run the full budget (see `rounds_logged`).
5. **One machine, no network.** All drivers share one host (22 CPUs, about 15 GB RAM),
   so synchronisation cost and scaling are not representative of a real cluster.
6. **Stale memory model.** The LogReg DOE drops configurations with a memory estimate
   fitted before the block-cache optimisation (12 GB budget). Check `sys_mem_avail_min_mb`.
7. **Memory parity is assumed.** Heap per driver is 75% of RAM divided by the number of
   workers, so total memory should match between baseline and multi-driver runs. Verify
   from the `[SparkSession] ... heap=` lines in a baseline and a multi-driver `run.log`.
8. **Pilot numbers are not results.** Smoke runs use one repetition and 4 rounds.

## Reproduce

```bash
bash scripts/run_collection.sh run --dry-run   # lists every stage command
bash scripts/run_collection.sh pilot           # tiny run of each stage type
bash scripts/run_collection.sh reset           # only when restarting from scratch
bash scripts/run_collection.sh start --archive-old
bash scripts/run_collection.sh status
```
