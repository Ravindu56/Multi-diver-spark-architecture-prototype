# Phase 6 resource predictor — progress README

| | |
|---|---|
| Date | 15 October 2026 |
| Branch | `feature/p6-resource-predictor` (fork `SinaliMarasinghe25/Multi-diver-spark-architecture-prototype`) |
| Data | your 14 October 2026 DOE files, collected after the Open MPI binding fix |
| Details | [`p6-evaluation-report.md`](p6-evaluation-report.md) (full results) · [`p6-model-card.md`](p6-model-card.md) (what the predictor is and its limits) |

> **This replaces the earlier progress README.** That one was built on data
> collected before the MPI binding fix, and all its findings are void. In
> particular, do not use these old findings:
> - "LogReg gets slower with more cores";
> - "memory use above Spark's setting";
> - "the BLAS CPU effect";
> - "K-Means CPU levels off at about 2 cores";
> - "use the lookup for CPU".
>
> Section 6 has what the new data shows. All old CSVs and zips were deleted
> and are never used.

## 1. Summary

- **The predictor exists, is tested and is evaluated on your new data
  (objective 2b).** Before a K-Means or LogReg job starts, it predicts each
  Spark driver's CPU (p95 and mean, in cores) and the memory of the largest
  driver (p95 and peak RSS, in MB), each with an upper bound for the
  allocator.
- **Accuracy:** on the frozen test runs the typical error is 0.47 cores for
  CPU p95 and 98 MB for memory p95. That is close to how much identical
  repeat runs differ anyway.
- **Bigger jobs than measured** (your E1 300–400 MB runs): errors stay small,
  but the predictions are a little low, so the allocator must use the upper
  bound there.
- **Transfer learning from Scout was done and tested.** It does not improve
  accuracy: Scout's labels describe whole AWS hosts, not single drivers. Its
  size trend is kept as a fallback.
- **Next:** the allocator (objective 2c), then the final experiment plan
  (2d), which you would run on the laptop.

## 2. What was built (all on the new data)

| Step | Commit | What it does |
|---|---|---|
| rebuild | `9ef4404` | dataset builder, frozen split and predictor rebuilt on the 14 October files |
| Scout size trend | `cf35c35` | can Scout's data-size trend predict bigger jobs? (option A experiment) |
| growth trends | `3dddd19` | demand ∝ x^β beyond the measured data per driver or cores |
| CPU forest per workload | `2f46403` | one random forest per workload for CPU; tests of model behaviour beyond the range |
| A1 | `ce1fe64` | Scout-informed size trend as the default for LogReg memory |
| A2 | `030e5b0` | power-law model for K-Means peak memory |
| A3 | `1131210` | calibrated upper bounds, wider bounds beyond the measured range, out-of-range flag for data per driver |
| A4 | `96b0978` | final evaluation report and model card |
| A2b | `df691bd` | same test for LogReg memory: current model kept |

The main files:

| File | Purpose |
|---|---|
| `mpj_spark/resource_prediction/local_data.py`, `scripts/build_local_dataset.py` | your CSVs → `data/local/local_drivers.csv` (one row per driver) |
| `config/phase6_splits.json` | frozen train / validation / test split (configuration names only, committed) |
| `mpj_spark/resource_prediction/models.py`, `transfer.py`, `calibration.py` | models, Scout trends, upper bounds |
| `mpj_spark/resource_prediction/predictor.py` | **the predictor**: `ResourcePredictor` with `PredictionRequest` → `PredictionResult` |
| `scripts/train_resource_predictor.py` | trains everything and writes `data/models/local_capped/` (model card + models) |
| `scripts/evaluate_resource_predictor.py` | final evaluation and the tables in the docs |
| `scripts/predict_resources.py` | command-line predictions |
| `scripts/*_test*.py`, `scripts/transfer_*.py` | the decision tests; each has its rules in its header |

## 3. Data used

| Your file | Used as | Notes |
|---|---|---|
| `doe_kmeans.csv` (Set A) | training / validation / test | K-Means designs, all at 4 cores |
| `doe_logreg.csv` (Set B) | training / validation / test | LogReg designs, all at 4 cores |
| `doe_rightsize.csv` (Set D) | training / validation / test | the core sweeps (K-Means 1–4, LogReg 1–8) |
| `doe_holdout.csv` E1 | holdout, scored separately | 75 / 125 MB (inside the range), 300 / 400 MB (beyond) |
| `doe_holdout.csv` E2 | noise ceiling only | third repetitions |
| `doe_holdout.csv` E3 | **excluded** | void: memory limits were not enforced |
| `pilot_doe_kmeans.csv`, `pilot_doe_logreg.csv` | **excluded** | pilot runs, not in the handoff inventory |
| `rightsize_labels.csv` | not yet | for checking the allocator (step B) |
| `compare_*_v2.csv`, `pilot_compare_kmeans.csv`, `logreg_iter_metrics.csv`, `doe_logreg_design.csv` | not used by the predictor | kept for later steps |
| anything `pre_*` / the old data | **never** | collected before the MPI binding fix |

- **Split.** Rows are split by configuration group, never by row: 88 groups,
  60 train / 14 validation / 14 test. Groups use `design_id` (A, B) and
  `workload_id` (D); groups that share an exact configuration are merged.
  - The test split is frozen in `config/phase6_splits.json`, so new runs never
    move old groups.
  - No test or E1 configuration appears in training (leakage audit in the
    evaluation report).
- **Labels:**
  - **CPU:** `cpu_p95_pct / 100` and `cpu_avg_pct / 100` per driver.
  - **Memory:** `rss_p95_mb` and `rss_peak_mb`, taking the largest driver of
    each run.
  - Heap is not an input: it was fixed per workload (3,072 MB K-Means,
    6,144 MB LogReg).
- **Scout:** 715 single-node AWS Spark runs (`data/scout/`, built from the
  Scout repository). It is used only for the transfer tests and the fallback
  trend.

## 4. Results

| set | target | typical error (MAE) | R² (noise ceiling) | real use within the upper bound |
|---|---|---|---|---|
| test | cpu_cores_p95 | 0.47 cores (8 %) | 0.925 (0.986) | 85 % |
| test | mem_mb_p95_max | 98 MB (6 %) | 0.942 (0.984) | 93 % |
| test | cpu_cores_mean | 0.11 cores (4 %) | 0.952 (0.995) | 93 % |
| test | mem_mb_peak_max | 100 MB (5 %) | 0.940 (0.986) | 90 % |
| E1 interior | cpu_cores_p95 | 0.42 cores (6 %) | 0.981 (0.986) | 83 % |
| E1 interior | mem_mb_p95_max | 146 MB (7 %) | 0.901 (0.984) | 94 % |
| E1 interior | cpu_cores_mean | 0.17 cores (5 %) | 0.919 (0.995) | 83 % |
| E1 interior | mem_mb_peak_max | 145 MB (6 %) | 0.910 (0.986) | 94 % |
| E1 extrap | cpu_cores_p95 | 0.51 cores (7 %) | 0.981 (0.986) | 100 % |
| E1 extrap | mem_mb_p95_max | 147 MB (6 %) | 0.977 (0.984) | 100 % |
| E1 extrap | cpu_cores_mean | 0.12 cores (3 %) | 0.970 (0.995) | 100 % |
| E1 extrap | mem_mb_peak_max | 155 MB (6 %) | 0.970 (0.986) | 83 % |

- **R² next to the noise ceiling:** the noise ceiling is the best any model
  can do, given how much identical repeats differ.
- **"Real use within the upper bound":** how often the measured value stayed
  under the predictor's upper bound. The target is 90 %.

Weak spots (details and reasons in the evaluation report):
- **K-Means CPU p95:** the upper bound covers 75 % of test runs. Identical
  K-Means runs differ by about ±0.41 cores, which no model can predict.
- **LogReg memory at 75 / 125 MB** (E1): about 206 MB typical error.
- **Bigger jobs than measured:**
  - CPU p95 and memory are under-predicted every time, by about 6–7 %;
  - the upper bound still covered all but one run;
  - only 3 such configurations exist.
- **Memory upper bounds are untested against real limits,** because no memory
  limit was enforced during collection. Step B must test them with
  `docker --memory`.

## 5. Transfer learning (Scout)

- **Scout labels → per-driver labels: does not work (negative result).**
  - Scout measured whole AWS hosts on inputs of at least 769 MiB; our drivers
    see 8–200 MB.
  - Memory errors got much worse with every Scout method, at every amount of
    local data.
  - For CPU, Scout never beat the local random forest.
- **Scout growth trend beyond the measured data size: kept as a fallback.**
  - When local data had no size trend, Scout's trend cut LogReg memory errors
    from 440 to 334 MB.
  - With all of your data, the local and Scout-informed trends agree, so
    today the predictions are the same.
- **Scout's core trend (bigger VMs) does not transfer:** a bigger AWS VM also
  has more memory. It is not used.

So the deployed predictor's accuracy comes from your local data. The thesis
should report the transfer result as negative / neutral, which the
architecture plan allows for.

## 6. What the new data shows (for the thesis)

These numbers come from `data/local/local_drivers.csv` (train + validation +
test rows, all runs OK):

1. **CPU use is not bounded by the core setting.** The thread cap is only a
   soft cap (your caveat 3).

   | Workload | Cores given | Mean CPU | CPU p95 | Highest CPU p95 |
   |---|---|---|---|---|
   | K-Means | 1 | 1.99 | 3.40 | 3.94 |
   | K-Means | 4 | 2.71 | 4.86 | 7.83 |
   | LogReg | 1 | 2.17 | 3.75 | 4.07 |
   | LogReg | 4 | 3.75 | 8.37 | 17.43 |
   | LogReg | 8 | 4.52 | 10.55 | 20.63 |

   At 1 core, both workloads average about 2 cores. At 4 cores, K-Means uses
   on average only 68 % of what it was given. Docker `--cpus` limits will
   therefore throttle short bursts, and step B has to decide whether limits
   follow the p95 or the mean.

2. **Memory grows with the cores given.** These are Set D sweeps, memory p95
   of the largest driver:
   - K-Means: 1,059 MB at 1 core → 1,441 MB at 4 cores.
   - LogReg: 1,606 MB at 1 core → 3,055 MB at 8 cores.

   So demand at the 4-core reference is not intrinsic demand (your caveat 4),
   and the predictor takes cores as an input.

3. **Memory stays well below the JVM heap setting.** Peak RSS is 18–74 % of
   the heap (median 46 % for K-Means and 37 % for LogReg). So the heap setting
   alone would over-reserve memory. This **reverses** the old-data finding.

4. **Growth rates measured on your data** (demand ∝ x^β, β from the local
   rows):

   | | Data per driver | Cores |
   |---|---|---|
   | LogReg memory p95 | β 0.36 | β 0.32 |
   | LogReg CPU p95 | β 0.26 | β 0.50 |
   | K-Means memory p95 | β 0.12 | β 0.22 |
   | K-Means CPU p95 | β −0.04 (flat) | β 0.29 |

5. **K-Means stopped after 2–4 iterations.** Where the count was logged
   (`iters_logged`: 119 rows, all runs with 20 planned iterations), it never
   ran more than 4. The 10- and 40-iteration runs have no count. LogReg always
   ran all its iterations. The predictor uses the planned count, as the
   architecture requires, so this does not leak. It does mean the iteration
   count may have little effect on K-Means (see the questions).

6. **Run-to-run noise.** Identical repeats differ by an SD of 0.36 cores
   (CPU p95) and 99 MB (memory p95). For K-Means CPU p95 it is 0.41 cores.

## 7. Running it on the laptop

Requirements: Python ≥ 3.11 (`pyproject.toml`), plus `requirements.txt`
(scikit-learn, pandas, numpy, scipy). The models were trained with
scikit-learn 1.9.1, pandas 3.0.6 and numpy 2.4.6; train on the laptop
rather than copying model files between machines.

```bash
git remote add sinali https://github.com/SinaliMarasinghe25/Multi-diver-spark-architecture-prototype.git
git fetch sinali feature/p6-resource-predictor
git checkout -b feature/p6-resource-predictor sinali/feature/p6-resource-predictor
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

# 1. your CSVs (data/ is git-ignored)
mkdir -p data/local/raw
cp results/doe_kmeans.csv results/doe_logreg.csv results/doe_rightsize.csv \
   results/doe_holdout.csv data/local/raw/          # never the pre_* files

# 2. build the per-driver table (uses the committed frozen split)
python scripts/build_local_dataset.py               # → data/local/local_drivers.csv

# 3. train (about 50 min on a 4-CPU machine; faster with more cores)
python scripts/train_resource_predictor.py          # → data/models/local_capped/

# 4. evaluate (about 5 min; also gives the latency on the laptop)
python scripts/evaluate_resource_predictor.py       # → data/models/evaluation/tables.md

# 5. predict
python scripts/predict_resources.py --workload kmeans --dataset-mb 100 --workers 2 \
    --cores 2,4 --iterations 20 --features 16 --k 5
```

Example output (from the current model):

```text
cores    CPU p95 (cores)    CPU avg (cores)         mem p95 (MB)        mem peak (MB)  status
    2       4.57 (≤5.25)       2.40 (≤2.58)         1273 (≤1376)         1279 (≤1384)  ok
    4       4.80 (≤5.48)       2.57 (≤2.75)         1537 (≤1640)         1566 (≤1671)  ok
```

The `(≤x)` values are the upper bounds. A request outside the measured range
comes back with status `out_of_domain` and the reason, for example "data per
driver 125 MB … beyond the largest measured 100 MB for kmeans". WordCount
comes back `unsupported`.

From Python:

```python
from mpj_spark.resource_prediction.predictor import PredictionRequest, ResourcePredictor

p = ResourcePredictor.load("data/models/local_capped")
r = p.predict(PredictionRequest("logreg", dataset_mb=100, num_workers=2, cores_per_driver=4,
                                iterations=30, num_features=20))
print(r.to_dict())
```

**Scout is optional.** It is needed only to re-run the transfer experiments:
clone `https://github.com/oxhead/scout` (about 3 GB), then run
`python scripts/build_scout_dataset.py --scout-dir ../scout`. Without it,
training still works, and the Scout-informed trend falls back to the local
trend (today they agree to within 0.001).

**Adding new runs:**
1. Drop the new CSVs into `data/local/raw/`.
2. Re-run steps 2–4.
   - Configurations already in `config/phase6_splits.json` keep their split.
     Only new configurations are assigned, so the test set never changes.
   - Commit the updated split file with the new runs.
   - Duplicate copies of a run count once.

**Tests:** run the Phase 6 suite with

```bash
python -m pytest tests/unit/test_local_data.py tests/unit/test_resource_models.py \
  tests/unit/test_predictor.py tests/unit/test_transfer.py tests/unit/test_transfer_size.py \
  tests/unit/test_evaluate_predictor.py tests/unit/test_scout_build.py
```

## 8. Questions for you

1. **K-Means early stop.** Every K-Means run with 20 planned iterations
   logged only 2–4. Is that convergence (expected), or does the job stop
   early for another reason? Also, why is `iters_logged` empty for the 10-
   and 40-iteration runs? If it is convergence, the "iterations" setting
   barely matters for K-Means demand.
2. **Sync mode.** Which sync mode did the DOE runs use? It is not recorded in
   the CSVs, and the thesis needs to state it.
3. **E1 sizes.** The handoff says the extrapolation runs cover 250–400 MB, but
   the file has only 300 and 400 MB (K-Means on 2 workers, LogReg 300 MB on
   1 worker). Were 250 MB runs planned or dropped?
4. **More runs beyond the range, if there is time.** The extrapolation
   evidence rests on 3 configurations. A few more would make the upper bounds
   much more trustworthy, for example:
   - LogReg 300–400 MB on 2–3 workers;
   - K-Means 250–300 MB on 1 worker.
5. **Docker memory limits.** For step B, can you run a few jobs with
   `docker --memory` set to the predictor's upper bound, to check that the
   limit is honoured and no job is OOM-killed?
6. **Latency on the laptop.** Running step 4 above prints the prediction time
   on the laptop. Please send the "Latency" table for the thesis.

## 9. Next steps and limitations

**Next:**
- **B: the allocator (objective 2c).** Your handoff interface: input `app`,
  `size_mb`, `workers`, `iters`, `features`, `k`; output `cores_per_driver`
  and `mem_mb_per_driver`. Built on the predictor's upper bounds, it will
  have:
  - its own safety margin;
  - extra care for out-of-range jobs;
  - a host-total check;
  - a fallback to static profiles;
  - a cap of at most 4 cores for K-Means;
  - Docker settings;
  - validation against `rightsize_labels.csv`.
- **C: the final experiment plan (2d).** Matched-budget comparisons that you
  would run on the laptop.

**Limitations:**
- one host, one job at a time, no contention;
- small data (88 configuration groups; 3 configurations beyond the range);
- valid only for the measured heaps and thread cap = cores;
- the memory label is RSS (PSS was empty);
- the test split and E1 were looked at while making decisions, so they are
  not blind.

The supervisor decisions needed (memory R² target, the Scout result, a random
forest for CPU, RSS as the label, the 90 % safety level, scope) are listed in
section 11 of the evaluation report.
