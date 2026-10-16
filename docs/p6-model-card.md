# Model card — Phase 6 per-driver resource predictor

| | |
|---|---|
| Model version | `capped-4fbf2d6566c2` (model card schema 4, `data/models/local_capped/model_card.json`) |
| Code | `mpj_spark/resource_prediction/` (`predictor.py` is the entry point); training in `scripts/train_resource_predictor.py` |
| Objective | 2b: estimate per-driver CPU and memory demand before launch; it feeds the allocator (2c) |
| Date | 15 October 2026 |
| Full evaluation | [`docs/p6-evaluation-report.md`](p6-evaluation-report.md) |

## In short

Give it a job description before the job starts (workload, data size,
workers, cores per driver, planned iterations, features, k). It returns, per
Spark driver:
- CPU p95 and CPU mean, in cores;
- memory p95 and memory peak of the largest driver, in MB;
- an upper bound for each of the four, for the allocator;
- where each number came from;
- whether the request is outside the measured range, and why.

On the frozen test split its typical error is 0.47 cores (CPU p95) and 98 MB
(memory p95), close to the run-to-run noise of identical repeats.

## Intended use

- **Use:** per-job, pre-launch estimates of driver demand for K-Means and
  LogReg on the multi-driver Spark / MPI / Docker prototype, on the host it
  was measured on. The allocator (step B) turns them into Docker limits, with
  its own safety factor and a host-total check.
- **Do not use:**
  - for WordCount or other workloads: they come back `unsupported`, and the
    allocator uses its static fallback;
  - for other hosts or VMs without re-measuring;
  - with a heap or thread cap other than the measured ones;
  - as a memory cap without the allocator's margin;
  - for predicting run time or model accuracy.

## How it works

| target | workload | model | growth trend beyond range | upper-bound method |
|---|---|---|---|---|
| cpu_cores_p95 | kmeans | local_rf_wl | cores: local | additive_rows |
| cpu_cores_p95 | logreg | local_rf_wl | model's own | relative_groups |
| mem_mb_p95_max | kmeans | local_ridge_wl | cores: local | additive_rows |
| mem_mb_p95_max | logreg | local_ridge_wl | size: combined | additive_rows |
| cpu_cores_mean | kmeans | local_rf_wl | cores: local | additive_rows |
| cpu_cores_mean | logreg | local_rf_wl | model's own | relative_groups |
| mem_mb_peak_max | kmeans | local_ridge_log_wl | cores: local | additive_rows |
| mem_mb_peak_max | logreg | local_ridge_wl | size: combined | additive_rows |

- **One entry point** (`ResourcePredictor`). It routes each target and
  workload to the model above. Every choice was made by a test on train +
  validation fixed in advance (see "Decision record"). The models:
  - `local_rf_wl`: a random forest per workload;
  - `local_ridge_wl`: a Ridge per workload;
  - `local_ridge_log_wl`: a per-workload Ridge on log memory (a power law).
- **Upper bound in range:** the 0.9 quantile of out-of-fold errors, per
  workload. It is additive, except for LogReg CPU, which is relative.
- **Beyond the largest measured data per driver or cores:**
  - the growth trend shown above is applied (`local` = β from local data;
    `combined` = the Scout-informed β);
  - the upper bound gets a stress-test floor;
  - the request is flagged out of range.

| target | beyond the largest measured | workload | floor | held-out rows (groups) |
|---|---|---|---|---|
| cpu_cores_p95 | data per driver | kmeans | prediction + 28 % | 72 (15) |
| cpu_cores_p95 | data per driver | logreg | prediction + 55 % | 20 (5) |
| cpu_cores_p95 | cores | kmeans | prediction + 32 % | 36 (5) |
| cpu_cores_p95 | cores | logreg | prediction + 49 % | 24 (4) |
| mem_mb_p95_max | data per driver | kmeans | prediction + 15 % | 38 (15) |
| mem_mb_p95_max | data per driver | logreg | prediction + 21 % | 20 (5) |
| mem_mb_p95_max | cores | kmeans | prediction + 9 % | 20 (5) |
| mem_mb_p95_max | cores | logreg | prediction + 12 % | 16 (4) |
| cpu_cores_mean | data per driver | kmeans | prediction + 37 % | 72 (15) |
| cpu_cores_mean | data per driver | logreg | prediction + 57 % | 20 (5) |
| cpu_cores_mean | cores | kmeans | prediction + 11 % | 36 (5) |
| cpu_cores_mean | cores | logreg | prediction + 53 % | 24 (4) |
| mem_mb_peak_max | data per driver | kmeans | prediction + 12 % | 38 (15) |
| mem_mb_peak_max | data per driver | logreg | prediction + 20 % | 20 (5) |
| mem_mb_peak_max | cores | kmeans | prediction + 10 % | 20 (5) |
| mem_mb_peak_max | cores | logreg | prediction + 12 % | 16 (4) |

## Inputs and valid ranges

Inputs are pre-launch only. The inputs are:
- `workload_type`, `dataset_mb`, `num_workers` (drivers), `cores_per_driver`;
- `iterations` (planned), `num_features`, `num_clusters` (k, K-Means only).

Data per driver (`dataset_mb / num_workers`) and its products with iterations
and features are derived inside the predictor.

Missing or invalid inputs are rejected. A request is flagged out of range
when:
- any input lies outside the measured values below; or
- the data per driver exceeds the largest measured (100 MB for K-Means,
  200 MB for LogReg), even when each input alone is in range.

Values between measured ones are interpolated, with a note.

| workload | input | measured values |
|---|---|---|
| kmeans | dataset_mb | 25, 50, 100, 150, 200 |
| kmeans | num_workers | 1, 2, 3 |
| kmeans | cores_per_driver | 1, 2, 3, 4 |
| kmeans | iterations | 10, 20, 40 |
| kmeans | num_features | 8, 16, 32 |
| kmeans | num_clusters | 3, 5, 10 |
| logreg | dataset_mb | 25, 50, 100, 150, 200 |
| logreg | num_workers | 1, 2, 3 |
| logreg | cores_per_driver | 1, 2, 3, 4, 6, 8 |
| logreg | iterations | 10, 30, 60 |
| logreg | num_features | 10, 20, 40 |
| logreg | num_clusters | 0 |

Fixed during all measurements, so the predictor is valid only for:
- heap: 3,072 MB per K-Means driver and 6,144 MB per LogReg driver;
- thread cap = cores per driver.

## Training data

| experiment | split | driver rows | runs | configuration groups |
|---|---|---|---|---|
| A | test | 24 | 12 | 6 |
| A | train | 118 | 56 | 28 |
| A | val | 28 | 12 | 6 |
| B | test | 24 | 14 | 7 |
| B | train | 102 | 54 | 27 |
| B | val | 18 | 12 | 6 |
| D | test | 40 | 16 | 2 |
| D | train | 124 | 76 | 8 |
| D | val | 36 | 20 | 2 |
| E1 | holdout | 34 | 22 | 11 |
| E2 | holdout | 22 | 12 | 12 |

- **Where it was measured:** one host (22 CPUs, about 15 GB RAM), headless,
  one job at a time, on 14 October 2026, after the Open MPI binding fix.
  - Sets A and B: K-Means and LogReg designs (size, workers, iterations,
    features, k), all at 4 cores.
  - Set D: core sweeps of 12 workloads.
  - E1: holdout sizes, at 4 cores. E2: third repetitions, used for the
    noise ceiling.
  - E3 is void; pilot files and any data from before the fix are never used.
- **Labels:**
  - CPU is the per-driver p95 and mean of the sampled CPU (core-equivalents,
    sampled about every 0.59 s).
  - Memory is the p95 and peak RSS of the largest driver of each run (PSS was
    empty).
- **Split:** by configuration group, never by row (60 train / 14 validation /
  14 test groups). Model selection and upper bounds use
  leave-one-configuration-out CV on train + validation only.

## Performance

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

The noise ceiling is the R² reached by the mean of replicate runs: no model
can do better on average. Details per workload, per number of workers and
per cores, with confidence intervals and the transfer ablation, are in the
evaluation report.

## Transfer learning

Scout (715 AWS single-node Spark runs) was audited and tested both as a
source of labels and as a source of growth trends.
- **Labels:** Scout's labels describe whole AWS hosts running inputs of at
  least 769 MiB. Transferring them to per-driver labels made memory
  predictions worse at every training-set size, and did not beat the local
  random forest for CPU. Negative result; not deployed.
- **Growth trends:** Scout's size trend lowered LogReg memory errors where
  local data had no trend. With the current data it equals the local trend,
  so it is deployed as an automatic fallback (`TREND_POLICY` size/logreg:
  `combined`). Scout's core trend does not transfer for K-Means and is not
  used.

## Limitations and risks

- **Beyond the measured range the point predictions are biased low**
  (E1 extrap: every CPU p95 and memory row under-predicted). Use the upper
  bound there. This was checked on only 3 configurations.
- **K-Means CPU p95 upper bound covers 75 % on test** (target 90 %), because
  identical K-Means runs vary by about 0.41 cores. CPU over-use only
  throttles.
- **Memory upper bounds have not been tested against real limits.** 7–10 % of
  test runs were above them, and no memory limit was enforced during
  measurement. The allocator must add a margin, and step B must validate
  with `docker --memory`.
- **LogReg memory at 75–125 MB** (E1 interior) has the largest in-range error:
  206 MB.
- **Scope:** one host, no contention, one sync mode (not recorded in the
  data), small data (88 configuration groups).
- **Not blind:** the test split and E1 were looked at during steps A1–A3.

## Decision record

| Decision | Script (protocol in its header) | Output |
|---|---|---|
| split groups and model selection | `scripts/train_resource_predictor.py` | `data/models/local_capped/` |
| CPU forest per workload (`FAMILY_POLICY`) | `scripts/per_workload_rf_test.py` | `data/models/per_workload_rf/` |
| CPU model beyond the measured data per driver (fix 1) | `scripts/extrapolation_model_tests.py` | `data/models/extrapolation/` |
| K-Means peak memory power law (`WORKLOAD_FAMILY_POLICY`, A2) | `scripts/kmeans_memory_test.py` | `data/models/kmeans_memory/` |
| LogReg memory keeps `ridge_wl` (A2b: power law clearly worse beyond the measured cores) | `scripts/logreg_memory_test.py` | `data/models/logreg_memory/` |
| Scout label transfer | `scripts/transfer_experiment.py` | `data/models/transfer_capped/` |
| Scout size trend | `scripts/transfer_size_trend.py` | `data/models/transfer_size/` |
| growth trends (`TREND_POLICY`, A1) | `scripts/transfer_trend_tests.py` | `data/models/transfer_trends/` |
| upper-bound methods and floors (`MARGIN_POLICY`, A3) | `scripts/upper_bound_tests.py` | `data/models/upper_bounds/` |
| final evaluation (A4) | `scripts/evaluate_resource_predictor.py` | `data/models/evaluation/` |

## Versions and use

Trained with Python 3.11.15, scikit-learn 1.9.1, pandas 3.0.6 and numpy 2.4.6.
The predictor adds a note when the scikit-learn version differs at load time.

```python
from mpj_spark.resource_prediction.predictor import PredictionRequest, ResourcePredictor

p = ResourcePredictor.load("data/models/local_capped")
r = p.predict(PredictionRequest("kmeans", dataset_mb=100, num_workers=2, cores_per_driver=4,
                                iterations=20, num_features=16, num_clusters=5))
r.cpu_cores_p95, r.cpu_cores_p95_upper, r.mem_mb_p95_max, r.mem_mb_p95_max_upper, r.status, r.reasons
```

Command line: `python scripts/predict_resources.py --help`.
