# MPJ-SPARK Multi-Driver Architecture Prototype

> **BScEng Research Prototype — University of Jaffna, EC6070**  
> **v0.2.0 — Phase 2: Iterative ML Workloads with Simulated Allreduce**  
> Implements a cloud-native multi-driver Spark architecture on a single machine using Python `multiprocessing` + PySpark, validating the MPJ-Spark execution model for iterative ML workloads.

---

## Overview

This prototype implements and benchmarks the **multi-driver Spark architecture** described in the state-of-the-art reference paper (Saleh et al., 2025). Each MPJ Worker owns an independent `SparkSession` and processes its data partition in parallel. A Root Process orchestrates the full pipeline — partition, launch, synchronise (Allreduce), collect, aggregate — mirroring how MPJ-Express coordinates processes across HPC cluster nodes.

**Phase 2 extends the architecture from batch analytics (WordCount) to iterative ML workloads** — K-Means clustering and binary Logistic Regression — with per-iteration cross-driver parameter synchronisation via a simulated Queue-based Allreduce.

> **State-of-the-Art Reference:**  
> Saleh et al. (2025). *MPJ-SPARK Integration-Based Technique to Enhance Big Data Analytics in High Performance Computing Environments.* IEEE Access. DOI: [10.1109/ACCESS.2025.3584744](https://doi.org/10.1109/ACCESS.2025.3584744)

---

## Research Objectives Addressed

| Objective | Status |
|---|---|
| **1a** — Adapt multi-driver Spark from HPC/SLURM to containerised cloud (Phase 4) | 🔜 Phase 4 |
| **1b** — Per-iteration cross-driver parameter synchronisation (Allreduce-based) | ✅ Phase 2 — Queue-simulated FedAvg |
| **1c** — Validate on iterative ML workloads (k-means, logistic regression) | ✅ Phase 2 |
| **2a** — Profile CPU/memory across heterogeneous ML workloads | ✅ `results/logreg_iter_metrics.csv` |
| **2b** — Prediction model for per-driver resource demand | 🔜 Phase 6 |
| **2c** — Workload-aware heuristic resource allocation | 🔜 Phase 6 |
| **2d** — Evaluate against single-driver and non-synchronised baselines | ✅ `--compare` flag |

---

## Architecture

### Multi-Driver Execution Model

```
[Root Process]
  │
  ├─ Phase 1 : dynamic_partition()      — O(1) RAM stream-split → N partition files
  │
  ├─ Phase 1b: global seed centroids    — isolated subprocess, 5% sample (k-means only)
  │
  ├─ Phase 2 : launch N workers         — each owns an independent SparkSession(local[K])
  │            JVM pre-warm barrier     — all N JVMs signal ready before timer starts
  │
  ├─ Phase 3 : fire go-signals          — all workers start simultaneously
  │            workers compute          — each processes its partition independently
  │            Allreduce coordinator    — background thread (logreg) or gossip loop (k-means)
  │
  ├─ Phase 4 : collect results          — Queue-based result collection
  │
  ├─ Phase 5 : aggregate                — FedAvg / Hungarian merge across all workers
  │
  └─ Phase 5b: re-assignment pass       — exact centroid correction (k-means + gossip only)
```

### Per-Iteration Allreduce — Two-Queue Design (Phase 2)

```
allreduce_up_queue   — workers → root  (local weight vectors / centroids, per iteration)
allreduce_down_queue — root → workers  (FedAvg-averaged result, broadcast back)
```

Two dedicated queues eliminate the livelock present in a single shared-queue design: the root coordinator can never read back its own broadcast messages. The coordinator runs in a **background thread** fired immediately after worker go-signals, so it is listening before workers push their first iteration.

---

## Supported Workloads

### WordCount (`--app wordcount`)
Batch word frequency aggregation. Validates Phase 1 multi-driver architecture against single-driver baseline.

### K-Means Clustering (`--app kmeans`)
Distributed iterative k-means via Spark MLlib on per-worker partitions.

- **Global seed centroids** — root samples 5% of full dataset in an isolated subprocess and broadcasts k anchor centroids to all workers before training begins, ensuring consistent cluster labelling across workers.
- **Adaptive Gossip aggregation** — root-coordinated gossip simulation over `gossip_queue`. Fanout adapts downward when `drift_ratio < 0.15` to reduce redundant communication. Convergence tracked per round.
- **Exact re-assignment pass** — after gossip, root broadcasts final centroids; each worker runs an assign-only scan and returns per-cluster `(sum, count)`; root computes exact weighted global centroids, eliminating gossip approximation error.

### Logistic Regression (`--app logreg`)
Distributed binary classification via Spark MLlib `LogisticRegression` with per-iteration FedAvg weight-vector Allreduce.

- Each worker runs `maxIter=1` per Allreduce round — one gradient step per synchronisation cycle.
- Per-iteration metrics (`iter_time_s`, `weight_norm`, `weight_delta`, `intercept`, `row_count`) written to `results/logreg_iter_metrics.csv` in append mode — this is the **Objective 2a workload characterisation dataset** for the Phase 6 prediction model.
- Baseline uses parity-iteration control: `baseline_maxIter = num_workers × logreg_iter` for a fair comparison.

---

## Repository Structure

```
mpj_spark/
├── core/
│   ├── file_manager.py          # MPJSparkFileManager — O(1) RAM streaming partition
│   ├── key_value.py             # KeyValueStructure (RDD ↔ MPJ buffer)
│   ├── root_process.py          # Root — 5-phase pipeline + Allreduce coordinator
│   └── gossip_aggregator.py     # Adaptive Gossip aggregation (k-means)
├── workers/
│   ├── spark_session.py         # SparkSession factory — fair local[N] core allocation
│   └── worker_process.py        # Worker — JVM pre-warm + go-signal + Allreduce
├── applications/
│   ├── wordcount.py             # WordCount RDD pipeline
│   ├── baseline_spark.py        # Single-driver WordCount baseline
│   ├── kmeans.py                # K-Means MLlib worker (gossip + re-assign)
│   ├── baseline_kmeans.py       # Single-driver K-Means baseline
│   ├── logreg.py                # LogisticRegression MLlib worker (FedAvg Allreduce)
│   └── baseline_logreg.py       # Single-driver LogReg baseline (parity-iter, OOM-safe)
├── utils/
│   ├── dataset_generator.py     # Synthetic dataset generators (text, numeric, classification)
│   └── dev_logger.py            # Persistent run logger → logs/dev/
└── config.py                    # Central config — TOTAL_CORES, paths, Spark settings
main.py                          # CLI entry point
requirements.txt
```

---

## Installation

**Requirements:** Java 11+, Python 3.8+

```bash
pip install -r requirements.txt
```

---

## Usage

### WordCount

```bash
# Basic multi-driver run
python main.py --app wordcount --workers 4 --generate 200

# With single-driver comparison
python main.py --app wordcount --workers 4 --generate 500 --compare
```

### K-Means Clustering

```bash
# Basic k-means
python main.py --app kmeans --workers 4 --generate 200 \
               --kmeans-k 5 --kmeans-iter 30

# With adaptive gossip aggregation
python main.py --app kmeans --workers 4 --generate 200 --gossip \
               --kmeans-k 5 --kmeans-iter 30 \
               --gossip-threshold 0.001 --gossip-max-rounds 10 --gossip-fanout 2

# Full comparative run
python main.py --app kmeans --workers 4 --generate 500 --compare --gossip \
               --kmeans-k 5 --kmeans-iter 30
```

### Logistic Regression

```bash
# Smoke test (fast)
python main.py --app logreg --workers 2 --generate 10 \
               --logreg-iter 5 --logreg-reg-param 0.01 --logreg-features 10

# Full comparative run
python main.py --app logreg --workers 4 --generate 100 --compare \
               --logreg-iter 15 --logreg-reg-param 0.001 --logreg-features 20

# Objective 2a parameter sweep — accumulates in results/logreg_iter_metrics.csv
for w in 2 3 4; do
  for r in 0.1 0.01 0.001; do
    python main.py --app logreg --workers $w --generate 100 \
                   --logreg-iter 10 --logreg-reg-param $r --logreg-features 10
  done
done
```

### Utility

```bash
# View all past run logs
python main.py --log-history
```

---

## CLI Reference

### Global Flags

| Flag | Default | Description |
|---|---|---|
| `--app NAME` | `wordcount` | Workload: `wordcount`, `kmeans`, `logreg` |
| `--workers N` | `2` | Number of parallel MPJ workers |
| `--generate N` | `None` | Generate synthetic dataset of N MB |
| `--input PATH` | — | Use existing input file (overrides `--generate`) |
| `--compare` | off | Run single-driver baseline and print comparison table |
| `--cores N` | auto | Cores per worker. Default: `TOTAL_CORES ÷ workers` |
| `--no-prewarm` | off | Cold-start mode — include JVM init in wall-clock |
| `--baseline-threads N` | — | Override baseline thread count for fair comparison |
| `--log-history` | off | Print all past run logs and exit |

### K-Means Flags

| Flag | Default | Description |
|---|---|---|
| `--kmeans-k N` | `3` | Number of clusters |
| `--kmeans-iter N` | `20` | Max K-Means iterations per worker |
| `--gossip` | off | Enable adaptive gossip aggregation |
| `--gossip-threshold F` | `0.001` | Convergence drift criterion |
| `--gossip-max-rounds N` | `10` | Hard cap on gossip rounds |
| `--gossip-fanout N` | `2` | Initial peer fan-out (adapts automatically) |

### Logistic Regression Flags

| Flag | Default | Description |
|---|---|---|
| `--logreg-iter N` | `10` | Allreduce iterations (gradient steps per sync round) |
| `--logreg-reg-param F` | `0.01` | L2 regularisation parameter |
| `--logreg-features N` | `10` | Number of feature columns in dataset |

---

## Performance Metrics

| Metric | Captured by |
|---|---|
| Execution time | `time.perf_counter()` around each phase |
| CPU / Memory utilisation | `psutil` per worker per iteration (Phase 6) |
| Synchronisation overhead | Queue `put()` → `get()` round-trip timestamp delta |
| Convergence rate | `‖w_t − w_{t−1}‖₂` per iteration (logreg), centroid drift (k-means) |
| Throughput | Rows processed per second per worker |
| Load vs. processing split | `T_Load` vs `T_Proc` reported separately |

---

## Benchmark Results

### K-Means — 500 MB, 4 Workers, k=3, iter=5, Gossip ON

| Metric | Multi-Driver | Baseline | Speedup |
|---|---|---|---|
| Load Time | 1.43 s | 6.43 s | **4.48×** |
| Proc Time (fit only) | 19.57 s | 11.54 s | 0.59× |
| Re-assign Pass | 19.50 s | — | — |
| Total Wall-clock | 41.11 s | 20.90 s | 0.51× |

> Single-node total wall-clock is expected to be slower. Load time speedup and architectural correctness (gossip convergence in 3 rounds, centroid drift < 0.0001) are the Phase 2 deliverables. Multi-node advantage is demonstrated in Phase 5 (Docker Swarm).

### WordCount — 500 MB, 2 Workers, Pre-warm mode

| Metric | Multi-Driver | Baseline | Speedup |
|---|---|---|---|
| Load Time | 1.22 s | 1.73 s | **1.42×** |
| Avg Worker Proc Time | 6.81 s | 10.51 s | **1.54×** |
| Total Wall-clock | 12.18 s | 30.94 s | **2.54×** |

---

## Branching Strategy

| Branch | Purpose |
|---|---|
| `master` | Stable, tagged releases |
| `dev` | Integration branch — always runnable |
| `release/vX.Y.Z` | Release staging branches |
| `feature/*` | Individual feature / research extensions |

### Phase Roadmap

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Single-machine prototype — WordCount | ✅ `v0.1.0` |
| **Phase 2** | Iterative ML workloads + Queue-simulated Allreduce | ✅ `v0.2.0` |
| **Phase 3** | Real MPI layer — replace Queue with `mpi4py` + OpenMPI | 🔜 |
| **Phase 4** | Docker containerisation — one container per MPI rank, NFS shared volume | 🔜 |
| **Phase 5** | Multi-node Docker Swarm cluster; validate at scale | 🔜 |
| **Phase 6** | ML-aware resource allocator integrated; full comparative evaluation | 🔜 |

---

## Recent Changes (v0.2.0)

| PR | Change | Impact |
|---|---|---|
| [#4](https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/pull/4) | LogisticRegression workload + FedAvg Allreduce | Objective 1b, 1c, 2a |
| [#4](https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/pull/4) | Two-queue Allreduce design | Eliminates single-queue livelock |
| [#4](https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/pull/4) | `results/logreg_iter_metrics.csv` per-iteration profiling | Objective 2a dataset |
| [#3](https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/pull/3) | K-Means + Adaptive Gossip + Re-assignment pass | Objective 1c |
| [#3](https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/pull/3) | Global seed centroid computation (isolated subprocess) | Eliminates centroid label misalignment |
