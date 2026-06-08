# MPJ-SPARK Multi-Driver Architecture Prototype

[![CI](https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/actions/workflows/ci.yml)
[![Lint](https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/actions/workflows/lint.yml/badge.svg?branch=dev)](https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/actions/workflows/lint.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Identity

| Field | Detail |
|---|---|
| **Title** | Resource Analysis and Optimization for Big Data Analytics in Cloud Environments |
| **Module** | EC6070 — BScEng, Department of Computer Engineering, University of Jaffna |
| **Student 1** | Dayarathna D.D.R.N. — 2022E033 |
| **Student 2** | Lawanya M.A.S. — 2022E090 |
| **Supervisor** | Dr. J. Jananie |
| **State-of-the-Art Paper** | Saleh et al. (2025) — MPJ-SPARK Integration-Based Technique to Enhance Big Data Analytics in High Performance Computing Environments. *IEEE Access*. DOI: [10.1109/ACCESS.2025.3584744](https://doi.org/10.1109/ACCESS.2025.3584744) |

---

## Research Gap & Objectives

### Research Gap

Existing scalable multi-driver execution frameworks (e.g., MPJ-Spark) are validated only for non-iterative, HPC-resident batch analytics workloads. They do not address:

- Cross-driver global state synchronisation required for iterative ML algorithms
- ML-aware dynamic resource allocation in containerised cloud environments
- Performance and convergence trade-offs introduced by synchronisation barriers under heterogeneous workload and resource conditions

### Primary Objectives

| # | Objective |
|---|---|
| **O1** | Adopt the state-of-the-art multi-driver Spark architecture and adapt it for iterative ML workloads in containerised cloud environments |
| **O2** | Develop a workload-aware resource allocation strategy to handle big data in a shared cluster |

### Secondary Objectives

| # | Objective | Maps to |
|---|---|---|
| 1a | Adapt and validate the multi-driver Spark execution model from HPC/SLURM to Docker (primary) / Kubernetes (secondary), using NFS shared volume as the functional equivalent of Lustre shared storage | O1 |
| 1b | Design and implement a per-iteration cross-driver parameter synchronisation mechanism (Allreduce-based or parameter-server-based) enabling iterative ML algorithms to converge on a shared global model state | O1 |
| 1c | Validate the adapted architecture on iterative ML workloads (K-Means, Logistic Regression) in addition to batch analytics (WordCount) | O1 |
| 2a | Profile CPU and memory behaviour across heterogeneous ML workloads to build a workload characterisation dataset | O2 |
| 2b | Develop a lightweight prediction model (LSTM or regression-based) for per-driver resource demand estimation | O2 |
| 2c | Implement a workload-aware heuristic resource allocation strategy that dynamically assigns CPU cores and memory to each Spark driver | O2 |
| 2d | Evaluate the full framework against two baselines: (i) single-driver Spark with static allocation, and (ii) multi-driver execution without workload-aware allocation or parameter synchronisation | O2 |

---

## Architecture

```
[Root Coordinator — rank 0]
  │
  ├─ Phase 1: dynamic_partition()
  │           O(1) RAM stream-split → N partition files on NFS shared volume
  │
  ├─ Phase 2: Dispatch N Spark driver workers (ranks 1..N)
  │           Send partition path + config via MPI (TAG_CONFIG)
  │           JVM pre-warm barrier — all N JVMs signal TAG_READY before timer starts
  │
  ├─ Phase 3: Fire go-signals (TAG_GO)
  │           Workers compute independently on their partition
  │           ┌─ WordCount  : RDD map/reduceByKey
  │           ├─ K-Means    : per-iteration local centroid update
  │           └─ LogReg     : per-iteration local gradient update
  │
  ├─ Phase 4: Per-iteration Allreduce sync  [ML workloads only]
  │           Workers send local model → root (TAG_ALLREDUCE_UP)
  │           Root computes FedAvg weighted mean
  │           Root broadcasts global model back (TAG_ALLREDUCE_DOWN)
  │           ── Gossip variant: adaptive peer-to-peer convergence
  │           ── (TAG_REASSIGN_BCAST / TAG_REASSIGN_STATS)
  │
  └─ Phase 5: Collect results (TAG_RESULT / TAG_TIMING)
              Aggregate KeyValueStructure across all workers
              Print comparison table + persist profiling CSVs
```

**Deployment targets:**

| Target | Stack | Status |
|---|---|---|
| Single-machine prototype | Python `multiprocessing` + PySpark | ✅ Phase 1–2 |
| MPI multi-process (single node) | `mpi4py` + OpenMPI + PySpark | ✅ Phase 3 |
| Docker multi-node cluster | Docker Swarm + OpenMPI + NFS volume | 🔧 Phase 4 |
| Kubernetes | Kubeflow MPI Operator (`MPIJob`) + Spark-on-K8s | 📋 Phase 5+ |

---

## Repository Structure

```
mpj_spark/
├── core/
│   ├── file_manager.py       # MPJSparkFileManager — O(1) RAM streaming partition
│   ├── gossip_aggregator.py  # Adaptive gossip Allreduce for K-Means centroid sync
│   ├── key_value.py          # KeyValueStructure   (RDD ↔ MPJ buffer)
│   ├── main_mpi.py           # Phase-3 MPI entry point — rank-dispatch shim
│   ├── root_mpi.py           # MPI root coordinator (rank 0) — replaces root_process
│   └── root_process.py       # Multiprocessing root coordinator (Phase 1–2)
├── workers/
│   ├── spark_session.py      # SparkSession factory — fair local[N] core allocation
│   ├── worker_mpi.py         # MPI worker (ranks 1..N) — Phase-3 driver
│   └── worker_process.py     # Multiprocessing worker — Phase 1–2 driver
├── applications/
│   ├── wordcount.py          # WordCount RDD pipeline
│   ├── kmeans.py             # K-Means iterative ML workload
│   ├── logreg.py             # Logistic Regression iterative ML workload
│   └── baseline_spark.py     # Single-driver Spark baseline (same core budget)
├── benchmarks/
│   ├── timing.py             # TimingCollector (T_Load, T_Init, T_Proc, T_Agg)
│   ├── reporter.py           # Console result + comparison tables
│   └── dev_logger.py         # Persistent run logger → logs/dev/
├── utils/
│   └── dataset_generator.py  # Synthetic dataset generator
└── config.py                 # Central config — TOTAL_CORES, paths, Spark settings

main.py                       # Phase 1–2 CLI entry point (multiprocessing)
mpj_spark_mpi.py              # Phase-3 parity launcher (MPI + Queue adapter)
mpj_spark_prototype.py        # Phase 1 single-file prototype (archived)
mpj_spark_prototype_v2.py     # Phase 2 single-file prototype with ML (archived)
requirements.txt
pytest.ini
```

**Tests:**

```
tests/
├── unit/
│   ├── test_file_manager.py          # MPJSparkFileManager — partition correctness
│   ├── test_root_process.py          # Root process pipeline helpers
│   ├── test_root_process_helpers.py  # Aggregation and timing helpers
│   └── test_worker_process.py        # Worker config and Spark session factory
└── integration/
    └── test_wordcount_pipeline.py    # End-to-end WordCount multi-driver run
```

---

## Phased Implementation Roadmap

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Single-machine prototype — Python `multiprocessing` + PySpark, WordCount | ✅ Complete |
| **Phase 2** | Iterative ML workloads (K-Means, LogReg) + simulated per-iteration Allreduce via `Queue` | ✅ Complete |
| **Phase 3** | Real MPI layer — `mpi4py` + OpenMPI replacing `Queue` simulation; MPI root + worker refactor | ✅ Complete |
| **Phase 4** | Docker containerisation — one container per MPI rank, NFS shared volume, multi-node Docker Swarm | 🔧 In progress |
| **Phase 5** | Multi-node Docker Swarm cluster validation at scale; full comparative evaluation | 📋 Planned |
| **Phase 6** | ML-aware resource allocator integrated; LSTM/regression demand prediction; full O1+O2 evaluation | 📋 Planned |

---

## Test Coverage

| Test Module | What It Covers | Phase |
|---|---|---|
| `test_file_manager.py` | `_count_lines()` accuracy, `dynamic_partition()` count/metadata/completeness/losslessness, `cleanup()` | P1 |
| `test_root_process.py` | Root pipeline: partition dispatch, barrier sync, result collection, aggregation | P1–P2 |
| `test_root_process_helpers.py` | `merge_word_counts()`, `compute_speedup()`, timing helpers | P1–P2 |
| `test_worker_process.py` | Worker config parsing, SparkSession factory, core allocation formula | P1–P2 |
| `test_wordcount_pipeline.py` | End-to-end multi-driver WordCount on a 10 MB synthetic dataset | P1 |

Run all tests:

```bash
pytest tests/ -v
```

Run unit tests only (no Spark required):

```bash
pytest tests/unit/ -v
```

---

## Installation

**Requirements:** Java 11+, Python 3.11+, OpenMPI 4+

```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify OpenMPI (required for Phase 3+)
mpirun --version
```

---

## Usage

### Phase 1–2 — Multiprocessing Entry Point (`main.py`)

```bash
# WordCount — fair comparison, pre-warmed JVM
python3 main.py --workers 4 --generate 500 --compare

# K-Means — 5 clusters, 30 iterations
python3 main.py --workers 4 --generate 500 --app kmeans --kmeans-k 5 --kmeans-iter 30

# Logistic Regression — 15 iterations
python3 main.py --workers 4 --generate 500 --app logreg --logreg-iter 15

# Manual core allocation
python3 main.py --workers 4 --generate 500 --compare --cores 4

# Cold-start mode (JVM init included in wall-clock)
python3 main.py --workers 4 --generate 500 --compare --no-prewarm

# View history of all past runs
python3 main.py --log-history
```

### Phase 3 — MPI Entry Point (`mpj_spark/core/main_mpi.py`)

```bash
# Single-node MPI run (dev / parity test)
mpirun --oversubscribe -np 5 python -m mpj_spark.core.main_mpi \
  --input ./data/dataset.txt --app wordcount

# K-Means with gossip Allreduce
mpirun --oversubscribe -np 5 python -m mpj_spark.core.main_mpi \
  --input ./data/dataset.txt --app kmeans --kmeans-k 5 --gossip

# Multi-node Docker Swarm (Phase 4+)
mpirun --hostfile hostfile.txt -np 5 python -m mpj_spark.core.main_mpi \
  --input /shared/data/dataset.txt --app logreg --logreg-iter 15
```

### Scaling Sweep (speedup curve)

```bash
for w in 1 2 4 8; do
  python3 main.py --workers $w --generate 500 --compare
done
```

---

## CLI Reference

### `main.py` (Phase 1–2)

| Flag | Default | Description |
|---|---|---|
| `--workers N` | `4` | Number of parallel Spark driver workers |
| `--generate N` | `50` | Auto-generate synthetic dataset of N MB |
| `--input PATH` | — | Use an existing input file |
| `--app NAME` | `wordcount` | Workload: `wordcount` \| `kmeans` \| `logreg` |
| `--compare` | off | Run single-driver baseline and print comparison table |
| `--cores N` | auto | Cores per worker and baseline (`0` = unconstrained `local[*]`) |
| `--no-prewarm` | off | Cold-start mode — include JVM init in parallel timer |
| `--no-log` | off | Disable automatic run logging to `logs/dev/` |
| `--log-history` | off | Print summary of all past dev runs and exit |
| `--kmeans-k N` | `3` | Number of K-Means clusters |
| `--kmeans-iter N` | `20` | K-Means maximum iterations |
| `--logreg-iter N` | `10` | Logistic Regression iterations |
| `--logreg-reg-param F` | `0.01` | Logistic Regression regularisation parameter |
| `--logreg-features N` | `10` | Feature vector dimensionality |

### `mpj_spark/core/main_mpi.py` (Phase 3)

| Flag | Default | Description |
|---|---|---|
| `--input PATH` | `./test_dataset.txt` | Path to input dataset |
| `--generate N` | `50` | Auto-generate N MB dataset if `--input` not found |
| `--app NAME` | `wordcount` | Workload: `wordcount` \| `kmeans` \| `logreg` |
| `--cores N` | auto | Override `local[N]` core count per worker |
| `--compare` | off | Run single-driver baseline |
| `--gossip` | off | Use gossip Allreduce for K-Means centroid sync |
| `--kmeans-k N` | `3` | Number of K-Means clusters |
| `--kmeans-iter N` | `20` | K-Means maximum iterations |
| `--logreg-iter N` | `10` | Logistic Regression iterations |
| `--results-dir PATH` | `results` | Directory for profiling CSVs |

---

## Performance Metrics

All runs report the following metrics:

| Metric | Description |
|---|---|
| `T_Load` | Input partition time (O(1) RAM stream split) |
| `T_Init` | JVM initialisation time per worker (excluded from `T_Proc` in pre-warm mode) |
| `T_Proc` | Pure computation time per worker |
| `T_Sync` | Per-iteration Allreduce synchronisation overhead (ML workloads) |
| `T_Total` | Full wall-clock including partition + compute + aggregate |
| `Speedup` | `T_Proc(baseline) / T_Proc(multi-driver)` |
| `Convergence` | Iterations to convergence + final model weight norm (ML workloads) |

---

## Benchmark Results

### WordCount — 500 MB, 4 Workers, 22-core machine

```
mpirun --oversubscribe -np 5 python -m mpj_spark.core.main_mpi \
  --generate 500 --app wordcount --compare --cores 4
```

| Metric | Multi-Driver | Std Spark | Speedup |
|---|---|---|---|
| Load Time | 1.22 s | 1.73 s | **1.42×** |
| Avg Worker Proc Time | 6.81 s | 10.51 s | **1.54×** |
| Total Wall-clock | 12.18 s | 30.94 s | **2.54×** |
| JVM Pre-warm (T_Init) | 3.04 s | — | excluded from T_Proc |

> Results produced on Phase 2 multiprocessing prototype. Phase 3 MPI results will be added upon Phase 4 Docker cluster validation.

---

## Branching Strategy

| Branch | Purpose |
|---|---|
| `master` | Stable, tagged release baselines |
| `dev` | Integration branch — always runnable, always lint-clean |
| `release/*` | Release snapshots (`v0.1.0`, `v0.2.0`) |
| `feature/*` | Individual feature / research extensions |

### Active Feature Branches

| Branch | Research Objective |
|---|---|
| `feature/phase3-mpi4py-openmpi-setup` | Phase 3 — mpi4py + OpenMPI environment |
| `feature/phase3-mpi-root-refactor` | Phase 3 — MPI root coordinator refactor |
| `feature/ml-kmeans-workload` | Objective 1c — K-Means iterative ML workload |
| `feature/ml-logreg-workload` | Objective 1c — Logistic Regression workload |
| `feature/adaptive-gossip-aggregation` | Objective 1b — Gossip Allreduce sync |
| `feature/phase1-phase2-tests` | Phase 1–2 unit and integration tests |
| `feature/ci-workflows` | GitHub Actions CI/lint/test workflows |
| `feature/docs-repo-setup` | This issue — LICENSE, README, branch rules |

---

## Key Literature

| # | Reference | Relevance |
|---|---|---|
| 1 | Saleh et al. (2025). MPJ-SPARK Integration-Based Technique to Enhance Big Data Analytics in HPC Environments. *IEEE Access*. DOI: 10.1109/ACCESS.2025.3584744 | **State-of-the-art** — multi-driver Spark architecture reference |
| 2 | Theodorakopoulos et al. (2025). Resource prediction for Spark MLlib workloads. *Algorithms*. | ML workload resource prediction |
| 3 | Kofi (2025). LSTM-based workload prediction for cloud resource management. *IJERET*. | LSTM demand estimation (Obj 2b) |
| 4 | Caderno et al. (2025). BigOPERA: Elastic Spark resource allocation. *Cluster Computing*. | Elastic allocation strategy (Obj 2c) |
| 5 | Zhu et al. (2025). Rockhopper: Automated Spark configuration tuning. *SIGMOD*. | Configuration optimisation |
| 6 | Verma et al. (2025). Deep reinforcement learning for Spark scheduling. *Journal of Cloud Computing*. | DRL-based scheduling (Obj 2c) |
| 7 | Zhou et al. (2025). StarData: Serverless MapReduce on cloud. *IEEE Access*. | Cloud-native execution model |
| 8 | Kim & Kim (2024). Hadoop data locality with WELM-FF. *Scalable Computing*. | Data locality in distributed frameworks |

---

## License

MIT License — see [LICENSE](LICENSE) for full text.

Copyright © 2026 Dayarathna D.D.R.N. (2022E033) & Lawanya M.A.S. (2022E090),
Department of Computer Engineering, Faculty of Engineering, University of Jaffna.
