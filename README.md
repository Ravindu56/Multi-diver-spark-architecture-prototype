# MPJ-SPARK Multi-Driver Architecture Prototype

> **BScEng Final Year Research Prototype — EC6070**  
> Department of Computer Engineering · Faculty of Engineering · University of Jaffna  
> **Supervisor:** Dr. J. Jananie

[![Tests](https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/actions/workflows/test.yml/badge.svg)](https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.x-orange)](https://spark.apache.org/)

---

## Project Identity

| Field | Detail |
|---|---|
| **Title** | Resource Analysis and Optimization for Big Data Analytics in Cloud Environments |
| **Students** | Dayarathna D.D.R.N. (2022E033) · Lawanya M.A.S. (2022E090) |
| **Supervisor** | Dr. J. Jananie |
| **Module** | EC6070 — Final Year Research Project |
| **Institution** | University of Jaffna, Sri Lanka |
| **State-of-the-Art Reference** | Saleh et al. (2025) — MPJ-SPARK Integration-Based Technique, IEEE Access. DOI: [10.1109/ACCESS.2025.3584744](https://doi.org/10.1109/ACCESS.2025.3584744) |

---

## Overview

This prototype implements and benchmarks a **cloud-native, ML-aware multi-driver Spark architecture** extended from the MPJ-SPARK reference paper. Each MPJ Worker owns an independent `SparkSession` and processes its data partition in parallel. A Root Process orchestrates the full pipeline — partition, launch, synchronise, collect, aggregate — and a Gossip Aggregator enables cross-driver parameter synchronisation for iterative ML workloads (k-means, logistic regression).

### Key Research Contributions

- **Multi-driver parallelism** delivers measurable speedup over single-driver Spark
- **Fair core allocation** (`local[N]`) makes single-machine comparison scientifically valid
- **JVM pre-warm barrier** cleanly separates initialisation cost from computation cost
- **Gossip Allreduce** (Queue-based, Phase 2) enables iterative ML convergence across independent Spark drivers
- **Hungarian alignment** corrects centroid label permutation across workers before weighted averaging
- **ML-aware resource allocation** (Phase 6) assigns CPU and memory per driver based on predicted workload properties

---

## Research Objectives

### Primary Objectives (Supervisor-Defined)
1. Adopt state-of-the-art architecture for machine learning workload
2. Develop a resource allocation strategy to handle big data in a cluster

### Secondary Objectives

| ID | Objective |
|---|---|
| **1a** | Adapt multi-driver Spark from HPC/SLURM → Docker containerised environment (NFS shared volume) |
| **1b** | Design per-iteration cross-driver parameter synchronisation (Allreduce / parameter-server) |
| **1c** | Validate on iterative ML workloads (k-means, logistic regression) + batch (WordCount) |
| **2a** | Profile CPU and memory across heterogeneous ML workloads → workload characterisation dataset |
| **2b** | Develop lightweight prediction model (LSTM / regression) for per-driver resource demand |
| **2c** | Implement workload-aware heuristic resource allocation strategy |
| **2d** | Evaluate against two baselines: (i) single-driver static, (ii) multi-driver without ML-aware allocation |

---

## Architecture

```
[Root Process]
  │
  ├─ Phase 1: dynamic_partition()         — O(1) RAM stream-split → N partition files
  │
  ├─ Phase 2: launch N workers            — each owns independent SparkSession(local[K])
  │           JVM pre-warm barrier        — all N JVMs signal ready before timer starts
  │
  ├─ Phase 3: fire go-signals             — all workers start simultaneously
  │           workers compute             — each processes its partition independently
  │           [ML] per-iteration sync     — Gossip Allreduce via Queue (Phase 2)
  │                                         MPI Allreduce via mpi4py (Phase 3+)
  │
  ├─ Phase 4: collect results             — Queue-based result collection
  │
  └─ Phase 5: aggregate                   — KeyValueStructure merge (WordCount)
                                            Hungarian-aligned weighted avg (K-Means)
                                            FedAvg / Allreduce weight fusion (LogReg)
```

---

## Phased Implementation Roadmap

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Single-machine prototype — Python multiprocessing + PySpark, WordCount | ✅ Complete |
| **Phase 2** | Iterative ML workloads + simulated Allreduce via Queue (k-means, logreg) | ✅ Complete |
| **Phase 3** | Real MPI layer — mpi4py + OpenMPI replaces Queue simulation | 🔄 In Progress |
| **Phase 4** | Docker containerisation — one container per MPI rank, NFS shared volume | ⏳ Planned |
| **Phase 5** | Multi-node Docker Swarm cluster — validate at scale | ⏳ Planned |
| **Phase 6** | ML-aware resource allocator integrated — full comparative evaluation | ⏳ Planned |

---

## Repository Structure

```
mpj_spark/
├── core/
│   ├── file_manager.py         # MPJSparkFileManager — O(1) RAM streaming partition
│   ├── key_value.py            # KeyValueStructure   (RDD ↔ MPJ buffer)
│   ├── gossip_aggregator.py    # Gossip Allreduce — Hungarian align + weighted avg
│   ├── root_process.py         # Root coordinator — barrier sync + 5-phase pipeline
│   ├── root_mpi.py             # MPI Root — mpi4py-based coordinator (Phase 3+)
│   └── main_mpi.py             # MPI entry point (Phase 3+)
├── workers/
│   ├── spark_session.py        # SparkSession factory — fair local[N] core allocation
│   ├── worker_process.py       # MPJ Worker — JVM pre-warm + go-signal barrier
│   └── worker_mpi.py           # MPI Worker (Phase 3+)
├── applications/
│   ├── wordcount.py            # WordCount RDD pipeline
│   ├── kmeans.py               # Distributed K-Means (iterative ML)
│   ├── logreg.py               # Distributed Logistic Regression (iterative ML)
│   ├── baseline_spark.py       # Single-driver Spark baseline (WordCount)
│   ├── baseline_kmeans.py      # Single-driver K-Means baseline
│   └── baseline_logreg.py      # Single-driver Logistic Regression baseline
├── benchmarks/
│   ├── timing.py               # TimingCollector (T_Load, T_Init, T_Proc, T_Agg)
│   ├── reporter.py             # Console result + comparison tables
│   └── dev_logger.py           # Persistent run logger → logs/dev/
├── utils/
│   └── dataset_generator.py    # Synthetic dataset generator
└── config.py                   # Central config — TOTAL_CORES, paths, Spark settings

tests/
├── unit/
│   ├── conftest.py
│   ├── test_file_manager.py
│   ├── test_file_manager_edge.py
│   ├── test_key_value.py
│   ├── test_gossip_aggregator.py
│   ├── test_root_process.py
│   ├── test_root_process_helpers.py
│   ├── test_baseline_applications.py
│   └── test_spark_session.py
└── pytest.ini

main.py                         # CLI entry point
requirements.txt
LICENSE
```

---

## Performance Metrics

| Metric | Description |
|---|---|
| **Execution Time** | Total wall-clock time per run |
| **CPU Utilisation** | Per-driver CPU usage during processing |
| **Memory Utilisation** | Per-driver heap and off-heap memory |
| **Throughput** | Records processed per second |
| **T_Load / T_Proc** | Partition load time vs. computation time |
| **Synchronisation Overhead** | Cost of cross-driver Allreduce per iteration |
| **Convergence Rate** | Iterations to convergence for iterative ML workloads |

---

## Installation

**Requirements:** Java 11+ · Python 3.8+

```bash
git clone https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype.git
cd Multi-diver-spark-architecture-prototype
pip install -r requirements.txt
```

---

## Usage

### WordCount (Phase 1)

```bash
# Recommended — fair comparison, pre-warmed, auto core allocation
python3 main.py --workers 2 --generate 500 --compare

# Manual core allocation
python3 main.py --workers 2 --generate 500 --compare --cores 3

# Cold-start mode — JVM init included in wall-clock
python3 main.py --workers 2 --generate 500 --compare --no-prewarm
```

### K-Means & Logistic Regression (Phase 2)

```bash
# Distributed K-Means
python3 main.py --app kmeans --workers 4 --generate 200 --compare

# Distributed Logistic Regression
python3 main.py --app logreg --workers 4 --generate 200 --compare
```

### Scaling Sweep

```bash
for w in 1 2 4 8; do
  python3 main.py --workers $w --generate 500 --compare
done
```

### View Run History

```bash
python3 main.py --log-history
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--workers N` | 4 | Number of parallel MPJ workers |
| `--generate N` | 50 | Generate a synthetic dataset of N MB |
| `--input PATH` | — | Use an existing input file |
| `--app NAME` | `wordcount` | Application: `wordcount`, `kmeans`, `logreg` |
| `--compare` | off | Run single-driver baseline and print comparison table |
| `--cores N` | auto | Cores per worker. `0` = unconstrained `local[*]` |
| `--no-prewarm` | off | Cold-start mode — include JVM init in wall-clock |
| `--no-log` | off | Disable automatic run logging |
| `--log-history` | off | Print summary table of all past runs and exit |

---

## Benchmark Results (500 MB WordCount, 2 Workers, 22-core machine)

### Pre-warm mode

| Metric | Multi-Driver | Std Spark | Speedup |
|---|---|---|---|
| Load Time | 1.22 s | 1.73 s | **1.42×** |
| Avg Worker Proc Time | 6.81 s | 10.51 s | **1.54×** |
| Total Wall-clock | 12.18 s | 30.94 s | **2.54×** |
| JVM Pre-warm (T_Init) | 3.04 s | — | excluded from T_Proc |

### Cold-start mode

| Metric | Multi-Driver | Std Spark | Speedup |
|---|---|---|---|
| Load Time | 1.22 s | 1.52 s | **1.24×** |
| Avg Worker Proc Time | 5.24 s | 5.73 s | **1.09×** |
| Total Wall-clock | 10.74 s | 10.33 s | 0.96× (JVM tax) |

> The 0.96× cold-start result quantifies the per-job JVM initialisation tax (~3.23 s/worker) that is amortised in production HPC deployments where Spark drivers are pre-resident on cluster nodes.

---

## Test Suite

```bash
pytest tests/unit/          # 168 tests, ~2.7 s
pytest tests/unit/ --cov    # with coverage report
```

| Module | Coverage |
|---|---|
| `core/key_value.py` | 100% |
| `workers/spark_session.py` | 100% |
| `applications/baseline_logreg.py` | 100% |
| `applications/baseline_spark.py` | 100% |
| `core/gossip_aggregator.py` | 96% |
| `core/file_manager.py` | 95% |
| `core/root_process.py` | 45% |

---

## Branching Strategy

| Branch | Purpose |
|---|---|
| `master` | Stable, tagged experiment baselines |
| `dev` | Integration branch — always runnable |
| `feature/*` | Individual feature / research extensions |
| `release/*` | Release candidates |

---

## Key Literature

1. Saleh et al. (2025) — MPJ-SPARK, IEEE Access **[State-of-the-Art]**
2. Theodorakopoulos et al. (2025) — Spark MLlib resource prediction, Algorithms
3. Kofi (2025) — LSTM workload prediction, IJERET
4. Caderno et al. (2025) — BigOPERA elastic Spark allocation, Cluster Computing
5. Zhu et al. (2025) — Rockhopper Spark config tuning, SIGMOD
6. Verma et al. (2025) — DRL Spark scheduling, Journal of Cloud Computing

---

## License

This project is licensed under the [MIT License](LICENSE).

© 2024–2026 Dayarathna D.D.R.N. & Lawanya M.A.S. — University of Jaffna
