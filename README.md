# MPJ-SPARK Multi-Driver Architecture Prototype

![CI](https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

> **BScEng Research Prototype — EC6070, University of Jaffna**
> Dayarathna D.D.R.N. (2022E033) · Lawanya M.A.S. (2022E090)
> Supervisor: Dr. J. Jananie, Department of Computer Engineering

Simulates the MPJ-SPARK multi-driver paper architecture on a single machine using Python `multiprocessing` + PySpark, with iterative ML workloads (k-means, logistic regression) and a gossip-based cross-driver parameter synchronisation layer.

---

## State-of-the-Art Reference

> Saleh et al. (2025). *MPJ-SPARK Integration-Based Technique to Enhance Big Data Analytics in High Performance Computing Environments.* IEEE Access. DOI: [10.1109/ACCESS.2025.3584744](https://doi.org/10.1109/ACCESS.2025.3584744)

---

## Research Gap Addressed

The state-of-the-art MPJ-Spark framework is validated only for **non-iterative, HPC-resident batch analytics**. This prototype extends it to:

1. **Iterative ML workloads** (k-means, logistic regression) in containerised cloud environments
2. **Cross-driver global state synchronisation** via gossip-based Allreduce per iteration
3. **ML-aware dynamic resource allocation** driven by workload characterisation

---

## Architecture

```
[Root Process]
  │
  ├─ Phase 1: dynamic_partition()     — O(1) RAM stream-split → N partition files
  │
  ├─ Phase 2: launch N workers        — each owns an independent SparkSession(local[K])
  │           JVM pre-warm barrier    — all N JVMs signal ready before timer starts
  │
  ├─ Phase 3: fire go-signals         — all workers start simultaneously
  │           workers compute         — each processes its partition independently
  │
  ├─ Phase 4: per-iteration sync      — gossip Allreduce aggregates model state
  │           (k-means / logreg)       across all drivers after every iteration
  │
  └─ Phase 5: collect & aggregate     — root merges final results
```

**Deployment targets:**
- **Primary (academic submission):** Docker multi-node cluster, mpi4py + OpenMPI, NFS-backed shared volume
- **Secondary (future work):** Kubernetes with Kubeflow MPI Operator

---

## Implementation Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Single-machine prototype — WordCount via `multiprocessing` + PySpark | ✅ Done |
| 2 | Iterative ML workloads + simulated Allreduce via `Queue` | ✅ Done |
| 3 | Real MPI layer (mpi4py) replacing Queue simulation | 🔄 In Progress |
| 4 | Docker containerisation — one container per MPI rank, NFS shared volume | ⏳ Planned |
| 5 | Multi-node Docker Swarm cluster; validate at scale | ⏳ Planned |
| 6 | ML-aware resource allocator integrated; full comparative evaluation | ⏳ Planned |

---

## Repository Structure

```
mpj_spark/
├── core/
│   ├── file_manager.py         # O(1) RAM streaming partition
│   ├── key_value.py            # KeyValueStructure (RDD ↔ MPJ buffer)
│   ├── gossip_aggregator.py    # Gossip Allreduce — cross-driver parameter sync
│   ├── root_process.py         # Root coordinator — barrier sync + pipeline
│   ├── root_mpi.py             # MPI-native root (Phase 3+)
│   └── main_mpi.py             # MPI entry point
├── workers/
│   ├── spark_session.py        # SparkSession factory — fair local[N] allocation
│   ├── worker_process.py       # Worker — JVM pre-warm + go-signal barrier
│   └── worker_mpi.py           # MPI-native worker (Phase 3+)
├── applications/
│   ├── wordcount.py            # WordCount RDD pipeline
│   ├── kmeans.py               # Distributed k-means (iterative)
│   ├── logreg.py               # Distributed logistic regression (iterative)
│   ├── baseline_spark.py       # Single-driver Spark baseline
│   ├── baseline_kmeans.py      # K-means baseline
│   └── baseline_logreg.py      # Logistic regression baseline
├── benchmarks/
│   ├── timing.py               # TimingCollector (T_Load, T_Init, T_Proc, T_Agg)
│   ├── reporter.py             # Console result + comparison tables
│   └── dev_logger.py           # Persistent run logger → logs/dev/
├── utils/
│   └── dataset_generator.py    # Synthetic dataset generator
└── config.py                   # Central config — cores, paths, Spark settings
main.py                         # CLI entry point
requirements.txt
```

---

## Test Coverage

```
pytest tests/unit/    # 89 tests, ~1.7 s
```

| Module | Coverage |
|---|---|
| `core/file_manager.py` | 94% |
| `core/gossip_aggregator.py` | 96% |
| `core/key_value.py` | 100% |
| `core/root_process.py` | 32% (Phase 3+ paths not yet tested) |
| `config.py` | 100% |

---

## Installation

**Requirements:** Java 11+ · Python 3.8+

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# WordCount — fair comparison, pre-warmed, auto core allocation
python3 main.py --workers 2 --generate 500 --compare

# K-Means
python3 main.py --app kmeans --workers 4 --generate 200 --compare

# Logistic Regression
python3 main.py --app logreg --workers 4 --generate 200 --compare

# Manual core allocation
python3 main.py --workers 2 --generate 500 --compare --cores 3

# Cold-start mode (JVM init included in wall-clock)
python3 main.py --workers 2 --generate 500 --compare --no-prewarm

# View full history of past runs
python3 main.py --log-history
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--workers N` | 4 | Number of parallel MPJ workers |
| `--app NAME` | `wordcount` | Application (`wordcount`, `kmeans`, `logreg`) |
| `--generate N` | 50 | Generate synthetic dataset of N MB |
| `--input PATH` | — | Use an existing input file |
| `--compare` | off | Run single-driver baseline and print comparison |
| `--cores N` | auto | Cores per worker; `0` = unconstrained `local[*]` |
| `--no-prewarm` | off | Cold-start — include JVM init in wall-clock |
| `--no-log` | off | Disable run logging to `logs/dev/` |
| `--log-history` | off | Print summary of all past runs and exit |

---

## Benchmark Results (WordCount, 500 MB, 2 Workers, 22-core machine)

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

---

## Branching Strategy

| Branch | Purpose |
|---|---|
| `master` | Stable, tagged experiment baselines |
| `dev` | Integration branch — always runnable |
| `feature/*` | Individual feature / research extensions |

---

## Key Literature

1. Saleh et al. (2025) — MPJ-Spark, IEEE Access **[State-of-Art]**
2. Theodorakopoulos et al. (2025) — Spark MLlib resource prediction, Algorithms
3. Kofi (2025) — LSTM workload prediction, IJERET
4. Caderno et al. (2025) — BigOPERA elastic Spark allocation, Cluster Computing
5. Zhu et al. (2025) — Rockhopper Spark config tuning, SIGMOD
6. Verma et al. (2025) — DRL Spark scheduling, Journal of Cloud Computing
