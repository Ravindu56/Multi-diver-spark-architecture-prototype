# MPJ-SPARK Multi-Driver Architecture Prototype

> **BScEng Research Prototype — University of Jaffna, EC6070**
> **Phase 3 — Real MPI Layer + Native MPI FedAvg (P3-08)**
> Implements a cloud-native multi-driver Spark architecture using `mpi4py` + OpenMPI + PySpark, validating the MPJ-Spark execution model for iterative ML workloads with pluggable cross-driver synchronization policies.

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

This prototype implements and benchmarks the **multi-driver Spark architecture** described in the state-of-the-art reference paper (Saleh et al., 2025). Each MPJ Worker owns an independent `SparkSession` and processes its data partition in parallel. A Root Process orchestrates the full pipeline — partition, launch, synchronise (Allreduce), collect, aggregate — mirroring how MPJ-Express coordinates processes across HPC cluster nodes.

**Phase 2 extends the architecture from batch analytics (WordCount) to iterative ML workloads** — K-Means clustering and binary Logistic Regression — with per-iteration cross-driver parameter synchronisation via a simulated Queue-based Allreduce.

> **State-of-the-Art Reference:**
> Saleh et al. (2025). *MPJ-SPARK Integration-Based Technique to Enhance Big Data Analytics in High Performance Computing Environments.* IEEE Access. DOI: [10.1109/ACCESS.2025.3584744](https://doi.org/10.1109/ACCESS.2025.3584744)

---

## Research Objectives Addressed

| Objective | Status |
|---|---|
| **1a** — Adapt multi-driver Spark from HPC/SLURM to containerised cloud (Phase 4) | 🔜 Phase 4 |
| **1b** — Per-iteration cross-driver parameter synchronisation (Allreduce-based) | ✅ Phase 2 — Queue-simulated FedAvg · Phase 3 — native MPI Allreduce + FedAvg (P3-08) |
| **1c** — Validate on iterative ML workloads (k-means, logistic regression) | ✅ Phase 2 |
| **2a** — Profile CPU/memory across heterogeneous ML workloads | ✅ `results/logreg_iter_metrics.csv` |
| **2b** — Prediction model for per-driver resource demand | 🔜 Phase 6 |
| **2c** — Workload-aware heuristic resource allocation | 🔜 Phase 6 |
| **2d** — Evaluate against single-driver and non-synchronised baselines | ✅ `--compare` flag |

---

## Architecture

### Multi-Driver Execution Model

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
  │           ── P3-08: native collective FedAvg (comm.gather / comm.bcast)
  │              over the worker sub-communicator (sync_mode=ps_sync_fedavg_mpi)
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
│   ├── sync_modes.py         # P3-08: central sync-mode registry (none / queue / mpi / allreduce)
│   ├── main_mpi.py           # Phase-3 MPI entry point — rank-dispatch shim
│   ├── root_mpi.py           # MPI root coordinator (rank 0) — replaces root_process
│   └── root_process.py       # Multiprocessing root coordinator (Phase 1–2)
├── workers/
│   ├── spark_session.py      # SparkSession factory — fair local[N] core allocation
│   ├── worker_mpi.py         # MPI worker (ranks 1..N) — Phase-3 driver
│   └── worker_process.py     # Multiprocessing worker — Phase 1–2 driver
├── applications/
│   ├── wordcount.py          # WordCount RDD pipeline
│   ├── kmeans/
│   │   ├── allreduce.py      # K-Means MPI Allreduce (Phase 3)
│   │   ├── driver.py         # Parity facade (dual-signature MPI/local)
│   │   ├── partition.py      # K-Means data partition + centroid init
│   │   └── metrics.py        # K-Means metrics collector
│   ├── logreg/
│   │   ├── allreduce.py      # LogReg MPI Allreduce — per-iteration SGD (Phase 3)
│   │   ├── fedavg_mpi_run.py # P3-08: LogReg periodic FedAvg over native MPI collectives
│   │   ├── queue_run.py      # LogReg Queue-based worker (Phase 2 — deprecated)
│   │   ├── nosync_run.py     # LogReg no-sync worker — M1 benchmark condition
│   │   ├── driver.py         # Parity facade (dual-signature MPI/local)
│   │   ├── partition.py      # LogReg data partition
│   │   └── metrics.py        # LogReg metrics collector
│   ├── baseline_kmeans.py    # Single-driver K-Means baseline
│   └── baseline_logreg.py    # Single-driver LogReg baseline (auto-detects CSV header)
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
scripts/
├── generate_datasets.py      # Generate fixed K-Means + LogReg datasets (run once)
├── run_docker.sh             # P4-09: Docker cluster lifecycle and validation launcher
├── validate_p4_05_wordcount.sh # P4-05 Docker WordCount acceptance validation
├── validate_p4_06_kmeans.sh  # P4-06 Docker K-Means acceptance validation
├── validate_p4_07_logreg.sh  # P4-07 Docker LogReg acceptance validation
├── validate_p4_09.sh         # P4-09 deployment acceptance checks
├── validate_parity.py        # Issue #10 — baseline Spark vs MPI parity validation
├── sync_overhead_benchmark.py # Issue #12 — MPI multi-driver vs baseline sync benchmark
└── timing_analysis.py        # Phase 3 timing decomposition + controller feature matrix
requirements.txt
pyproject.toml
pytest.ini
```

**Tests:**

```
tests/
├── unit/
│   ├── conftest.py                          # Shared fixtures
│   ├── test_file_manager.py                 # MPJSparkFileManager — partition correctness
│   ├── test_file_manager_edge.py            # Edge cases — empty file, single-line, unicode
│   ├── test_root_process.py                 # Root process pipeline helpers
│   ├── test_root_process_helpers.py         # Aggregation and timing helpers
│   ├── test_spark_session.py                # SparkSession factory — core allocation
│   ├── test_key_value.py                    # KeyValueStructure serialisation
│   ├── test_gossip_aggregator.py            # Gossip Allreduce correctness
│   ├── test_baseline_applications.py        # Baseline K-Means + LogReg return shapes
│   ├── test_sync_modes.py                   # P3-08 sync-mode registry
│   └── test_fedavg_mpi_run.py               # P3-08 FedAvg MPI helpers + mock collectives
├── phase3/
│   ├── test_kmeans_allreduce.py             # K-Means MPI Allreduce — guarded @NEEDS_MPI
│   ├── test_kmeans_convergence.py           # K-Means convergence over iterations
│   ├── test_kmeans_metrics.py               # K-Means metrics collector
│   ├── test_kmeans_partition.py             # K-Means partition + centroid init
│   ├── test_mpi_verify.py                   # MPI environment sanity checks
│   └── test_wordcount_mpi_vs_baseline.py    # Issue #14 — WordCount MPI regression test
└── logreg/
    ├── test_allreduce.py                    # LogReg Allreduce + convergence check
    ├── test_local_gradient.py               # Gradient computation (real local[1] Spark)
    └── test_metrics.py                      # LogRegMetricsCollector
```

---

## Phase 3 Prerequisites

> **Phase 3 requires OpenMPI installed at the OS level.** `mpi4py` is the Python binding — it links against the system OpenMPI shared library and cannot install it via pip.

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y openmpi-bin libopenmpi-dev
```

### macOS (Homebrew)

```bash
brew install open-mpi
```

### Docker (per-container)

Add to your `Dockerfile`:

```dockerfile
RUN apt-get update && apt-get install -y \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*
```

### Verify installation

```bash
mpirun --version           # e.g. Open MPI 4.1.x
python -c "from mpi4py import MPI; print('MPI OK, version:', MPI.Get_version())"
```

> **Note — Phase 2 Queue simulation is deprecated.** `mpj_spark/applications/logreg/queue_run.py` remains in the codebase as the M2 benchmark condition (Queue FedAvg) for comparative evaluation (Objective 2d), but it is no longer the primary execution path. Phase 3 `mpi4py` + OpenMPI is the current production execution model, with two selectable synchronisation policies: per-iteration Allreduce (`allreduce.py`) and periodic FedAvg over native MPI collectives (`fedavg_mpi_run.py`, P3-08).

---

## Phased Implementation Roadmap

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Single-machine prototype — Python `multiprocessing` + PySpark, WordCount | ✅ Complete |
| **Phase 2** | Iterative ML workloads (K-Means, LogReg) + simulated per-iteration Allreduce via `Queue` | ✅ Complete |
| **Phase 3** | Real MPI layer — `mpi4py` + OpenMPI replacing `Queue` simulation; MPI root + worker refactor; K-Means + LogReg Allreduce validated; FedAvg over native MPI collectives (P3-08) | ✅ Complete |
| **Phase 4** | Docker containerisation — one container per MPI rank, NFS shared volume, multi-node Docker Swarm | 🔧 In progress |
| **Phase 5** | Multi-node Docker Swarm cluster validation at scale; full comparative evaluation | 📋 Planned |
| **Phase 6** | ML-aware resource allocator integrated; LSTM/regression demand prediction; full O1+O2 evaluation | 📋 Planned |

```
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

## Installation

**Requirements:** Java 11+, Python 3.11+, OpenMPI 4+ (required for Phase 3 — see [Phase 3 Prerequisites](#phase-3-prerequisites))

```bash
# 1. Install OpenMPI at OS level (see Phase 3 Prerequisites above)

# 2. Install Python dependencies
pip install -e .

# 3. Verify
mpirun --version
python -c "from mpi4py import MPI; print('MPI OK')"
python -c "import pyspark; print('PySpark', pyspark.__version__)"
```

---

## Usage

### Phase 3 — MPI Entry Point (primary execution model)

#### Step 1 — Generate datasets (run once before any mpirun command)

```bash
python scripts/generate_datasets.py
```

This creates fixed-seed datasets for K-Means and LogReg under `data/` (paths defined in `mpj_spark/config.py`).

#### Step 2 — Run workloads with `mpirun -n 5`

**K-Means — MPI Allreduce, N=5 workers:**

```bash
# Default: k=3, max_iter=20
mpirun -n 5 python -m mpj_spark.applications.kmeans.allreduce

# Custom: k=5, 30 iterations
mpirun -n 5 python -m mpj_spark.applications.kmeans.allreduce \
  --k 5 --max-iter 30

# With gossip Allreduce variant
mpirun --oversubscribe -n 5 python -m mpj_spark.core.main_mpi \
  --input ./data/kmeans_dataset.csv --app kmeans --kmeans-k 3 --gossip
```

**LogReg — MPI Allreduce (FedAvg), N=5 workers:**

```bash
# Default: 30 epochs
mpirun -n 5 python -m mpj_spark.applications.logreg.allreduce

# Custom epochs
mpirun -n 5 python -m mpj_spark.applications.logreg.allreduce \
  --epochs 50
```

**LogReg — Periodic FedAvg over native MPI collectives (P3-08):**

```bash
# Native MPI FedAvg (default sync mode on main_mpi.py)
mpirun --oversubscribe -np 3 python -m mpj_spark.core.main_mpi \
  --app logreg --sync-mode ps_sync_fedavg_mpi \
  --input ./shared_storage/logreg_data.csv \
  --logreg-iter 10 --logreg-features 10

# Legacy Queue-transport FedAvg fallback (M2 over MPI P2P adapters)
mpirun --oversubscribe -np 3 python -m mpj_spark.core.main_mpi \
  --app logreg --sync-mode ps_sync_fedavg_queue \
  --input ./shared_storage/logreg_data.csv \
  --logreg-iter 10 --logreg-features 10
```

> **FedAvg over Native MPI (P3-08, Issue #65).** Phase 3 FedAvg aggregation runs entirely on the real mpi4py transport — `comm.gather()` of per-worker `(weights, intercept, row_count)` to the aggregator rank and `comm.bcast()` of the row-weighted global model back, once per synchronisation round (every E local epochs). No `multiprocessing.Queue` or go-signal simulation is involved when `--sync-mode ps_sync_fedavg_mpi` is selected. Validated 2026-08-16 (540 K rows, 10 rounds, reg_param=0.01): identical convergence across transports (|w| = 0.3228, accuracy = 0.6308 for both `ps_sync_fedavg_mpi` and `ps_sync_fedavg_queue`), native-MPI scaling verified at 2, 3, and 4 workers. Per-iteration metrics are tagged via the `sync_mode` column in `results/logreg_iter_metrics.csv`.

**WordCount — MPI multi-driver, N=5 workers:**

```bash
mpirun --oversubscribe -n 5 python -m mpj_spark.core.main_mpi \
  --input ./data/dataset.txt --app wordcount
```

#### Step 3 — Data integrity validation (Issue #10)

```bash
# Full parity check: baseline single-driver Spark vs MPI multi-driver
# Runs K-Means + LogReg, writes results/parity_report.csv
mpirun -n 5 python scripts/validate_parity.py

# K-Means only
mpirun -n 5 python scripts/validate_parity.py --skip-logreg

# LogReg only
mpirun -n 5 python scripts/validate_parity.py --skip-kmeans

# Custom tolerance and parameters
mpirun -n 5 python scripts/validate_parity.py \
  --tolerance 1e-3 --kmeans-k 3 --logreg-epochs 30

# View report
column -t -s, results/parity_report.csv
```

#### Step 4 — Sync overhead benchmark (Issue #12)

```bash
# Full benchmark: MPI multi-driver vs single-driver baseline
# Reads existing metrics CSVs + runs baselines, writes results/sync_overhead_benchmark.csv
# NOTE: Run workloads first (Step 2) so metrics CSVs are present in metrics/
mpirun --oversubscribe -n 5 python scripts/sync_overhead_benchmark.py

# K-Means only
mpirun --oversubscribe -n 5 python scripts/sync_overhead_benchmark.py --skip-logreg

# LogReg only
mpirun --oversubscribe -n 5 python scripts/sync_overhead_benchmark.py --skip-kmeans

# Custom parameters
mpirun --oversubscribe -n 5 python scripts/sync_overhead_benchmark.py \
  --kmeans-k 3 --kmeans-iter 20 --logreg-epochs 14

# View report
column -t -s, results/sync_overhead_benchmark.csv
```

#### Step 5 — WordCount MPI regression test (Issue #14)

```bash
# Run only in full MPI environment (skipped automatically in CI)
mpirun --oversubscribe -n 5 pytest tests/phase3/test_wordcount_mpi_vs_baseline.py -v
```

### Phase 1–2 — Multiprocessing Entry Point (`main.py`)

> Phase 2 Queue simulation (`--sync queue`) is retained as benchmark condition M2 for Objective 2d comparative evaluation.

```bash
# WordCount — fair comparison, pre-warmed JVM
python3 main.py --workers 4 --generate 500 --compare

# K-Means — 5 clusters, 30 iterations
python3 main.py --workers 4 --generate 500 --app kmeans --kmeans-k 5 --kmeans-iter 30

# Logistic Regression — M1 (no sync)
python3 main.py --workers 4 --generate 500 --app logreg --sync none

# Logistic Regression — M2 (Queue FedAvg)
python3 main.py --workers 4 --generate 500 --app logreg --sync queue

# Logistic Regression — M3 (MPI Allreduce)
python3 main.py --workers 4 --generate 500 --app logreg --sync mpi

# Logistic Regression — B2 (standalone Spark baseline, fair benchmark)
python3 main.py --workers 4 --generate 500 --app logreg --compare \
  --baseline-master spark://spark-master:7077

# View history of all past runs
python3 main.py --log-history
```

### Logistic Regression

```bash
for w in 1 2 4 8; do
  python3 main.py --workers $w --generate 500 --compare
done
```

---

## Docker Deployment — Phase 4

Phase 4 deploys the MPI-enabled multi-driver Spark prototype as a
three-container Docker cluster:

- `mpi-root`: MPI launcher and coordinator
- `mpi-worker-1`, `mpi-worker-2`: Spark/MPI worker containers
- Shared Docker-mounted `/data` storage for input datasets, metrics, and results
- OpenMPI over the Docker bridge network and the generated MPI hostfile

### Prerequisites

- Docker Engine 24+ recommended
- Docker Compose v2
- At least 8 GB available RAM; 16 GB is recommended for the K-Means and
  Logistic Regression validation workloads

### Quick Start

```bash
chmod +x scripts/run_docker.sh scripts/validate_p4_09.sh

# Build images and start the Docker MPI cluster
./scripts/run_docker.sh up

# Inspect services
./scripts/run_docker.sh status

# Validate Docker deployment configuration
./scripts/validate_p4_09.sh
```

### Phase 4 Workload Validation

```bash
# P4-05: WordCount Docker validation
./scripts/run_docker.sh validate-p4-05

# P4-06: K-Means convergence validation
./scripts/run_docker.sh validate-p4-06

# P4-07: Logistic Regression parity validation
./scripts/run_docker.sh validate-p4-07

# P4-08: Baseline versus MPI synchronization-overhead benchmark
./scripts/run_docker.sh benchmark-p4-08
```

The P4-08 CSV is written inside the shared Docker data volume:

```text
/data/results/p4_08_sync/sync_overhead_benchmark.csv
```

Print it without requiring the optional `column` package:

```bash
docker exec mpi-root cat /data/results/p4_08_sync/sync_overhead_benchmark.csv
```

### Cleanup

```bash
./scripts/run_docker.sh down
```

The Docker deployment is intended for functional validation and controlled
resource experiments. Performance optimization, dynamic allocation, and
Kubernetes deployment are later-phase work.

---

## Test Coverage

Run all tests:

```bash
pytest tests/ -v
```

Run unit tests only (no MPI or Spark required):

```bash
pytest tests/unit/ -v
```

Run with coverage:

```bash
pytest tests/unit/ -v --cov=mpj_spark --cov-report=term-missing
```

| Test Module | What It Covers | Requires MPI? |
|---|---|---|
| `unit/test_file_manager.py` | `_count_lines()`, `dynamic_partition()` correctness, losslessness, cleanup | No |
| `unit/test_file_manager_edge.py` | Edge cases — empty file, single-line, unicode | No |
| `unit/test_root_process.py` | Root pipeline: dispatch, barrier, aggregation | No |
| `unit/test_root_process_helpers.py` | `merge_word_counts()`, `compute_speedup()`, timing | No |
| `unit/test_spark_session.py` | SparkSession factory, core allocation formula | No |
| `unit/test_key_value.py` | `KeyValueStructure` serialisation | No |
| `unit/test_gossip_aggregator.py` | Gossip Allreduce centroid convergence | No |
| `unit/test_baseline_applications.py` | Baseline K-Means + LogReg return shapes, accuracy range | No |
| `unit/test_sync_modes.py` | P3-08 sync-mode registry — canonical names, aliases, descriptors | No |
| `unit/test_fedavg_mpi_run.py` | P3-08 FedAvg MPI helpers + simulated gather/bcast FedAvg math | No |
| `phase3/test_kmeans_allreduce.py` | K-Means MPI Allreduce correctness | Yes (skipped in CI) |
| `phase3/test_kmeans_convergence.py` | K-Means convergence over iterations | No |
| `phase3/test_kmeans_metrics.py` | K-Means metrics collector | No |
| `phase3/test_kmeans_partition.py` | K-Means partition + centroid init | Yes (skipped in CI) |
| `phase3/test_mpi_verify.py` | MPI environment sanity (barrier, allreduce) | Partial |
| `phase3/test_wordcount_mpi_vs_baseline.py` | WordCount MPI vs baseline top-N match | Yes (skipped in CI) |
| `logreg/test_allreduce.py` | `allreduce_gradients()`, `check_loss_convergence()` | No |
| `logreg/test_local_gradient.py` | `compute_gradient_spark()` (real `local[1]` Spark) | No |
| `logreg/test_metrics.py` | `LogRegMetricsCollector` all methods | No |

---

## CLI Reference

### `main.py` (Phase 1–2)

| Flag | Default | Description |
|---|---|---|
| `--workers N` | `4` | Number of parallel Spark driver workers |
| `--generate N` | `50` | Auto-generate synthetic dataset of N MB |
| `--input PATH` | — | Use an existing input file |
| `--app NAME` | `wordcount` | Workload: `wordcount` \| `kmeans` \| `logreg` |
| `--sync MODE` | `queue` | Sync mode: `queue` (M2) \| `none` (M1) \| `mpi` (M3) |
| `--compare` | off | Run single-driver baseline and print comparison table |
| `--baseline-master URL` | `None` | Spark master URL for fair standalone baseline (e.g. `spark://spark-master:7077`) |
| `--cores N` | auto | Cores per worker (`0` = unconstrained `local[*]`) |
| `--no-prewarm` | off | Cold-start mode — include JVM init in wall-clock |
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
| `--sync-mode MODE` | `ps_sync_fedavg_mpi` | Sync strategy: `ps_sync_fedavg_mpi` (native MPI gather/bcast FedAvg) \| `ps_sync_fedavg_queue` (legacy P2P fallback) \| `allreduce_mpi` (per-iteration collective) \| `none` (M1 no-sync) |
| `--cores N` | auto | Override `local[N]` core count per worker |
| `--compare` | off | Run single-driver baseline |
| `--gossip` | off | Use gossip Allreduce for K-Means centroid sync |
| `--kmeans-k N` | `3` | Number of K-Means clusters |
| `--kmeans-iter N` | `20` | K-Means maximum iterations |
| `--logreg-iter N` | `10` | Logistic Regression iterations |
| `--results-dir PATH` | `results` | Directory for profiling CSVs |

### `scripts/validate_parity.py` (Issue #10)

| Flag | Default | Description |
|---|---|---|
| `--tolerance F` | `1e-3` | L2 delta tolerance for centroid and weight comparisons |
| `--kmeans-k N` | `3` | K for K-Means |
| `--kmeans-iter N` | `20` | max_iter for K-Means |
| `--logreg-epochs N` | `30` | Epochs for LogReg MPI |
| `--skip-kmeans` | off | Skip K-Means validation |
| `--skip-logreg` | off | Skip LogReg validation |

### `scripts/sync_overhead_benchmark.py` (Issue #12)

| Flag | Default | Description |
|---|---|---|
| `--ranks N` | `5` | Expected MPI size (informational) |
| `--kmeans-k N` | `3` | K for K-Means |
| `--kmeans-iter N` | `20` | max_iter for K-Means baseline |
| `--logreg-epochs N` | `14` | Epochs for LogReg baseline |
| `--metrics-dir PATH` | `metrics` | Directory containing workload metrics CSVs |
| `--results-dir PATH` | `results` | Output directory for benchmark CSV |
| `--data-dir PATH` | `data` | Dataset directory |
| `--skip-kmeans` | off | Skip K-Means benchmark |
| `--skip-logreg` | off | Skip LogReg benchmark |

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

---

## Benchmark Results

### Phase 3 Timing Analysis — K-Means & LogReg (N=5, 540 K rows)

| Workload | Bottleneck | Spark fraction | Sync overhead (mean) | Iterations |
|---|---|---|---|---|
| K-Means | compute | 0.963 | 3.74% | 4 (converged) |
| LogReg | sync | 0.375 | 62.51% | 14 (not converged) |

> Full timing decomposition in `results/timing/timing_summary.csv`.
> Sync overhead benchmark in `results/sync_overhead_benchmark.csv`.
> Controller feature matrix in `results/timing/controller_feature_matrix.csv` — input for Objective 2b predictor.

### P3-08 — FedAvg Transport Comparison (540 K rows, 10 rounds, reg_param=0.01)

| Run | Workers | Cores/Worker | Wall-Clock | Proc Time | Final \|w\| | Accuracy |
|---|---|---|---|---|---|---|
| FedAvg — native MPI (`ps_sync_fedavg_mpi`) | 2 | 11 | 26.05 s | 25.70 s | 0.3228 | 0.6308 |
| FedAvg — Queue fallback (`ps_sync_fedavg_queue`) | 2 | 11 | 26.41 s | 26.13 s | 0.3228 | 0.6308 |
| FedAvg — native MPI | 3 | 8 | 32.37 s | 31.75 s | 0.3228 | 0.6308 |
| FedAvg — native MPI | 4 | 6 | 40.21 s | 39.55 s | 0.3228 | 0.6308 |

> Identical final model across transports (convergence parity); transport overhead within noise at this compute-bound scale. Single-machine wall-clock rises with worker count due to fixed-core subdivision — multi-node scaling is validated in Phase 5.

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
| `feature/p3-ml-workload-parity` | Issues #10 #12 #13 #14 — parity + sync benchmark + docs + WordCount regression |
| `feature/adaptive-gossip-aggregation` | Objective 1b — Gossip Allreduce sync |
| `feature/p3-fedavg-mpi` | Issue #65 — FedAvg aggregation over native MPI transport |

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
