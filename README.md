# MPJ-SPARK Multi-Driver Architecture Prototype

> **BScEng Research Prototype — University of Jaffna**  
> Simulates the MPJ-SPARK multi-driver paper architecture on a single machine using Python `multiprocessing` + PySpark.

---

## Overview

This prototype implements and benchmarks the **multi-driver Spark architecture** described in the state-of-the-art reference paper. Each MPJ Worker owns an independent `SparkSession` and processes its data partition in parallel. A Root Process orchestrates the full pipeline — partition, launch, synchronise, collect, aggregate — mirroring how MPJ-Express coordinates processes across HPC cluster nodes.

The key research contributions validated by this prototype:
- **Multi-driver parallelism** delivers measurable speedup over single-driver Spark
- **Fair core allocation** (`local[N]`) makes the single-machine comparison scientifically valid
- **JVM pre-warm barrier** cleanly separates initialisation cost from computation cost
- **Persistent dev logging** accumulates all experiment results for paper analysis

> **Paper Reference:**  
> *MPJ-SPARK Integration-Based Technique to Enhance Big Data Analytics in High Performance Computing Environments*  
> IEEE Access, 2025 — DOI: [10.1109/ACCESS.2025.3584744](https://doi.org/10.1109/ACCESS.2025.3584744)

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
  ├─ Phase 4: collect results         — Queue-based result collection
  │
  └─ Phase 5: aggregate               — KeyValueStructure merge across all workers
```

---

## Repository Structure

```
mpj_spark/
├── core/
│   ├── file_manager.py       # MPJSparkFileManager — O(1) RAM streaming partition
│   ├── key_value.py          # KeyValueStructure   (RDD ↔ MPJ buffer)
│   └── root_process.py       # MPJ Root — barrier sync + 5-phase pipeline
├── workers/
│   ├── spark_session.py      # SparkSession factory — fair local[N] core allocation
│   └── worker_process.py     # MPJ Worker — JVM pre-warm + go-signal barrier
├── applications/
│   ├── wordcount.py          # WordCount RDD pipeline (current benchmark)
│   └── baseline_spark.py     # Single-driver Spark baseline (same core budget)
├── benchmarks/
│   ├── timing.py             # TimingCollector (T_Load, T_Init, T_Proc, T_Agg)
│   ├── reporter.py           # Console result + comparison tables
│   └── dev_logger.py         # Persistent run logger → logs/dev/
├── utils/
│   └── dataset_generator.py  # Synthetic dataset generator (Paper §VI.B)
└── config.py                 # Central config — TOTAL_CORES, paths, Spark settings
main.py                       # CLI entry point
requirements.txt
run_prototype.bat             # Windows convenience runner
```

---

## Features

### Core Allocation — Manual & Automatic

Cores are allocated per entity (each worker **and** the baseline) using a fair division formula:

```
cores_per_entity = max(1, TOTAL_CORES ÷ num_workers)
```

You can control cores three ways:

| Mode | CLI Flag | Behaviour |
|---|---|---|
| **Auto** (default) | _(omit `--cores`)_ | `TOTAL_CORES ÷ --workers` — fair HPC mirror |
| **Manual** | `--cores N` | Exact core count per worker and baseline |
| **Unconstrained** | `--cores 0` | Restores `local[*]` for baseline — legacy/unfair mode |

Every run prints the resolved allocation at startup:
```
[CONFIG] Machine cores: 22  |  Workers: 4  |  Cores/entity: 5  |  JVM mode: pre-warmed
```

The same `cores_per_entity` is applied to both the multi-driver workers and the single-driver baseline, ensuring the speedup measurement is a fair comparison of *architecture*, not of *resource advantage*.

---

### O(1) RAM Streaming Partition

`MPJSparkFileManager.dynamic_partition()` splits the input file into N partition files using a **two-pass stream** — it never loads the full file into memory. This keeps T_Load constant regardless of dataset size and enables large-file benchmarks on machines with limited RAM.

---

### JVM Pre-Warm Barrier

Two JVM timing modes are supported:

| Mode | Timer starts | Models | Use for |
|---|---|---|---|
| **Pre-warm** (default) | After all N JVMs ready | HPC cluster — drivers pre-resident on nodes | Primary paper metrics (T_Proc comparison) |
| **Cold-start** (`--no-prewarm`) | Immediately at worker launch | Serverless / fresh-boot batch | Limitations section; real single-machine cost |

In pre-warm mode, `T_Init` (~3.2 s per worker) is **reported separately** but excluded from T_Proc comparison. This directly reflects the Aziz Supercomputer deployment in Paper §IV.A.

---

### Persistent Dev Run Logger

Every run is automatically saved to `logs/dev/` unless `--no-log` is passed:

| File | Format | Purpose |
|---|---|---|
| `logs/dev/dev_runs.jsonl` | JSON Lines | Machine-readable — import directly for paper plots |
| `logs/dev/dev_runs.txt` | Plain text | Human-readable — mirrors console output |

The log file is **append-only** — no run is ever overwritten. Each record captures: run ID, timestamp, hostname, total cores, worker count, dataset size, JVM mode, cores per entity, T_Load, T_Proc, T_Init, T_Parallel, T_Total, and speedup ratios.

```bash
# View a summary table of all past runs
python3 main.py --log-history
```

```bash
# Load all runs in Python for plotting
import json
with open('logs/dev/dev_runs.jsonl') as f:
    runs = [json.loads(line) for line in f]
```

---

### Benchmark Timing — Four Metrics

`TimingCollector` tracks four named phases independently per run:

| Metric | Measures |
|---|---|
| `T_Load` | Input partition time (stream split) |
| `T_Init` | JVM initialisation time per worker (excluded from T_Proc in pre-warm mode) |
| `T_Proc` | Pure computation time (WordCount RDD pipeline) |
| `T_Total` | Full wall-clock including partition + aggregate |

---

### Multi-Driver vs. Baseline Comparison

Passing `--compare` runs the single-driver Spark baseline immediately after the multi-driver run on the same input file with the same core budget, then prints a side-by-side comparison table:

```
╔══════════════════════════════════════════════════════════════════════╗
║              MULTI-DRIVER vs STANDARD SPARK — COMPARISON            ║
╠═══════════════════════╦════════════════╦════════════════╦═══════════╣
║ Metric                ║  Multi-Driver  ║   Std Spark    ║  Speedup  ║
╠═══════════════════════╬════════════════╬════════════════╬═══════════╣
║ Load Time             ║       1.22 s   ║       1.73 s   ║   1.42×   ║
║ Avg Worker Proc Time  ║       6.81 s   ║      10.51 s   ║   1.54×   ║
║ Total Wall-clock      ║      12.18 s   ║      30.94 s   ║   2.54×   ║
║ JVM Pre-warm (T_Init) ║       3.04 s   ║          —     ║     —     ║
╚═══════════════════════╩════════════════╩════════════════╩═══════════╝
```

---

## Installation

**Requirements:** Java 11+ and Python 3.8+

```bash
pip install -r requirements.txt
```

---

## Usage

### Basic Runs

```bash
# Recommended — fair comparison, pre-warmed, auto core allocation
python3 main.py --workers 2 --generate 500 --compare

# Manual core allocation — 3 cores per worker and baseline
python3 main.py --workers 2 --generate 500 --compare --cores 3

# Cold-start mode — JVM init included in wall-clock
python3 main.py --workers 2 --generate 500 --compare --no-prewarm

# Unconstrained baseline — legacy unfair mode (for contrast demonstration)
python3 main.py --workers 2 --generate 500 --compare --cores 0

# Use your own input file
python3 main.py --workers 4 --input /path/to/data.txt --compare

# Run without saving to log
python3 main.py --workers 2 --generate 500 --compare --no-log

# View full history of past runs
python3 main.py --log-history
```

### Scaling Sweep (speedup curve)

```bash
python3 main.py --workers 1 --generate 500 --compare
python3 main.py --workers 2 --generate 500 --compare
python3 main.py --workers 4 --generate 500 --compare
python3 main.py --workers 8 --generate 500 --compare
```

### Core Allocation Experiments

```bash
# Compare different manual core budgets at fixed worker count
python3 main.py --workers 4 --generate 500 --compare --cores 2
python3 main.py --workers 4 --generate 500 --compare --cores 4
python3 main.py --workers 4 --generate 500 --compare --cores 8
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--workers N` | 4 | Number of parallel MPJ workers |
| `--generate N` | 50 | Generate a synthetic dataset of N MB |
| `--input PATH` | — | Use an existing input file instead of generating |
| `--compare` | off | Run single-driver Spark baseline and print comparison table |
| `--app NAME` | `wordcount` | Application to run (`wordcount`; `kmeans` planned) |
| `--cores N` | auto | Cores per worker and baseline. `0` = unconstrained `local[*]`. Default: `TOTAL_CORES ÷ workers` |
| `--no-prewarm` | off | Cold-start mode — include JVM init inside parallel timer |
| `--no-log` | off | Disable automatic run logging to `logs/dev/` |
| `--log-history` | off | Print summary table of all past dev runs and exit |

---

## Benchmark Results (500 MB WordCount, 2 Workers, 22-core machine)

### Pre-warm mode — `--workers 2 --generate 500 --compare --cores 4`

| Metric | Multi-Driver | Std Spark | Speedup |
|---|---|---|---|
| Load Time | 1.22 s | 1.73 s | **1.42×** |
| Avg Worker Proc Time | 6.81 s | 10.51 s | **1.54×** |
| Total Wall-clock | 12.18 s | 30.94 s | **2.54×** |
| JVM Pre-warm (T_Init) | 3.04 s | — | excluded from T_Proc |

### Cold-start mode — `--workers 2 --generate 500 --compare --no-prewarm`

| Metric | Multi-Driver | Std Spark | Speedup |
|---|---|---|---|
| Load Time | 1.22 s | 1.52 s | **1.24×** |
| Avg Worker Proc Time | 5.24 s | 5.73 s | **1.09×** |
| Total Wall-clock | 10.74 s | 10.33 s | 0.96× (JVM tax) |
| Avg Worker JVM Init | 3.23 s | — | amortised in production |

> The 0.96× cold-start result is expected and meaningful — it quantifies the per-job JVM initialisation tax (~3.23 s/worker) that is amortised in production HPC deployments where Spark drivers are pre-resident on cluster nodes.

---

## Branching Strategy

| Branch | Purpose |
|---|---|
| `master` | Stable, tagged experiment baselines |
| `dev` | Integration branch — always runnable |
| `feature/*` | Individual feature / research extensions |

### Planned Feature Branches

| Branch | Research Objective |
|---|---|
| `feature/ml-kmeans-workload` | Objective 1 — ML workload (K-Means) integration |
| `feature/dynamic-resource-alloc` | Objective 2 — Adaptive worker resource allocation |
| `feature/scaling-benchmark` | Speedup curve (1→2→4→8 workers) |
| `feature/mpi4py-comms` | Authentic MPI message passing |

---

## Recent Fixes (dev branch)

| Commit | Fix | Impact |
|---|---|---|
| `709395c` | Streaming partition — `dynamic_partition()` uses O(1) RAM two-pass stream | T_Load: ~1.8 s → ~0.2 s on 500 MB |
| `b49354d` | Fair thread budget — baseline and workers both constrained to `local[N]` | Processing speedup corrected to 1.54× |
| `04f51e2` | `cores_override` param — `run_baseline()` accepts and passes `--cores` flag | `--cores N` works end-to-end |
| earlier | JVM pre-warm barrier — computation timer starts only after all JVMs are ready | JVM init (~3.2 s) excluded from T_Proc |
