# MPJ-SPARK Multi-Driver Architecture Prototype

> BScEng Research Prototype — University of Jaffna  
> Simulates the MPJ-SPARK paper architecture on a single machine using Python `multiprocessing` + PySpark.

---

## Recent Fixes (dev branch)

| Commit | Fix | Impact |
|---|---|---|
| `709395c` | **Streaming partition** — `dynamic_partition()` now uses O(1) RAM two-pass stream instead of loading the full file into memory | T_Load drops from ~1.8 s → ~0.2 s on 500 MB |
| `b49354d` | **Fair thread budget** — baseline and each worker constrained to `local[N]` where `N = TOTAL_CORES // num_workers` | Eliminates unfair `local[*]` advantage for baseline; processing speedup flips to 1.54× ✓ |
| `04f51e2` | **`cores_override` param** — `run_baseline()` signature fixed to accept and pass `--cores` CLI override | `--cores N` flag now works end-to-end |
| (earlier) | **JVM pre-warm barrier** — workers signal Root when JVM is ready; computation timer starts only after all JVMs are warm | JVM init (~3.2 s) excluded from T_Proc comparison; models HPC production deployment |

---

## Architecture Overview

```
[Root Process]
  │
  ├─ Phase 1: dynamic_partition()   — stream-split input → N partition files (O(1) RAM)
  │
  ├─ Phase 2: launch N workers      — each owns an independent SparkSession (local[K])
  │           wait for JVM barrier  — all N JVMs warm before timer starts (pre-warm mode)
  │
  ├─ Phase 3: fire go-signals       — all workers start simultaneously
  │           workers compute       — each processes its partition independently
  │
  ├─ Phase 4: collect results       — Queue-based result collection
  │
  └─ Phase 5: aggregate             — KeyValueStructure merge across workers
```

**Thread budget formula (fair single-machine comparison):**
```
cores_per_entity = max(1, TOTAL_CORES // num_workers)
```
Each worker AND the baseline both run `local[cores_per_entity]` — equal CPU budget per entity, mirroring an HPC cluster where each node has a fixed dedicated core count.

---

## Repository Structure

```
mpj_spark/
├── core/
│   ├── file_manager.py       # MPJSparkFileManager — streaming dynamic partition (O(1) RAM)
│   ├── key_value.py          # KeyValueStructure   (RDD ↔ MPJ buffer)
│   └── root_process.py       # MPJ Root orchestrator — barrier sync + phase pipeline
├── workers/
│   ├── spark_session.py      # SparkSession factory — local[N] fair thread budget
│   └── worker_process.py     # MPJ Worker — JVM pre-warm + go-signal barrier
├── applications/
│   ├── wordcount.py          # WordCount RDD pipeline (current benchmark)
│   └── baseline_spark.py     # Single-driver Spark baseline — constrained to same core budget
├── benchmarks/
│   ├── timing.py             # TimingCollector (T_Load, T_Init, T_Proc, T_Agg)
│   └── reporter.py           # Console result + comparison tables
├── utils/
│   └── dataset_generator.py  # Synthetic dataset generator (Paper §VI.B)
├── config.py                 # Central config — TOTAL_CORES, paths, Spark settings
main.py                       # CLI entry point
requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

> Requires Java 11+ and Python 3.8+

---

## Usage

### Basic Runs

```bash
# Standard fair comparison — 2 workers, 500 MB, pre-warmed (recommended for paper metrics)
python3 main.py --workers 2 --generate 500 --compare

# Cold-start mode — JVM init included in wall-clock (honest single-machine cost)
python3 main.py --workers 2 --generate 500 --compare --no-prewarm

# Force explicit core budget (overrides auto formula)
python3 main.py --workers 2 --generate 500 --compare --cores 4

# Restore unconstrained local[*] for baseline (legacy / unfair mode)
python3 main.py --workers 2 --generate 500 --compare --cores 0

# Use your own input file
python3 main.py --workers 4 --input /path/to/data.txt --compare
```

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--workers N` | 4 | Number of parallel MPJ workers |
| `--generate N` | 50 | Generate synthetic dataset of N MB |
| `--input PATH` | — | Use existing input file instead of generating |
| `--compare` | off | Run standard Spark baseline and print comparison table |
| `--app NAME` | wordcount | Application to run (`wordcount`; `kmeans` planned) |
| `--no-prewarm` | off | Cold-start mode — include JVM init inside parallel timer |
| `--cores N` | auto | Override cores per entity; `0` = unconstrained `local[*]` |

### Scaling Sweep (for speedup curve)

```bash
python3 main.py --workers 1 --generate 500 --compare
python3 main.py --workers 2 --generate 500 --compare
python3 main.py --workers 4 --generate 500 --compare
python3 main.py --workers 8 --generate 500 --compare
```

---

## JVM Modes Explained

| Mode | Timer starts | Models | Use for |
|---|---|---|---|
| **Pre-warm** (default) | After all JVMs ready | HPC cluster — drivers pre-resident on nodes | Primary paper metrics (T_Proc comparison) |
| **Cold-start** (`--no-prewarm`) | Immediately at worker launch | Serverless / fresh-boot batch job | Limitations section; real prototype cost |

In pre-warm mode, `T_Init` (~3.2 s per worker) is **reported separately** but excluded from the T_Proc comparison. This directly reflects the Aziz Supercomputer deployment described in the paper (§IV.A) where Spark drivers are already resident before a job is dispatched.

---

## Benchmark Results (500 MB WordCount, 2 Workers, 22-core machine)

### Pre-warm mode — `--workers 2 --generate 500 --compare --cores 4`

| Metric | Multi-Driver | Std Spark | Speedup |
|---|---|---|---|
| Load Time | 1.22 s | 1.73 s | **1.42×** ✓ |
| Avg Worker Proc Time | 6.81 s | 10.51 s | **1.54×** ✓ |
| Total Wall-clock | 12.18 s | 30.94 s | **2.54×** ✓ |
| JVM Pre-warm (T_Init) | 3.04 s | — | excluded from T_Proc |

### Cold-start mode — `--workers 2 --generate 500 --compare --no-prewarm`

| Metric | Multi-Driver | Std Spark | Speedup |
|---|---|---|---|
| Load Time | 1.22 s | 1.52 s | **1.24×** ✓ |
| Avg Worker Proc Time | 5.24 s | 5.73 s | **1.09×** ✓ |
| Total Wall-clock | 10.74 s | 10.33 s | 0.96× (JVM tax) |
| Avg Worker JVM Init | 3.23 s | — | amortised in production |

---

## Branching Strategy

| Branch | Purpose |
|---|---|
| `master` | Stable, tagged experiment baselines |
| `dev` | Integration branch — always runnable |
| `feature/*` | Individual feature / research extensions |

---

## Planned Extensions (from `dev`)

| Feature Branch | Research Objective |
|---|---|
| `feature/ml-kmeans-workload` | Objective 1 — ML workload integration |
| `feature/dynamic-resource-alloc` | Objective 2 — Adaptive worker allocation |
| `feature/scaling-benchmark` | Speedup curve (1→2→4→8 workers) |
| `feature/mpi4py-comms` | Authentic MPI message passing |

---

## Paper Reference

> MPJ-SPARK Integration-Based Technique to Enhance Big Data Analytics in High Performance Computing Environments  
> IEEE Access, 2025 — DOI: [10.1109/ACCESS.2025.3584744](https://doi.org/10.1109/ACCESS.2025.3584744)
