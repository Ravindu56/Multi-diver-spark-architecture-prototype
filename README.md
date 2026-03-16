# MPJ-SPARK Multi-Driver Architecture Prototype

> BScEng Research Prototype — University of Jaffna  
> Simulates the MPJ-SPARK paper architecture on a single machine using Python `multiprocessing` + PySpark.

---

## Repository Structure

```
mpj_spark/
├── core/
│   ├── file_manager.py       # MPJSparkFileManager (shared storage, dynamic partitioning)
│   ├── key_value.py          # KeyValueStructure   (RDD ↔ MPJ buffer)
│   └── root_process.py       # MPJ Root orchestrator (Algorithm 1)
├── workers/
│   ├── spark_session.py      # SparkSession factory (central Spark config)
│   └── worker_process.py     # MPJ Worker — independent Spark driver per partition
├── applications/
│   ├── wordcount.py          # WordCount RDD pipeline (current benchmark)
│   └── baseline_spark.py     # Single-driver Spark baseline for comparison
├── benchmarks/
│   ├── timing.py             # TimingCollector (T_Load, T_Init, T_Proc, T_Agg)
│   └── reporter.py           # Console result + comparison tables
├── utils/
│   └── dataset_generator.py  # Synthetic dataset generator (Section VI.B)
├── config.py                 # Central config (paths, Spark settings, defaults)
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

```bash
# Basic run — 4 workers, auto-generate 50 MB dataset
python main.py

# 2 workers, 500 MB dataset, with baseline comparison
python main.py --workers 2 --generate 500 --compare

# Use your own input file
python main.py --workers 4 --input /path/to/data.txt --compare

# Scaling sweep (for speedup curve)
python main.py --workers 1 --generate 500 --compare
python main.py --workers 2 --generate 500 --compare
python main.py --workers 4 --generate 500 --compare
```

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
> IEEE Access, 2025 — DOI: 10.1109/ACCESS.2025.3584744
