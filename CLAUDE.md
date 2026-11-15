# CLAUDE.md — Multi-driver Spark prototype (EC6070)

Context for Claude Code. It mirrors the claude.ai Project "Academic Research" so local
sessions start with the same background. Update the status section as work lands.

## Project

- **Title:** Resource Analysis and Optimization for Big Data Analytics in Cloud Environments
- **Course:** EC6070, BScEng Computer Engineering, University of Jaffna
- **Students:** Dayarathna D.D.R.N. (2022E033), Lawanya M.A.S. (2022E090)
- **Supervisor:** Dr. J. Jananie
- **Reference architecture:** Saleh et al. (2025), MPJ-SPARK, IEEE Access,
  DOI 10.1109/ACCESS.2025.3584744. It is an architectural reference, not an implementation
  template: Python/mpi4py is preferred over Java/MPJ-Express.

**Research gap:** multi-driver frameworks like MPJ-Spark are validated only for
non-iterative HPC batch jobs. They do not cover (i) cross-driver global state sync for
iterative ML, (ii) ML-aware dynamic resource allocation in containers, or (iii) the
performance/convergence trade-offs of sync barriers under heterogeneous load.

**Aim:** a cloud-native, ML-aware multi-driver Spark framework for distributed ML
training in containerized clusters, with resource allocation driven by workload
characterization.

## Objectives

Primary (supervisor-defined, do not change):
1. Adopt a state-of-the-art architecture for ML workloads.
2. Develop a resource allocation strategy for big data in a cluster.

Secondary:
- 1a Adapt multi-driver Spark from HPC/SLURM to Docker (primary) / Kubernetes (secondary),
  with an NFS volume standing in for Lustre.
- 1b Per-iteration cross-driver parameter sync (Allreduce or parameter server).
- 1c Validate on iterative ML (K-Means, LogReg) and batch (WordCount).
- 2a Profile CPU and memory across heterogeneous ML workloads.
- 2b Lightweight resource-demand predictor per driver.
- 2c Workload-aware allocator assigning CPU and memory per driver.
- 2d Evaluate against (i) single-driver static allocation and (ii) multi-driver without
  allocator.

## Architecture

Root coordinator → partitions the dataset → N independent Spark drivers (one per MPI rank)
on shared NFS storage → per-iteration cross-driver sync → root aggregation.

Sync modes implemented: Allreduce, FedAvg, async parameter server (staleness-damped),
hybrid PS + Allreduce, gossip.

Metrics: execution time, CPU and memory utilization, throughput, load vs processing time,
sync overhead, convergence rate.

## Hard constraints

- Cloud/container deployment only. Do not suggest or default to HPC/SLURM.
- Must run on undergraduate lab resources. The current testbed is three KVM VMs on one
  laptop (Docker Swarm). Required worker axis is {2, 4}; eight workers is out of scope
  (see `docs/p5-four-worker-design-decision.md`).
- Out of scope: improving ML accuracy, new ML algorithms, production-grade cloud service,
  stream processing.
- Do not claim multi-host scale-out or networked sync cost from co-located VM results.
  Separate useful iteration time from startup overhead.

## Status (as of 2026-11-15, `dev` @ dc0c727)

| Phase | Status |
|---|---|
| 1 WordCount prototype | Done (v0.1.0) |
| 2 Iterative ML + simulated Allreduce | Done (v0.2.0) |
| 3 Real MPI, six sync modes | Done (v0.3.0, v0.3.1) |
| 4 Docker + NFS | Done (v0.3.1) |
| 5 Multi-node Swarm | Partial: infra merged (PR #96); only a 4-worker 100 MB K-Means smoke run. The {2,4} × {100 MB, 500 MB, 1 GB} grid with repetitions is still to do |
| 6 Allocator + evaluation | Partial: 2a data and 2b predictor done (`mpj_spark/resource_prediction/`, `docs/p6-evaluation-report.md`). Allocator #48 and Docker limit enforcement #49 not started. Full 2d comparison not run |

Known issues to respect:
- The predictor under-predicts beyond the measured range: the allocator must use the upper
  bound there, not the point estimate. K-Means CPU p95 upper-bound coverage is 75% vs a
  90% target.
- Docker memory caps were not enforced during data collection. Re-validate once #49 lands.
- Baselines before v0.4.0 were affected by an Open MPI binding bug and are not comparable.
  Keep `OMPI_MCA_hwloc_base_binding_policy=none`.
- Uneven baseline partitions may inflate multi-driver speedup by up to ~1.3×
  (`docs/measurement_notes.md`).
- #97 (single subcommand CLI) is a design direction only.

## Repo conventions

- Python ≥ 3.11. Lint and format with ruff (line length 100, config in `pyproject.toml`).
- Tests: `pytest tests/unit/` (what CI runs). Stub `mpi4py` in unit tests that must pass
  in CI without libmpi.
- Work on feature branches (`feature/pN-...`, `fix/...`, `docs/...`) and open PRs into
  `dev`; releases are merged to `master` and recorded in `CHANGELOG.md` (pre-1.0, minor
  bump = new phase).
- Commit style: conventional commits with a phase scope, for example `feat(p6-03): ...`.
- Generated numbers in reports come from scripts, never typed in by hand.

## Literature guidance

When suggesting sources: peer-reviewed only (IEEE Access, ACM, Springer, Elsevier,
Journal of Big Data, 2022–2026), available on IEEE Xplore, ACM DL, Springer Link or open
access. Use IEEE reference style.
