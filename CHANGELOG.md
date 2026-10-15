# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor bump = new project phase).

## [0.4.0] - 2026-10-15 (pre-release)

Phase 6 groundwork: data collection and a fair single-driver baseline (Objectives 2b, 2d).

### Added
- `scripts/run_collection.sh`: sequential, resumable headless collection driver
  (pilot / start / status / stop / reset / run --dry-run), sleep inhibit, per-stage `.done`
  markers, manifest with commit hash.
- `scripts/baseline_eval.py` and `scripts/compare_baseline.py`: fair single-driver baseline
  (same framework and kernels, 1 driver with the same total cores as W drivers x c cores).
- Fixed-budget K-Means evaluation across the four sync modes (`sync_eval.py`), with the round
  budget recorded in run ids and tables.
- Workload profiling for the predictor dataset.
- `docs/measurement_notes.md`: baseline definition and eight known caveats.
- `docs/predictor_handoff.md`.

### Fixed
- Open MPI binding: with np=2 each rank was bound to one core, so single-driver runs used
  1 of 22 CPUs. The collection now exports `OMPI_MCA_hwloc_base_binding_policy=none` and a
  probe aborts unless every rank sees all CPUs.
- K-Means PS thread labels, Spark progress bar disabled, G1 Old Generation collector name,
  WordCount MPI regression test, `pandas<3` cap, ruff formatting.
- Package/project version strings aligned (`pyproject.toml` was 0.3.0 and
  `mpj_spark.__version__` was 2.0.0).

### Changed
- Baselines measured before this release were under-provisioned by the binding issue and are
  not comparable with 0.4.0 numbers (pilot: baseline round time 4.37 s -> 1.90 s; multi-driver
  round speedup 2.5x -> 1.08-1.15x; WCSS unchanged).

### Known limitations
- Pilot evidence only (single repetition); no measurement data is included.
- Uneven baseline partitions (345937/345921/345922/43245 rows) may flatter the multi-driver
  result by up to about 1.3x. Documented in `docs/measurement_notes.md`.

## [0.3.1] - 2026-09-03 (pre-release)

Phase 3 sync modes complete, plus Phase 4 Docker containerization.

### Added
- Docker worker image (python:3.11-slim, OpenJDK 17, OpenMPI), NFS-backed shared volume,
  N-worker compose cluster, `scripts/run_docker.sh` (PR #60).
- FedAvg over native MPI collectives (PRs #69, #71).
- Asynchronous Parameter Server with staleness-damped mixing (PR #72).
- Hybrid PS + Allreduce (PR #73).
- Decentralized Gossip (PR #74).
- Six-mode synchronization benchmark and `docs/p3-12-sync-benchmark.md` (PR #75).

## [0.3.0] - 2026-06-30

Phase 3: real MPI Allreduce and ML workload parity (PR #55).

## [0.2.0] - 2026-05-13 (pre-release)

Phase 2: iterative ML workloads (K-Means, LogisticRegression) with simulated Allreduce.

## [0.1.0] - 2026-05-13 (pre-release)

Phase 1: single-machine multi-driver WordCount prototype.

[0.4.0]: https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/releases/tag/v0.3.1
[0.3.0]: https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/releases/tag/v0.3.0
[0.2.0]: https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/releases/tag/v0.2.0
[0.1.0]: https://github.com/Ravindu56/Multi-diver-spark-architecture-prototype/releases/tag/v0.1.0
