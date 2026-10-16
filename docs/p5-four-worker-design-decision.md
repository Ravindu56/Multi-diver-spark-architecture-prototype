# Design Decision: Phase 5 Maximum of Four Non-root Workers

Date: 2026-10-15

Status: Research-team scope decision requested by the project student. Supervisor agreement is not asserted by this record.

## Decision

Use two and four non-root MPI/Spark workers for required Phase 5 experiments on the three-VM laptop testbed. Exclude eight-worker execution from required acceptance criteria. Use the same worker range for linked Phase 6 multi-driver comparisons. Preserve the two primary research objectives and the six-phase implementation structure.

## Context and rationale

The available host is an ASUS Vivobook Pro 15 Creator Q533MJ running Ubuntu 24.04 LTS. Three KVM VMs share its physical compute, memory, storage, and virtual network. The student reports approximately 40–50% host memory utilization with the three VMs running; this is an observation, not proof of exhaustion or a measured root cause.

The October 6 evidence in #82 and #29 records a successful four-worker, 100 MB K-Means smoke run, but an eight-worker workload experienced Swarm membership instability, task replacement, and MPI termination. No OOM kill was recorded and the root cause remains unmeasured. The decision is based on limited available capacity and the need for repeatable, instrumented experiments, not a claim that eight workers are intrinsically unsupported.

## Terminology

- Three nodes means three VMs, not three independent physical hosts.
- Two workers correspond to three MPI ranks including the root.
- Four workers correspond to five MPI ranks including the root.
- Eight workers correspond to nine MPI ranks including the root.
- Whether the root also computes must be reported per application.

## Consequences

- Required worker axis: {2, 4}, replacing {2, 4, 8}.
- K-Means dataset candidates remain 100 MB, 500 MB, and 1 GB, subject to explicit feasibility testing and documented exclusions.
- Four-worker stability and sampled repetitions remain to be established; a successful smoke run does not complete the evaluation.
- Eight-worker configurations and logs may remain as optional/historical artifacts. They are not required runs and must not be counted as completed performance results.
- Findings are framed as resource behavior and process-density comparisons on a fixed-capacity virtualized testbed, not independent-host cloud scale-out.
- Synchronization correctness, convergence behavior, profiling, Phase 4 comparison, and Phase 6 allocator evaluation remain in scope.

## Related issue revisions

Apply this decision to #29 (K-Means matrix), #30 (LogReg worker range), #31 (synchronization measurements), #32 (Phase 4 comparison), and the current-scope notice on closed preparation issue #82. Align #34 (characterization) and #50–#52 (baseline/proposed-system evaluation) with the same feasible testbed range. These are follow-up issue edits, not changes performed by adding this document.

Do not close experimental issues merely because eight-worker work is removed. Preserve historical comments and failure evidence. Keep baseline synchronization policy explicit, preserve existing no-synchronization baseline requirements, and do not silently substitute a different baseline condition.

## Documentation precedence and future work

The revised p5-04-experimental-design.md defines the current required experiment matrix. Older setup runbooks, eight-worker examples, and historical issue comments do not expand the required worker range. Update active README guidance and runbook notices through review without deleting historical evidence.

Independent physical-host testing, larger stable datasets, and eight-worker evaluation remain future work when resources are available. No claim of supervisor approval or completed large-scale validation is made.
