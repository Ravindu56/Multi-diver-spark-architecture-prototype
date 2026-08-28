#!/usr/bin/env python3
"""apply_p3_10_wiring.py — P3-10 (Issue #62) hybrid_ps_allreduce dispatch wiring patch.

One-shot migration script.  Run ONCE from the repo root:

    python scripts/apply_p3_10_wiring.py

Patches, with assertion-guarded anchors (each anchor must match EXACTLY
ONCE or the edit is skipped and reported):

  STRICT (functional):
    mpj_spark/workers/worker_process.py
        1. import MODE_HYBRID_PS_ALLREDUCE
        2. new elif branch routing sync_mode=hybrid_ps_allreduce to
           applications/logreg/hybrid_run.run() with BOTH communicators
           (comm = worker sub-comm for the dense Allreduce,
           root_comm = COMM_WORLD for the scalar PS channel)
    mpj_spark/core/root_mpi.py
        3. import MODE_HYBRID_PS_ALLREDUCE
        4. do_logreg_hybrid derived flag
        5. scalar-PS coordinator daemon thread (reuses the
           allreduce_thread/_allreduce_store join + aggregation path)
    mpj_spark/core/root_process.py
        6. import MODE_HYBRID_PS_ALLREDUCE
        7. aggregate_logreg_results: hybrid branch — dense weights from
           worker_results[0] (identical post-Allreduce), intercept from
           the scalar PS result (must precede the generic
           allreduce_result branch, whose weight_vector is None here)

  LENIENT (cosmetic — warn and continue if the anchor misses):
    mpj_spark/core/root_mpi.py    8. agg_mode header label
    mpj_spark/core/main_mpi.py     9. --sync-mode CLI help text

After patching, every touched file is py_compile-checked.
Then: ruff check . && pytest tests/unit/ -v
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

FAILURES: list[str] = []
TOUCHED: list[str] = []


def patch_first(path: str, candidates: list[tuple[str, str]], label: str, strict: bool = True) -> bool:
    """Apply the first candidate whose anchor matches exactly once."""
    p = Path(path)
    src = p.read_text(encoding="utf-8")
    for old, new in candidates:
        n = src.count(old)
        if n == 1:
            p.write_text(src.replace(old, new), encoding="utf-8")
            print(f"OK    {path}: {label}")
            if path not in TOUCHED:
                TOUCHED.append(path)
            return True
    print(f"{'FAIL' if strict else 'WARN'}  {path}: '{label}' — no unique anchor match")
    if strict:
        FAILURES.append(f"{path}: {label}")
    return False


# ───────────────────────── 1–2. worker_process.py (strict) ─────────────────────────
WP = "mpj_spark/workers/worker_process.py"

patch_first(
    WP,
    [
        (
            """from mpj_spark.core.sync_modes import (
    MODE_NONE,
    MODE_PS_ASYNC,
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    normalize_sync_mode,
)""",
            """from mpj_spark.core.sync_modes import (
    MODE_HYBRID_PS_ALLREDUCE,
    MODE_NONE,
    MODE_PS_ASYNC,
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    normalize_sync_mode,
)""",
        )
    ],
    "import MODE_HYBRID_PS_ALLREDUCE",
)

patch_first(
    WP,
    [
        (
            "        elif sync_mode == MODE_PS_ASYNC:",
            '''        elif sync_mode == MODE_HYBRID_PS_ALLREDUCE:
            if comm is None or root_comm is None:
                raise RuntimeError(
                    f"[W{worker_id}] sync_mode='hybrid_ps_allreduce' requires both the "
                    "worker sub-communicator and COMM_WORLD — run via "
                    "python -m mpj_spark.core.main_mpi."
                )
            from mpj_spark.applications.logreg import hybrid_run

            result = hybrid_run.run(
                partition_path=partition_path,
                comm=comm,  # worker sub-comm: dense-weight Allreduce channel
                rank=worker_id,  # 0-based sub-comm rank
                num_workers=num_workers,
                root_comm=root_comm,  # COMM_WORLD: scalar PS channel
                world_rank=worker_id + 1,
                max_iter=worker_config.get("logreg_iter", 10),
                reg_param=worker_config.get("logreg_reg_param", 0.01),
                num_features=worker_config.get("logreg_features", 10),
                results_dir=results_dir,
                local_epochs=worker_config.get("logreg_local_epochs", 5),
            )
        elif sync_mode == MODE_PS_ASYNC:''',
        )
    ],
    "hybrid_ps_allreduce dispatch branch (dual-channel)",
)

# ───────────────────────── 3–5. root_mpi.py (strict) ─────────────────────────
RM = "mpj_spark/core/root_mpi.py"

patch_first(
    RM,
    [
        (
            """from mpj_spark.core.sync_modes import (
    MODE_PS_ASYNC,
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    normalize_sync_mode,
)""",
            """from mpj_spark.core.sync_modes import (
    MODE_HYBRID_PS_ALLREDUCE,
    MODE_PS_ASYNC,
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    normalize_sync_mode,
)""",
        )
    ],
    "import MODE_HYBRID_PS_ALLREDUCE",
)

patch_first(
    RM,
    [
        (
            '    do_logreg_async_ps = app == "logreg" and sync_mode == MODE_PS_ASYNC\n',
            '    do_logreg_async_ps = app == "logreg" and sync_mode == MODE_PS_ASYNC\n'
            '    do_logreg_hybrid = app == "logreg" and sync_mode == MODE_HYBRID_PS_ALLREDUCE\n',
        )
    ],
    "do_logreg_hybrid derived flag",
)

patch_first(
    RM,
    [
        (
            '        allreduce_thread.start()\n'
            '        print("  [LogReg Async PS] Coordinator thread started (P3-09, non-blocking P2P)")',
            '        allreduce_thread.start()\n'
            '        print("  [LogReg Async PS] Coordinator thread started (P3-09, non-blocking P2P)")\n'
            """
    if do_logreg_hybrid:
        import threading

        from mpj_spark.core.hybrid_ps import run_logreg_hybrid_scalar_ps

        def _hybrid_ps_thread_fn():
            res = run_logreg_hybrid_scalar_ps(
                comm,  # COMM_WORLD on root — scalar P2P with worker ranks 1..N
                num_workers=num_workers,
                num_iterations=logreg_iter,
                results_dir=results_dir,
            )
            _allreduce_store.append(res)  # intercept-only result; weights via Allreduce

        allreduce_thread = threading.Thread(
            target=_hybrid_ps_thread_fn, daemon=True, name="logreg-hybrid-ps"
        )
        allreduce_thread.start()
        print("  [LogReg Hybrid PS] Scalar coordinator thread started (P3-10)")""",
        )
    ],
    "hybrid scalar-PS coordinator daemon thread",
)

# ───────────────────────── 6–7. root_process.py (strict) ─────────────────────────
RP = "mpj_spark/core/root_process.py"

patch_first(
    RP,
    [
        (
            """from mpj_spark.core.sync_modes import (
    MODE_NONE,
    MODE_PS_ASYNC,
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    normalize_sync_mode,
)""",
            """from mpj_spark.core.sync_modes import (
    MODE_HYBRID_PS_ALLREDUCE,
    MODE_NONE,
    MODE_PS_ASYNC,
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    normalize_sync_mode,
)""",
        )
    ],
    "import MODE_HYBRID_PS_ALLREDUCE",
)

patch_first(
    RP,
    [
        (
            '''    if allreduce_result is not None:
        final_weights = allreduce_result["weight_vector"]
        final_intercept = allreduce_result["intercept"]
        agg_mode = (
            "Async Parameter Server — FedAsync mixing (P3-09)"
            if sync_mode == MODE_PS_ASYNC
            else "Allreduce — FedAvg per-iteration (M2)"
        )''',
            '''    if sync_mode == MODE_HYBRID_PS_ALLREDUCE:
        # Dense weights are identical across workers post-Allreduce (collective
        # result); the scalar PS result carries the global intercept.  Must
        # precede the generic allreduce_result branch: the hybrid PS result
        # has weight_vector=None by design.
        final_weights = worker_results[0]["weight_vector"]
        final_intercept = (
            allreduce_result["intercept"]
            if allreduce_result is not None
            else worker_results[0].get("intercept", 0.0)
        )
        agg_mode = "Hybrid PS+Allreduce (P3-10)"
    elif allreduce_result is not None:
        final_weights = allreduce_result["weight_vector"]
        final_intercept = allreduce_result["intercept"]
        agg_mode = (
            "Async Parameter Server — FedAsync mixing (P3-09)"
            if sync_mode == MODE_PS_ASYNC
            else "Allreduce — FedAvg per-iteration (M2)"
        )''',
        )
    ],
    "aggregate_logreg_results hybrid branch",
)

# ───────────────────────── 8–9. cosmetic edits (lenient) ─────────────────────────
patch_first(
    RM,
    [
        (
            '        else f"Async Parameter Server ({logreg_iter} rounds, FedAsync)"\n'
            '        if (app == "logreg" and sync_mode == MODE_PS_ASYNC)\n',
            '        else f"Hybrid PS+Allreduce ({logreg_iter} iters)"\n'
            '        if (app == "logreg" and sync_mode == MODE_HYBRID_PS_ALLREDUCE)\n'
            '        else f"Async Parameter Server ({logreg_iter} rounds, FedAsync)"\n'
            '        if (app == "logreg" and sync_mode == MODE_PS_ASYNC)\n',
        )
    ],
    "agg_mode header label for hybrid",
    strict=False,
)

patch_first(
    "mpj_spark/core/main_mpi.py",
    [
        (
            'help="Cross-driver synchronization mode (ps_sync_fedavg_mpi | ps_sync_fedavg_queue | allreduce_mpi | ps_async | none)"',
            'help="Cross-driver synchronization mode (ps_sync_fedavg_mpi | ps_sync_fedavg_queue | allreduce_mpi | ps_async | hybrid_ps_allreduce | none)"',
        )
    ],
    "--sync-mode help text lists hybrid_ps_allreduce",
    strict=False,
)

# ───────────────────────── compile check + summary ─────────────────────────
print()
for path in TOUCHED:
    try:
        py_compile.compile(path, doraise=True)
        print(f"compile OK  {path}")
    except py_compile.PyCompileError as exc:
        FAILURES.append(f"{path}: compile error")
        print(f"compile FAIL  {path}\n{exc}")

print()
if FAILURES:
    print("PATCH INCOMPLETE — unresolved anchors:")
    for f in FAILURES:
        print(f"  - {f}")
    print("Paste the surrounding lines of each failing anchor back to the chat for a corrected patch.")
    sys.exit(1)
print("All anchors applied. Next: ruff check . && pytest tests/unit/ -v")
