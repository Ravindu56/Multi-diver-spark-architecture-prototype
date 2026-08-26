#!/usr/bin/env python3
"""apply_p3_09_wiring.py — P3-09 (Issue #61) ps_async dispatch wiring patch.

One-shot migration script.  Run ONCE from the repo root:

    python scripts/apply_p3_09_wiring.py

Patches, with assertion-guarded anchors (each anchor must match EXACTLY
ONCE or the edit is skipped and reported):

  STRICT (functional — required for the contract tests + E2E):
    mpj_spark/workers/worker_process.py
        1. import MODE_PS_ASYNC + MODE_PS_SYNC_FEDAVG_QUEUE
        2. run_worker_core(...) gains root_comm=None
        3. new elif branch routing sync_mode=ps_async to
           applications/logreg/async_ps_run.run() over COMM_WORLD
           (root_comm) with one-based COMM_WORLD rank
        4. fail-fast guard: unwired modes no longer fall through to
           queue_run (silent M1 degradation seen on 2026-08-26 run)
    mpj_spark/workers/worker_mpi.py
        5. pass root_comm=comm (COMM_WORLD) into run_worker_core
    mpj_spark/core/root_mpi.py
        6. import MODE_PS_ASYNC
        7. do_logreg_async_ps derived flag
        8. async PS coordinator daemon thread (reuses the existing
           allreduce_thread/_allreduce_store join + aggregation path)

  LENIENT (cosmetic — warn and continue if the anchor misses):
    mpj_spark/core/root_mpi.py    9.  agg_mode header label
    mpj_spark/core/root_process.py 10. aggregation agg_mode label
    mpj_spark/core/main_mpi.py     11. --sync-mode CLI help text

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


# ───────────────────────── 1–4. worker_process.py (strict) ─────────────────────────
WP = "mpj_spark/workers/worker_process.py"

patch_first(
    WP,
    [
        (
            """from mpj_spark.core.sync_modes import (
    MODE_NONE,
    MODE_PS_SYNC_FEDAVG_MPI,
    normalize_sync_mode,
)""",
            """from mpj_spark.core.sync_modes import (
    MODE_NONE,
    MODE_PS_ASYNC,
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    normalize_sync_mode,
)""",
        )
    ],
    "import MODE_PS_ASYNC + MODE_PS_SYNC_FEDAVG_QUEUE",
)

patch_first(
    WP,
    [
        (
            "    reassign_adapter=None,\n    comm=None,\n):",
            "    reassign_adapter=None,\n    comm=None,\n    root_comm=None,\n):",
        ),
        (
            "    reassign_adapter=None,\n    comm=None\n):",
            "    reassign_adapter=None,\n    comm=None,\n    root_comm=None\n):",
        ),
        (
            "reassign_adapter=None, comm=None):",
            "reassign_adapter=None, comm=None, root_comm=None):",
        ),
    ],
    "run_worker_core signature gains root_comm=None",
)

patch_first(
    WP,
    [
        (
            "        elif sync_mode == MODE_PS_SYNC_FEDAVG_MPI and comm is not None:",
            '''        elif sync_mode == MODE_PS_ASYNC:
            if root_comm is None:
                raise RuntimeError(
                    f"[W{worker_id}] sync_mode='ps_async' requires the MPI execution path "
                    "(python -m mpj_spark.core.main_mpi) — root_comm is None on the "
                    "multiprocessing transport."
                )
            from mpj_spark.applications.logreg import async_ps_run

            result = async_ps_run.run(
                partition_path=partition_path,
                comm=root_comm,  # COMM_WORLD: root PS is rank 0
                rank=worker_id + 1,  # COMM_WORLD rank (workers are 1..N)
                num_workers=num_workers,
                max_iter=worker_config.get("logreg_iter", 10),
                reg_param=worker_config.get("logreg_reg_param", 0.01),
                num_features=worker_config.get("logreg_features", 10),
                results_dir=results_dir,
                local_epochs=worker_config.get("logreg_local_epochs", 5),
            )
        elif sync_mode == MODE_PS_SYNC_FEDAVG_MPI and comm is not None:''',
        )
    ],
    "ps_async dispatch branch (COMM_WORLD, one-based rank)",
)

patch_first(
    WP,
    [
        (
            "        else:\n            from mpj_spark.applications.logreg import queue_run",
            '''        else:
            if sync_mode != MODE_PS_SYNC_FEDAVG_QUEUE:
                raise RuntimeError(
                    f"[W{worker_id}] sync_mode='{sync_mode}' is registered but not wired "
                    "for logreg on this transport — refusing silent fallback to queue_run."
                )
            from mpj_spark.applications.logreg import queue_run''',
        )
    ],
    "fail-fast guard against silent queue_run fallback",
)

# ───────────────────────── 5. worker_mpi.py (strict) ─────────────────────────
patch_first(
    "mpj_spark/workers/worker_mpi.py",
    [
        (
            "        reassign_adapter=reassign,\n        comm=worker_comm,\n    )",
            "        reassign_adapter=reassign,\n        comm=worker_comm,\n"
            "        root_comm=comm,  # P3-09: COMM_WORLD channel for the async-PS P2P protocol\n"
            "    )",
        )
    ],
    "pass root_comm=comm (COMM_WORLD) into run_worker_core",
)

# ───────────────────────── 6–8. root_mpi.py (strict) ─────────────────────────
RM = "mpj_spark/core/root_mpi.py"

patch_first(
    RM,
    [
        (
            """from mpj_spark.core.sync_modes import (
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    normalize_sync_mode,
)""",
            """from mpj_spark.core.sync_modes import (
    MODE_PS_ASYNC,
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    normalize_sync_mode,
)""",
        )
    ],
    "import MODE_PS_ASYNC",
)

patch_first(
    RM,
    [
        (
            '    do_logreg_allreduce_p2p = app == "logreg" and sync_mode == MODE_PS_SYNC_FEDAVG_QUEUE\n',
            '    do_logreg_allreduce_p2p = app == "logreg" and sync_mode == MODE_PS_SYNC_FEDAVG_QUEUE\n'
            '    do_logreg_async_ps = app == "logreg" and sync_mode == MODE_PS_ASYNC\n',
        )
    ],
    "do_logreg_async_ps derived flag",
)

patch_first(
    RM,
    [
        (
            '        allreduce_thread.start()\n'
            '        print("  [LogReg Allreduce MPI] Coordinator thread started (P2P Queue-fallback)")',
            '        allreduce_thread.start()\n'
            '        print("  [LogReg Allreduce MPI] Coordinator thread started (P2P Queue-fallback)")\n'
            """
    if do_logreg_async_ps:
        import threading

        from mpj_spark.core.async_ps import run_logreg_async_ps

        def _async_ps_thread_fn():
            res = run_logreg_async_ps(
                comm,  # COMM_WORLD on root — P2P with worker ranks 1..N
                num_workers=num_workers,
                num_iterations=logreg_iter,
                num_features=logreg_features,
                results_dir=results_dir,
            )
            _allreduce_store.append(res)  # same result shape as run_logreg_allreduce_mpi()

        allreduce_thread = threading.Thread(
            target=_async_ps_thread_fn, daemon=True, name="logreg-async-ps"
        )
        allreduce_thread.start()
        print("  [LogReg Async PS] Coordinator thread started (P3-09, non-blocking P2P)")""",
        )
    ],
    "async PS coordinator daemon thread",
)

# ───────────────────────── 9–11. cosmetic edits (lenient) ─────────────────────────
patch_first(
    RM,
    [
        (
            '        else f"Allreduce FedAvg MPI ({logreg_iter} iters)"\n        if app == "logreg"\n',
            '        else f"Async Parameter Server ({logreg_iter} rounds, FedAsync)"\n'
            "        if (app == \"logreg\" and sync_mode == MODE_PS_ASYNC)\n"
            '        else f"Allreduce FedAvg MPI ({logreg_iter} iters)"\n        if app == "logreg"\n',
        )
    ],
    "agg_mode header label for ps_async",
    strict=False,
)

patch_first(
    "mpj_spark/core/root_process.py",
    [
        (
            """from mpj_spark.core.sync_modes import (
    MODE_NONE,
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    normalize_sync_mode,
)""",
            """from mpj_spark.core.sync_modes import (
    MODE_NONE,
    MODE_PS_ASYNC,
    MODE_PS_SYNC_FEDAVG_MPI,
    MODE_PS_SYNC_FEDAVG_QUEUE,
    normalize_sync_mode,
)""",
        )
    ],
    "root_process import MODE_PS_ASYNC",
    strict=False,
)

patch_first(
    "mpj_spark/core/root_process.py",
    [
        (
            '        agg_mode = "Allreduce — FedAvg per-iteration (M2)"',
            '''        agg_mode = (
            "Async Parameter Server — FedAsync mixing (P3-09)"
            if sync_mode == MODE_PS_ASYNC
            else "Allreduce — FedAvg per-iteration (M2)"
        )''',
        )
    ],
    "aggregation agg_mode label for ps_async",
    strict=False,
)

patch_first(
    "mpj_spark/core/main_mpi.py",
    [
        (
            'help="Cross-driver synchronization mode (ps_sync_fedavg_mpi | ps_sync_fedavg_queue | allreduce_mpi | none)"',
            'help="Cross-driver synchronization mode (ps_sync_fedavg_mpi | ps_sync_fedavg_queue | allreduce_mpi | ps_async | none)"',
        )
    ],
    "--sync-mode help text lists ps_async",
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
