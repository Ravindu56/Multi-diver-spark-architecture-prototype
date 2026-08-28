#!/usr/bin/env python3
"""apply_p3_11_wiring.py — P3-11 (Issue #63) gossip dispatch wiring patch.

One-shot migration script.  Run ONCE from the repo root:

    python scripts/apply_p3_11_wiring.py

Patches, with assertion-guarded anchors (each anchor must match EXACTLY
ONCE or the edit is skipped and reported):

  STRICT (functional):
    mpj_spark/workers/worker_process.py
        1. import MODE_GOSSIP
        2. new elif branch routing sync_mode=gossip to
           applications/logreg/gossip_run.run() over the worker
           sub-comm ONLY (decentralized: root_comm not required)
    mpj_spark/core/root_mpi.py
        3. import MODE_GOSSIP
        4. do_logreg_gossip derived flag
        5. info print: gossip starts NO root coordinator thread
    mpj_spark/core/root_process.py
        6. import MODE_GOSSIP

  LENIENT (cosmetic — warn and continue if the anchor misses):
    mpj_spark/core/root_mpi.py     7. agg_mode header label
                                   8. worker_cfg_base gains gossip_fanout
    mpj_spark/core/root_process.py 9. aggregation agg_mode label
    mpj_spark/core/main_mpi.py     10. --sync-mode CLI help text

Gossip needs no root-side coordinator: workers exchange parameter states
with ring neighbours over the worker sub-communicator, and the existing
post-hoc row-weighted aggregation branch handles the (near-consensus)
final models.

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
    MODE_HYBRID_PS_ALLREDUCE,""",
            """from mpj_spark.core.sync_modes import (
    MODE_GOSSIP,
    MODE_HYBRID_PS_ALLREDUCE,""",
        )
    ],
    "import MODE_GOSSIP",
)

patch_first(
    WP,
    [
        (
            "        elif sync_mode == MODE_HYBRID_PS_ALLREDUCE:",
            '''        elif sync_mode == MODE_GOSSIP:
            if comm is None:
                raise RuntimeError(
                    f"[W{worker_id}] sync_mode='gossip' requires the MPI worker "
                    "sub-communicator — run via python -m mpj_spark.core.main_mpi."
                )
            from mpj_spark.applications.logreg import gossip_run

            result = gossip_run.run(
                partition_path=partition_path,
                comm=comm,  # worker sub-comm: decentralized ring exchange
                rank=worker_id,  # 0-based sub-comm rank
                num_workers=num_workers,
                max_iter=worker_config.get("logreg_iter", 10),
                reg_param=worker_config.get("logreg_reg_param", 0.01),
                num_features=worker_config.get("logreg_features", 10),
                results_dir=results_dir,
                local_epochs=worker_config.get("logreg_local_epochs", 5),
                fanout=worker_config.get("gossip_fanout", 1),
            )
        elif sync_mode == MODE_HYBRID_PS_ALLREDUCE:''',
        )
    ],
    "gossip dispatch branch (sub-comm only, decentralized)",
)

# ───────────────────────── 3–5. root_mpi.py (strict) ─────────────────────────
RM = "mpj_spark/core/root_mpi.py"

patch_first(
    RM,
    [
        (
            """from mpj_spark.core.sync_modes import (
    MODE_HYBRID_PS_ALLREDUCE,""",
            """from mpj_spark.core.sync_modes import (
    MODE_GOSSIP,
    MODE_HYBRID_PS_ALLREDUCE,""",
        )
    ],
    "import MODE_GOSSIP",
)

patch_first(
    RM,
    [
        (
            '    do_logreg_hybrid = app == "logreg" and sync_mode == MODE_HYBRID_PS_ALLREDUCE\n',
            '    do_logreg_hybrid = app == "logreg" and sync_mode == MODE_HYBRID_PS_ALLREDUCE\n'
            '    do_logreg_gossip = app == "logreg" and sync_mode == MODE_GOSSIP\n',
        )
    ],
    "do_logreg_gossip derived flag",
)

patch_first(
    RM,
    [
        (
            '        print("  [LogReg Hybrid PS] Scalar coordinator thread started (P3-10)")',
            '        print("  [LogReg Hybrid PS] Scalar coordinator thread started (P3-10)")\n'
            """
    if do_logreg_gossip:
        print(
            "  [LogReg Gossip] Decentralized mode — workers exchange over the ring; "
            "no root coordinator thread (P3-11)"
        )""",
        )
    ],
    "gossip no-coordinator info print",
)

# ───────────────────────── 6. root_process.py (strict) ─────────────────────────
RP = "mpj_spark/core/root_process.py"

patch_first(
    RP,
    [
        (
            """from mpj_spark.core.sync_modes import (
    MODE_HYBRID_PS_ALLREDUCE,""",
            """from mpj_spark.core.sync_modes import (
    MODE_GOSSIP,
    MODE_HYBRID_PS_ALLREDUCE,""",
        )
    ],
    "import MODE_GOSSIP",
)

# ───────────────────────── 7–10. cosmetic edits (lenient) ─────────────────────────
patch_first(
    RM,
    [
        (
            '        else f"Hybrid PS+Allreduce ({logreg_iter} iters)"\n'
            '        if (app == "logreg" and sync_mode == MODE_HYBRID_PS_ALLREDUCE)\n',
            '        else f"Decentralized Gossip ({logreg_iter} rounds)"\n'
            '        if (app == "logreg" and sync_mode == MODE_GOSSIP)\n'
            '        else f"Hybrid PS+Allreduce ({logreg_iter} iters)"\n'
            '        if (app == "logreg" and sync_mode == MODE_HYBRID_PS_ALLREDUCE)\n',
        )
    ],
    "agg_mode header label for gossip",
    strict=False,
)

patch_first(
    RM,
    [
        (
            '        "logreg_features": logreg_features,\n        "results_dir": results_dir,',
            '        "logreg_features": logreg_features,\n'
            '        "gossip_fanout": gossip_fanout,\n'
            '        "results_dir": results_dir,',
        )
    ],
    "worker_cfg_base gains gossip_fanout",
    strict=False,
)

patch_first(
    RP,
    [
        (
            '"Post-hoc row-weighted average — no sync (M1)"',
            '("Decentralized gossip consensus (P3-11)" if sync_mode == MODE_GOSSIP '
            'else "Post-hoc row-weighted average — no sync (M1)")',
        )
    ],
    "aggregation agg_mode label for gossip",
    strict=False,
)

patch_first(
    "mpj_spark/core/main_mpi.py",
    [
        (
            'help="Cross-driver synchronization mode (ps_sync_fedavg_mpi | ps_sync_fedavg_queue | allreduce_mpi | ps_async | hybrid_ps_allreduce | none)"',
            'help="Cross-driver synchronization mode (ps_sync_fedavg_mpi | ps_sync_fedavg_queue | allreduce_mpi | ps_async | hybrid_ps_allreduce | gossip | none)"',
        )
    ],
    "--sync-mode help text lists gossip",
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
