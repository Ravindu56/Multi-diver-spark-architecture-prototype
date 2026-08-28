#!/usr/bin/env python3
"""apply_p3_12_wiring.py — P3-12 (Issue #64) --gossip-fanout CLI patch.

One-shot migration script.  Run ONCE from the repo root:

    python scripts/apply_p3_12_wiring.py

Adds a --gossip-fanout CLI flag to mpj_spark/core/main_mpi.py so the
benchmark harness can override the ring fanout per run (the 4-worker
partial-consensus arm needs fanout=1; the inherited run_root_mpi
default is 3).  Behaviour is unchanged when the flag is absent.

  STRICT anchors (must match exactly once):
    1. insert the --gossip-fanout add_argument after the --sync-mode block
    2. conditional gossip_fanout pass-through in the run_root_mpi call

Then: ruff check . && pytest tests/unit/ -v
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

FAILURES: list[str] = []


def patch_first(path, candidates, label):
    p = Path(path)
    src = p.read_text(encoding="utf-8")
    for old, new in candidates:
        n = src.count(old)
        if n == 1:
            p.write_text(src.replace(old, new), encoding="utf-8")
            print(f"OK    {path}: {label}")
            return True
    print(f"FAIL  {path}: '{label}' — no unique anchor match")
    FAILURES.append(f"{path}: {label}")
    return False


MP = "mpj_spark/core/main_mpi.py"

patch_first(
    MP,
    [
        (
            'help="Cross-driver synchronization mode (ps_sync_fedavg_mpi | ps_sync_fedavg_queue | allreduce_mpi | ps_async | hybrid_ps_allreduce | gossip | none)",\n    )',
            'help="Cross-driver synchronization mode (ps_sync_fedavg_mpi | ps_sync_fedavg_queue | allreduce_mpi | ps_async | hybrid_ps_allreduce | gossip | none)",\n'
            "    )\n"
            "\n"
            "    p.add_argument(\n"
            '        "--gossip-fanout",\n'
            "        type=int,\n"
            "        default=None,\n"
            '        help="Ring distance contacted per gossip round (default: inherit the run_root_mpi gossip_fanout default).",\n'
            "    )",
        ),
        (
            'help="Cross-driver synchronization mode (ps_sync_fedavg_mpi | ps_sync_fedavg_queue | allreduce_mpi | ps_async | hybrid_ps_allreduce | none)",\n    )',
            'help="Cross-driver synchronization mode (ps_sync_fedavg_mpi | ps_sync_fedavg_queue | allreduce_mpi | ps_async | hybrid_ps_allreduce | gossip | none)",\n'
            "    )\n"
            "\n"
            "    p.add_argument(\n"
            '        "--gossip-fanout",\n'
            "        type=int,\n"
            "        default=None,\n"
            '        help="Ring distance contacted per gossip round (default: inherit the run_root_mpi gossip_fanout default).",\n'
            "    )",
        ),
    ],
    "--gossip-fanout CLI argument",
)

patch_first(
    MP,
    [
        (
            "        sync_mode=sync_mode,\n",
            "        sync_mode=sync_mode,\n"
            "        **({\"gossip_fanout\": args.gossip_fanout} "
            "if args.gossip_fanout is not None else {}),\n",
        )
    ],
    "conditional gossip_fanout pass-through to run_root_mpi",
)

print()
try:
    py_compile.compile(MP, doraise=True)
    print(f"compile OK  {MP}")
except py_compile.PyCompileError as exc:
    FAILURES.append(f"{MP}: compile error")
    print(f"compile FAIL  {MP}\n{exc}")

if FAILURES:
    print("\nPATCH INCOMPLETE:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("\nDone. Next: ruff check . && pytest tests/unit/ -v")
