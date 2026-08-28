#!/usr/bin/env python3
"""scripts/run_sync_benchmark.py
P3-12 (Issue #64): controlled benchmark of LogReg synchronization strategies.

Runs every selected sync mode over an identical workload configuration
(fixed dataset, feature count, iteration count, worker count) and records
a manifest row per run.  Two arms:

  homogeneous - all workers get the same core budget (default)
  throttled   - an OpenMPI rankfile binds one worker rank to a reduced
                slot set, inducing worker heterogeneity at the OS level
                (research gap iii: sync-barrier behaviour under
                heterogeneous resource conditions).  Note the throttle is
                real: the worker's Spark local[N] budget is unchanged, so
                the pinned rank is genuinely over-subscribed.

Usage (from repo root):

    python scripts/run_sync_benchmark.py \
        --input ./shared_storage/logreg_data.csv \
        --workers 2 4 --arms homogeneous throttled \
        --logreg-iter 10 --logreg-features 10 --gossip-fanout 1

Per-run artifacts land in results/benchmark/<arm>/<mode>_w<workers>/
(log.txt plus the mode's metrics CSVs); the manifest is
results/benchmark/manifest.csv.  Analyze with
scripts/analyze_sync_benchmark.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Modes dispatched through main_mpi.  M3 (allreduce_mpi) runs via its own
# standalone module entry and is only included with --include-m3.
WIRED_MODES = [
    "none",
    "ps_sync_fedavg_queue",
    "ps_sync_fedavg_mpi",
    "ps_async",
    "hybrid_ps_allreduce",
    "gossip",
]
M3_MODE = "allreduce_mpi"

MANIFEST_FIELDS = [
    "arm",
    "mode",
    "workers",
    "np",
    "gossip_fanout",
    "exit_code",
    "elapsed_s",
    "final_weight_norm",
    "weighted_accuracy",
    "wall_clock_s",
    "proc_time_s",
    "run_dir",
    "log_path",
]


@dataclass(frozen=True)
class RunSpec:
    arm: str
    mode: str
    workers: int
    np_: int  # workers + 1 (root rank)
    run_dir: str
    log_path: str
    gossip_fanout: int | None


def build_run_plan(modes, workers_list, arms, base_dir, gossip_fanout=None) -> list[RunSpec]:
    """Full grid: arms x modes x worker counts, one RunSpec per cell."""
    plan = []
    for arm in arms:
        for mode in modes:
            for workers in workers_list:
                run_dir = os.path.join(base_dir, arm, f"{mode}_w{workers}")
                plan.append(
                    RunSpec(
                        arm=arm,
                        mode=mode,
                        workers=workers,
                        np_=workers + 1,
                        run_dir=run_dir,
                        log_path=os.path.join(run_dir, "log.txt"),
                        gossip_fanout=gossip_fanout if mode == "gossip" else None,
                    )
                )
    return plan


def render_rankfile(np_ranks, total_cores, throttle_rank=None, throttle_slots=2) -> str:
    """OpenMPI rankfile binding each rank to a contiguous slot range.

    Balanced split of total_cores across ranks; when throttle_rank is
    set, that rank is pinned to throttle_slots cores and the remaining
    ranks keep their even share (the throttled worker's Spark fit phase
    is then genuinely compute-starved).
    """
    base = max(1, total_cores // np_ranks)
    lines = []
    cursor = 0
    for r in range(np_ranks):
        n = max(1, throttle_slots) if (throttle_rank is not None and r == throttle_rank) else base
        hi = cursor + n - 1
        lines.append(f"rank {r}=localhost slot={cursor}-{hi}")
        cursor = hi + 1
    return "\n".join(lines) + "\n"


def build_command(
    spec: RunSpec,
    input_path: str,
    logreg_iter: int,
    logreg_features: int,
    rankfile_path: str | None = None,
    python: str = sys.executable,
) -> list[str]:
    """mpirun command for one run.  Wired modes go through main_mpi;
    M3 runs via its standalone all-ranks collective module entry."""
    cmd = ["mpirun", "--oversubscribe"]
    if rankfile_path:
        cmd += ["--rankfile", rankfile_path]
    cmd += ["-np", str(spec.np_), python]
    if spec.mode == M3_MODE:
        # NOTE: verify the module's argparse via --help before enabling;
        # the M3 runner is a standalone entry point, not main_mpi dispatch.
        cmd += [
            "-m",
            "mpj_spark.applications.logreg.allreduce",
            "--input",
            input_path,
            "--epochs",
            str(logreg_iter),
        ]
        return cmd
    cmd += [
        "-m",
        "mpj_spark.core.main_mpi",
        "--app",
        "logreg",
        "--sync-mode",
        spec.mode,
        "--input",
        input_path,
        "--logreg-iter",
        str(logreg_iter),
        "--logreg-features",
        str(logreg_features),
        "--results-dir",
        spec.run_dir,
    ]
    if spec.gossip_fanout is not None:
        cmd += ["--gossip-fanout", str(spec.gossip_fanout)]
    return cmd


_PATTERNS = {
    "final_weight_norm": r"Final \|w\|\s*:\s*([0-9.]+)",
    "weighted_accuracy": r"Weighted accuracy:\s+([0-9.]+)",
    "wall_clock_s": r"Total Wall-clock Time\s+([0-9.]+)\s*s",
    "proc_time_s": r"Processing Time \(avg fit\)\s+([0-9.]+)\s*s",
}


def parse_log_metrics(log_text: str) -> dict:
    """Best-effort parse of the root timing/aggregation block."""
    out = {}
    for key, pat in _PATTERNS.items():
        m = re.search(pat, log_text)
        out[key] = float(m.group(1)) if m else None
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python scripts/run_sync_benchmark.py",
        description="P3-12 sync-mode benchmark orchestrator (Issue #64).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default="./shared_storage/logreg_data.csv")
    p.add_argument("--modes", nargs="*", default=WIRED_MODES, choices=WIRED_MODES + [M3_MODE])
    p.add_argument(
        "--include-m3",
        action="store_true",
        help="Also run the M3 standalone Allreduce module entry (verify its CLI via --help first).",
    )
    p.add_argument("--workers", nargs="*", type=int, default=[2, 4])
    p.add_argument("--arms", nargs="*", default=["homogeneous"], choices=["homogeneous", "throttled"])
    p.add_argument(
        "--throttle-rank",
        type=int,
        default=1,
        help="MPI rank pinned to the reduced slot set in the throttled arm (rank 1 = worker 0).",
    )
    p.add_argument("--throttle-slots", type=int, default=2)
    p.add_argument("--total-cores", type=int, default=os.cpu_count() or 8)
    p.add_argument("--logreg-iter", type=int, default=10)
    p.add_argument("--logreg-features", type=int, default=10)
    p.add_argument(
        "--gossip-fanout",
        type=int,
        default=None,
        help="Forwarded to gossip runs (use 1 to expose ring partial consensus at 4+ workers).",
    )
    p.add_argument("--base-dir", default="results/benchmark")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and commands without running.")
    args = p.parse_args(argv)

    modes = list(args.modes)
    if args.include_m3 and M3_MODE not in modes:
        modes.append(M3_MODE)

    plan = build_run_plan(modes, args.workers, args.arms, args.base_dir, args.gossip_fanout)
    manifest_path = os.path.join(args.base_dir, "manifest.csv")
    write_header = not os.path.exists(manifest_path)
    os.makedirs(args.base_dir, exist_ok=True)

    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        for spec in plan:
            os.makedirs(spec.run_dir, exist_ok=True)
            rankfile_path = None
            if spec.arm == "throttled":
                rf = render_rankfile(
                    spec.np_,
                    args.total_cores,
                    throttle_rank=args.throttle_rank,
                    throttle_slots=args.throttle_slots,
                )
                rankfile_path = os.path.join(spec.run_dir, "rankfile")
                Path(rankfile_path).write_text(rf, encoding="utf-8")
            cmd = build_command(spec, args.input, args.logreg_iter, args.logreg_features, rankfile_path)
            print(f"\n=== [{spec.arm}] {spec.mode} | workers={spec.workers} ===")
            print(" ".join(cmd))
            if args.dry_run:
                continue
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.perf_counter() - t0
            Path(spec.log_path).write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
            metrics = parse_log_metrics(proc.stdout)
            writer.writerow(
                {
                    "arm": spec.arm,
                    "mode": spec.mode,
                    "workers": spec.workers,
                    "np": spec.np_,
                    "gossip_fanout": spec.gossip_fanout if spec.gossip_fanout is not None else "",
                    "exit_code": proc.returncode,
                    "elapsed_s": round(elapsed, 3),
                    "final_weight_norm": metrics["final_weight_norm"],
                    "weighted_accuracy": metrics["weighted_accuracy"],
                    "wall_clock_s": metrics["wall_clock_s"],
                    "proc_time_s": metrics["proc_time_s"],
                    "run_dir": spec.run_dir,
                    "log_path": spec.log_path,
                }
            )
            mf.flush()
            print(f"exit={proc.returncode}  elapsed={elapsed:.1f}s  -> {spec.log_path}")

    print(f"\nManifest: {manifest_path}")
    print(f"Next: python scripts/analyze_sync_benchmark.py --base-dir {args.base_dir}")


if __name__ == "__main__":
    main()
