#!/usr/bin/env python3
"""scripts/analyze_sync_benchmark.py
P3-12 (Issue #64): analyzer for the sync-mode benchmark output.

Reads results/benchmark/manifest.csv plus the per-run metrics CSVs and
produces, under results/benchmark/analysis/:

  summary.csv      - one row per (arm, mode, workers): wall-clock,
                     accuracy, final |w|, mean per-round iter time,
                     instrumented sync-channel time, async staleness
  convergence.csv  - per (arm, mode, iteration): mean/min/max weight_norm
                     across workers (input for the convergence-curve plot)
  report.md        - README/thesis-ready results fragment
  plots/*.png      - convergence curves + sync-overhead bars
                     (only when matplotlib is installed)

Sync-overhead attribution is direct where the mode instruments it:
  hybrid_ps_allreduce -> allreduce_time_s + ps_time_s per round
  gossip              -> gossip_time_s per round
  ps_async            -> no barrier; staleness distribution instead
  M3 allreduce_mpi    -> the module logs per-epoch spark=/sync= splits
For the remaining modes the sync phase is folded into iter_time_s, so
the cross-mode comparison uses wall-clock + round-time distributions.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # plots are optional; tables are the deliverable
    plt = None

SYNC_COLUMNS = {
    "hybrid_ps_allreduce": ["allreduce_time_s", "ps_time_s"],
    "gossip": ["gossip_time_s"],
}

SUMMARY_FIELDS = [
    "arm",
    "mode",
    "workers",
    "wall_clock_s",
    "weighted_accuracy",
    "final_weight_norm",
    "mean_iter_time_s",
    "sync_channel_time_s",
    "mean_staleness",
    "max_staleness",
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def collect_worker_rows(run_dir: str) -> list[dict]:
    """All worker metrics CSVs in a run dir, as raw dict rows.

    Covers worker_*_metrics.csv (fedavg/async/hybrid/gossip) and
    worker_*_iter_metrics.csv (queue/none), deduplicated by path.
    """
    rd = Path(run_dir)
    paths = set(rd.glob("worker_*_metrics.csv")) | set(rd.glob("worker_*_iter_metrics.csv"))
    rows = []
    for path in sorted(paths):
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _staleness_stats(run_dir: str):
    path = Path(run_dir) / "logreg_async_ps_staleness.csv"
    if not path.exists():
        return None, None
    with open(path, newline="", encoding="utf-8") as f:
        vals = [_f(r.get("staleness")) for r in csv.DictReader(f)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return round(statistics.fmean(vals), 4), int(max(vals))


def build_summary_rows(manifest_rows: list[dict], base_dir: str) -> list[dict]:
    out = []
    for m in manifest_rows:
        if str(m.get("exit_code", "0")) not in ("0", "0.0"):
            continue
        run_dir = m["run_dir"]
        rows = collect_worker_rows(run_dir)
        iter_times = [_f(r.get("iter_time_s")) for r in rows]
        iter_times = [t for t in iter_times if t is not None]

        sync_time = None
        cols = SYNC_COLUMNS.get(m["mode"], [])
        if cols and rows:
            per_worker = len({r.get("worker_id") for r in rows}) or 1
            sync_time = round(
                sum(_f(r.get(c)) or 0.0 for r in rows for c in cols) / per_worker, 6
            )

        mean_stale, max_stale = _staleness_stats(run_dir)
        out.append(
            {
                "arm": m["arm"],
                "mode": m["mode"],
                "workers": m["workers"],
                "wall_clock_s": m.get("wall_clock_s") or m.get("elapsed_s"),
                "weighted_accuracy": m.get("weighted_accuracy"),
                "final_weight_norm": m.get("final_weight_norm"),
                "mean_iter_time_s": round(statistics.fmean(iter_times), 6) if iter_times else None,
                "sync_channel_time_s": sync_time,
                "mean_staleness": mean_stale,
                "max_staleness": max_stale,
            }
        )
    return out


def build_convergence_rows(manifest_rows: list[dict]) -> list[dict]:
    """Per (arm, mode, iteration): weight_norm spread across workers."""
    groups = defaultdict(list)
    for m in manifest_rows:
        if str(m.get("exit_code", "0")) not in ("0", "0.0"):
            continue
        for r in collect_worker_rows(m["run_dir"]):
            it, wn = _f(r.get("iteration")), _f(r.get("weight_norm"))
            if it is not None and wn is not None:
                groups[(m["arm"], m["mode"], m["workers"], int(it))].append(wn)
    rows = []
    for (arm, mode, workers, it), vals in sorted(groups.items()):
        rows.append(
            {
                "arm": arm,
                "mode": mode,
                "workers": workers,
                "iteration": it,
                "mean_weight_norm": round(statistics.fmean(vals), 8),
                "min_weight_norm": round(min(vals), 8),
                "max_weight_norm": round(max(vals), 8),
            }
        )
    return rows


def render_markdown(summary_rows: list[dict]) -> str:
    """README/thesis-ready results fragment (one table per arm)."""
    lines = [
        "## P3-12 — Sync-Strategy Benchmark (Issue #64)",
        "",
        "Identical workload per run (fixed dataset, features, iterations, worker",
        "count).  sync_channel_time_s is the instrumented per-round sync cost",
        "(hybrid: allreduce+ps; gossip: ring exchange); ps_async carries no",
        "barrier, so its cost appears as staleness instead.",
        "",
    ]
    for arm in sorted({r["arm"] for r in summary_rows}):
        lines += [
            f"### Arm: {arm}",
            "",
            "| Mode | Workers | Wall-clock (s) | Accuracy | Final \\|w\\| | Mean iter (s) | Sync channel (s) | Mean staleness |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in summary_rows:
            if r["arm"] != arm:
                continue
            lines.append(
                "| {mode} | {workers} | {wall} | {acc} | {wn} | {it} | {sync} | {stale} |".format(
                    mode=r["mode"],
                    workers=r["workers"],
                    wall=r["wall_clock_s"],
                    acc=r["weighted_accuracy"],
                    wn=r["final_weight_norm"],
                    it=r["mean_iter_time_s"],
                    sync=r["sync_channel_time_s"] if r["sync_channel_time_s"] is not None else "—",
                    stale=r["mean_staleness"] if r["mean_staleness"] is not None else "—",
                )
            )
        lines.append("")
    lines += [
        "### Findings (fill after analysis)",
        "",
        "- Homogeneous arm: ...",
        "- Throttled arm (rankfile-pinned worker): ...",
        "- Cross-reference vs literature (Chen et al. stragglers; Xie et al.",
        "  FedAsync staleness damping; Lian et al. AD-PSGD heterogeneity): ...",
        "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_plots(convergence_rows, summary_rows, plots_dir: Path) -> None:
    if plt is None:
        print("matplotlib not installed - skipping plots (tables are complete)")
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    by_arm_mode = defaultdict(list)
    for r in convergence_rows:
        by_arm_mode[(r["arm"], r["mode"], r["workers"])].append(r)
    for (arm, mode, workers), rows in by_arm_mode.items():
        rows = sorted(rows, key=lambda r: r["iteration"])
        fig, ax = plt.subplots()
        ax.plot([r["iteration"] for r in rows], [r["mean_weight_norm"] for r in rows], marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean |w|")
        ax.set_title(f"{mode} | {arm} | workers={workers}")
        fig.savefig(plots_dir / f"convergence_{arm}_{mode}_w{workers}.png", dpi=120)
        plt.close(fig)

    sync_rows = [r for r in summary_rows if r["sync_channel_time_s"] is not None]
    if sync_rows:
        fig, ax = plt.subplots()
        labels = [f"{r['mode']}|{r['arm']}|w{r['workers']}" for r in sync_rows]
        ax.bar(range(len(sync_rows)), [r["sync_channel_time_s"] for r in sync_rows])
        ax.set_xticks(range(len(sync_rows)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("instrumented sync time (s, per worker)")
        ax.set_title("Per-round sync-channel cost by mode")
        fig.tight_layout()
        fig.savefig(plots_dir / "sync_channel_time.png", dpi=120)
        plt.close(fig)
    print(f"plots -> {plots_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python scripts/analyze_sync_benchmark.py",
        description="P3-12 benchmark analyzer (Issue #64).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-dir", default="results/benchmark")
    args = p.parse_args(argv)

    base = Path(args.base_dir)
    manifest_path = base / "manifest.csv"
    with open(manifest_path, newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    summary = build_summary_rows(manifest_rows, str(base))
    convergence = build_convergence_rows(manifest_rows)

    out_dir = base / "analysis"
    _write_csv(out_dir / "summary.csv", summary, SUMMARY_FIELDS)
    _write_csv(
        out_dir / "convergence.csv",
        convergence,
        ["arm", "mode", "workers", "iteration", "mean_weight_norm", "min_weight_norm", "max_weight_norm"],
    )
    (out_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    _make_plots(convergence, summary, out_dir / "plots")

    print(f"summary rows: {len(summary)}  convergence rows: {len(convergence)}")
    print(f"analysis -> {out_dir}")


if __name__ == "__main__":
    main()
