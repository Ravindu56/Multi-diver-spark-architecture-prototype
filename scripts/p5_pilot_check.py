"""Apply the P5-04 VM RAM decision rule to pilot runs from scripts/p5_run_cell.sh.

Reads one runner output directory ($HOME/p5-runs/<timestamp>-cell<X>) and, for each
pilot run (rep 0, or every rep with --all-reps), checks:

  1. exit code 0 (the run finished inside the runner timeout);
  2. no new Swarm tasks between stack-before.txt and stack-after.txt (no task
     replacement or restart during the run);
  3. memory "full" pressure stall on every node at or below --max-mem-full-pct of the
     sampled interval, from the cumulative total= counters in psi-*.log.

A run passes only if all three hold. A node with no usable PSI samples makes the run
INCOMPLETE rather than PASS. See docs/p5-04-experimental-design.md (VM RAM decision).

Usage:
    python scripts/p5_pilot_check.py ~/p5-runs/20261115-120000-cellB
    python scripts/p5_pilot_check.py RUN_DIR --csv pilot_check.csv

Exit status: 0 if every checked run passes, 1 otherwise, 2 on bad input.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MAX_MEM_FULL_PCT = 5.0


@dataclass
class PsiStall:
    """Stall percentage over the sampled interval, per resource and kind."""

    span_s: float
    pct: dict[tuple[str, str], float]


@dataclass
class RunCheck:
    name: str
    rc: int
    wall_s: int
    new_tasks: int | None
    mem_full_pct: dict[str, float | None] = field(default_factory=dict)
    mem_some_pct: dict[str, float | None] = field(default_factory=dict)
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)


def parse_psi(path: Path) -> PsiStall | None:
    """Parse a psi-*.log file written by the runner.

    Lines look like "<epoch_s> <cpu|memory|io> <some|full> avg10=.. ... total=<us>".
    Returns None if the file has fewer than two samples (or says PSI is unavailable).
    """
    first: dict[tuple[str, str], tuple[int, int]] = {}
    last: dict[tuple[str, str], tuple[int, int]] = {}
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0].isdigit() or not parts[-1].startswith("total="):
            continue
        try:
            ts, total = int(parts[0]), int(parts[-1].split("=", 1)[1])
        except ValueError:
            continue
        key = (parts[1], parts[2])
        first.setdefault(key, (ts, total))
        last[key] = (ts, total)
    pct: dict[tuple[str, str], float] = {}
    span = 0.0
    for key, (t0, v0) in first.items():
        t1, v1 = last[key]
        if t1 > t0:
            span = max(span, float(t1 - t0))
            pct[key] = 100.0 * (v1 - v0) / 1e6 / (t1 - t0)
    return PsiStall(span, pct) if pct else None


def count_tasks(path: Path) -> int | None:
    if not path.is_file():
        return None
    return sum(1 for line in path.read_text(errors="replace").splitlines() if line.strip())


def check_run(run_dir: Path, row: dict[str, str], max_mem_full_pct: float) -> RunCheck:
    name = row["name"]
    d = run_dir / name
    result = RunCheck(name=name, rc=int(row["rc"]), wall_s=int(row["wall_s"]), new_tasks=None)

    before, after = count_tasks(d / "stack-before.txt"), count_tasks(d / "stack-after.txt")
    if before is not None and after is not None:
        result.new_tasks = after - before

    psi_files = sorted(d.glob("psi-*.log"))
    for p in psi_files:
        node = p.stem.removeprefix("psi-")
        stall = parse_psi(p)
        result.mem_full_pct[node] = stall.pct.get(("memory", "full")) if stall else None
        result.mem_some_pct[node] = stall.pct.get(("memory", "some")) if stall else None

    if result.rc != 0:
        result.reasons.append(f"exit code {result.rc}")
    if result.new_tasks is None:
        result.reasons.append("missing stack snapshot")
    elif result.new_tasks > 0:
        result.reasons.append(f"{result.new_tasks} new Swarm task(s)")
    over = [n for n, v in result.mem_full_pct.items() if v is not None and v > max_mem_full_pct]
    if over:
        result.reasons.append(f"memory full stall > {max_mem_full_pct}% on {', '.join(over)}")
    missing = [n for n, v in result.mem_full_pct.items() if v is None]
    if not psi_files:
        missing = ["all nodes"]

    if result.reasons:
        result.verdict = "FAIL"
    elif missing:
        result.verdict = "INCOMPLETE"
        result.reasons.append(f"no PSI data for {', '.join(missing)}")
    else:
        result.verdict = "PASS"
    return result


def check_dir(run_dir: Path, max_mem_full_pct: float, all_reps: bool) -> list[RunCheck]:
    runlog = run_dir / "runlog.csv"
    if not runlog.is_file():
        raise FileNotFoundError(f"{runlog} not found")
    with runlog.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if all_reps or r["rep"] == "0"]
    return [check_run(run_dir, r, max_mem_full_pct) for r in rows]


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--max-mem-full-pct", type=float, default=DEFAULT_MAX_MEM_FULL_PCT)
    ap.add_argument("--all-reps", action="store_true", help="check every rep, not only rep 0")
    ap.add_argument("--csv", type=Path, help="also write the results to this CSV file")
    args = ap.parse_args(argv)

    try:
        checks = check_dir(args.run_dir, args.max_mem_full_pct, args.all_reps)
    except FileNotFoundError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if not checks:
        print("error: no matching runs in runlog.csv", file=sys.stderr)
        return 2

    print(f"threshold: memory full stall <= {args.max_mem_full_pct}% of the sampled interval")
    for c in checks:
        full = " ".join(f"{n}={_fmt(v)}" for n, v in sorted(c.mem_full_pct.items()))
        print(f"{c.verdict:<10} {c.name}  rc={c.rc} wall={c.wall_s}s new_tasks={c.new_tasks}")
        print(f"           mem_full_pct: {full or 'n/a'}")
        if c.reasons:
            print(f"           reasons: {'; '.join(c.reasons)}")

    if args.csv:
        nodes = sorted({n for c in checks for n in c.mem_full_pct})
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["name", "verdict", "rc", "wall_s", "new_tasks"]
                + [f"mem_full_pct_{n}" for n in nodes]
                + [f"mem_some_pct_{n}" for n in nodes]
                + ["reasons"]
            )
            for c in checks:
                w.writerow(
                    [c.name, c.verdict, c.rc, c.wall_s, c.new_tasks]
                    + [_fmt(c.mem_full_pct.get(n)) for n in nodes]
                    + [_fmt(c.mem_some_pct.get(n)) for n in nodes]
                    + ["; ".join(c.reasons)]
                )

    return 0 if all(c.verdict == "PASS" for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
