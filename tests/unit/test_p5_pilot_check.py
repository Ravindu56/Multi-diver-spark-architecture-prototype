"""Tests for scripts/p5_pilot_check.py (P5-04 VM RAM decision rule)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pc = _load("p5_pilot_check")

STACK = "mpj_mpi-root.1 swarm-manager Running 1 hour ago\n"


def _psi(t0: int, secs: int, mem_full_us_per_s: int) -> str:
    lines = []
    for i in range(secs + 1):
        t = t0 + i
        for res, kind, rate in [
            ("cpu", "some", 1000),
            ("memory", "some", 2 * mem_full_us_per_s),
            ("memory", "full", mem_full_us_per_s),
            ("io", "some", 0),
        ]:
            lines.append(f"{t} {res} {kind} avg10=0.00 avg60=0.00 avg300=0.00 total={rate * i}")
    return "\n".join(lines) + "\n"


def _make_run(
    root: Path,
    name: str,
    rc: int = 0,
    rep: int = 0,
    *,
    extra_tasks: int = 0,
    mem_full_us_per_s: int = 0,
    psi_nodes=("host", "10.0.0.1"),
    write_stack=True,
):
    d = root / name
    d.mkdir(parents=True)
    if write_stack:
        (d / "stack-before.txt").write_text(STACK)
        (d / "stack-after.txt").write_text(STACK * (1 + extra_tasks))
    for node in psi_nodes:
        (d / f"psi-{node}.log").write_text(_psi(1_000, 10, mem_full_us_per_s))
    runlog = root / "runlog.csv"
    if not runlog.exists():
        runlog.write_text("name,cell,np,workload,param,rep,rc,wall_s,start\n")
    with runlog.open("a") as fh:
        fh.write(f"{name},B,5,kmeans,500mb,{rep},{rc},120,2026-11-15T12:00:00+05:30\n")


def test_parse_psi_computes_stall_percentage(tmp_path):
    p = tmp_path / "psi-host.log"
    p.write_text(_psi(1_000, 10, 30_000))  # 30 ms stalled per second = 3%
    stall = pc.parse_psi(p)
    assert stall.span_s == 10
    assert abs(stall.pct[("memory", "full")] - 3.0) < 1e-9
    assert abs(stall.pct[("memory", "some")] - 6.0) < 1e-9


def test_parse_psi_unavailable_returns_none(tmp_path):
    p = tmp_path / "psi-host.log"
    p.write_text("PSI unavailable: /proc/pressure not readable\n")
    assert pc.parse_psi(p) is None


def test_clean_pilot_passes(tmp_path):
    _make_run(tmp_path, "km-B-500mb-r0", mem_full_us_per_s=10_000)  # 1%
    (c,) = pc.check_dir(tmp_path, 5.0, all_reps=False)
    assert c.verdict == "PASS"
    assert c.new_tasks == 0


def test_failure_modes(tmp_path):
    _make_run(tmp_path, "rc", rc=124)
    _make_run(tmp_path, "tasks", extra_tasks=2)
    _make_run(tmp_path, "pressure", mem_full_us_per_s=80_000)  # 8%
    checks = {c.name: c for c in pc.check_dir(tmp_path, 5.0, all_reps=False)}
    assert checks["rc"].verdict == "FAIL" and "exit code 124" in checks["rc"].reasons
    assert checks["tasks"].verdict == "FAIL" and checks["tasks"].new_tasks == 2
    assert checks["pressure"].verdict == "FAIL"
    assert "host" in checks["pressure"].reasons[0]


def test_missing_psi_is_incomplete_not_pass(tmp_path):
    _make_run(tmp_path, "nopsi", psi_nodes=())
    (c,) = pc.check_dir(tmp_path, 5.0, all_reps=False)
    assert c.verdict == "INCOMPLETE"


def test_only_rep0_unless_all_reps(tmp_path):
    _make_run(tmp_path, "r0", rep=0)
    _make_run(tmp_path, "r1", rep=1, rc=1)
    assert [c.name for c in pc.check_dir(tmp_path, 5.0, all_reps=False)] == ["r0"]
    assert len(pc.check_dir(tmp_path, 5.0, all_reps=True)) == 2


def test_main_exit_codes_and_csv(tmp_path):
    _make_run(tmp_path, "ok")
    out = tmp_path / "check.csv"
    assert pc.main([str(tmp_path), "--csv", str(out)]) == 0
    assert out.read_text().startswith("name,verdict,rc,wall_s,new_tasks,")
    _make_run(tmp_path, "bad", rc=1)
    assert pc.main([str(tmp_path)]) == 1
    assert pc.main([str(tmp_path / "missing")]) == 2
