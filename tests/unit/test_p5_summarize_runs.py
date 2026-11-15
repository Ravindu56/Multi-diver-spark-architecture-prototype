"""Tests for scripts/p5_summarize_runs.py (P5 per-configuration run summary)."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sr = _load("p5_summarize_runs")

RUNLOG_HEADER = "name,cell,np,workload,param,rep,rc,wall_s,start\n"


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _lr_run(run_dir: Path, name: str, total: float, final_loss: str, ranks: int = 2, **kw):
    """Write a LogReg run in the runner layout: <run_dir>/<name>/<name>/..."""
    m = run_dir / name / name
    m.mkdir(parents=True)
    epochs = kw.get("epochs", 3)
    fields = [
        "epoch",
        "spark_time_s",
        "sync_time_s",
        "epoch_time_s",
        "grad_norm",
        "global_loss",
        "weight_norm",
        "loss_time_s",
    ]
    for r in range(ranks):
        weight = kw.get("weights", ["0.5"] * ranks)[r]
        rows = [
            {
                "epoch": e,
                "spark_time_s": 1.0,
                "sync_time_s": 0.25,
                "epoch_time_s": 2.0,
                "grad_norm": 0.1,
                "global_loss": final_loss if e == epochs - 1 else "0.7",
                "weight_norm": weight,
                "loss_time_s": 0.75,
            }
            for e in range(epochs)
        ]
        _write_csv(m / f"logreg_rank{r}_epochs.csv", fields, rows)
    (m / "logreg_rank0_run.json").write_text(
        json.dumps({"total_time_s": total, "epochs_run": epochs, "converged": True})
    )


def _km_run(run_dir: Path, name: str, total: float, wcss: str, iters: int = 2):
    m = run_dir / name / name
    m.mkdir(parents=True)
    fields = ["iteration", "spark_time_s", "sync_time_s", "iter_time_s", "centroid_shift"]
    fields += ["global_wcss", "sync_overhead_pct"]
    for r in range(2):
        rows = [
            {
                "iteration": i + 1,
                "spark_time_s": 4.0,
                "sync_time_s": 1.0,
                "iter_time_s": 5.0,
                "centroid_shift": "0.01",
                "global_wcss": wcss,
                "sync_overhead_pct": 20.0,
            }
            for i in range(iters)
        ]
        _write_csv(m / f"kmeans_metrics_rank{r}.csv", fields, rows)
    # aggregated file must not be picked up as a rank file
    _write_csv(m / "kmeans_metrics_aggregated.csv", ["rank"] + fields, [])
    run = {"total_time_s": total, "iterations_run": iters, "converged": False}
    (m / "kmeans_metrics_rank0.json").write_text(json.dumps({"run": run, "derived": {}}))


@pytest.fixture
def lr_dir(tmp_path):
    d = tmp_path / "20261115-000000-cellA"
    d.mkdir()
    log = [RUNLOG_HEADER]
    # pilot (excluded by default), two measured reps, one failed measured rep
    for rep, rc, wall, total in [(0, 0, 30, 25.0), (1, 0, 20, 17.0), (2, 0, 22, 19.0)]:
        name = f"lr-A-f10-r{rep}"
        log.append(f"{name},A,3,logreg,f10,{rep},{rc},{wall},2026-11-15T00:00:00+05:30\n")
        _lr_run(d, name, total, "0.65")
    log.append("lr-A-f10-r3,A,3,logreg,f10,3,1,5,2026-11-15T00:00:00+05:30\n")
    (d / "runlog.csv").write_text("".join(log))
    return d


def test_measured_reps_only_and_failed_counted(lr_dir):
    [c] = sr.summarize([str(lr_dir)])
    row = sr.to_row(c)
    assert (row["workload"], row["cell"], row["param"]) == ("logreg", "A", "f10")
    assert row["reps"] == 2 and row["failed"] == 1
    assert row["wall_s_mean"] == 21.0 and row["app_s_mean"] == 18.0
    assert row["startup_s_mean"] == 3.0
    # setup = app - sum(epoch_time_s) = 17 - 6 and 19 - 6
    assert row["setup_s_mean"] == 12.0
    assert (row["spark_s"], row["loss_s"], row["sync_s"], row["iter_s"]) == (1.0, 0.75, 0.25, 2.0)
    assert row["sync_pct"] == 12.5
    assert row["iterations"] == "3,3"
    assert row["all_converged"] is True and row["same_model_all_ranks"] is True
    assert row["final_objective"] == "0.65"


def test_include_pilot(lr_dir):
    [c] = sr.summarize([str(lr_dir)], include_pilot=True)
    assert len(c.apps) == 3


def test_different_final_model_across_ranks_is_flagged(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "runlog.csv").write_text(RUNLOG_HEADER + "lr-B-f50-r1,B,5,logreg,f50,1,0,10,x\n")
    _lr_run(d, "lr-B-f50-r1", 8.0, "0.6", ranks=2, weights=["0.5", "0.51"])
    [c] = sr.summarize([str(d)])
    assert sr.to_row(c)["same_model_all_ranks"] is False


def test_kmeans_and_multiple_dirs_sorted(tmp_path, lr_dir):
    k = tmp_path / "km"
    k.mkdir()
    (k / "runlog.csv").write_text(
        RUNLOG_HEADER
        + "km-B-1024mb-r1,B,5,kmeans,1024mb,1,0,40,x\n"
        + "km-B-500mb-r1,B,5,kmeans,500mb,1,0,20,x\n"
    )
    _km_run(k, "km-B-1024mb-r1", 36.0, "200.5")
    _km_run(k, "km-B-500mb-r1", 18.0, "100.5")
    rows = [sr.to_row(c) for c in sr.summarize([str(lr_dir), str(k)])]
    assert [(r["workload"], r["param"]) for r in rows] == [
        ("kmeans", "500mb"),
        ("kmeans", "1024mb"),
        ("logreg", "f10"),
    ]
    km = rows[0]
    assert km["loss_s"] == "" and km["sync_pct"] == 20.0
    assert km["setup_s_mean"] == 8.0  # 18 - 2 * 5
    assert km["all_converged"] is False and km["final_objective"] == "100.5"


def test_main_writes_csv_and_bad_dir_returns_2(tmp_path, lr_dir, capsys):
    out = tmp_path / "s.csv"
    assert sr.main([str(lr_dir), "--csv", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert list(rows[0].keys()) == sr.CSV_FIELDS and rows[0]["reps"] == "2"
    assert sr.main([str(tmp_path / "missing")]) == 2
    assert "no runlog.csv" in capsys.readouterr().err
