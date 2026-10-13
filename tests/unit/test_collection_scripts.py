"""Tests for scripts/compare_baseline.py and scripts/baseline_eval.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cb = _load("compare_baseline")
be = _load("baseline_eval")


def _runs(rows, **defaults):
    base = {"exit_code": 0, "rounds_cfg": 8, **defaults}
    return pd.DataFrame([{**base, **r} for r in rows])


def _curve(run_id, mode, workers, elapsed, losses):
    return pd.DataFrame(
        {
            "run_id": run_id,
            "mode": mode,
            "workers": workers,
            "round": list(range(1, len(losses) + 1)),
            "elapsed_s": elapsed,
            "loss_after": losses,
        }
    )


def test_kmeans_comparison_matches_total_cores_and_budget():
    fw = _runs(
        [
            {
                "workers": 2,
                "cores": 2,
                "mode": "gossip",
                "mean_round_s": 1.0,
                "wall_s": 20.0,
                "final_wcss": 100.0,
            },
            {
                "workers": 2,
                "cores": 2,
                "mode": "gossip",
                "mean_round_s": 1.2,
                "wall_s": 22.0,
                "final_wcss": 100.0,
            },
        ]
    )
    ps = "ps_sync_fedavg_mpi"
    base = _runs(
        [
            {
                "workers": 1,
                "cores": 4,
                "mode": ps,
                "mean_round_s": 2.2,
                "wall_s": 30.0,
                "final_wcss": 101.0,
            },
            # wrong total core budget: must not match
            {
                "workers": 1,
                "cores": 2,
                "mode": ps,
                "mean_round_s": 9.9,
                "wall_s": 99.0,
                "final_wcss": 1.0,
            },
            # wrong round budget: must not match
            {
                "workers": 1,
                "cores": 4,
                "mode": ps,
                "mean_round_s": 9.9,
                "wall_s": 99.0,
                "final_wcss": 1.0,
                "rounds_cfg": 4,
            },
            # failed run: must not match
            {
                "workers": 1,
                "cores": 4,
                "mode": ps,
                "mean_round_s": 9.9,
                "wall_s": 99.0,
                "final_wcss": 1.0,
                "exit_code": 1,
            },
        ]
    )
    out = cb.compare_kmeans(fw, base)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["baseline_cores"] == 4 and row["n_base"] == 1 and row["n_ok"] == 2
    assert row["round_speedup"] == pytest.approx(2.0)
    assert row["wall_speedup"] == pytest.approx(30 / 21)
    assert row["wcss_vs_base_pct"] == pytest.approx(100 * (100 / 101 - 1))


def test_kmeans_without_matching_baseline_gives_no_rows():
    fw = _runs(
        [
            {
                "workers": 4,
                "cores": 2,
                "mode": "gossip",
                "mean_round_s": 1.0,
                "wall_s": 20.0,
                "final_wcss": 100.0,
            }
        ]
    )
    base = _runs(
        [
            {
                "workers": 1,
                "cores": 4,
                "mode": "ps_sync_fedavg_mpi",
                "mean_round_s": 2.0,
                "wall_s": 30.0,
                "final_wcss": 101.0,
            }
        ]
    )
    assert cb.compare_kmeans(fw, base).empty


def test_logreg_speedup_uses_one_target_for_both_sides():
    ps = "ps_sync_fedavg_mpi"
    fw_runs = _runs(
        [
            {"run_id": "f1", "workers": 2, "cores": 2, "mode": ps},
            {"run_id": "f2", "workers": 2, "cores": 2, "mode": ps},
        ],
        rounds_cfg=10,
    )
    fw_rounds = pd.concat(
        [
            _curve("f1", ps, 2, [8, 16, 24], [0.5, 0.0229, 0.0229]),
            _curve("f2", ps, 2, [8, 16, 24], [0.5, 0.0229, 0.0229]),
        ]
    )
    base_runs = _runs(
        [
            {"run_id": "b1", "workers": 1, "cores": 4, "mode": ps},
            {"run_id": "b2", "workers": 1, "cores": 4, "mode": ps},
        ],
        rounds_cfg=10,
    )
    base_rounds = pd.concat(
        [
            _curve("b1", ps, 1, [15, 30, 45], [0.5, 0.5, 0.0229]),
            _curve("b2", ps, 1, [15, 30, 45], [0.5, 0.5, 0.4]),  # never reaches the target
        ]
    )
    out = cb.compare_logreg(fw_runs, fw_rounds, base_runs, base_rounds)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["median_ttt_s"] == 16.0 and row["n_reached"] == 2
    assert row["base_median_ttt_s"] == 45.0
    assert row["base_reached"] == 1 and row["n_base"] == 2
    assert row["speedup"] == pytest.approx(45 / 16)


def test_logreg_baseline_that_never_reaches_target_gives_nan_speedup():
    ps = "ps_sync_fedavg_mpi"
    fw_runs = _runs([{"run_id": "f1", "workers": 2, "cores": 2, "mode": ps}], rounds_cfg=10)
    fw_rounds = _curve("f1", ps, 2, [8, 16, 24], [0.5, 0.0229, 0.0229])
    base_runs = _runs([{"run_id": "b1", "workers": 1, "cores": 4, "mode": ps}], rounds_cfg=10)
    base_rounds = _curve("b1", ps, 1, [15, 30, 45], [0.5, 0.5, 0.4])
    out = cb.compare_logreg(fw_runs, fw_rounds, base_runs, base_rounds)
    assert len(out) == 1 and np.isnan(out.iloc[0]["speedup"])


def test_baseline_plan_gives_equal_total_threads():
    assert be.plan([1, 2, 4], 2) == [(2, 4), (4, 8)]
    assert be.plan([1], 2) == []


def test_baseline_dry_run_prints_single_driver_commands(capsys, tmp_path):
    modes_before = list(be.se.MODES)
    be.main(
        [
            "--app",
            "kmeans",
            "--compare-workers",
            "2",
            "4",
            "--cores",
            "2",
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "--workers 1 --cores 4" in out and "--workers 1 --cores 8" in out
    assert "--app kmeans" in out
    assert list(be.se.MODES) == modes_before


def test_baseline_runs_only_fedavg_and_restores_modes(monkeypatch, tmp_path):
    calls = []
    modes_before = list(be.se.MODES)

    def fake_main():
        calls.append((list(sys.argv), list(be.se.MODES)))

    monkeypatch.setattr(be.se, "main", fake_main)
    be.main(
        [
            "--app",
            "logreg",
            "--compare-workers",
            "2",
            "4",
            "--cores",
            "2",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert len(calls) == 2
    for argv, modes in calls:
        assert modes == [be.BASELINE_MODE]
        assert argv[argv.index("--workers") + 1] == "1"
        assert argv[argv.index("--app") + 1] == "logreg"
    assert [c[0][c[0].index("--cores") + 1] for c in calls] == ["4", "8"]
    assert list(be.se.MODES) == modes_before
