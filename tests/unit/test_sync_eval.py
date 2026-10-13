"""Tests for scripts/sync_eval.py: metrics parsing, time-to-target, selection, resume."""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sync_eval.py"
_spec = importlib.util.spec_from_file_location("sync_eval", SCRIPT)
se = importlib.util.module_from_spec(_spec)
sys.modules["sync_eval"] = se
_spec.loader.exec_module(se)

# loss ENTERING each round (round 1 starts at zero weights)
FAST = [0.693147, 0.023133, 0.022901, 0.022901]
SLOW = [0.693147, 0.30, 0.15, 0.08, 0.05, 0.035, 0.028, 0.025, 0.0235, 0.0232]


def _write(run_dir, mode, losses, times, rows=(1000, 1000), sync=None):
    """Write per-worker metrics CSVs the way each mode does."""
    os.makedirs(run_dir, exist_ok=True)
    names = {
        "ps_sync_fedavg_mpi": "worker_{r}_fedavg_mpi_metrics.csv",
        "gossip": "worker_{r}_gossip_metrics.csv",
        "ps_async": "worker_{r}_async_ps_metrics.csv",
        "hybrid_ps_allreduce": "worker_{r}_hybrid_metrics.csv",
    }
    for w in range(len(rows)):
        recs = []
        for t, (loss, tm) in enumerate(zip(losses, times, strict=True)):
            rec = {"worker_id": w, "iteration": t + 1, "iter_time_s": tm, "row_count": rows[w]}
            if mode == "ps_sync_fedavg_mpi":
                rec["global_loss"] = loss
                rec["loss_start"] = loss + 0.001 * w  # local view differs; must be ignored
            else:
                rec["loss_start"] = loss
            if sync is not None:
                if mode == "gossip":
                    rec["gossip_time_s"] = sync[t][w]
                if mode == "hybrid_ps_allreduce":
                    rec["allreduce_time_s"], rec["ps_time_s"] = sync[t][w], 0.0
            recs.append(rec)
        pd.DataFrame(recs).to_csv(Path(run_dir) / names[mode].format(r=w), index=False)


def test_exact_global_loss_and_cumulative_time_for_fedavg(tmp_path):
    _write(tmp_path, "ps_sync_fedavg_mpi", FAST, [8, 8, 5, 5])
    after, s = se.summarise_run(se.load_rounds(str(tmp_path), "ps_sync_fedavg_mpi"))
    assert list(after["round"]) == [1, 2, 3]  # R-1: the last model is not evaluated
    assert list(after["loss_after"]) == FAST[1:]  # global_loss, not the differing local views
    assert list(after["elapsed_s"]) == [8, 16, 21]
    assert s["rounds_logged"] == 4 and np.isnan(s["wait_s"])


def test_row_weighted_loss_for_local_views(tmp_path):
    d = tmp_path
    pd.DataFrame(
        [
            {"worker_id": 0, "iteration": 1, "iter_time_s": 1, "row_count": 100, "loss_start": 1.0},
            {"worker_id": 1, "iteration": 1, "iter_time_s": 1, "row_count": 300, "loss_start": 3.0},
            {"worker_id": 0, "iteration": 2, "iter_time_s": 1, "row_count": 100, "loss_start": 1.0},
            {"worker_id": 1, "iteration": 2, "iter_time_s": 1, "row_count": 300, "loss_start": 3.0},
        ]
    ).assign(worker_id=lambda x: x["worker_id"]).to_csv(
        d / "worker_0_async_ps_metrics.csv", index=False
    )
    after, _ = se.summarise_run(se.load_rounds(str(d), "ps_async"))
    assert after["loss_after"].iloc[0] == pytest.approx((1 * 100 + 3 * 300) / 400)


def test_sync_time_split_into_transfer_and_wait(tmp_path):
    sync = [(0.0002, 2.8), (2.8, 0.0002), (0.0002, 2.8)]  # workers take turns being the straggler
    _write(tmp_path, "gossip", FAST[:3], [8, 5, 5], sync=sync)
    _, s = se.summarise_run(se.load_rounds(str(tmp_path), "gossip"))
    assert s["transfer_s"] == pytest.approx(0.0006)
    assert s["wait_s"] == pytest.approx(3 * 1.4, rel=1e-3)  # mean over the two workers per round
    assert s["wait_share"] == pytest.approx(s["wait_s"] / 18)


def test_time_to_target_hit_and_miss():
    after = pd.DataFrame(
        {"round": [1, 2, 3], "elapsed_s": [8.0, 16.0, 21.0], "loss_after": [0.5, 0.0229, 0.0229]}
    )
    assert se.time_to_target(after, 0.023) == 16.0
    assert np.isnan(se.time_to_target(after, 0.01))


def _fake_mpirun(path):
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, os, random\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        "a = sys.argv\n"
        "mode = a[a.index('--sync-mode') + 1]\n"
        "rd = a[a.index('--results-dir') + 1]\n"
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('t', {str(Path(__file__))!r})\n"
        "t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)\n"
        "rep = int(rd.rsplit('_r', 1)[1])\n"
        "k = 1 + 0.01 * rep\n"
        "if mode == 'ps_sync_fedavg_mpi':\n"
        "    t._write(rd, mode, t.FAST, [7.5 * k, 7.5 * k, 5 * k, 5 * k])\n"
        "elif mode == 'hybrid_ps_allreduce':\n"
        "    t._write(rd, mode, t.FAST * 3, [8 * k] * 2 + [5 * k] * 10, sync=[(0.0, 2.0)] * 12)\n"
        "elif mode == 'gossip':\n"
        "    t._write(rd, mode, t.FAST * 3, [9 * k] * 2 + [6 * k] * 10, sync=[(0.0, 2.8)] * 12)\n"
        "else:\n"
        "    t._write(rd, mode, t.SLOW, [6 * k] * 10)\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run_cli(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sync_eval.py", *argv])
    se.main()


def test_end_to_end_selection_resume_and_gates(tmp_path, monkeypatch, capsys):
    inp = tmp_path / "logreg.csv"
    inp.write_text("1,2,0\n")
    mpirun = tmp_path / "fakempirun"
    _fake_mpirun(mpirun)
    out = tmp_path / "out"
    args = [
        "run",
        "--mpirun",
        str(mpirun),
        "--input",
        str(inp),
        "--reps",
        "3",
        "--workers",
        "2",
        "--settle",
        "0",
        "--out-dir",
        str(out),
    ]
    _run_cli(args, monkeypatch)
    runs = pd.read_csv(out / "runs.csv")
    assert len(runs) == 12 and (runs["exit_code"] == 0).all()
    assert set(runs["mode"]) == set(se.MODES)
    assert runs["dataset_sha256"].nunique() == 1

    _run_cli(args, monkeypatch)  # resume: nothing is run twice
    assert len(pd.read_csv(out / "runs.csv")) == 12

    capsys.readouterr()
    _run_cli(["select", "--out-dir", str(out)], monkeypatch)
    text = capsys.readouterr().out
    assert "best=ps_sync_fedavg_mpi" in text and "SEPARABLE" in text
    table = pd.read_csv(out / "mode_table.csv").set_index("mode")
    assert table.loc["ps_async", "n_reached"] == 0  # never reaches the target in 9 measured rounds
    assert table.loc["ps_sync_fedavg_mpi", "n_reached"] == 3
    assert table.loc["gossip", "median_wait_share"] > 0
    assert np.isnan(table.loc["ps_sync_fedavg_mpi", "median_wait_share"])


def test_unknown_app_is_refused(monkeypatch):
    with pytest.raises(SystemExit):
        _run_cli(["run", "--app", "wordcount"], monkeypatch)


# ───────────────────────── K-Means (fixed round budget) ─────────────────────────


def test_parse_total_wcss_takes_last_and_handles_missing(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("x\n     Total WCSS : 5.5000\nmore\n     Total WCSS : 2,616,692.1636\n")
    assert se.parse_total_wcss(str(log)) == pytest.approx(2616692.1636)
    log.write_text("nothing to see\n")
    assert np.isnan(se.parse_total_wcss(str(log)))
    assert np.isnan(se.parse_total_wcss(str(tmp_path / "absent.log")))


def test_summarise_kmeans_transfer_and_wait_split(tmp_path):
    syncs = [(0.0002, 2.8, 0.0002), (2.8, 0.0002, 2.8)]  # workers take turns being the straggler
    for w, sync in enumerate(syncs):
        pd.DataFrame(
            {
                "iteration": [1, 2, 3],
                "spark_time_s": [1.0, 1.0, 1.0],
                "sync_time_s": list(sync),
                "iter_time_s": [5.0, 5.0, 5.0],
                "centroid_shift": [0.1, 0.05, 0.02],
                "global_wcss": [3.0, 2.0, 1.0],
                "sync_overhead_pct": [0.0, 0.0, 0.0],
            }
        ).to_csv(tmp_path / f"kmeans_metrics_rank{w}.csv", index=False)
    s = se.summarise_kmeans(se.load_kmeans_rounds(str(tmp_path)))
    assert s["rounds_logged"] == 3
    assert s["transfer_s"] == pytest.approx(0.0006)
    assert s["wait_s"] == pytest.approx(3 * 1.4, rel=1e-3)
    assert s["wait_share"] == pytest.approx(s["wait_s"] / 15)
    assert se.summarise_kmeans(pd.DataFrame()) == {}


def test_build_cmd_kmeans_uses_kmeans_flags():
    cmd = se.build_cmd("gossip", 4, 2, "in.csv", "py", "/r", 8, 10, [], app="kmeans")
    assert cmd[cmd.index("-np") + 1] == "5"
    assert cmd[cmd.index("--app") + 1] == "kmeans"
    assert cmd[cmd.index("--kmeans-iter") + 1] == "8"
    assert "--logreg-iter" not in cmd and "--logreg-features" not in cmd
    default = se.build_cmd("gossip", 4, 2, "in.csv", "py", "/r", 8, 10, [])
    assert "--logreg-iter" in default and "--kmeans-iter" not in default


def _fake_kmeans_mpirun(path):
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, os, csv\n"
        "a = sys.argv\n"
        "mode = a[a.index('--sync-mode') + 1]\n"
        "rd = a[a.index('--results-dir') + 1]\n"
        "assert a[a.index('--app') + 1] == 'kmeans' and '--kmeans-iter' in a\n"
        "rounds = int(a[a.index('--kmeans-iter') + 1])\n"
        "os.makedirs(rd, exist_ok=True)\n"
        "wcss = {'ps_sync_fedavg_mpi': 2616692.0, 'gossip': 2616692.0,\n"
        "        'hybrid_ps_allreduce': 2616692.0, 'ps_async': 2616808.0}[mode]\n"
        "head = ['iteration', 'spark_time_s', 'sync_time_s', 'iter_time_s',\n"
        "        'centroid_shift', 'global_wcss', 'sync_overhead_pct']\n"
        "for w in range(2):\n"
        "    p = os.path.join(rd, 'kmeans_metrics_rank%d.csv' % w)\n"
        "    with open(p, 'w', newline='') as f:\n"
        "        wr = csv.writer(f)\n"
        "        wr.writerow(head)\n"
        "        for t in range(rounds):\n"
        "            wr.writerow([t + 1, 1.0, 0.0 if w == 0 else 0.5, 2.0, 0.01, 1.0, 0.0])\n"
        "print('     Total WCSS : 1.0000')\n"
        "print('     Total WCSS : %.4f' % wcss)\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_kmeans_end_to_end_table_and_resume(tmp_path, monkeypatch, capsys):
    inp = tmp_path / "km.csv"
    inp.write_text("1,2,0\n")
    mpirun = tmp_path / "fakempirun"
    _fake_kmeans_mpirun(mpirun)
    out = tmp_path / "out"
    args = [
        "run",
        "--app",
        "kmeans",
        "--mpirun",
        str(mpirun),
        "--input",
        str(inp),
        "--reps",
        "2",
        "--workers",
        "2",
        "--rounds",
        "4",
        "--settle",
        "0",
        "--out-dir",
        str(out),
    ]
    _run_cli(args, monkeypatch)
    runs = pd.read_csv(out / "runs.csv")
    assert len(runs) == 8 and (runs["exit_code"] == 0).all()
    assert runs["run_id"].str.startswith("kmeans_").all()
    assert (runs["rounds_logged"] == 4).all()
    assert runs["final_wcss"].notna().all()
    assert not (out / "rounds.csv").exists()  # no per-round loss file for K-Means

    _run_cli(args, monkeypatch)  # resume: nothing is run twice
    assert len(pd.read_csv(out / "runs.csv")) == 8

    capsys.readouterr()
    _run_cli(["select", "--app", "kmeans", "--out-dir", str(out)], monkeypatch)
    assert "no winner declared" in capsys.readouterr().out
    table = pd.read_csv(out / "mode_table.csv").set_index("mode")
    assert table.loc["gossip", "wcss_vs_best_pct"] == pytest.approx(0.0)
    assert table.loc["ps_async", "wcss_vs_best_pct"] > 0
    assert table.loc["gossip", "median_wait_share"] > 0
    assert table.loc["ps_sync_fedavg_mpi", "median_rounds"] == 4
