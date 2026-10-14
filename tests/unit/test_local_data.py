# =============================================================
# tests/unit/test_local_data.py
# Phase 6 — local per-driver resource tables.
# Small synthetic CSVs in the layouts of the DOE result files
# (Sets A, B, D and the holdout file with E1 / E2 / E3).
# =============================================================
import itertools
import json
import os
import sys

import pandas as pd
import pytest

from mpj_spark.resource_prediction import local_data as ld

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
_CLOCK = itertools.count()


def _run(prefix, gid, app, size, workers, cores, rep, iters=20, feats=16, k=5, **extra):
    """One run: a root row plus one row per driver, in the raw column layout."""
    exit_code = extra.pop("exit_code", 0)
    ts = f"2026-10-14T10:{next(_CLOCK) // 60:02d}:{next(_CLOCK) % 60:02d}"
    run_id = f"{prefix}_{gid}_s{size}_w{workers}_c{cores}_i{iters}_r{rep}"
    heap = 3072 if app == "kmeans" else 6144
    rows = []
    for drv in range(workers + 1):
        driver = drv > 0
        rows.append(
            {
                "run_id": run_id,
                "timestamp": ts,
                "host": "h",
                "app": app,
                "size_mb": size,
                "np": workers + 1,
                "workers": workers,
                "iters": iters,
                "features": feats,
                "k": k if app == "kmeans" else 0,
                "cores": cores,
                "heap_mb": heap,
                "cap_threads": cores,
                "rep": rep,
                "driver_idx": drv,
                "role": "driver" if driver else "root",
                "pid": 1000 + drv,
                "wall_s": 20.0,
                "exit_code": exit_code,
                "n_samples": 30,
                "cpu_avg_pct": 100.0 * cores + 10 * drv if driver else 95.0,
                "cpu_peak_pct": 300.0 * cores if driver else 120.0,
                "cpu_p95_pct": 200.0 * cores + 10 * drv if driver else 101.0,
                "rss_avg_mb": 600 + 100 * drv if driver else 43,
                "rss_peak_mb": 1200 + 100 * drv if driver else 53,
                "rss_p95_mb": 1100 + 100 * drv if driver else 50,
                "sys_cpu_avg_pct": 30.0,
                "sys_mem_avg_mb": 3000,
                "sys_mem_peak_mb": 4000,
                "sys_mem_avail_min_mb": 11000,
                "cores_alloc": cores,
                "heap_mb_alloc": heap,
                **extra,
            }
        )
    return rows


def _write(path, rows, drop=()):
    pd.DataFrame(rows).drop(columns=list(drop)).to_csv(path, index=False)


def _set_a():
    rows = []
    for i, (size, workers, iters) in enumerate(
        [(25, 2, 10), (50, 3, 20), (100, 1, 20), (150, 2, 40), (200, 3, 10), (200, 1, 40)], 1
    ):
        for rep in (1, 2):
            rows += _run(
                "A", f"d{i:02d}", "kmeans", size, workers, 4, rep, iters, design_id=f"d{i:02d}"
            )
    return rows


def _set_b():
    rows = []
    for i, (size, workers, iters) in enumerate(
        [(25, 2, 10), (50, 1, 30), (100, 2, 60), (150, 3, 30), (200, 1, 10), (200, 2, 30)], 1
    ):
        for rep in (1, 2):
            rows += _run(
                "B", f"b{i:02d}", "logreg", size, workers, 4, rep, iters, 20,
                design_id=f"b{i:02d}", attempt=1, iters_logged=iters,
            )  # fmt: skip
    rows += _run("B", "b07", "logreg", 200, 3, 4, 1, 60, 20, design_id="b07", exit_code=124)
    return rows


# Set D: K2 at 4 cores is identical to Set A design d03 (kmeans 100 MB, 1 worker)
_SWEEPS = {
    "K1": ("kmeans", 25, 2, 20),
    "K2": ("kmeans", 100, 1, 20),
    "K3": ("kmeans", 200, 2, 20),
    "L1": ("logreg", 25, 2, 30),
    "L2": ("logreg", 100, 2, 30),
    "L3": ("logreg", 150, 1, 30),
}


def _set_d():
    rows = []
    for wid, (app, size, workers, iters) in _SWEEPS.items():
        for cores in (1, 2, 3, 4):
            for rep in (1, 2):
                feats = 16 if app == "kmeans" else 20
                rows += _run(
                    "D", wid, app, size, workers, cores, rep, iters, feats,
                    workload_id=wid, attempt=1,
                )  # fmt: skip
    return rows


def _holdout():
    rows = []
    for rep in (1, 2):
        rows += _run(
            "E1", "HK75", "kmeans", 75, 2, 4, rep, workload_id="HK75", set="E1", kind="interior"
        )
        rows += _run(
            "E1", "HL300", "logreg", 300, 1, 4, rep, 30, 20,
            workload_id="HL300", set="E1", kind="extrap",
        )  # fmt: skip
    rows += _run("E2", "K1", "kmeans", 25, 2, 2, 3, workload_id="K1", set="E2", kind="rep3")
    rows += _run(
        "E3", "L1", "logreg", 25, 2, 2, 1, 30, 20,
        workload_id="L1", set="E3", kind="memresp", mem_factor=0.5, mem_limit_mb=900,
    )  # fmt: skip
    return rows


@pytest.fixture
def raw_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    _write(d / "doe_kmeans.csv", _set_a())
    _write(d / "doe_logreg.csv", _set_b(), drop=["k"])
    _write(d / "doe_rightsize.csv", _set_d())
    _write(d / "doe_holdout.csv", _holdout())
    _write(d / "pilot_doe_kmeans.csv", _run("A", "d01", "kmeans", 25, 2, 4, 1, design_id="d01"))
    return d


# ── sets, roles and exclusions ────────────────────────────────


def test_sets_get_their_roles(raw_dir):
    drivers, runs, report = ld.build_tables(str(raw_dir))
    assert set(runs["experiment"]) == {"A", "B", "D", "E1", "E2"}
    assert runs.loc[runs["experiment"].isin(["A", "B", "D"]), "in_main_set"].all()
    holdout = runs[runs["experiment"].isin(["E1", "E2"])]
    assert not holdout["in_main_set"].any() and (holdout["split"] == "holdout").all()
    assert set(holdout["holdout_kind"]) == {"interior", "extrap", "rep3"}
    assert "E3" in report["excluded_sets"] and "VOID" in report["excluded_sets"]["E3"]["reason"]
    assert "pilot_doe_kmeans.csv" in report["excluded_files"]


def test_root_rows_dropped_and_units(raw_dir):
    drivers, _, report = ld.build_tables(str(raw_dir))
    assert (drivers["driver_idx"] >= 1).all() and report["root_rows_dropped"] > 0
    row = drivers[(drivers["experiment"] == "D") & (drivers["cores_per_driver"] == 2)].iloc[0]
    assert row["cpu_cores_mean"] == pytest.approx((200 + 10 * row["driver_idx"]) / 100)
    assert row["cpu_cores_p95"] == pytest.approx((400 + 10 * row["driver_idx"]) / 100)
    assert row["mem_mb_p95"] == 1100 + 100 * row["driver_idx"]
    assert row["thread_cap"] == 2 and row["design_known"]


def test_inputs_are_mapped(raw_dir):
    drivers, _, _ = ld.build_tables(str(raw_dir))
    km = drivers[drivers["workload_type"] == "kmeans"]
    lr = drivers[drivers["workload_type"] == "logreg"]
    assert (km["num_clusters"] == 5).all() and (lr["num_clusters"] == 0).all()
    b = drivers[(drivers["experiment"] == "B") & (drivers["status"] == "ok")]  # no k column
    assert b["num_clusters"].eq(0).all() and b["iters_logged"].notna().all()
    assert set(drivers["dataset_mb"]) >= {25, 75, 300}
    assert {"iterations", "num_features", "num_workers"} <= set(drivers.columns)


def test_memory_labels_are_the_largest_driver_once_per_run(raw_dir):
    drivers, _, _ = ld.build_tables(str(raw_dir))
    for label, col in ld.RUN_MAX_LABELS.items():
        per_run = drivers.groupby("run_key")[label]
        assert (per_run.count() == 1).all()  # stored once per run
        expected = drivers.groupby("run_key")[col].max()
        assert per_run.max().equals(expected.astype(float))


def test_timeouts_are_censored(raw_dir):
    _, runs, _ = ld.build_tables(str(raw_dir))
    assert runs.loc[runs["run_id"].str.contains("b07"), "status"].eq("timeout").all()


# ── split groups (handoff: design_id / workload_id) ───────────


def test_core_sweep_of_one_workload_stays_together(raw_dir):
    drivers, _, _ = ld.build_tables(str(raw_dir))
    k1 = drivers[drivers["source_group"] == "D:K1"]
    assert k1["cores_per_driver"].nunique() == 4
    assert k1["config_group"].nunique() == 1 and k1["split"].nunique() == 1
    a = drivers[drivers["source_group"] == "A:d01"]
    assert a["rep"].nunique() == 2 and a["config_group"].nunique() == 1


def test_identical_configurations_are_linked(raw_dir):
    drivers, _, report = ld.build_tables(str(raw_dir))
    assert report["linked_groups"] == ["A:d03+D:K2"]
    linked = drivers[drivers["source_group"].isin(["A:d03", "D:K2"])]
    assert set(linked["config_group"]) == {"A:d03+D:K2"} and linked["split"].nunique() == 1


def test_every_split_holds_whole_groups(raw_dir):
    _, runs, report = ld.build_tables(str(raw_dir))
    main = runs[runs["in_main_set"]]
    assert main.groupby("config_group")["split"].nunique().max() == 1
    assert report["split_map"] and set(report["split_map"].values()) <= {"train", "val", "test"}


def test_stratified_split_gives_each_workload_a_test_group():
    groups = pd.DataFrame(
        {
            "config_group": [f"kmeans|{i}" for i in range(10)] + [f"wc|{i}" for i in range(3)],
            "stratum": ["kmeans"] * 10 + ["wc"] * 3,
        }
    )
    split = ld.stratified_split(groups)
    for prefix in ("kmeans|", "wc|"):
        vals = [s for k, s in split.items() if k.startswith(prefix)]
        assert "test" in vals and "val" in vals and "train" in vals
    assert split == ld.stratified_split(groups.iloc[::-1])  # order-independent


def test_core_sweeps_are_stratified_separately(raw_dir):
    drivers, _, _ = ld.build_tables(str(raw_dir))
    main = drivers[drivers["in_main_set"]].drop_duplicates("config_group")
    sweeps = main[main["config_group"].str.contains("D:")]
    for wl, g in sweeps.groupby("workload_type"):  # 3 sweep groups per workload
        assert set(g["split"]) == {"train", "val", "test"}, wl


def test_small_stratum_uses_hash_split_not_all_train():
    groups = pd.DataFrame({"config_group": [f"x|{i}" for i in range(2)], "stratum": ["x"] * 2})
    split = ld.stratified_split(groups)
    assert split == {k: ld._hash_split(k) for k in split}


# ── robustness ────────────────────────────────────────────────


def test_duplicate_file_copy_counts_once(raw_dir):
    before, _, _ = ld.build_tables(str(raw_dir))
    (raw_dir / "doe_kmeans (1).csv").write_text((raw_dir / "doe_kmeans.csv").read_text())
    after, _, report = ld.build_tables(str(raw_dir))
    assert len(after) == len(before) and report["duplicate_rows_dropped"] > 0


def test_summary_file_in_raw_folder_is_refused(raw_dir):
    pd.DataFrame({"workload_id": ["K1"], "knee_cores": [2]}).to_csv(
        raw_dir / "rightsize_labels.csv", index=False
    )
    with pytest.raises(ValueError, match="not a per-rank measurement file"):
        ld.build_tables(str(raw_dir))


def test_unknown_set_needs_a_role(raw_dir):
    _write(raw_dir / "doe_new.csv", _run("F", "x1", "kmeans", 25, 1, 4, 1, workload_id="x1"))
    with pytest.raises(ValueError, match="add them to SET_ROLES"):
        ld.build_tables(str(raw_dir))


# ── frozen splits ─────────────────────────────────────────────


def test_frozen_split_wins_over_recomputed_split(raw_dir):
    _, _, first = ld.build_tables(str(raw_dir))
    group = sorted(first["split_map"])[0]
    flipped = "test" if first["split_map"][group] != "test" else "train"
    _, runs, second = ld.build_tables(str(raw_dir), {group: flipped})
    assert second["split_map"][group] == flipped
    assert runs.loc[runs["config_group"] == group, "split"].eq(flipped).all()


def test_frozen_splits_survive_new_data(raw_dir):
    _, _, first = ld.build_tables(str(raw_dir))
    rows = []
    for i in range(7, 12):
        rows += _run("A", f"d{i:02d}", "kmeans", 50, 2, 4, 1, 10 * i, design_id=f"d{i:02d}")
    pd.concat([pd.read_csv(raw_dir / "doe_kmeans.csv"), pd.DataFrame(rows)]).to_csv(
        raw_dir / "doe_kmeans.csv", index=False
    )
    _, _, second = ld.build_tables(str(raw_dir), first["split_map"])
    assert all(second["split_map"][g] == s for g, s in first["split_map"].items())
    assert len(second["split_map"]) == len(first["split_map"]) + 5


# ── build script ──────────────────────────────────────────────


def test_build_script_end_to_end(raw_dir, tmp_path):
    sys.path.insert(0, os.path.abspath(_SCRIPTS))
    import build_local_dataset

    out, splits = tmp_path / "out", tmp_path / "splits.json"
    args = ["--raw-dir", str(raw_dir), "--out-dir", str(out), "--splits-file", str(splits)]
    report = build_local_dataset.main(args)
    assert (out / "local_drivers.csv").exists() and (out / "local_runs.csv").exists()
    assert report["runs_by_set"]["E1"] == 4 and "E3" in report["excluded_sets"]
    assert report["checks"]["thread_cap_equals_cores"]
    saved = json.loads(splits.read_text())
    assert saved["groups"] and "design_id" in saved["policy"]
    again = build_local_dataset.main(args)  # rebuild keeps every group in place
    assert again["split_groups_kept_from_file"] == len(saved["groups"])
    assert json.loads(splits.read_text())["groups"] == saved["groups"]


def test_default_split_file_is_version_controlled_path():
    sys.path.insert(0, os.path.abspath(_SCRIPTS))
    import build_local_dataset

    assert build_local_dataset.DEFAULT_SPLITS_FILE.endswith(
        os.path.join("config", "phase6_splits.json")
    )
