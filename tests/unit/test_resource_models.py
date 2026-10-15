# =============================================================
# tests/unit/test_resource_models.py
# Phase 6 — per-driver resource models and the training script.
# Synthetic driver tables only.
# =============================================================
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from mpj_spark.resource_prediction import models as m

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")


def _frame(n_per_group=2, seed=0):
    """Configs × repeats with a known demand formula per workload."""
    rng = np.random.default_rng(seed)
    rows = []
    for wl, base in (("kmeans", 2.0), ("logreg", 6.0)):
        for size in (50, 100):
            for cores in (1, 2, 4, 8):
                for heap in (1536, 3072):
                    for rep in range(n_per_group):
                        cpu = base + 0.1 * cores + rng.normal(0, 0.02)
                        mem = 500 + 0.3 * heap + 4 * size + rng.normal(0, 5)
                        rows.append(
                            {
                                "workload_type": wl,
                                "dataset_mb": size,
                                "num_workers": 2,
                                "cores_per_driver": cores,
                                "heap_mb_per_driver": heap,
                                "cpu_cores_mean": cpu,
                                "cpu_cores_p95": cpu * 1.2,
                                "mem_mb_peak": mem,
                                "mem_mb_p95_max": 0.95 * mem,
                                "mem_mb_peak_max": mem,
                                "rep": rep,
                            }
                        )
    return pd.DataFrame(rows)


# ── feature contract ──────────────────────────────────────────


def test_check_features_rejects_missing_inputs():
    df = _frame()
    m.check_features(df)
    df.loc[0, "heap_mb_per_driver"] = np.nan
    with pytest.raises(ValueError, match="missing pre-run inputs"):
        m.check_features(df)
    with pytest.raises(ValueError, match="missing feature columns"):
        m.check_features(df.drop(columns=["num_workers"]))


# ── models ────────────────────────────────────────────────────


@pytest.mark.parametrize("factory", [lambda: m.make_ridge(1.0), lambda: m.make_rf(5, 2)])
def test_pipelines_learn_the_signal(factory):
    df = _frame()
    model = factory().fit(df[m.FEATURES], df["mem_mb_peak"])
    pred = model.predict(df[m.FEATURES])
    assert m.regression_metrics(df["mem_mb_peak"], pred)["mape_pct"] < 5


def test_ridge_scaler_fitted_on_training_rows_only():
    df = _frame()
    train = df[df["dataset_mb"] == 50]
    model = m.make_ridge().fit(train[m.FEATURES], train["cpu_cores_mean"])
    scaler = model.named_steps["pre"].named_transformers_["log"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(np.log2(50))  # never saw size 100


def test_lookup_falls_back_to_coarser_keys():
    df = _frame()
    train = df[df["cores_per_driver"] != 8]
    lk = m.LookupModel().fit(train[m.FEATURES], train["cpu_cores_mean"])
    seen = df[(df["cores_per_driver"] == 2)].iloc[[0]]
    unseen_cores = df[(df["cores_per_driver"] == 8)].iloc[[0]]
    new_workload = seen.assign(workload_type="wordcount")
    lk.predict(seen)
    assert lk.level_used_[0] == 0
    lk.predict(unseen_cores)
    assert lk.level_used_[0] == 2  # workload + size + workers
    pred = lk.predict(new_workload)
    assert lk.level_used_[0] == 4 and pred[0] == pytest.approx(train["cpu_cores_mean"].median())


# ── metrics ───────────────────────────────────────────────────


def test_regression_metrics_units_and_signs():
    met = m.regression_metrics([10, 20], [12, 18], upper=[15, 19])
    assert met["mae"] == 2 and met["mean_signed_error"] == 0
    assert met["underprediction_rate"] == 0.5
    assert met["upper_bound_coverage"] == 0.5
    assert m.regression_metrics([5, 5], [5, 5])["r2"] is None


def test_upper_margin_covers_quantile():
    rng = np.random.default_rng(1)
    y = rng.normal(100, 10, 1000)
    pred = np.full_like(y, 100)
    margin = m.upper_margin(y, pred, 0.9)
    assert np.mean(pred + margin >= y) == pytest.approx(0.9, abs=0.01)
    assert m.upper_margin([1, 2], [5, 5]) == 0.0  # never negative


# ── training script end-to-end ────────────────────────────────


def test_training_script_end_to_end(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.abspath(_SCRIPTS))
    import train_resource_predictor as trp

    # keep the CV grid small so the test stays fast
    monkeypatch.setattr(m, "RIDGE_ALPHAS", [1.0, 10.0])
    monkeypatch.setattr(m, "RF_GRID", [{"max_depth": 3, "min_samples_leaf": 2}])
    full_rf = m.make_rf
    monkeypatch.setattr(m, "make_rf", lambda **kw: full_rf(n_estimators=10, **kw))

    df = _frame(n_per_group=2)
    df["config_group"] = df[m.FEATURES].astype(str).agg("|".join, axis=1)
    groups = sorted(df["config_group"].unique())
    split = {
        g: ("test" if i % 7 == 0 else "val" if i % 7 == 1 else "train")
        for i, g in enumerate(groups)
    }
    df["split"] = df["config_group"].map(split)
    df["in_main_set"], df["status"], df["alloc_known"] = True, "ok", True
    df["run_key"], df["driver_idx"] = df.index.astype(str), 1
    extra = df.iloc[:2].assign(alloc_known=False, workload_type="wordcount")
    path = tmp_path / "drivers.csv"
    pd.concat([df, extra]).to_csv(path, index=False)

    out = tmp_path / "models"
    metrics = trp.main(
        ["--drivers", str(path), "--out-dir", str(out), "--training-set", "uncapped"]
    )
    assert set(metrics["model"]) == set(m.MODEL_FAMILIES)
    assert set(metrics["target"]) == set(m.TARGETS)
    card = json.loads((out / "model_card.json").read_text())
    assert card["rows"]["rows_dropped_unknown_inputs"] == 2
    assert card["rows"]["workloads_dropped_entirely"] == ["wordcount"]
    assert (out / "mem_mb_peak_max__rf.joblib").exists()
    cv = pd.read_csv(out / "cv_metrics.csv")
    assert set(cv["model"]) == set(m.MODEL_FAMILIES)
    sel = card["choices"]["mem_mb_peak_max"]["selected"]
    assert sel["ridge"].startswith("ridge|") and sel["rf"].startswith("rf|")
    assert "mae_diff_ci95" in card["choices"]["mem_mb_peak_max"]["cv_uncertainty"]["rf"]
    preds = pd.read_csv(out / "predictions_test.csv")
    test_keys = set(df.loc[df["split"] == "test", "run_key"])
    assert set(preds["run_key"].astype(str)) == test_keys  # only TEST rows scored


# ── training sets: capped vs uncapped ─────────────────────────


def _env_frame():
    """Uncapped rows without design + capped rows with design inputs."""
    old = _frame().assign(thread_cap=np.nan, design_known=False)
    old[m.DESIGN_FEATURES] = np.nan
    new = _frame(seed=1).assign(
        thread_cap=4.0, design_known=True, iterations=20.0, num_features=10.0
    )
    new["num_clusters"] = np.where(new["workload_type"] == "kmeans", 5.0, 0.0)
    df = pd.concat([old, new], ignore_index=True)
    df["in_main_set"], df["status"], df["alloc_known"] = True, "ok", True
    return df


def test_training_sets_never_mix_environments():
    df = _env_frame()
    capped, info = m.select_training_rows(df, "capped")
    uncapped, _ = m.select_training_rows(df, "uncapped")
    assert capped["thread_cap"].notna().all() and uncapped["thread_cap"].isna().all()
    assert len(capped) + len(uncapped) == len(df)
    capped_inputs = m.features_for("capped")
    assert set(m.DESIGN_FEATURES + m.DERIVED_FEATURES) <= set(capped_inputs)
    assert "heap_mb_per_driver" not in capped_inputs  # fixed per workload in the runs
    assert set(m.DERIVED_FEATURES) <= set(capped.columns)  # added by the selection
    assert m.features_for("uncapped") == m.FEATURES


def test_capped_set_drops_rows_without_design():
    df = _env_frame()
    df.loc[df["thread_cap"].notna() & (df["workload_type"] == "logreg"), "design_known"] = False
    capped, info = m.select_training_rows(df, "capped")
    assert set(capped["workload_type"]) == {"kmeans"}
    assert info["workloads_dropped_entirely"] == ["logreg"]


def test_derived_inputs_follow_the_handoff():
    df = pd.DataFrame(
        {"dataset_mb": [100.0], "num_workers": [2], "iterations": [30], "num_features": [20]}
    )
    d = m.add_derived(df)
    assert d[m.DERIVED_FEATURES].iloc[0].tolist() == [50.0, 1500.0, 1000.0]


def test_lookup_skips_levels_with_columns_that_are_not_inputs():
    feats = m.features_for("capped")
    levels = m.LookupModel(feats).levels
    assert levels[0] == feats and levels[-1] == []
    assert all(set(lvl) <= set(feats) for lvl in levels)
    assert m.LookupModel().levels[0] == m.FEATURES  # base inputs unchanged


def test_design_features_reach_every_model():
    rng = np.random.default_rng(0)
    df = _env_frame()
    df = df[df["thread_cap"].notna()].copy()
    df["iterations"] = rng.choice([10.0, 20.0, 40.0], len(df))
    df = m.add_derived(df)
    df["cpu_cores_mean"] = np.log2(df["iterations"]) + (df["workload_type"] == "logreg")
    feats = m.features_for("capped")
    num = m.TRAINING_SETS["capped"]["numeric"]
    for model in (
        m.make_ridge(0.01, num),
        m.make_ridge_wl(0.01, num),
        m.make_rf(5, 1, n_estimators=10),
        m.LookupModel(feats),
    ):
        pred = model.fit(df[feats], df["cpu_cores_mean"]).predict(df[feats])
        assert m.regression_metrics(df["cpu_cores_mean"], pred)["mae"] < 0.1


# ── group cross-validation ────────────────────────────────────


def _grouped(df):
    df = df.copy()
    df["config_group"] = df[m.FEATURES].astype(str).agg("|".join, axis=1)
    return df


def test_group_cv_never_trains_on_held_out_configuration():
    df = _grouped(_frame(n_per_group=2))
    oof = m.group_cv_predictions(df, "cpu_cores_mean", {"lookup": m.LookupModel})
    assert len(oof) == len(df)  # every row predicted exactly once
    # lookup can never hit the exact configuration (level 0): it was held out
    for group, held in df.groupby("config_group"):
        rest = df[df["config_group"] != group]
        lk = m.LookupModel().fit(rest[m.FEATURES], rest["cpu_cores_mean"])
        lk.predict(held)
        assert (lk.level_used_ > 0).all()


def test_group_bootstrap_interval_and_paired_difference():
    df = _grouped(_frame(n_per_group=2))
    oof = m.group_cv_predictions(
        df, "mem_mb_peak", {"lookup": m.LookupModel, "ridge": lambda: m.make_ridge(1.0)}
    )
    res = m.group_bootstrap(oof, "ridge", baseline="lookup", n_boot=300)
    lo, hi = res["mae_ci95"]
    assert lo <= res["mae"] <= hi
    assert res["n_groups"] == df["config_group"].nunique()
    # linear signal: ridge must beat the coarse lookup clearly
    assert res["mae_diff_vs_baseline"] < 0 and res["mae_diff_ci95"][1] < 0


# ── per-workload Ridge ────────────────────────────────────────


def _slopes_differ(n=4, seed=3):
    """Memory rises with heap for logreg only; flat for kmeans."""
    rng = np.random.default_rng(seed)
    rows = []
    for wl, slope in (("kmeans", 0.0), ("logreg", 0.8)):
        for heap in (1024, 2048, 3072, 4096, 6144):
            for cores in (2, 4):
                for _ in range(n):
                    rows.append(
                        {
                            "workload_type": wl,
                            "dataset_mb": 100,
                            "num_workers": 2,
                            "cores_per_driver": cores,
                            "heap_mb_per_driver": heap,
                            "mem_mb_peak": 1000 + slope * heap + rng.normal(0, 20),
                        }
                    )
    return pd.DataFrame(rows)


def test_workload_interactions_shape_and_training_stats():
    df = _slopes_differ()
    t = m.WorkloadInteractions().fit(df[m.FEATURES])
    out = t.transform(df[m.FEATURES])
    k, p = 2, len(m.NUMERIC_FEATURES)
    assert out.shape == (len(df), k + p + k * p)
    assert t.std_[m.NUMERIC_FEATURES.index("num_workers")] == 1.0  # constant column
    unseen = t.transform(df[m.FEATURES].assign(workload_type="wordcount").iloc[:3])
    assert np.all(unseen[:, :k] == 0) and np.all(unseen[:, k + p :] == 0)


def test_ridge_wl_captures_per_workload_slopes():
    df = _slopes_differ()
    shared = m.make_ridge(0.1).fit(df[m.FEATURES], df["mem_mb_peak"])
    per_wl = m.make_ridge_wl(0.1).fit(df[m.FEATURES], df["mem_mb_peak"])
    mae_shared = m.regression_metrics(df["mem_mb_peak"], shared.predict(df[m.FEATURES]))["mae"]
    mae_wl = m.regression_metrics(df["mem_mb_peak"], per_wl.predict(df[m.FEATURES]))["mae"]
    assert mae_wl < 0.5 * mae_shared


# ── per-workload model ────────────────────────────────────────


def _two_workloads():
    """K-Means measured at small sizes, LogReg at large sizes with much higher CPU."""
    rows = []
    for wl, sizes, level in (("kmeans", (25, 50, 100), 5.0), ("logreg", (100, 200, 400), 15.0)):
        for size in sizes:
            for cores in (1, 2, 4):
                rows.append(
                    {
                        "workload_type": wl,
                        "dataset_mb": size,
                        "num_workers": 1,
                        "cores_per_driver": cores,
                        "heap_mb_per_driver": 3072,
                        "cpu_cores_p95": level + 0.1 * cores,
                    }
                )
    return pd.DataFrame(rows)


def test_per_workload_model_never_borrows_another_workload():
    df = _two_workloads()
    big_kmeans = df[df["workload_type"] == "logreg"].assign(workload_type="kmeans")
    per_wl = m.make_rf_wl(5, 1, n_estimators=20).fit(df[m.FEATURES], df["cpu_cores_p95"])
    # K-Means sizes beyond the K-Means range are answered from K-Means rows only
    assert per_wl.predict(big_kmeans[m.FEATURES]) == pytest.approx(5.2, abs=0.3)
    assert set(per_wl.models_) == {"kmeans", "logreg"}


def test_per_workload_model_routes_rows_and_refuses_unseen_workloads():
    df = _two_workloads()
    per_wl = m.make_rf_wl(5, 1, n_estimators=20).fit(df[m.FEATURES], df["cpu_cores_p95"])
    mixed = pd.concat([df.head(2), df.tail(2)])
    pred = per_wl.predict(mixed[m.FEATURES])
    assert pred[:2] == pytest.approx(5.1, abs=0.3) and pred[2:] == pytest.approx(15.4, abs=0.3)
    with pytest.raises(ValueError, match="no per-workload model"):
        per_wl.predict(df[m.FEATURES].head(1).assign(workload_type="wordcount"))


def test_per_workload_model_is_one_saveable_object(tmp_path):
    import joblib
    from sklearn.base import clone

    df = _two_workloads()
    per_wl = m.make_rf_wl(5, 1, n_estimators=10).fit(df[m.FEATURES], df["cpu_cores_p95"])
    path = tmp_path / "cpu.joblib"
    joblib.dump(per_wl, path)
    back = joblib.load(path)
    assert back.predict(df[m.FEATURES]) == pytest.approx(per_wl.predict(df[m.FEATURES]))
    assert clone(per_wl).get_params()["base"] is not None  # scikit-learn compatible


def test_power_law_ridge_recovers_a_power_law_and_stays_positive():
    rows = []
    for wl in ("kmeans", "logreg"):
        for size in (25, 50, 100, 200):
            for cores in (1, 2, 4):
                rows.append(
                    {
                        "workload_type": wl,
                        "dataset_mb": size,
                        "num_workers": 2,
                        "cores_per_driver": cores,
                        "heap_mb_per_driver": 3072,
                        "mem_mb_peak": 300 * size**0.4 * (2 if wl == "logreg" else 1),
                    }
                )
    df = pd.DataFrame(rows)
    model = m.make_ridge_log_wl(0.01).fit(df[m.FEATURES], df["mem_mb_peak"])
    pred = model.predict(df[m.FEATURES])
    assert m.regression_metrics(df["mem_mb_peak"], pred)["mape_pct"] < 1
    big = df[df["dataset_mb"] == 200].assign(dataset_mb=800)
    expected = 300 * 800**0.4 * np.where(big["workload_type"] == "logreg", 2, 1)
    assert model.predict(big[m.FEATURES]) == pytest.approx(expected, rel=0.05)  # extrapolates
    assert (model.predict(big[m.FEATURES]) > 0).all()


# ── calibrated margins (step A3) ──────────────────────────────


def _residual_rows():
    """One big group with large residuals, nine small groups with small ones."""
    rows = [{"config_group": "big", "actual": 20.0, "predicted": 10.0} for _ in range(30)]
    for g in range(9):
        rows += [{"config_group": f"g{g}", "actual": 11.0, "predicted": 10.0} for _ in range(2)]
    df = pd.DataFrame(rows)
    df["workload_type"] = "kmeans"
    return df


def test_weighted_quantile():
    assert m.weighted_quantile([1, 2, 3, 4], [1, 1, 1, 1], 0.5) == 2
    assert m.weighted_quantile([1, 2, 3, 4], [3, 1, 1, 1], 0.5) == 1


def test_group_weighting_stops_one_big_group_dominating():
    rows = _residual_rows()
    by_rows = m.calibrate_margin(rows, 0.5, "additive_rows")
    by_groups = m.calibrate_margin(rows, 0.5, "additive_groups")
    assert by_rows == pytest.approx(10.0)  # 30 of 48 rows come from one group
    assert by_groups == pytest.approx(1.0)  # each group counts once


def test_conformal_level_and_relative_margin():
    rows = _residual_rows()
    # level q·(1 + 1/G) with G = 10 groups: 0.9 → 0.99, reaching the big group
    assert m.calibrate_margin(rows, 0.9, "additive_groups") == pytest.approx(10.0)
    rel = m.calibrate_margin(rows, 0.5, "relative_groups")
    assert rel == pytest.approx(0.1)  # (11 - 10) / 10
    assert m.apply_margin(200.0, rel, "relative_groups") == pytest.approx(220.0)
    assert m.apply_margin(200.0, 5.0, "additive_rows") == pytest.approx(205.0)
    out = m.calibrated_margins(rows, 0.9)
    assert set(out) == set(m.MARGIN_METHODS) and set(out["additive_rows"]) == {"pooled", "kmeans"}
