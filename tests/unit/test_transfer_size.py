# =============================================================
# tests/unit/test_transfer_size.py
# Phase 6 — transfer option A: Scout data-size trend.
# Synthetic Scout hosts and local rows with known elasticities.
# =============================================================
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from mpj_spark.resource_prediction import models as m
from mpj_spark.resource_prediction import transfer as t

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")


def _scout(beta=0.5):
    """LogReg hosts: y = c_vm · size^beta (different level per VM type); K-Means without size."""
    rows = []
    for vm, level in (("m4.large", 1.0), ("c4.xlarge", 3.0), ("r4.large", 10.0)):
        for size in (769.0, 1153.0, 1537.0):
            y = level * size**beta
            rows.append(
                {
                    "setting": "single_node",
                    "workload_type": "logreg",
                    "vm_type": vm,
                    "vcpus_per_host": 4,
                    "input_mib": size,
                    "host_cpu_busy_p95_cores": y,
                    "host_cpu_busy_mean_pct": y,
                    "host_mem_app_peak_mib": 100 * y,
                }
            )
        rows.append(rows[-1] | {"workload_type": "kmeans", "input_mib": np.nan})
    return pd.DataFrame(rows)


def _local(beta=0.3, seed=0):
    """Local rows: target = level(config) · shard^beta for every configuration."""
    rng = np.random.default_rng(seed)
    rows = []
    for wl in ("kmeans", "logreg"):
        for workers in (1, 2):
            for cores in (1, 2, 4):
                for size in (25, 50, 100, 150, 200):
                    level = 0.5 * cores + workers
                    shard = size / workers
                    for rep in range(2):
                        y = level * shard**beta * (1 + rng.normal(0, 0.01))
                        rows.append(
                            {
                                "workload_type": wl,
                                "dataset_mb": size,
                                "num_workers": workers,
                                "cores_per_driver": cores,
                                "heap_mb_per_driver": 3072,
                                "iterations": 20,
                                "num_features": 16,
                                "num_clusters": 5 if wl == "kmeans" else 0,
                                "thread_cap": cores,
                                "cpu_cores_p95": y,
                                "cpu_cores_mean": y / 2,
                                "mem_mb_p95_max": 1000 * y,
                                "mem_mb_peak_max": 1100 * y,
                                "rep": rep,
                            }
                        )
    df = pd.DataFrame(rows)
    key = ["workload_type", "dataset_mb", "num_workers", "cores_per_driver"]
    df["config_group"] = df[key].astype(str).agg("|".join, axis=1)
    return df


# ── elasticities ──────────────────────────────────────────────


def test_scout_elasticity_recovers_beta_within_vm_type():
    out = t.scout_size_elasticity(_scout(0.5), "cpu_cores_p95", "logreg")
    assert out["beta"] == pytest.approx(0.5, abs=1e-6)  # despite very different VM levels
    assert out["vm_types"] == 3 and out["rows"] == 9


def test_scout_has_no_size_trend_for_kmeans():
    assert t.scout_size_elasticity(_scout(), "cpu_cores_p95", "kmeans") is None


def test_local_elasticity_recovers_beta_within_configuration():
    beta = t.local_size_elasticity(_local(0.3), "cpu_cores_p95", "logreg")
    assert beta == pytest.approx(0.3, abs=0.02)


# ── anchor and scale ──────────────────────────────────────────


class _Echo:
    """Stand-in model: predicts the shard (dataset_mb / num_workers)."""

    def predict(self, X):
        return (X["dataset_mb"] / X["num_workers"]).to_numpy(float)


def test_anchor_and_scale_only_touches_rows_beyond_s_max():
    X = m.add_derived(
        pd.DataFrame(
            {
                "dataset_mb": [50.0, 100.0, 400.0],
                "num_workers": [1, 1, 1],
                "iterations": 1,
                "num_features": 1,
            }
        )
    )
    feats = ["dataset_mb", "num_workers"]
    own = t.anchor_and_scale(_Echo(), X, feats, 100.0, None)
    flat = t.anchor_and_scale(_Echo(), X, feats, 100.0, 0.0)
    half = t.anchor_and_scale(_Echo(), X, feats, 100.0, 0.5)
    assert own.tolist() == [50, 100, 400]
    assert flat.tolist() == [50, 100, 100]  # anchored at s_max, no growth
    assert half.tolist() == pytest.approx([50, 100, 100 * 4**0.5])


# ── script end-to-end ─────────────────────────────────────────


def test_size_trend_script_end_to_end(tmp_path):
    sys.path.insert(0, os.path.abspath(_SCRIPTS))
    import transfer_size_trend as ts

    local = _local()
    groups = sorted(local["config_group"].unique())
    local["split"] = local["config_group"].map(
        {g: ("test" if i % 7 == 0 else "train") for i, g in enumerate(groups)}
    )
    local["in_main_set"], local["status"] = True, "ok"
    local["alloc_known"], local["design_known"] = True, True
    local["experiment"], local["run_key"], local["driver_idx"] = "A", local.index.astype(str), 1
    holdout = local[local["dataset_mb"] == 200].assign(
        dataset_mb=300, split="holdout", experiment="E1", in_main_set=False
    )
    lp, sp = tmp_path / "drivers.csv", tmp_path / "scout.csv"
    pd.concat([local, holdout]).to_csv(lp, index=False)
    _scout(0.3).to_csv(sp, index=False)
    out = tmp_path / "out"
    report = ts.main(
        ["--drivers", str(lp), "--scout-hosts", str(sp), "--out-dir", str(out), "--rf-trees", "10",
         "--n-boot", "200"]
    )  # fmt: skip
    assert set(report["targets"]) == set(m.TARGETS)
    assert report["not_applicable"] == {"kmeans": "Scout records no input size for this workload"}
    stress = report["targets"]["mem_mb_p95_max"]["size_stress"]
    assert stress["rows_beyond_s_max"] > 0 and stress["s_max_shard_mb"] == 100
    assert set(stress["paired"]) == {"scout_trend_vs_local", "scout_trend_vs_local_trend"}
    assert isinstance(stress["scout_helps"], bool)
    metrics = pd.read_csv(out / "size_trend_metrics.csv")
    assert set(metrics["method"]) == set(ts.METHODS) and set(metrics["test"]) == {
        "size_stress",
        "E1",
    }
    saved = json.loads((out / "report.json").read_text())
    assert saved["targets"]["cpu_cores_p95"]["scout"]["beta"] == pytest.approx(0.3, abs=1e-6)


# ── cores trend and Scout-informed combination ────────────────


def _scout_both(size_beta=0.3, cores_beta=0.6):
    """Hosts varying in input size (LogReg) and in vCPUs within VM family and size label."""
    rows = []
    for wl in ("kmeans", "logreg"):
        for fam, level in (("c", 1.0), ("m", 2.0), ("r", 5.0)):
            for label, size in (("small", 769.0), ("large", 1537.0)):
                for vcpus in (2, 4, 8):
                    y = level * size**size_beta * vcpus**cores_beta
                    rows.append(
                        {
                            "setting": "single_node",
                            "workload_type": wl,
                            "vm_type": f"{fam}{vcpus}",
                            "vm_family": fam,
                            "datasize_label": label,
                            "vcpus_per_host": vcpus,
                            "input_mib": size if wl == "logreg" else np.nan,
                            "host_cpu_busy_p95_cores": y,
                            "host_cpu_busy_mean_pct": y,
                            "host_mem_app_peak_mib": 100 * y,
                        }
                    )
    return pd.DataFrame(rows)


def test_scout_cores_elasticity_within_family_and_size():
    out = t.scout_elasticity(_scout_both(cores_beta=0.6), "cpu_cores_p95", "kmeans", "cores")
    assert out["beta"] == pytest.approx(0.6, abs=1e-6) and out["groups"] == 6


def test_local_cores_elasticity():
    df = _local()
    df["cpu_cores_p95"] = (df["num_workers"] + 1) * df["cores_per_driver"] ** 0.6
    out = t.local_elasticity(df, "cpu_cores_p95", "kmeans", "cores")
    assert out["beta"] == pytest.approx(0.6, abs=1e-6) and out["groups"] > 0


def test_combine_elasticity_weights_scout_like_k_local_families():
    local, scout = {"beta": 0.4, "groups": 3}, {"beta": 0.1, "groups": 18}
    comb = t.combine_elasticity(local, scout, k=3)
    assert comb["beta"] == pytest.approx(0.25) and comb["scout_weight"] == pytest.approx(0.5)
    assert t.combine_elasticity(local, None)["source"] == "local"
    assert t.combine_elasticity(None, scout)["beta"] == 0.1
    assert t.combine_elasticity(None, None) is None
    many = t.combine_elasticity({"beta": 0.4, "groups": 297}, scout, k=3)
    assert many["beta"] == pytest.approx(0.397)  # plenty of local data → mostly local


def test_anchor_and_scale_on_cores():
    class _Cores:
        def predict(self, X):
            return X["cores_per_driver"].to_numpy(float)

    X = pd.DataFrame({"cores_per_driver": [1.0, 4.0, 8.0], "dataset_mb": 10.0, "num_workers": 1})
    out = t.anchor_and_scale(_Cores(), X, ["cores_per_driver"], 4.0, 1.0, axis="cores")
    assert out.tolist() == [1, 4, 8]  # anchored at 4 cores, grown linearly
    flat = t.anchor_and_scale(_Cores(), X, ["cores_per_driver"], 4.0, 0.0, axis="cores")
    assert flat.tolist() == [1, 4, 4]


def test_trend_tests_script_end_to_end(tmp_path):
    sys.path.insert(0, os.path.abspath(_SCRIPTS))
    import transfer_trend_tests as tt

    local = _local()
    local["split"], local["in_main_set"], local["status"] = "train", True, "ok"
    local["alloc_known"], local["design_known"] = True, True
    local["experiment"], local["run_key"], local["driver_idx"] = "D", local.index.astype(str), 1
    lp, sp = tmp_path / "drivers.csv", tmp_path / "scout.csv"
    local.to_csv(lp, index=False)
    _scout_both().to_csv(sp, index=False)
    out = tmp_path / "out"
    report = tt.main(
        ["--drivers", str(lp), "--scout-hosts", str(sp), "--out-dir", str(out),
         "--rf-trees", "10", "--n-boot", "200"]
    )  # fmt: skip
    policy = report["policy"]
    assert set(policy) == {"size", "cores"}
    decision = policy["size"]["logreg"]["mem_mb_p95_max"]
    assert set(decision) == {"source", "use_trend", "rule"} and isinstance(
        decision["use_trend"], bool
    )
    assert "kmeans" in policy["cores"]  # K-Means 4 cores predicted from 1-2
    assert (out / "trend_metrics.csv").exists() and (out / "trend_paired.csv").exists()
    case = report["cases"]["size_S2|size|logreg|mem_mb_p95_max"]
    assert case["beta_scout"]["beta"] == pytest.approx(0.3, abs=1e-6)
    assert case["beta_combined"]["source"] == "scout_informed"


# ── beyond-range margins (calibration.py) ─────────────────────


def test_stress_margins_score_core_sweeps_only():
    from mpj_spark.resource_prediction import calibration as cal

    local = m.add_derived(_local())  # cores 1, 2, 4 for both workloads
    local["experiment"] = np.where(local["cores_per_driver"] == 4, "A", "D")
    local.loc[(local["cores_per_driver"] == 4) & (local["rep"] == 1), "experiment"] = "D"
    keys = {"kmeans": "ridge_wl|1.0", "logreg": "ridge_wl|1.0"}
    out = cal.stress_margins(local, "mem_mb_p95_max", keys, {}, None, "capped")
    km_cores = out["cores"]["kmeans"]
    held_d = local[(local["workload_type"] == "kmeans") & (local["cores_per_driver"] > 2)]
    assert km_cores["rows"] == int((held_d["experiment"] == "D").sum())  # Set A not scored
    assert out["size"]["kmeans"]["rows"] > 0 and km_cores["margin"] >= 0
    assert "cores" not in out or "logreg" not in out["cores"]  # LogReg has nothing above 4 cores


def test_stress_predict_grows_beyond_the_training_range_only():
    from mpj_spark.resource_prediction import calibration as cal

    local = m.add_derived(_local(beta=0.5))
    train, ev = cal.stress_split(local, "size", "logreg")  # LogReg above 100 MB per driver
    assert (t.axis_value(ev, "size") > 100).all() and len(ev) > 0
    assert not ((train["workload_type"] == "logreg") & (t.axis_value(train, "size") > 100)).any()
    features = m.features_for("capped")
    policy = {"size": {"logreg": {"mem_mb_p95_max": "local"}}}
    grown, source = cal.stress_predict(
        m.make_ridge_wl(1.0, m.TRAINING_SETS["capped"]["numeric"]),
        train, ev, "mem_mb_p95_max", "logreg", "size", policy, None, features,
    )  # fmt: skip
    own, none = cal.stress_predict(
        m.make_ridge_wl(1.0, m.TRAINING_SETS["capped"]["numeric"]),
        train, ev, "mem_mb_p95_max", "logreg", "size", {}, None, features,
    )  # fmt: skip
    assert source == "local" and none is None
    assert not np.allclose(grown, own)  # the trend replaces the model's own extrapolation
    err_trend = np.abs(grown - ev["mem_mb_p95_max"]).mean()
    assert err_trend < np.abs(own - ev["mem_mb_p95_max"]).mean()  # true law is a power law


def test_a2b_power_law_candidate_uses_the_card_alpha():
    sys.path.insert(0, os.path.abspath(_SCRIPTS))
    import logreg_memory_test as a2b

    card = {"choices": {"mem_mb_p95_max": {"selected": {"ridge_log_wl": "ridge_log_wl|10.0"}}}}
    lr_mae = pd.Series(
        {"ridge_int_wl|0.1": 5.0, "ridge_int_wl|1.0": 3.0,
         "ridge_log_wl|0.1": 1.0, "ridge_log_wl|10.0": 9.0}
    )  # fmt: skip
    keys = a2b.chosen_keys(card, "mem_mb_p95_max", lr_mae)
    assert keys["ridge_log_wl"] == "ridge_log_wl|10.0"  # deployed alpha, not the CV-best one
    assert keys["ridge_int_wl"] == "ridge_int_wl|1.0"  # lowest LogReg CV MAE, as in A2
    assert keys["ridge_wl"] == "ridge_wl" and keys["rf_wl"] == "rf_wl"
