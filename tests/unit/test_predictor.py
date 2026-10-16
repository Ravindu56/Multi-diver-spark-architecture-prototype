# =============================================================
# tests/unit/test_predictor.py
# Phase 6 — pre-launch predictor: request checks, domain flags,
# per-workload upper margins, model choice and the CLI.
# Models are trained once on a small synthetic capped table.
# =============================================================
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from mpj_spark.resource_prediction import models as m
from mpj_spark.resource_prediction import predictor as pr

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
sys.path.insert(0, os.path.abspath(_SCRIPTS))

_CORES = {"kmeans": (1, 2, 4), "logreg": (1, 2, 4, 8)}
_HEAP = {"kmeans": 3072, "logreg": 6144}
_NOISE = {"kmeans": 0.02, "logreg": 0.4}  # LogReg much noisier → wider margin


def _capped_frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for wl in ("kmeans", "logreg"):
        for cores in _CORES[wl]:
            for size in (50, 100, 200):
                for iters in (10, 20):
                    for rep in range(2):
                        cpu = 1 + 0.5 * np.log2(cores) + rng.normal(0, _NOISE[wl])
                        rows.append(
                            {
                                "workload_type": wl,
                                "dataset_mb": size,
                                "num_workers": 2,
                                "cores_per_driver": cores,
                                "heap_mb_per_driver": _HEAP[wl],
                                "iterations": iters,
                                "num_features": 16,
                                "num_clusters": 5 if wl == "kmeans" else 0,
                                "thread_cap": cores,
                                "cpu_cores_mean": cpu,
                                "cpu_cores_p95": 1.3 * cpu,
                                "mem_mb_peak": 800 + 4 * size + 100 * cores + rng.normal(0, 20),
                                "mem_mb_p95_max": 750 + 4 * size + 100 * cores,
                                "mem_mb_peak_max": 800 + 4 * size + 100 * cores,
                                "rep": rep,
                            }
                        )
    df = pd.DataFrame(rows)
    key_cols = [c for c in m.features_for("capped") if c not in m.DERIVED_FEATURES]
    df["config_group"] = df[key_cols].astype(str).agg("|".join, axis=1)
    groups = sorted(df["config_group"].unique())
    split = {
        g: ("test" if i % 6 == 0 else "val" if i % 6 == 1 else "train")
        for i, g in enumerate(groups)
    }
    df["split"] = df["config_group"].map(split)
    df["in_main_set"], df["status"] = True, "ok"
    df["alloc_known"], df["design_known"] = True, True
    df["run_key"], df["driver_idx"] = df.index.astype(str), 1
    return df


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    import train_resource_predictor as trp

    root = tmp_path_factory.mktemp("predictor")
    path = root / "drivers.csv"
    _capped_frame().to_csv(path, index=False)
    out = root / "capped"
    with pytest.MonkeyPatch.context() as mp:  # small grids keep the test fast
        mp.setattr(m, "RIDGE_ALPHAS", [1.0])
        mp.setattr(m, "RF_GRID", [{"max_depth": 5, "min_samples_leaf": 1}])
        full_rf = m.make_rf
        mp.setattr(m, "make_rf", lambda **kw: full_rf(n_estimators=20, **kw))
        trp.main(
            ["--drivers", str(path), "--out-dir", str(out), "--training-set", "capped",
             "--scout-hosts", str(root / "no_scout.csv")]
        )  # fmt: skip
    return out


def _req(**kw):
    base = {
        "workload_type": "logreg",
        "dataset_mb": 100,
        "num_workers": 2,
        "cores_per_driver": 4,
        "iterations": 20,
        "num_features": 16,
    }
    return pr.PredictionRequest(**(base | kw))


def _same(a: dict, b: dict) -> None:
    """Equal results; floats may differ in the last bits (parallel tree sums)."""
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], float):
            assert a[key] == pytest.approx(b[key], rel=1e-9), key
        else:
            assert a[key] == b[key], key


# ── model card written by the training script ─────────────────


def test_card_records_margins_inputs_and_cap_rule(model_dir):
    card = json.loads((model_dir / "model_card.json").read_text())
    assert card["card_schema_version"] == pr.CARD_SCHEMA_VERSION
    assert card["model_version"].startswith("capped-")
    assert card["thread_cap_policy"] == "equals_cores"
    margins = card["choices"]["cpu_cores_mean"]["upper_margins"]["rf"]
    assert set(margins) == {"kmeans", "logreg", "pooled"}
    assert margins["logreg"] > margins["kmeans"]
    cores = card["valid_inputs"]["logreg"]["cores_per_driver"]
    assert cores["min"] == 1 and cores["max"] == 8
    assert set(cores["values"]) <= {1, 2, 4, 8}
    assert "heap_mb_per_driver" not in card["valid_inputs"]["kmeans"]  # not an input
    assert not set(m.DERIVED_FEATURES) & set(card["valid_inputs"]["kmeans"])  # follow inputs


def test_upper_margins_fall_back_to_pooled_for_small_workloads():
    oof = pd.DataFrame(
        {
            "workload_type": ["a"] * 30 + ["b"] * 5,
            "actual": np.r_[np.arange(30.0), np.full(5, 10.0)],
            "predicted": np.r_[np.arange(30.0) - 1, np.full(5, 0.0)],
        }
    )
    margins = m.upper_margins(oof, 0.9)
    assert margins["a"] == pytest.approx(1.0) and "b" not in margins
    assert m.margin_for(margins, "b") == margins["pooled"]


# ── predictions ───────────────────────────────────────────────


def test_in_domain_request_gets_predictions_and_upper_bounds(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir), "rf")
    res = p.predict(_req())
    assert res.status == "ok" and not res.out_of_domain and not res.reasons
    card = p.card["choices"]
    for target in m.TARGETS:
        value, upper = getattr(res, target), getattr(res, f"{target}_upper")
        assert value > 0
        method = pr.MARGIN_POLICY.get(target, {}).get("logreg", pr.DEFAULT_MARGIN_METHOD)
        margin = m.margin_for(card[target]["calibrated_margins"]["rf"][method], "logreg")
        assert upper == pytest.approx(m.apply_margin(value, margin, method))
    assert res.source == {t: "local_rf" for t in m.TARGETS}
    assert res.model_version == p.card["model_version"]
    assert res.cpu_cores_mean == pytest.approx(2.0, abs=0.5)  # 1 + 0.5·log2(4)


def test_predict_many_matches_single_requests(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir), "ridge_wl")
    reqs = [_req(cores_per_driver=c) for c in (1, 2, 8)] + [_req(workload_type="wordcount")]
    batch = p.predict_many(reqs)
    for got, req in zip(batch, reqs, strict=True):
        _same(got.to_dict(), p.predict(req).to_dict())
    assert batch[0].source["mem_mb_peak_max"] == "local_ridge_wl"


def test_best_cv_picks_lowest_out_of_fold_error(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir))
    for target in m.TARGETS:
        cv = p.card["choices"][target]["cv_uncertainty"]
        expected = pr.FAMILY_POLICY.get(target) or min(
            pr.BEST_CV_FAMILIES, key=lambda f: cv[f]["mae"]
        )
        assert p.families[target] == expected
    assert p.families["cpu_cores_p95"] == "rf_wl"  # forest per workload for CPU


# ── request checks and domain flags ───────────────────────────


def test_missing_or_invalid_inputs_are_rejected(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir), "rf")
    with pytest.raises(ValueError, match="iterations is missing"):
        p.predict(_req(iterations=None))
    with pytest.raises(ValueError, match="dataset_mb must be > 0"):
        p.predict(_req(dataset_mb=0))
    with pytest.raises(ValueError, match="num_clusters"):
        p.predict(_req(workload_type="kmeans", num_clusters=0))


def test_workload_without_training_data_is_unsupported(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir), "rf")
    res = p.predict(pr.PredictionRequest("wordcount", 50, 2, 4))
    assert res.status == "unsupported" and res.cpu_cores_mean is None
    assert "wordcount" in res.reasons[0] and "fallback" in res.reasons[0]
    assert set(res.source.values()) == {"none"}


@pytest.mark.parametrize(
    "change, flagged",
    [
        ({"cores_per_driver": 16, "thread_cap": 16}, "cores_per_driver=16 outside"),
        ({"dataset_mb": 1000}, "dataset_mb=1000 outside"),
        ({"iterations": 100}, "iterations=100 outside"),
        ({"thread_cap": 2}, "thread_cap=2 differs"),
    ],
)
def test_out_of_domain_requests_are_flagged(model_dir, change, flagged):
    p = pr.ResourcePredictor.load(str(model_dir), "rf")
    res = p.predict(_req(**change))
    assert res.status == "out_of_domain" and res.out_of_domain
    assert any(flagged in r for r in res.reasons)
    assert res.cpu_cores_mean is not None  # numbers still given; allocator decides


def test_data_per_driver_beyond_measured_is_flagged(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir), "rf")
    assert p.predict(_req()).status == "ok"  # 100 MB / 2 workers = 50 MB per driver
    for entry in p.card["growth_trends"]["size"]["logreg"].values():
        entry["x_max"] = 40.0  # pretend the largest measured was 40 MB per driver
    res = p.predict(_req())  # dataset_mb and num_workers each still in range
    assert res.status == "out_of_domain"
    assert any("data per driver 50 MB" in r and "largest measured 40 MB" in r for r in res.reasons)
    assert not any("dataset_mb=" in r or "num_workers=" in r for r in res.reasons)


def test_unmeasured_value_inside_range_is_only_a_note(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir), "rf")
    res = p.predict(_req(cores_per_driver=3))
    assert res.status == "ok" and not res.reasons
    assert any("cores_per_driver=3 was not measured" in n for n in res.notes)


def test_request_computes_the_derived_inputs():
    row = _req(dataset_mb=100, num_workers=2, iterations=30, num_features=20).feature_row()
    assert (row["shard_mb"], row["shard_x_iters"], row["shard_x_features"]) == (50, 1500, 1000)
    assert pr.PredictionRequest("kmeans", 100, 2, 4).feature_row()["shard_x_iters"] is None


def test_non_kmeans_ignores_cluster_count(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir), "rf")
    _same(p.predict(_req(num_clusters=7)).to_dict(), p.predict(_req()).to_dict())


def test_old_model_card_is_refused(tmp_path):
    (tmp_path / "model_card.json").write_text(json.dumps({"features": []}))
    with pytest.raises(ValueError, match="re-run scripts/train_resource_predictor.py"):
        pr.ResourcePredictor.load(str(tmp_path))


def test_unknown_model_name_is_refused(model_dir):
    with pytest.raises(ValueError, match="unknown model"):
        pr.ResourcePredictor.load(str(model_dir), "lstm")


# ── growth trends beyond the measured range ───────────────────


def test_card_records_growth_trends(model_dir):
    card = json.loads((model_dir / "model_card.json").read_text())
    entry = card["growth_trends"]["cores"]["kmeans"]["mem_mb_p95_max"]
    assert entry["x_max"] == 4 and entry["local"]["beta"] > 0
    assert entry["scout"] is None and entry["combined"]["source"] == "local"  # no Scout file


def test_kmeans_beyond_measured_cores_grows_with_local_trend(model_dir):
    km = {"workload_type": "kmeans", "num_clusters": 5}
    on = pr.ResourcePredictor.load(str(model_dir), "rf")
    off = pr.ResourcePredictor.load(str(model_dir), "rf", trends="off")
    res = on.predict(_req(cores_per_driver=8, thread_cap=8, **km))
    at4 = off.predict(_req(cores_per_driver=4, **km))
    for target in m.TARGETS:
        beta = on.card["growth_trends"]["cores"]["kmeans"][target]["local"]["beta"]
        assert getattr(res, target) == pytest.approx(getattr(at4, target) * 2**beta)
        assert res.source[target].endswith("+trend")
    assert res.out_of_domain  # still extrapolation, still flagged
    assert any("grown" in n and "cores^" in n for n in res.notes)


def test_size_trend_only_for_logreg_memory(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir), "rf")
    res = p.predict(_req(dataset_mb=400, num_workers=1))  # 400 MB per driver > measured 100
    assert res.source["mem_mb_p95_max"].endswith("+trend")
    assert res.source["mem_mb_peak_max"].endswith("+trend")
    assert not res.source["cpu_cores_p95"].endswith("+trend")


def test_kmeans_peak_memory_uses_the_power_law_model(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir))
    assert p.family_for("mem_mb_peak_max", "kmeans") == "ridge_log_wl"
    assert p.family_for("mem_mb_peak_max", "logreg") == p.families["mem_mb_peak_max"]
    assert p.family_for("mem_mb_p95_max", "kmeans") == p.families["mem_mb_p95_max"]
    km = p.predict(_req(workload_type="kmeans", num_clusters=5))
    lr = p.predict(_req())
    assert km.source["mem_mb_peak_max"] == "local_ridge_log_wl"
    assert lr.source["mem_mb_peak_max"] == pr.SOURCE_NAMES[p.families["mem_mb_peak_max"]]
    explicit = pr.ResourcePredictor.load(str(model_dir), "rf")  # explicit model: no exceptions
    assert explicit.family_for("mem_mb_peak_max", "kmeans") == "rf"


def test_card_has_calibrated_and_beyond_range_margins(model_dir):
    card = json.loads((model_dir / "model_card.json").read_text())
    cal = card["choices"]["cpu_cores_p95"]["calibrated_margins"]["rf_wl"]
    assert set(cal) == set(m.MARGIN_METHODS)
    assert "cores" in card["extrapolation_margins"]["cpu_cores_p95"]


def test_margin_method_follows_the_policy(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir))
    cal = p.card["choices"]["cpu_cores_p95"]["calibrated_margins"]["rf_wl"]
    lr = p.predict(_req())  # LogReg: relative margin
    r = m.margin_for(cal["relative_groups"], "logreg")
    assert lr.cpu_cores_p95_upper == pytest.approx(lr.cpu_cores_p95 * (1 + r))
    km = p.predict(_req(workload_type="kmeans", num_clusters=5))  # K-Means: additive margin
    a = m.margin_for(cal["additive_rows"], "kmeans")
    assert km.cpu_cores_p95_upper == pytest.approx(km.cpu_cores_p95 + a)


def test_beyond_range_requests_get_the_stress_test_floor(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir))
    km = {"workload_type": "kmeans", "num_clusters": 5}
    # K-Means: the synthetic table has data per driver above the 50 MB stress limit
    p.card["extrapolation_margins"]["mem_mb_p95_max"]["size"]["kmeans"]["margin"] = 2.0
    res = p.predict(_req(dataset_mb=400, num_workers=1, **km))  # beyond the measured size
    assert res.mem_mb_p95_max_upper == pytest.approx(res.mem_mb_p95_max * 3.0)
    assert any("upper bound widened" in n for n in res.notes)
    inside = p.predict(_req(**km))
    assert not any("upper bound widened" in n for n in inside.notes)


def test_size_trend_is_scout_informed_by_default(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir), "rf")
    assert pr.TREND_POLICY["size"]["logreg"]["mem_mb_p95_max"] == "combined"
    assert pr.TREND_POLICY["cores"]["kmeans"]["mem_mb_p95_max"] == "local"
    res = p.predict(_req(dataset_mb=400, num_workers=1))
    assert any("Scout-informed trend" in n for n in res.notes)


def test_scout_trend_is_the_fallback_without_local_trend(model_dir):
    p = pr.ResourcePredictor.load(str(model_dir), "rf")
    off = pr.ResourcePredictor.load(str(model_dir), "rf", trends="off")
    entry = p.card["growth_trends"]["size"]["logreg"]["mem_mb_p95_max"]
    entry["local"] = None  # no local size trend, Scout only (as in the scarce test)
    entry["combined"] = {"beta": 0.5, "scout_weight": 1.0, "source": "scout"}
    x_max = entry["x_max"]
    res = p.predict(_req(dataset_mb=4 * x_max, num_workers=1))
    at_anchor = off.predict(_req(dataset_mb=x_max, num_workers=1))
    assert res.mem_mb_p95_max == pytest.approx(at_anchor.mem_mb_p95_max * 4**0.5)
    local_only = pr.ResourcePredictor.load(str(model_dir), "rf", trends="local_only")
    local_only.card = p.card  # same edited card: the local trend is missing
    kept = local_only.predict(_req(dataset_mb=4 * x_max, num_workers=1))
    assert any("no local size trend available" in n for n in kept.notes)


def test_trends_off_and_unknown_choice(model_dir):
    off = pr.ResourcePredictor.load(str(model_dir), "rf", trends="off")
    res = off.predict(_req(dataset_mb=400, num_workers=1))
    assert not any(v.endswith("+trend") for v in res.source.values())
    with pytest.raises(ValueError, match="unknown trends"):
        pr.ResourcePredictor.load(str(model_dir), "rf", trends="always")


# ── command line ──────────────────────────────────────────────


def test_cli_table_and_json(model_dir, capsys):
    import predict_resources as cli

    args = [
        "--model-dir",
        str(model_dir),
        "--model",
        "rf",
        "--workload",
        "logreg",
        "--dataset-mb",
        "100",
        "--workers",
        "2",
        "--cores",
        "1,3,16",
        "--heap-mb",
        "6144",
        "--iterations",
        "20",
        "--features",
        "16",
    ]
    results = cli.main(args)
    out = capsys.readouterr().out
    assert [r.status for r in results] == ["ok", "ok", "out_of_domain"]
    assert "memory of the largest driver" in out and "cores_per_driver=16 outside" in out
    cli.main([*args, "--json"])
    records = json.loads(capsys.readouterr().out)
    assert len(records) == 3 and records[0]["source"]["cpu_cores_mean"] == "local_rf"
