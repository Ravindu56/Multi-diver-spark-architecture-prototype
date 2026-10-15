# =============================================================
# tests/unit/test_evaluate_predictor.py
# Phase 6 — step A4 evaluation helpers: scored sets, metrics
# (R² rule, upper-bound rates), paired bootstrap, leakage audit
# and the markdown tables.  Small synthetic frames, no models.
# =============================================================
import os
import sys

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
sys.path.insert(0, os.path.abspath(_SCRIPTS))

import evaluate_resource_predictor as ev  # noqa: E402


def _long(groups=3, rows_per_group=2, offset=0.0, variant="deployed"):
    """Rows of one target: actual = 10·g, predicted = actual + offset."""
    g = np.repeat(np.arange(groups), rows_per_group)
    actual = 10.0 * (g + 1)
    return pd.DataFrame(
        {
            "variant": variant,
            "set": "test",
            "target": "cpu_cores_p95",
            "workload_type": "kmeans",
            "num_workers": 2,
            "cores_per_driver": 4.0,
            "config_group": [f"g{i}" for i in g],
            "actual": actual,
            "predicted": actual + offset,
            "upper": actual + offset + 1.0,
            "status": "ok",
            "source": "local_rf_wl",
        }
    )


def test_scored_sets_keep_finished_runs_of_each_set():
    raw = pd.DataFrame(
        {
            "status": ["ok", "ok", "ok", "ok", "failed"],
            "split": ["test", "train", "holdout", "holdout", "test"],
            "experiment": ["A", "A", "E1", "E1", "B"],
            "holdout_kind": [None, None, "interior", "extrap", None],
        }
    )
    sets = ev.scored_sets(raw)
    assert [len(sets[k]) for k in ("test", "E1 interior", "E1 extrap")] == [1, 1, 1]


def test_metrics_report_r2_only_where_meaningful():
    three = ev.metrics(_long(groups=3, offset=1.0))
    assert three["r2"] is not None and three["n_groups"] == 3
    assert ev.metrics(_long(groups=2, offset=1.0))["r2"] is None  # too few groups
    assert ev.metrics(_long(groups=3, offset=1.0), r2=False)["r2"] is None  # narrow slice
    assert three["mean_upper_width"] == pytest.approx(1.0)
    assert three["above_upper_rate"] == 0.0 and three["flagged_out_of_range"] == 0.0


def test_metrics_count_rows_above_the_upper_bound():
    d = _long(groups=3, offset=-2.0)  # under-predicted by 2, upper = actual - 1
    r = ev.metrics(d)
    assert r["above_upper_rate"] == 1.0 and r["underprediction_rate"] == 1.0
    assert r["mean_signed_error"] == pytest.approx(-2.0)


def test_metric_table_slices_and_variants():
    long = pd.concat([_long(offset=1.0), _long(offset=3.0, variant="lookup")])
    mt = ev.metric_table(long)
    assert set(mt["variant"]) == {"deployed", "lookup"}
    assert set(mt["by"]) == {"all", *ev.BY}
    assert mt[mt["by"] == "num_workers"]["r2"].isna().all()


def test_paired_differences_favour_the_better_variant():
    long = pd.concat(
        [_long(offset=1.0)]
        + [_long(offset=3.0, variant=v) for v in ("lookup", "local_trends_only", "no_trends")]
    )
    pt = ev.paired_table(long)
    vs_lookup = pt[(pt["baseline"] == "lookup") & (pt["workload"] == "all")].iloc[0]
    assert vs_lookup["mae_diff"] == pytest.approx(-2.0)
    assert vs_lookup["diff_ci95_high"] < 0


def test_leakage_audit_finds_overlap_and_post_run_inputs():
    raw = pd.DataFrame(
        {
            "split": ["train", "val", "test", "holdout", "holdout"],
            "experiment": ["A", "A", "A", "E1", "E2"],
            "exact_config": ["c1", "c2", "c3", "c4", "c1"],
            "config_group": ["g1", "g2", "g3", "E1:g4", "E2:g1"],
        }
    )
    sets = {"test": raw[raw["split"] == "test"], "E1": raw[raw["experiment"] == "E1"]}
    audit = ev.leakage_audit(raw, sets, ["workload_type", "dataset_mb", "wall_s"])
    overlap = audit["overlap_with_train_val"]
    assert overlap["test"]["exact_configs_in_train_val"] == 0
    assert overlap["E2 (noise ceiling only, not scored)"]["exact_configs_in_train_val"] == 1
    assert audit["groups_in_more_than_one_split"] == 0
    assert audit["post_run_columns_among_inputs"] == ["wall_s"]


def test_markdown_table_has_header_separator_and_rows():
    md = ev._md(pd.DataFrame({"a": [1, None], "b": ["x", "y"]}))
    lines = md.splitlines()
    assert lines[0] == "| a | b |" and lines[1] == "|---|---|" and len(lines) == 4
