# =============================================================================
# mpj_spark/resource_prediction/local_data.py
# Phase 6 — local target-domain tables from per-rank profiling CSVs (P6-01)
#
# PURPOSE
# -------
# Turn the per-rank resource CSVs of the design-of-experiments runs (MPI runs
# of K-Means and LogReg with a per-process CPU / RSS sampler, collected on
# 14 October 2026 after the Open MPI binding fix) into the TARGET-DOMAIN
# tables of the Phase 6 resource predictor:
#
#   local_drivers.csv   one row per Spark driver (role == driver) per run
#   local_runs.csv      one row per run (driver aggregates + run outcome)
#
# DATA SETS  (see the predictor handoff; one row per driver rank + one root row)
#   A   doe_kmeans.csv     K-Means at the 4-core reference allocation   train
#   B   doe_logreg.csv     LogReg  at the 4-core reference allocation   train
#   D   doe_rightsize.csv  core sweep (K-Means 1-4, LogReg 1-8 cores)   train
#   E1  doe_holdout.csv    unseen sizes: interior / extrapolation       holdout
#   E2  doe_holdout.csv    third repetition at each knee (noise)        holdout
#   E3  doe_holdout.csv    memory-limit runs — VOID (limits not enforced)
# The set of a row comes from the `set` column, or else from the run_id
# prefix (A_..., B_..., D_...).
#
# SPLIT GROUPS  (handoff: never split by row)
#   design_id for Sets A and B, workload_id for Set D, so repetitions, drivers
#   and — for Set D — all core counts of one workload stay on the same side.
#   Groups that contain an IDENTICAL configuration (a Set D workload at
#   4 cores can equal a Set A/B design point) are merged, otherwise the same
#   job could sit in train and test.  `config_group` holds this split unit;
#   `exact_config` the exact configuration.
#
# MEASUREMENT SCOPE
# -----------------
# Labels are PER DRIVER PROCESS TREE as recorded by the sampler:
#   cpu_*_pct  — process CPU percent, 100 % = one core  →  divided by 100
#   rss_*_mb   — resident memory of the driver process
# Memory targets of the handoff are per RUN: the largest driver
# (mem_mb_p95_max, mem_mb_peak_max), because every driver of a job gets the
# same memory limit.  They are stored once per run (on the lowest driver
# rank, NaN on the others) so each run counts once in training.
# Host-wide columns (sys_*) and iters_logged are measured DURING the run, so
# they are diagnostics only and never predictor inputs.
#
# CHANGING THE SCOPE
# ------------------
# Everything a researcher may adjust lives in the CONFIGURATION block below.
# Rebuild with scripts/build_local_dataset.py after editing.
# =============================================================================

from __future__ import annotations

import glob
import hashlib
import os

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION — edit here to change what the dataset contains
# =============================================================================

# Files left out completely, by exact name or by name prefix, with the reason
# recorded in the build report.
EXCLUDED_FILES: dict[str, str] = {}
EXCLUDED_FILE_PREFIXES: dict[str, str] = {
    "pre_": "collected before the Open MPI binding fix (handoff: must not be used)",
    "pilot_": "pilot trial runs before Set A; not part of the handoff data inventory",
}

# Role of every data set: "train" (training / CV / frozen test split),
# "holdout" (scored separately, never trained on) or "excluded".
SET_ROLES: dict[str, str] = {
    "A": "train",
    "B": "train",
    "D": "train",
    "E1": "holdout",
    "E2": "holdout",
    "E3": "excluded",
}
SET_NOTES: dict[str, str] = {
    "E1": "holdout sizes: interior (75, 125 MB) and extrapolation (>200 MB)",
    "E2": "third repetition at each knee: noise ceiling",
    "E3": "VOID: memory limits were not enforced (handoff caveat 5)",
}

# exit codes that mean "stopped by the time limit" → censored, not a label.
TIMEOUT_EXIT_CODES: set[int] = {124}

WORKLOAD_CLASS: dict[str, str] = {
    "wordcount": "batch",
    "kmeans": "iterative",
    "logreg": "iterative",
}

SPLIT_FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

# =============================================================================
# Output schema
# =============================================================================

TRACKING_COLUMNS = [
    "source",
    "file",
    "experiment",
    "holdout_kind",
    "run_id",
    "run_key",
    "timestamp",
    "host",
    "rep",
    "source_group",
    "exact_config",
    "config_group",
    "split",
    "in_main_set",
]
INPUT_COLUMNS = [
    "workload_type",
    "workload_class",
    "dataset_mb",
    "num_workers",
    "cores_per_driver",
    "heap_mb_per_driver",
    "iterations",
    "num_features",
    "num_clusters",
    "thread_cap",
    "alloc_known",
    "design_known",
]
DRIVER_LABEL_COLUMNS = [
    "cpu_cores_mean",
    "cpu_cores_p95",
    "cpu_cores_peak",
    "mem_mb_mean",
    "mem_mb_p95",
    "mem_mb_peak",
]
# run-level labels: the largest driver of the run (stored once per run)
RUN_MAX_LABELS: dict[str, str] = {"mem_mb_p95_max": "mem_mb_p95", "mem_mb_peak_max": "mem_mb_peak"}
OUTCOME_COLUMNS = ["wall_s", "exit_code", "status"]
DIAGNOSTIC_COLUMNS = [
    "n_samples",
    "iters_logged",
    "host_cpu_avg_pct",
    "host_mem_avg_mb",
    "host_mem_peak_mb",
    "host_mem_avail_min_mb",
]

_RAW_RENAME = {
    "app": "workload_type",
    "size_mb": "dataset_mb",
    "workers": "num_workers",
    "iters": "iterations",
    "features": "num_features",
    "cores_alloc": "cores_per_driver",
    "heap_mb_alloc": "heap_mb_per_driver",
    "cap_threads": "thread_cap",
    "sys_cpu_avg_pct": "host_cpu_avg_pct",
    "sys_mem_avg_mb": "host_mem_avg_mb",
    "sys_mem_peak_mb": "host_mem_peak_mb",
    "sys_mem_avail_min_mb": "host_mem_avail_min_mb",
    "rss_avg_mb": "mem_mb_mean",
    "rss_p95_mb": "mem_mb_p95",
    "rss_peak_mb": "mem_mb_peak",
}

# columns every per-rank measurement file must have
REQUIRED_RAW_COLUMNS = [
    "run_id",
    "timestamp",
    "app",
    "size_mb",
    "workers",
    "iters",
    "features",
    "cap_threads",
    "driver_idx",
    "role",
    "exit_code",
    "wall_s",
    "cpu_avg_pct",
    "cpu_p95_pct",
    "cpu_peak_pct",
    "rss_avg_mb",
    "rss_p95_mb",
    "rss_peak_mb",
]

# =============================================================================
# Loading
# =============================================================================


def excluded_reason(filename: str) -> str | None:
    if filename in EXCLUDED_FILES:
        return EXCLUDED_FILES[filename]
    for prefix, reason in EXCLUDED_FILE_PREFIXES.items():
        if filename.startswith(prefix):
            return reason
    return None


def load_file(path: str) -> pd.DataFrame:
    """Read one per-rank measurement CSV and bring it to the common column set."""
    df = pd.read_csv(path)
    name = os.path.basename(path)
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df]
    if "cores_alloc" not in df and "cores" not in df:
        missing.append("cores_alloc or cores")
    if "design_id" not in df and "workload_id" not in df:
        missing.append("design_id or workload_id")
    if missing:
        raise ValueError(
            f"{name}: not a per-rank measurement file (missing {missing}); "
            "keep summary / comparison files outside the raw folder"
        )
    if "cores_alloc" not in df:
        df["cores_alloc"] = df["cores"]
    if "heap_mb_alloc" not in df:
        df["heap_mb_alloc"] = df["heap_mb"] if "heap_mb" in df else np.nan
    for col in ("k", "iters_logged", "kind"):
        if col not in df:
            df[col] = np.nan
    if "set" not in df:
        df["set"] = df["run_id"].str.split("_").str[0]
    # handoff split groups: design_id (Sets A, B) or workload_id (Sets D, E)
    group_id = df["design_id"] if "design_id" in df else df["workload_id"]
    df = df.rename(columns=_RAW_RENAME)
    df["file"] = name
    df["experiment"] = df["set"].astype(str)
    df["source_group"] = df["experiment"] + ":" + group_id.astype(str)
    df["holdout_kind"] = df["kind"].fillna("")
    # k only exists for K-Means; other workloads have no clusters (0, not missing)
    df["num_clusters"] = np.where(df["workload_type"] == "kmeans", df["k"], 0.0)
    unknown = sorted(set(df["experiment"]) - set(SET_ROLES))
    if unknown:
        raise ValueError(f"{name}: data set(s) {unknown} have no role; add them to SET_ROLES")
    return df


def exact_config(row) -> str:
    """Key of one exact configuration (all repetitions of a job share it)."""
    parts = [
        row["workload_type"],
        f"s{_int_or_na(row['dataset_mb'])}",
        f"w{_int_or_na(row['num_workers'])}",
        f"c{_int_or_na(row['cores_per_driver'])}",
        f"h{_int_or_na(row['heap_mb_per_driver'])}",
        f"cap{_int_or_na(row['thread_cap'])}",
        f"i{_int_or_na(row['iterations'])}",
        f"f{_int_or_na(row['num_features'])}",
        f"k{_int_or_na(row['num_clusters'])}",
    ]
    return "|".join(parts)


def _int_or_na(v) -> str:
    return "na" if pd.isna(v) else str(int(v))


def link_groups(rows: pd.DataFrame) -> dict[str, str]:
    """
    {source_group: split group}.  Source groups that share an exact
    configuration are merged (union-find), so one configuration can never be
    on both sides of a split.  The merged name lists its members.
    """
    parent = {g: g for g in rows["source_group"].unique()}

    def find(g: str) -> str:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    for _, members in rows.groupby("exact_config")["source_group"]:
        first, *rest = sorted(set(members))
        for other in rest:
            a, b = find(first), find(other)
            if a != b:
                parent[max(a, b)] = min(a, b)
    clusters: dict[str, list[str]] = {}
    for g in parent:
        clusters.setdefault(find(g), []).append(g)
    return {g: "+".join(sorted(members)) for members in clusters.values() for g in members}


def _hash_split(key: str, fractions: dict[str, float] = SPLIT_FRACTIONS) -> str:
    """Per-group hash split used when a stratum has too few groups to stratify."""
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < fractions["test"]:
        return "test"
    if h < fractions["test"] + fractions["val"]:
        return "val"
    return "train"


def stratified_split(groups: pd.DataFrame, fractions: dict[str, float] = SPLIT_FRACTIONS) -> dict:
    """
    {config_group: split}, stratified (column `stratum`) so every workload —
    and the core-sweep workloads of Set D separately — appears in val and
    test.  Groups are ordered by a hash of their key, so the split is
    deterministic and does not depend on file order.  Strata with fewer than
    3 groups fall back to a per-group hash split.
    """
    out = {}
    for _, g in groups.groupby("stratum"):
        keys = sorted(
            g["config_group"].unique(), key=lambda k: hashlib.sha256(k.encode()).hexdigest()
        )
        n = len(keys)
        if n < 3:
            out.update({k: _hash_split(k, fractions) for k in keys})
            continue
        n_test = max(1, round(n * fractions["test"]))
        n_val = max(1, round(n * fractions["val"]))
        for i, key in enumerate(keys):
            out[key] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def assign_splits(groups: pd.DataFrame, frozen: dict[str, str] | None = None) -> dict[str, str]:
    """
    Keep every group already recorded in `frozen` in its original split and
    split only the NEW groups (stratified among themselves).  This keeps the
    held-out test set fixed when new result files are added.
    """
    frozen = frozen or {}
    new = groups[~groups["config_group"].isin(frozen)]
    out = {g: frozen[g] for g in groups["config_group"] if g in frozen}
    out.update(stratified_split(new) if len(new) else {})
    return out


def status_for(exit_code: int) -> str:
    if exit_code == 0:
        return "ok"
    if exit_code in TIMEOUT_EXIT_CODES:
        return "timeout"
    return "failed"


# =============================================================================
# Table assembly
# =============================================================================


def build_tables(
    raw_dir: str, frozen_splits: dict[str, str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Return (drivers, runs, report) from every *.csv in raw_dir.

    `frozen_splits` ({config_group: split}) from an earlier build keeps those
    groups in their split; report["split_map"] holds the mapping to save.
    """
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    report = {"files": {}, "excluded_files": {}, "excluded_sets": {}}
    frames = []
    for path in files:
        name = os.path.basename(path)
        reason = excluded_reason(name)
        if reason:
            report["excluded_files"][name] = reason
            continue
        df = load_file(path)
        report["files"][name] = {
            "rows": len(df),
            "runs": int(df["run_id"].nunique()),
            "sets": sorted(df["experiment"].unique()),
        }
        frames.append(df)
    if not frames:
        raise ValueError(f"no usable CSV files in {raw_dir!r}")
    raw = pd.concat(frames, ignore_index=True)

    # The same run copied into two files must count once: a run is identified
    # by run_id + timestamp, a row by + driver_idx (first file in name order kept).
    dup = raw.duplicated(["run_id", "timestamp", "driver_idx"])
    report["duplicate_rows_dropped"] = int(dup.sum())
    report["files_with_duplicate_rows"] = sorted(raw.loc[dup, "file"].unique())
    raw = raw[~dup].copy()

    role = raw["experiment"].map(SET_ROLES)
    for s in sorted(raw.loc[role == "excluded", "experiment"].unique()):
        report["excluded_sets"][s] = {
            "runs": int(raw.loc[raw["experiment"] == s, "run_id"].nunique()),
            "reason": SET_NOTES.get(s, "excluded in SET_ROLES"),
        }
    raw = raw[role != "excluded"].copy()
    role = raw["experiment"].map(SET_ROLES)

    raw["run_key"] = raw["file"] + ":" + raw["run_id"] + "@" + raw["timestamp"].astype(str)
    raw["source"] = "local"
    raw["workload_class"] = raw["workload_type"].map(WORKLOAD_CLASS)
    raw["alloc_known"] = raw["cores_per_driver"].notna()
    raw["design_known"] = raw[["iterations", "num_features", "num_clusters"]].notna().all(axis=1)
    raw["exact_config"] = raw.apply(exact_config, axis=1)
    raw["in_main_set"] = role == "train"

    main = raw["in_main_set"]
    linked = link_groups(raw.loc[main, ["source_group", "exact_config"]])
    raw["config_group"] = np.where(main, raw["source_group"].map(linked), raw["source_group"])
    family = np.where(raw["config_group"].str.contains("D:"), "sweep", "reference")
    raw["stratum"] = raw["workload_type"] + "|" + family
    split_map = assign_splits(
        raw.loc[main, ["config_group", "stratum"]].drop_duplicates(), frozen_splits
    )
    raw["split"] = np.where(main, raw["config_group"].map(split_map), "holdout")
    frozen_splits = frozen_splits or {}
    report["split_groups_kept_from_file"] = int(sum(g in frozen_splits for g in split_map))
    report["split_groups_newly_assigned"] = int(sum(g not in frozen_splits for g in split_map))
    report["linked_groups"] = sorted({g for g in linked.values() if "+" in g})
    report["split_map"] = split_map
    raw["status"] = raw["exit_code"].map(status_for)
    for stat in ("avg", "p95", "peak"):
        name = "mean" if stat == "avg" else stat
        raw[f"cpu_cores_{name}"] = raw[f"cpu_{stat}_pct"] / 100.0

    drivers = raw[raw["role"] == "driver"].copy()
    report["root_rows_dropped"] = int((raw["role"] != "driver").sum())
    # run-level memory labels: the largest driver, stored on one row per run
    first = drivers["driver_idx"] == drivers.groupby("run_key")["driver_idx"].transform("min")
    for label, col in RUN_MAX_LABELS.items():
        drivers[label] = drivers.groupby("run_key")[col].transform("max").where(first)
    driver_cols = (
        TRACKING_COLUMNS
        + ["driver_idx"]
        + INPUT_COLUMNS
        + DRIVER_LABEL_COLUMNS
        + list(RUN_MAX_LABELS)
        + OUTCOME_COLUMNS
        + DIAGNOSTIC_COLUMNS
    )
    drivers = drivers[driver_cols].reset_index(drop=True)

    agg = drivers.groupby("run_key").agg(
        drivers_recorded=("driver_idx", "count"),
        **{f"{c}_avg_drivers": (c, "mean") for c in DRIVER_LABEL_COLUMNS},
        **{f"{c}_max_drivers": (c, "max") for c in DRIVER_LABEL_COLUMNS},
    )
    run_first = (
        drivers.drop(columns=["driver_idx", *DRIVER_LABEL_COLUMNS, *RUN_MAX_LABELS])
        .groupby("run_key", as_index=False)
        .first()
    )
    runs = run_first.merge(agg, on="run_key").reset_index(drop=True)
    return drivers, runs, report


def summarise(drivers: pd.DataFrame, runs: pd.DataFrame, report: dict) -> dict:
    """Counts and sanity checks for the build report."""
    ok = drivers[drivers["status"] == "ok"]
    main = ok[ok["in_main_set"]]
    groups = runs.groupby("config_group")["split"].nunique()
    report.update(
        {
            "runs": len(runs),
            "driver_rows": len(drivers),
            "runs_by_status": runs["status"].value_counts().to_dict(),
            "runs_by_set": runs["experiment"].value_counts().sort_index().to_dict(),
            "runs_by_workload": runs["workload_type"].value_counts().to_dict(),
            "runs_by_split": runs["split"].value_counts().to_dict(),
            "holdout_runs_by_kind": {
                f"{s}/{k}": int(n)
                for (s, k), n in runs.loc[runs["split"] == "holdout"]
                .groupby(["experiment", "holdout_kind"])
                .size()
                .items()
            },
            "split_groups": int(runs.loc[runs["in_main_set"], "config_group"].nunique()),
            "split_groups_by_split": runs.loc[runs["in_main_set"]]
            .drop_duplicates("config_group")["split"]
            .value_counts()
            .to_dict(),
            "config_groups_spanning_splits": int((groups > 1).sum()),
            "main_set_ok_driver_rows": len(main),
            "checks": {
                "share_cpu_p95_above_cores_given": (
                    round(float((main["cpu_cores_p95"] > main["cores_per_driver"]).mean()), 3)
                    if len(main)
                    else None
                ),
                "share_cpu_mean_above_cores_given": (
                    round(float((main["cpu_cores_mean"] > main["cores_per_driver"]).mean()), 3)
                    if len(main)
                    else None
                ),
                "thread_cap_equals_cores": bool((ok["thread_cap"] == ok["cores_per_driver"]).all()),
                "missing_cpu_p95_rows": int(ok["cpu_cores_p95"].isna().sum()),
                "missing_mem_p95_rows": int(ok["mem_mb_p95"].isna().sum()),
                "missing_design_rows": int((~ok["design_known"]).sum()),
            },
        }
    )
    return report
