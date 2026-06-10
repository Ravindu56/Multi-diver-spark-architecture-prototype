#!/usr/bin/env python3
"""
generate_data.py
~~~~~~~~~~~~~~~~
Regenerates ALL datasets under shared_storage/ with parameters tuned
for visible multi-iteration convergence curves.

Run once before any test sweep:
    python generate_data.py

Outputs
-------
shared_storage/logreg_data.csv   — 540 000 rows, 10 features + label
shared_storage/kmeans_data.csv   — 540 000 rows, 8 features (no label)
shared_storage/wordcount_data.txt — ~1 M words, realistic vocabulary

Design rationale
----------------
logreg:
  class_sep=0.5   Overlapping class boundaries -> L-BFGS cannot converge
                  in a single step; weight_delta stays > 0 across all
                  30 iterations, giving Objective 2a a visible gradient.
  n_informative=8 Signal spread across 8 of 10 features -> no single
                  dominant weight coefficient; FedAvg must average a
                  truly multi-dimensional weight vector each round.
  flip_y=0.05     5% label noise prevents perfect accuracy, keeping the
                  optimiser active through all iterations.
  n_clusters_per_class=2  Non-convex local structure per class forces
                  iterative refinement rather than one-shot solution.

kmeans:
  cluster_std=1.8 Wide overlapping clusters -> centroids shift each
                  iteration rather than snapping to solution in round 1.
  n_clusters=5    Odd number of clusters with uneven counts exposes
                  centroid drift differences across workers.
"""

import os
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs

OUT_DIR = os.path.join(os.path.dirname(__file__), "shared_storage")
N_SAMPLES = 540_000
RANDOM_STATE = 42


def ensure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[generate_data] Output directory: {OUT_DIR}")


def generate_logreg(path: str):
    """
    Binary classification dataset tuned for multi-iteration convergence.

    Parameters chosen so that:
      - All 10 features contribute to the decision boundary
      - Classes overlap -> accuracy < 1.0 -> weight_delta > 0 each iter
      - Different reg_param values produce clearly different convergence
        trajectories (needed for Objective 2b predictor training)
    """
    print(f"[generate_data] Generating logreg dataset ({N_SAMPLES:,} rows) ...")
    t0 = time.perf_counter()

    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=10,
        n_informative=8,        # 8/10 features carry real signal
        n_redundant=2,          # 2 features are linear combos of informative
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2, # non-convex per-class structure
        class_sep=0.5,          # KEY: overlapping classes
        flip_y=0.05,            # 5% label noise
        random_state=RANDOM_STATE,
    )

    cols = [f"f{i}" for i in range(10)] + ["label"]
    df = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=cols)
    df["label"] = df["label"].astype(int)

    # Write with header (logreg.py / queue_run.py drops the stray header
    # via the schema+filter approach, so header=True is correct here)
    df.to_csv(path, index=False, header=True)

    elapsed = time.perf_counter() - t0
    balance = df["label"].mean()
    print(f"[generate_data] logreg_data.csv  {len(df):,} rows  "
          f"class_balance={balance:.3f}  ({elapsed:.1f}s)")
    print(f"[generate_data] Feature means: "
          f"{np.abs(X).mean(axis=0).round(3).tolist()}")


def generate_kmeans(path: str):
    """
    Clustering dataset tuned for multi-iteration centroid drift.

    cluster_std=1.8 means clusters overlap significantly -> centroids
    require many iterations to stabilise, giving Objective 2a a
    non-trivial convergence signal for k-means workloads.
    """
    print(f"[generate_data] Generating kmeans dataset ({N_SAMPLES:,} rows) ...")
    t0 = time.perf_counter()

    N_FEATURES = 8
    N_CLUSTERS = 5

    X, _ = make_blobs(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        centers=N_CLUSTERS,
        cluster_std=1.8,        # KEY: wide overlapping clusters
        random_state=RANDOM_STATE,
    )

    cols = [f"f{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=cols)
    df.to_csv(path, index=False, header=True)

    elapsed = time.perf_counter() - t0
    print(f"[generate_data] kmeans_data.csv  {len(df):,} rows  "
          f"{N_FEATURES} features  {N_CLUSTERS} clusters  ({elapsed:.1f}s)")


def generate_wordcount(path: str):
    """
    Synthetic text corpus for WordCount baseline.
    ~1 M words drawn from a Zipf-distributed vocabulary of 5 000 words,
    written as plain text lines of 10-20 words each.
    """
    print(f"[generate_data] Generating wordcount corpus ...")
    t0 = time.perf_counter()

    rng = random.Random(RANDOM_STATE)
    VOCAB_SIZE = 5_000
    TARGET_WORDS = 1_000_000
    LINE_LEN_MIN, LINE_LEN_MAX = 10, 20

    # Zipf-distributed vocabulary (rank -> frequency proportional to 1/rank)
    vocab = [f"word{i:05d}" for i in range(1, VOCAB_SIZE + 1)]
    weights = [1.0 / i for i in range(1, VOCAB_SIZE + 1)]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    lines = []
    words_written = 0
    while words_written < TARGET_WORDS:
        line_len = rng.randint(LINE_LEN_MIN, LINE_LEN_MAX)
        chosen = rng.choices(vocab, weights=weights, k=line_len)
        lines.append(" ".join(chosen))
        words_written += line_len

    with open(path, "w") as f:
        f.write("\n".join(lines))

    elapsed = time.perf_counter() - t0
    print(f"[generate_data] wordcount_data.txt  {words_written:,} words  "
          f"{len(lines):,} lines  ({elapsed:.1f}s)")


def main():
    ensure_dir()

    logreg_path   = os.path.join(OUT_DIR, "logreg_data.csv")
    kmeans_path   = os.path.join(OUT_DIR, "kmeans_data.csv")
    wordcount_path = os.path.join(OUT_DIR, "wordcount_data.txt")

    generate_logreg(logreg_path)
    generate_kmeans(kmeans_path)
    generate_wordcount(wordcount_path)

    print("\n[generate_data] All datasets written:")
    for p in [logreg_path, kmeans_path, wordcount_path]:
        size_mb = os.path.getsize(p) / 1024 / 1024
        print(f"  {p}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
