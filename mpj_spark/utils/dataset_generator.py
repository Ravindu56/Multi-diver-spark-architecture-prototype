import os
import random
import math


def generate_text_dataset(output_path: str, size_mb: int):
    words = ["spark","hadoop","cluster","data","parallel","worker",
             "driver","partition","memory","compute","node","core",
             "task","job","stage","rdd","dataframe","shuffle","agg","pipeline"]
    target = size_mb * 1024 * 1024
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"[DatasetGen] Generating text → {output_path} (~{size_mb} MB)")
    written = 0
    with open(output_path, "w") as f:
        while written < target:
            line = " ".join(random.choices(words, k=20)) + "\n"
            f.write(line)
            written += len(line.encode())
    print(f"[DatasetGen] Written {os.path.getsize(output_path)/(1024*1024):.1f} MB")


def generate_numeric_dataset(output_path: str, size_mb: int, num_features: int = 10):
    target = size_mb * 1024 * 1024
    row_template = ",".join(["{:.6f}"] * num_features) + "\n"
    CENTRES = [[2.0]*num_features, [8.0]*num_features, [15.0]*num_features]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"[DatasetGen] Generating numeric → {output_path} (~{size_mb} MB, {num_features} features, 3 clusters)")
    written = 0
    with open(output_path, "w") as f:
        while written < target:
            centre = random.choice(CENTRES)
            row = row_template.format(*[c + random.gauss(0, 1.5) for c in centre])
            f.write(row)
            written += len(row.encode())
    print(f"[DatasetGen] Written {os.path.getsize(output_path)/(1024*1024):.1f} MB")


def generate_classification_dataset(
    output_path: str,
    size_mb: int,
    num_features: int = 10,
    noise: float = 0.5,
    seed: int = 42,
):
    """
    Generate a linearly separable binary classification CSV with a header row.

    CSV format:
        f0,f1,...,f{num_features-1},label
    where label ∈ {0, 1}.

    Strategy
    --------
    Two Gaussian clusters separated along the first feature axis by a margin
    of `2 * num_features` units, ensuring linear separability with default
    noise=0.5. Each row is assigned label=0 (cluster A, centred at [-margin,...,0])
    or label=1 (cluster B, centred at [+margin,...,0]). The remaining features
    are drawn from N(0, noise) to add realistic correlation structure.

    Parameters
    ----------
    output_path  : str   — absolute path for output CSV
    size_mb      : int   — approximate file size in MB
    num_features : int   — number of numeric feature columns (default 10)
    noise        : float — Gaussian noise std for feature dimensions (default 0.5)
    seed         : int   — random seed for reproducibility (default 42)
    """
    random.seed(seed)
    target   = size_mb * 1024 * 1024
    margin   = max(2.0, num_features * 0.3)   # linear separability guarantee

    # Class centres: class 0 at -margin on f0, class 1 at +margin on f0
    centres = [
        [-margin] + [0.0] * (num_features - 1),   # label 0
        [+margin] + [0.0] * (num_features - 1),   # label 1
    ]

    header       = ",".join([f"f{i}" for i in range(num_features)] + ["label"]) + "\n"
    row_template = ",".join(["{:.6f}"] * num_features) + ",{}\n"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(
        f"[DatasetGen] Generating classification → {output_path} "
        f"(~{size_mb} MB, {num_features} features, binary label, margin={margin:.2f})"
    )

    written = 0
    with open(output_path, "w") as f:
        f.write(header)
        written += len(header.encode())
        while written < target:
            label  = random.randint(0, 1)
            centre = centres[label]
            feats  = [centre[d] + random.gauss(0, noise) for d in range(num_features)]
            row    = row_template.format(*feats, label)
            f.write(row)
            written += len(row.encode())

    actual_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[DatasetGen] Written {actual_mb:.1f} MB  (label balance ~50/50)")
