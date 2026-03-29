import os
import random


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
