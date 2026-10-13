#!/usr/bin/env python3
"""scripts/baseline_eval.py -- the FAIR single-driver baseline for the sync evaluation.

What makes it fair (Objective 2d, baseline i): it is the SAME framework code path,
algorithm and vectorised kernels as the multi-driver runs, executed with ONE driver
(np=2: the root plus one worker, so there is nothing to synchronise) and the SAME
resources in total:

    W drivers x c cores   ->   1 driver x (W*c) cores    (static allocation)

Same dataset, same --rounds (hence the same number of local epochs over the data).
Only the execution architecture differs.  The stock MLlib baselines behind
`main_mpi --compare` use a different algorithm and objective (k-means|| and
standardised L-BFGS), so they are a useful reference but not a like-for-like
comparison.

This wrapper drives scripts/sync_eval.py with only ps_sync_fedavg_mpi (with a single
worker the four modes are equivalent) once per compared worker count.  Results land
in --out-dir in sync_eval's own CSV layout, with workers=1 and cores=W*c.  Compare
with scripts/compare_baseline.py.

  python scripts/baseline_eval.py --app kmeans --compare-workers 2 4 --cores 2 --dry-run
  python scripts/baseline_eval.py --app kmeans --compare-workers 2 4 --cores 2 --rounds 8 --reps 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sync_eval as se  # noqa: E402

BASELINE_MODE = "ps_sync_fedavg_mpi"


def plan(compare_workers, cores_per_driver):
    """[(workers, baseline_cores)]: one driver with the same total cores."""
    return [(w, cores_per_driver * w) for w in compare_workers if w > 1]


def default_out_dir(app):
    return f"results/baseline_{app}_v2"


def build_argv(a, cores_total, out_dir):
    argv = ["sync_eval.py", "run", "--app", a.app, "--workers", "1", "--cores", str(cores_total),
            "--reps", str(a.reps), "--seed", str(a.seed), "--rounds", str(a.rounds),
            "--features", str(a.features), "--input", a.input, "--pybin", a.pybin,
            "--mpirun", a.mpirun, "--timeout", str(a.timeout), "--settle", str(a.settle),
            "--out-dir", out_dir]
    if a.smoke:
        argv.append("--smoke")
    return argv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", default="kmeans", choices=["kmeans", "logreg"])
    ap.add_argument("--compare-workers", nargs="+", type=int, default=[2, 4],
                    help="worker counts of the multi-driver runs this baseline is compared with")
    ap.add_argument("--cores", type=int, default=2,
                    help="cores per driver in those multi-driver runs")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--features", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--input", default="shared_storage/logreg_100mb.csv")
    ap.add_argument("--pybin", default=sys.executable)
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true", help="1 rep per baseline, to check wiring")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    a = ap.parse_args(argv)

    out_dir = a.out_dir or default_out_dir(a.app)
    jobs = plan(a.compare_workers, a.cores)
    if not jobs:
        sys.exit("nothing to do: --compare-workers needs at least one value > 1")
    ncpu = os.cpu_count() or 1
    for w, c in jobs:
        if c > ncpu:
            print(f"[!] baseline for {w} workers wants {c} threads; this host has {ncpu} CPUs")

    saved_argv, saved_modes = sys.argv, list(se.MODES)
    try:
        if not a.dry_run:
            se.MODES[:] = [BASELINE_MODE]
        for w, c in jobs:
            argv_se = build_argv(a, c, out_dir)
            print(f"[baseline] {w} workers x {a.cores} cores -> 1 driver x {c} cores: "
                  + " ".join(argv_se[1:]), flush=True)
            if a.dry_run:
                continue
            sys.argv = argv_se
            se.main()
    finally:
        sys.argv = saved_argv
        se.MODES[:] = saved_modes


if __name__ == "__main__":
    main()
