#!/usr/bin/env bash

export OMPI_MCA_hwloc_base_binding_policy=none
python scripts/doe_rightsize.py --design-only
python scripts/doe_rightsize.py --pybin "$(which python)" --prepare-only
nohup python scripts/doe_rightsize.py --pybin "$(which python)" > results/doe_rightsize_log.txt 2>&1 &
python scripts/doe_rightsize.py --labels results/doe_rightsize.csv --tol 0.05 0.10
python scripts/doe_holdout.py --design-only
nohup python scripts/doe_holdout.py --pybin "$(which python)" > results/doe_holdout_log.txt 2>&1 &
python scripts/doe_holdout.py --summarize results/doe_holdout.csv --d-csv results/doe_rightsize.csv