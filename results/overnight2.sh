#!/usr/bin/env bash
cd ~/dev-env/Research
source .venv/bin/activate
PY="$(which python)"
RB="python scripts/resource_benchmark.py --pybin $PY"

# 1. Is the extra logreg CPU from BLAS threads? (cap them at 2, run with 2 cores)
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $RB \
  --apps logreg --sizes 50 --np 3 --cores 2 --heap-mb 3072 --reps 2 \
  --out results/blas_test.csv || echo "step 1 failed"

# 2. Heap floor, logreg (heaps descending; a failure ends this step only)
$RB --apps logreg --sizes 50 --np 3 --cores 4 \
  --heap-mb 2048 1536 1024 768 --reps 2 \
  --out results/floor_logreg50.csv || echo "step 2 ended"

# 3. Heap floor, kmeans
$RB --apps kmeans --sizes 100 --np 3 --cores 2 \
  --heap-mb 1024 768 512 --reps 2 \
  --out results/floor_kmeans100.csv || echo "step 3 ended"

# 4. Size dependence, logreg
$RB --apps logreg --sizes 100 200 --np 3 --cores 4 6 11 \
  --heap-mb 4096 3072 --reps 2 \
  --out results/grid_logreg_big.csv || echo "step 4 ended"

echo "QUEUE DONE $(date)"
