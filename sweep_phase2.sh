#!/bin/bash
# Phase 2 nested sweep: dataset size x worker count, all workloads + sync variants
SIZES=(50 100 200 400 1000 1600)      # MB
WORKERS=(2 4 8)
LOGDIR=benchmark_logs
mkdir -p $LOGDIR

for app in wordcount kmeans logreg; do
  for size in "${SIZES[@]}"; do
    for w in "${WORKERS[@]}"; do
      fair_threads=$((w * 5))   # match --baseline-threads to total MPJ threads

      # Default sync (Hungarian for kmeans, Allreduce for logreg, none for wordcount)
      python main.py --app $app --workers $w --generate $size \
          --compare --baseline-threads $fair_threads \
          | tee $LOGDIR/${app}_default_${size}MB_w${w}.log

      # Gossip variant — only valid for kmeans
      if [ "$app" == "kmeans" ]; then
        python main.py --app kmeans --workers $w --generate $size \
            --kmeans-iter 20 --gossip \
            | tee $LOGDIR/kmeans_gossip_${size}MB_w${w}.log
      fi
    done
  done
done

python main.py --log-history | tee $LOGDIR/all_runs_summary.log