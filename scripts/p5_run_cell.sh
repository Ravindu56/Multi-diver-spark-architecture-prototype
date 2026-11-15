#!/usr/bin/env bash
# p5_run_cell.sh - headless runner for P5-04 (#29) K-Means and P5-05 (#30) LogReg
# on the Swarm stack. Runs ONE cell (A = 3 ranks, B = 5 ranks); deploy the cell first:
#   ./scripts/bootstrap_stack.sh        # Cell A
#   ./scripts/bootstrap_stack.sh 4w     # Cell B
# Usage:
#   setsid nohup systemd-inhibit --what=sleep:idle --why="P5 runs" \
#     ./scripts/p5_run_cell.sh A > cellA.driver.log 2>&1 < /dev/null &
# rep 0 of each configuration is the PILOT (not a measured sample); reps 1..N-1 are measured.
# A failed pilot marks the configuration infeasible and skips its remaining reps.
# No automatic retries: failed runs keep their logs and are excluded from timing.
# Per run, for the host and each VM: vmstat-*.log and psi-*.log (1 s samples),
# mem-before.txt / mem-after.txt (free -m, nproc), stack-before/after.txt (task states).
# LogReg runs through mpj_spark.applications.logreg.cli (added for #30); hand-run it once
# in Cell A before leaving the runner unattended.
# Defaults: KM_SIZES = 50 100 200 500 1024 (Cell A) / 50 100 200 (Cell B), LR_FEATS = 10 50 200.
# Empty KM_SIZES= or LR_FEATS= skips that workload. DRY_RUN=1 prints the plan only.
# The plan (sizes, features, reps) is printed and recorded in meta.txt before the first run.
set -u -o pipefail
CELL=${1:?usage: p5_run_cell.sh A|B}
case $CELL in A) NP=3;; B) NP=5;; *) exit 2;; esac
SSHU=${SSHU:-ubuntu}; MGR=${MGR:-192.168.122.224}
VMS=${VMS:-"192.168.122.224 192.168.122.27 192.168.122.7"}
CIDR=${CIDR:-10.0.1.0/24}
REPS=${REPS:-4}; TMO=${TMO:-5400}; COOL=${COOL:-60}
# Default K-Means sizes (MB) per cell follow the 2026-11-15 scope revision in
# docs/p5-04-experimental-design.md: at 3072 MB per VM Cell B is infeasible at 500 MB and
# 1 GB (documented exclusion), so its grid is 50/100/200 MB; Cell A runs all five sizes.
case $CELL in A) KM_DEFAULT="50 100 200 500 1024";; B) KM_DEFAULT="50 100 200";; esac
KM_SIZES=${KM_SIZES-$KM_DEFAULT}
LR_FEATS=${LR_FEATS-"10 50 200"}; LR_SIZE=${LR_SIZE:-100}
LR_MODULE=${LR_MODULE:-mpj_spark.applications.logreg.cli}
# Guard: Cell B K-Means above 200 MB is a recorded exclusion, not part of the grid.
# Set ALLOW_EXCLUDED=1 to run it anyway (for example a deliberate re-pilot).
if [ "$CELL" = B ] && [ "${ALLOW_EXCLUDED:-0}" != 1 ]; then
  for s in $KM_SIZES; do
    [ "$s" -gt 200 ] && { echo "refusing Cell B K-Means ${s} MB: excluded by the 2026-11-15 scope revision (set ALLOW_EXCLUDED=1 to override)" >&2; exit 2; }
  done
fi
# DRY_RUN=1: print the plan and exit without touching the cluster.
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "cell=$CELL np=$NP reps=$REPS tmo=${TMO}s cool=${COOL}s"
  echo "kmeans sizes (MB): ${KM_SIZES:-<none>}"
  echo "logreg features:   ${LR_FEATS:-<none>} (size ${LR_SIZE} MB)"
  exit 0
fi
OUT=$HOME/p5-runs/$(date +%Y%m%d-%H%M%S)-cell$CELL; mkdir -p "$OUT"
LOG=$OUT/runlog.csv
echo "name,cell,np,workload,param,rep,rc,wall_s,start" > "$LOG"
echo "commit=$(git rev-parse HEAD 2>/dev/null) cell=$CELL np=$NP" > "$OUT/meta.txt"
echo "plan: kmeans_mb=\"$KM_SIZES\" logreg_feats=\"$LR_FEATS\" logreg_mb=$LR_SIZE reps=$REPS tmo=$TMO cool=$COOL" \
  | tee -a "$OUT/meta.txt"
# VM CPU pinning in force for this cell (see scripts/pin_swarm_vms.sh).
{ echo "== vcpu pinning"; "$(dirname "$0")/pin_swarm_vms.sh" status 2>&1; } >> "$OUT/meta.txt" || true
ROOT='docker exec $(docker ps -q -f name=mpj_mpi-root | head -1)'

# Linux pressure stall information (PSI), sampled once per second. Each line is
# "<epoch_s> <cpu|memory|io> <some|full> avg10=.. avg60=.. avg300=.. total=<us>".
# total= is a cumulative stall counter in microseconds: the difference between the
# first and last sample of a run gives the exact stall time for that run.
# The trailing p5-psi-sampler argument names the process so stop_samplers can find it.
PSI_LOOP='if [ -r /proc/pressure/memory ]; then while :; do t=$(date +%s); for r in cpu memory io; do sed "s/^/$t $r /" /proc/pressure/$r; done; sleep 1; done; else echo "PSI unavailable: /proc/pressure not readable"; fi'

start_samplers() {
  vmstat -t 1 > "$1/vmstat-host.log" 2>&1 & HOSTPID=$!
  bash -c "$PSI_LOOP" p5-psi-sampler > "$1/psi-host.log" 2>&1 & HOSTPSI=$!
  for ip in $VMS; do
    ssh -n $SSHU@$ip "timeout $TMO vmstat -t 1" > "$1/vmstat-$ip.log" 2>&1 &
    ssh -n $SSHU@$ip "timeout $TMO bash -c '$PSI_LOOP' p5-psi-sampler" > "$1/psi-$ip.log" 2>&1 &
  done
}
stop_samplers() {
  kill $HOSTPID $HOSTPSI 2>/dev/null
  # [r] keeps the pattern from matching the remote shell that runs this pkill.
  for ip in $VMS; do
    ssh -n $SSHU@$ip 'pkill -u $USER -x vmstat; pkill -u $USER -f "p5-psi-sample[r]"' 2>/dev/null
  done
  wait 2>/dev/null
}

mem_snapshot() { # file: free -m (plus vCPU count) for the host and every VM
  { echo "== host $(date -Is) nproc=$(nproc)"; free -m
    for ip in $VMS; do
      echo "== $ip"; ssh -n $SSHU@$ip 'echo "$(date -Is) nproc=$(nproc)"; free -m' 2>&1
    done
  } > "$1"
}

run_one() { # name workload param rep module args...
  local name=$1 wl=$2 param=$3 rep=$4 module=$5; shift 5
  local d=$OUT/$name; mkdir -p "$d"
  ssh -n $SSHU@$MGR "docker stack ps mpj --format '{{.Name}} {{.Node}} {{.CurrentState}}'" > "$d/stack-before.txt"
  ssh -n $SSHU@$MGR "$ROOT printenv SPARK_DRIVER_MEMORY SLOTS_OVERRIDE" > "$d/env.txt" 2>&1
cat <<EOF | ssh $SSHU@$MGR "mkdir -p /srv/mpj-share/run_cmd /srv/mpj-share/results/p5-04 && cat > /srv/mpj-share/run_cmd/$name.sh"
mpirun -n $NP --hostfile /etc/mpi/hostfile \\
  -x PYTHONPATH -x PYTHONUNBUFFERED -x SPARK_DRIVER_MEMORY -x SLOTS_OVERRIDE \\
  -mca pml ob1 -mca btl tcp,self \\
  -mca oob_tcp_if_include $CIDR -mca btl_tcp_if_include $CIDR \\
  -mca plm_rsh_args "-p 22" \\
  python -m $module $* --output /data/results/p5-04/$name
EOF
  mem_snapshot "$d/mem-before.txt"
  start_samplers "$d"
  local t0; t0=$(date +%s)
  timeout $TMO ssh -n $SSHU@$MGR "$ROOT bash -lc 'bash /data/run_cmd/$name.sh'" > "$d/run.log" 2>&1
  local rc=$? t1; t1=$(date +%s)
  [ $rc -ne 0 ] && ssh -n $SSHU@$MGR "$ROOT pkill -f mpirun" 2>/dev/null
  stop_samplers
  mem_snapshot "$d/mem-after.txt"
  ssh -n $SSHU@$MGR "docker stack ps mpj --format '{{.Name}} {{.Node}} {{.CurrentState}}'" > "$d/stack-after.txt"
  scp -q -r $SSHU@$MGR:/srv/mpj-share/results/p5-04/$name "$d/" 2>/dev/null
  echo "$name,$CELL,$NP,$wl,$param,$rep,$rc,$((t1-t0)),$(date -d @$t0 -Is)" >> "$LOG"
  sleep $COOL
  return $rc
}

for size in $KM_SIZES; do
  for rep in $(seq 0 $((REPS-1))); do
    run_one "km-$CELL-${size}mb-r$rep" kmeans ${size}mb $rep mpj_spark.applications.kmeans.allreduce \
      --input /data/input/kmeans_data_${size}mb.csv --k 5 --max-iter 20 \
      || { [ $rep -eq 0 ] && echo "INFEASIBLE km $CELL ${size}mb" >> "$OUT/infeasible.txt" && break; }
  done
done
for f in $LR_FEATS; do
  for rep in $(seq 0 $((REPS-1))); do
    run_one "lr-$CELL-f${f}-r$rep" logreg f$f $rep $LR_MODULE \
      --input /data/input/logreg_f${f}_${LR_SIZE}mb.csv --epochs 30 --lr 0.01 \
      || { [ $rep -eq 0 ] && echo "INFEASIBLE lr $CELL f$f" >> "$OUT/infeasible.txt" && break; }
  done
done
echo "cell $CELL finished" >> "$OUT/DONE"
