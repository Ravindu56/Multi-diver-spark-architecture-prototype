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
# UNVERIFIED: the LogReg entry point (LR_MODULE) - confirm it has a CLI with --help first.
set -u -o pipefail
CELL=${1:?usage: p5_run_cell.sh A|B}
case $CELL in A) NP=3;; B) NP=5;; *) exit 2;; esac
SSHU=${SSHU:-ubuntu}; MGR=${MGR:-192.168.122.224}
VMS=${VMS:-"192.168.122.224 192.168.122.27 192.168.122.7"}
CIDR=${CIDR:-10.0.1.0/24}
REPS=${REPS:-4}; TMO=${TMO:-5400}; COOL=${COOL:-60}
KM_SIZES=${KM_SIZES:-"100 500 1024"}
LR_FEATS=${LR_FEATS:-"10 50 200"}; LR_SIZE=${LR_SIZE:-100}
LR_MODULE=${LR_MODULE:-mpj_spark.applications.logreg.allreduce}
OUT=$HOME/p5-runs/$(date +%Y%m%d-%H%M%S)-cell$CELL; mkdir -p "$OUT"
LOG=$OUT/runlog.csv
echo "name,cell,np,workload,param,rep,rc,wall_s,start" > "$LOG"
echo "commit=$(git rev-parse HEAD 2>/dev/null) cell=$CELL np=$NP" > "$OUT/meta.txt"
ROOT='docker exec $(docker ps -q -f name=mpj_mpi-root | head -1)'

start_samplers() {
  vmstat -t 1 > "$1/vmstat-host.log" 2>&1 & HOSTPID=$!
  for ip in $VMS; do
    ssh -n $SSHU@$ip "timeout $TMO vmstat -t 1" > "$1/vmstat-$ip.log" 2>&1 &
  done
}
stop_samplers() {
  kill $HOSTPID 2>/dev/null
  for ip in $VMS; do ssh -n $SSHU@$ip 'pkill -u $USER -x vmstat' 2>/dev/null; done
  wait 2>/dev/null
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
  start_samplers "$d"
  local t0; t0=$(date +%s)
  timeout $TMO ssh -n $SSHU@$MGR "$ROOT bash -lc 'bash /data/run_cmd/$name.sh'" > "$d/run.log" 2>&1
  local rc=$? t1; t1=$(date +%s)
  [ $rc -ne 0 ] && ssh -n $SSHU@$MGR "$ROOT pkill -f mpirun" 2>/dev/null
  stop_samplers
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
