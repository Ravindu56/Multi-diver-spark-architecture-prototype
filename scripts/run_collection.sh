#!/usr/bin/env bash
# scripts/run_collection.sh - headless data collection for the resource predictor
# (Objective 2b) and the sync-mode evaluation with a fair baseline (Objective 2d).
#
# Jobs run STRICTLY ONE AT A TIME: the DOE runs sample per-driver CPU/RSS, so a
# second job on the same machine would contaminate both measurements.
#
# Usage (the exec bit is not required):
#   bash scripts/run_collection.sh pilot                 # tiny run of every stage type
#   bash scripts/run_collection.sh start --archive-old   # detached (setsid nohup), all stages
#   bash scripts/run_collection.sh status                # running? manifest tail, log tails
#   bash scripts/run_collection.sh stop                  # stop it, clean up mpirun/JVMs
#   bash scripts/run_collection.sh run [--dry-run]       # foreground (what start launches)
#
# Stages (default order; override with STAGES="doe_kmeans eval_kmeans ..."):
#   doe_kmeans   predictor training data, K-Means (Set A)    scripts/doe_kmeans.py
#   doe_logreg   predictor training data, LogReg  (Set B)    scripts/doe_logreg.py
#   eval_kmeans  4 sync modes, K-Means, fixed round budget   scripts/sync_eval.py
#   base_kmeans  fair single-driver baseline for eval_kmeans scripts/baseline_eval.py
#   eval_logreg  4 sync modes, LogReg, time to target loss   scripts/sync_eval.py
#   base_logreg  fair single-driver baseline for eval_logreg scripts/baseline_eval.py
#   compare      select tables + scripts/compare_baseline.py
#
# Settings (environment variables, defaults in brackets): WORKERS [2 4], CORES [2],
#   ROUNDS_KM [8], ROUNDS_LR [10], REPS_KM [3], REPS_LR [5], DOE_CONFIGS [40],
#   DOE_REPS [2], INPUT [shared_storage/logreg_100mb.csv], COOLDOWN_S [30],
#   FAIL_FAST [0], FORCE [0], PYBIN [python of the activated venv].
# A finished stage leaves results/collection_logs/<stage>.done and is skipped on the
# next run (FORCE=1 reruns it); every stage script is itself resumable.
# Old pre-optimisation DOE / sync results must be archived first (--archive-old):
# the DOE scripts would otherwise treat the old rows as already done and skip them.
set -u -o pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SELF")/.." && pwd)"
cd "$REPO" || exit 1

: "${STAGES:=doe_kmeans doe_logreg eval_kmeans base_kmeans eval_logreg base_logreg compare}"
: "${WORKERS:=2 4}"
: "${CORES:=2}"
: "${ROUNDS_KM:=8}"
: "${ROUNDS_LR:=10}"
: "${REPS_KM:=3}"
: "${REPS_LR:=5}"
: "${DOE_CONFIGS:=40}"
: "${DOE_REPS:=2}"
: "${INPUT:=shared_storage/logreg_100mb.csv}"
: "${COOLDOWN_S:=30}"
: "${FAIL_FAST:=0}"
: "${FORCE:=0}"
read -r -a WARR <<< "$WORKERS"

LOGDIR=results/collection_logs
MANIFEST=results/collection_manifest.txt
PIDFILE=results/collection.pid
MARKER=results/.collection_v2
mkdir -p results "$LOGDIR"

die() { echo "[collection] ERROR: $*" >&2; exit 2; }
note() { local m; m="[$(date '+%F %T')] $*"; echo "$m"; echo "$m" >> "$MANIFEST"; }
usage() { sed -n '2,/^set -u/p' "$SELF" | sed '$d'; }

CMD="${1:-help}"
[ $# -gt 0 ] && shift
ORIG_ARGS=("$@")
ARCHIVE=0
DRYRUN=0
for arg in "$@"; do
  case "$arg" in
    --archive-old) ARCHIVE=1 ;;
    --dry-run) DRYRUN=1 ;;
    *) die "unknown option: $arg" ;;
  esac
done

setup_python() {
  if [ -z "${PYBIN:-}" ] && [ -f .venv/bin/activate ]; then
    . .venv/bin/activate
  fi
  PY="${PYBIN:-$(command -v python)}"
  if [ -z "$PY" ] || [ ! -x "$PY" ]; then die "no python found (activate the venv or set PYBIN)"; fi
  command -v mpirun >/dev/null 2>&1 || die "mpirun not found on PATH"
}

stage_argv() {
  case "$1" in
    doe_kmeans) STAGE_ARGV=("$PY" scripts/doe_kmeans.py --pybin "$PY" --n-configs "$DOE_CONFIGS" --reps "$DOE_REPS") ;;
    doe_logreg) STAGE_ARGV=("$PY" scripts/doe_logreg.py --pybin "$PY" --n-configs "$DOE_CONFIGS" --reps "$DOE_REPS") ;;
    eval_kmeans) STAGE_ARGV=("$PY" scripts/sync_eval.py run --app kmeans --pybin "$PY" --input "$INPUT" --rounds "$ROUNDS_KM" --workers "${WARR[@]}" --cores "$CORES" --reps "$REPS_KM" --out-dir results/sync_eval_kmeans_v2) ;;
    base_kmeans) STAGE_ARGV=("$PY" scripts/baseline_eval.py --app kmeans --pybin "$PY" --input "$INPUT" --rounds "$ROUNDS_KM" --compare-workers "${WARR[@]}" --cores "$CORES" --reps "$REPS_KM" --out-dir results/baseline_kmeans_v2) ;;
    eval_logreg) STAGE_ARGV=("$PY" scripts/sync_eval.py run --app logreg --pybin "$PY" --input "$INPUT" --rounds "$ROUNDS_LR" --workers "${WARR[@]}" --cores "$CORES" --reps "$REPS_LR" --out-dir results/sync_eval_logreg_v2) ;;
    base_logreg) STAGE_ARGV=("$PY" scripts/baseline_eval.py --app logreg --pybin "$PY" --input "$INPUT" --rounds "$ROUNDS_LR" --compare-workers "${WARR[@]}" --cores "$CORES" --reps "$REPS_LR" --out-dir results/baseline_logreg_v2) ;;
    compare)
      STAGE_ARGV=(bash -c '
        PY=$1; rc=0
        "$PY" scripts/sync_eval.py select --app kmeans --out-dir results/sync_eval_kmeans_v2 || rc=1
        "$PY" scripts/compare_baseline.py --app kmeans --framework-dir results/sync_eval_kmeans_v2 --baseline-dir results/baseline_kmeans_v2 --out results/compare_kmeans_v2.csv || rc=1
        "$PY" scripts/sync_eval.py select --app logreg --out-dir results/sync_eval_logreg_v2 || rc=1
        "$PY" scripts/compare_baseline.py --app logreg --framework-dir results/sync_eval_logreg_v2 --baseline-dir results/baseline_logreg_v2 --out results/compare_logreg_v2.csv || rc=1
        exit $rc' _ "$PY") ;;
    *) return 1 ;;
  esac
}

run_stage() {
  local name="$1" done_f log_f rc
  STAGE_SKIPPED=0
  stage_argv "$name" || { note "unknown stage: $name"; return 2; }
  done_f="$LOGDIR/$name.done"
  log_f="$LOGDIR/$name.log"
  if [ -f "$done_f" ] && [ "$FORCE" != 1 ]; then
    note "skip $name (already done; FORCE=1 to rerun)"
    STAGE_SKIPPED=1
    return 0
  fi
  note "START $name: ${STAGE_ARGV[*]}"
  if [ "$DRYRUN" = 1 ]; then STAGE_SKIPPED=1; return 0; fi
  "${STAGE_ARGV[@]}" > "$log_f" 2>&1
  rc=$?
  note "END   $name exit=$rc (log: $log_f)"
  if [ "$rc" -eq 0 ]; then : > "$done_f"; fi
  return "$rc"
}

preflight() {
  local need_input=0 st stale=() move=() p dest avail
  for st in $STAGES; do
    case "$st" in eval_*|base_*) need_input=1 ;; esac
  done
  if [ "$need_input" = 1 ] && [ ! -f "$INPUT" ]; then die "missing input file: $INPUT"; fi
  avail=$(df -BG --output=avail . 2>/dev/null | tail -n 1 | tr -dc '0-9')
  if [ -n "$avail" ] && [ "$avail" -lt 10 ]; then note "WARNING: only ${avail} GB free on this disk"; fi
  if [ ! -f "$MARKER" ]; then
    for p in results/doe_kmeans.csv results/doe_logreg.csv results/sync_eval; do
      if [ -e "$p" ]; then stale+=("$p"); fi
    done
    if [ "${#stale[@]}" -gt 0 ] && [ "$ARCHIVE" != 1 ]; then
      die "old results present (${stale[*]}). Re-run with --archive-old (moves them to results/pre_opt_*); otherwise the DOE scripts would skip designs already recorded."
    fi
    move=("${stale[@]+"${stale[@]}"}")
    for p in results/logs results/traces; do
      if [ -e "$p" ]; then move+=("$p"); fi
    done
    if [ "${#move[@]}" -gt 0 ]; then
      dest="results/pre_opt_$(date +%Y%m%d_%H%M%S)"
      if [ "$DRYRUN" = 1 ]; then
        note "would archive: ${move[*]} -> $dest"
      else
        mkdir -p "$dest" && mv "${move[@]}" "$dest"/ && note "archived: ${move[*]} -> $dest"
      fi
    fi
    if [ "$DRYRUN" != 1 ]; then : > "$MARKER"; fi
  fi
  return 0
}

cmd_start() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    die "already running (pid $(cat "$PIDFILE"))"
  fi
  setsid nohup bash "$SELF" run ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"} > results/collection_driver.log 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  echo "started pid $(cat "$PIDFILE"). Follow: tail -f results/collection_driver.log  |  bash scripts/run_collection.sh status"
}

cmd_status() {
  local f
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "running (pid $(cat "$PIDFILE"))"
  else
    echo "not running"
  fi
  tail -n 15 "$MANIFEST" 2>/dev/null
  for f in "$LOGDIR"/*.log; do
    [ -e "$f" ] || continue
    echo "--- $f"
    tail -n 2 "$f"
  done
}

cmd_stop() {
  local pid
  if [ -f "$PIDFILE" ]; then
    pid="$(cat "$PIDFILE")"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  fi
  sleep 2
  pkill -u "$USER" -TERM -f "mpj_spark.core.main_mpi" 2>/dev/null
  pkill -u "$USER" -TERM -x mpirun 2>/dev/null
  rm -f "$PIDFILE"
  echo "stopped; finished runs are saved, rerun 'start' to resume"
}

cmd_run() {
  local st failed=()
  if [ "$DRYRUN" != 1 ] && [ -z "${COLLECTION_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1 && systemd-inhibit --what=sleep:idle --why=probe true >/dev/null 2>&1; then
    export COLLECTION_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --why="MPJ-Spark data collection" bash "$SELF" run ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}
  fi
  exec 9> results/.collection.lock
  flock -n 9 || die "another collection is already running"
  setup_python
  preflight
  note "collection start: commit=$(git rev-parse HEAD 2>/dev/null) uncommitted_files=$(git status --short 2>/dev/null | wc -l) dry_run=$DRYRUN"
  note "host: $(nproc) CPUs, $(free -m | awk '/Mem:/ {print $2}') MB RAM; stages: $STAGES"
  note "config: WORKERS='$WORKERS' CORES=$CORES ROUNDS_KM=$ROUNDS_KM ROUNDS_LR=$ROUNDS_LR REPS_KM=$REPS_KM REPS_LR=$REPS_LR DOE_CONFIGS=$DOE_CONFIGS DOE_REPS=$DOE_REPS INPUT=$INPUT"
  for st in $STAGES; do
    if ! run_stage "$st"; then
      failed+=("$st")
      if [ "$FAIL_FAST" = 1 ]; then break; fi
    fi
    if [ "$DRYRUN" != 1 ] && [ "$STAGE_SKIPPED" != 1 ]; then sleep "$COOLDOWN_S"; fi
  done
  note "collection finished; failed stages: ${failed[*]:-none}"
  [ "${#failed[@]}" -eq 0 ]
}

cmd_pilot() {
  local rc=0
  exec 9> results/.collection.lock
  flock -n 9 || die "another collection is already running"
  setup_python
  note "pilot start: commit=$(git rev-parse HEAD 2>/dev/null)"
  "$PY" scripts/doe_kmeans.py --pybin "$PY" --n-configs 3 --reps 1 --out results/pilot_doe_kmeans.csv || rc=1
  "$PY" scripts/doe_logreg.py --pybin "$PY" --n-configs 3 --reps 1 --out results/pilot_doe_logreg.csv || rc=1
  "$PY" scripts/sync_eval.py run --app kmeans --smoke --rounds 4 --workers 2 --settle 2 --pybin "$PY" --input "$INPUT" --out-dir results/pilot_sync_kmeans || rc=1
  "$PY" scripts/baseline_eval.py --app kmeans --smoke --rounds 4 --compare-workers 2 --settle 2 --pybin "$PY" --input "$INPUT" --out-dir results/pilot_baseline_kmeans || rc=1
  note "pilot end rc=$rc: inspect results/pilot_* (check sys_mem_avail_min_mb on the largest DOE rows)"
  return "$rc"
}

case "$CMD" in
  start) cmd_start ;;
  run) cmd_run ;;
  pilot) cmd_pilot ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  help|-h|--help) usage ;;
  *) usage; exit 2 ;;
esac
