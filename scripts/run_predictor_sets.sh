#!/usr/bin/env bash
# Set D (right-sizing) then Set E (holdout), strictly one job at a time. Resumable.
# bash scripts/run_predictor_sets.sh --archive-old
# detached: setsid nohup bash scripts/run_predictor_sets.sh --archive-old > results/predictor_sets_driver.log 2>&1 < /dev/null &
set -u -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export OMPI_MCA_hwloc_base_binding_policy=none
mkdir -p results
MANIFEST=results/predictor_sets_manifest.txt
MARKER=results/.predictor_sets_v2
note() { echo "[$(date '+%F %T')] $*" | tee -a "$MANIFEST"; }
die() { note "ERROR: $*"; exit 2; }
ARCHIVE=0; [ "${1:-}" = "--archive-old" ] && ARCHIVE=1

exec 9> results/.collection.lock
flock -n 9 || die "another collection or predictor-set run is active"

[ -f .venv/bin/activate ] && . .venv/bin/activate
PY="$(command -v python)"; [ -x "$PY" ] || die "no python found"
command -v mpirun >/dev/null || die "mpirun not found"

seen=$(mpirun --oversubscribe -np 2 "$PY" -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null | grep -E '^[0-9]+$' | sort -n | head -n 1)
[ "${seen:-0}" -ge "$(nproc)" ] || die "MPI binding probe: ranks see ${seen:-?} of $(nproc) CPUs"
note "binding probe ok ($seen of $(nproc)); commit=$(git rev-parse HEAD) uncommitted=$(git status --short | wc -l)"

if [ ! -f "$MARKER" ]; then
  stale=()
  for p in results/doe_rightsize.csv results/rightsize_labels.csv results/doe_holdout.csv; do
    [ -e "$p" ] && stale+=("$p")
  done
  if [ "${#stale[@]}" -gt 0 ]; then
    [ "$ARCHIVE" = 1 ] || die "old results present (${stale[*]}); rerun with --archive-old"
    dest="results/pre_binding_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$dest" && mv "${stale[@]}" "$dest"/ && note "archived ${stale[*]} -> $dest"
  fi
  : > "$MARKER"
fi

step() { local name="$1"; shift; note "START $name"; "$@" >> "results/${name}.log" 2>&1; local rc=$?; note "END   $name rc=$rc"; return $rc; }

step rightsize_design  "$PY" scripts/doe_rightsize.py --design-only || exit 1
step holdout_e1_design "$PY" scripts/doe_holdout.py --sets E1 --design-only || exit 1
step rightsize_prepare "$PY" scripts/doe_rightsize.py --pybin "$PY" --prepare-only || exit 1
step doe_rightsize     "$PY" scripts/doe_rightsize.py --pybin "$PY" || exit 1
step rightsize_labels  "$PY" scripts/doe_rightsize.py --labels results/doe_rightsize.csv --tol 0.05 0.10 || exit 1
step doe_holdout       "$PY" scripts/doe_holdout.py --pybin "$PY" || exit 1
step holdout_summary   "$PY" scripts/doe_holdout.py --summarize results/doe_holdout.csv --d-csv results/doe_rightsize.csv || exit 1
note "predictor sets finished"
