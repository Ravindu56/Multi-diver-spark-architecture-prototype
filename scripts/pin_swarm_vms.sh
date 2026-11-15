#!/usr/bin/env bash
# pin_swarm_vms.sh - P5-04: pin the Swarm VMs' vCPUs to dedicated performance cores.
#
# The project host (Intel Core Ultra 7 155H) mixes performance cores (P), efficiency
# cores (E) and low-power cores. Unpinned vCPUs can migrate between core types, which
# adds run-to-run variance and can bias the 2-vs-4 worker comparison. This script pins
# each vCPU to its own P-core (one hyperthread per physical core, so no two vCPUs share
# a core) and pins each VM's QEMU emulator threads to the E-cores, away from the
# measured cores.
#
# Default map for the project host (logical CPU -> physical core, from lscpu -e):
#   P-cores: core0 = 0,5  core1 = 1,2  core2 = 3,4  core3 = 6,7  core4 = 8,9  core5 = 10,11
#   cores 1 and 2 boost to 4.8 GHz, the others to 4.5 GHz.
#   swarm-manager  vcpu0 -> 0  (core0)   vcpu1 -> 10 (core5)
#   swarm-worker1  vcpu0 -> 1  (core1)   vcpu1 -> 6  (core3)
#   swarm-worker2  vcpu0 -> 3  (core2)   vcpu1 -> 8  (core4)
# Each worker VM gets one 4.8 GHz and one 4.5 GHz core, so the workers are symmetric.
#
# Usage:
#   ./scripts/pin_swarm_vms.sh status          # show current pinning (default)
#   ./scripts/pin_swarm_vms.sh plan            # show what apply would run
#   ./scripts/pin_swarm_vms.sh apply           # pin (persistent; live too if running)
#   ./scripts/pin_swarm_vms.sh unpin           # allow every host CPU again
#
# Override the map for another host, e.g.:
#   PIN_MAP="swarm-manager:0,10 swarm-worker1:1,6 swarm-worker2:3,8" EMU_CPUS=12-19 ...
# apply refuses any vCPU target that is not a P-core thread, or two targets on one core.
set -euo pipefail

VIRSH="${VIRSH:-virsh --connect qemu:///system}"
PIN_MAP="${PIN_MAP:-swarm-manager:0,10 swarm-worker1:1,6 swarm-worker2:3,8}"
EMU_CPUS="${EMU_CPUS:-12-19}"
MODE="${1:-status}"

log() { printf '[pin] %s\n' "$*"; }
die() { printf '[pin] ERROR: %s\n' "$*" >&2; exit 1; }

# Expand a cpulist such as "0-3,8" into one number per line.
expand() {
  local part
  for part in ${1//,/ }; do
    if [[ $part == *-* ]]; then seq "${part%-*}" "${part#*-}"; else echo "$part"; fi
  done
}

pcore_cpus() {
  [[ -r /sys/devices/cpu_core/cpus ]] || die "no /sys/devices/cpu_core/cpus: not a hybrid Intel host; set PIN_MAP by hand"
  expand "$(cat /sys/devices/cpu_core/cpus)"
}

core_of() { cat "/sys/devices/system/cpu/cpu$1/topology/core_id"; }

validate() {
  local pcores seen_cores="" entry dom cpus cpu core
  pcores=" $(pcore_cpus | tr '\n' ' ') "
  for entry in $PIN_MAP; do
    dom=${entry%%:*}; cpus=${entry#*:}
    $VIRSH dominfo "$dom" >/dev/null 2>&1 || die "domain not found: $dom"
    local nvcpu; nvcpu=$($VIRSH vcpucount "$dom" --maximum --config)
    [[ $(echo "${cpus//,/ }" | wc -w) -eq $nvcpu ]] || die "$dom has $nvcpu vCPUs but the map lists '$cpus'"
    for cpu in ${cpus//,/ }; do
      [[ $pcores == *" $cpu "* ]] || die "CPU $cpu ($dom) is not a P-core thread"
      core=$(core_of "$cpu")
      [[ " $seen_cores " != *" $core "* ]] || die "CPU $cpu ($dom) shares physical core (core_id $core) with another vCPU"
      seen_cores+=" $core"
    done
  done
  for cpu in $(expand "$EMU_CPUS"); do
    [[ $pcores != *" $cpu "* ]] || die "EMU_CPUS includes P-core thread $cpu"
  done
}

# --config always; --live as well when the domain is running.
flags() {
  if [[ $($VIRSH domstate "$1") == running ]]; then echo "--config --live"; else echo "--config"; fi
}

run() { # dry-run aware
  if [[ $MODE == plan ]]; then echo "  $*"; else "$@" >/dev/null; fi
}

pin_all() {
  local entry dom cpus i cpu
  for entry in $PIN_MAP; do
    dom=${entry%%:*}; cpus=${entry#*:}; i=0
    for cpu in ${cpus//,/ }; do
      # shellcheck disable=SC2046  # flags expands to separate options on purpose
      run $VIRSH vcpupin "$dom" "$i" "$cpu" $(flags "$dom")
      i=$((i + 1))
    done
    # shellcheck disable=SC2046
    run $VIRSH emulatorpin "$dom" "$EMU_CPUS" $(flags "$dom")
    [[ $MODE == plan ]] || log "$dom: vCPUs -> $cpus, emulator -> $EMU_CPUS"
  done
}

unpin_all() {
  local all entry dom i n
  all="0-$(($(nproc --all) - 1))"
  for entry in $PIN_MAP; do
    dom=${entry%%:*}
    n=$($VIRSH vcpucount "$dom" --maximum --config)
    for ((i = 0; i < n; i++)); do
      # shellcheck disable=SC2046
      $VIRSH vcpupin "$dom" "$i" "$all" $(flags "$dom") >/dev/null
    done
    # shellcheck disable=SC2046
    $VIRSH emulatorpin "$dom" "$all" $(flags "$dom") >/dev/null
    log "$dom: unpinned (vCPUs and emulator -> $all)"
  done
}

status() {
  local entry dom
  for entry in $PIN_MAP; do
    dom=${entry%%:*}
    echo "== $dom ($($VIRSH domstate "$dom" 2>/dev/null || echo missing)) config:"
    $VIRSH vcpupin "$dom" --config 2>&1 | sed -n '3,$p' | sed '/^$/d'
    $VIRSH emulatorpin "$dom" --config 2>&1 | sed -n '3,$p' | sed '/^$/d'
  done
}

case $MODE in
  status) status ;;
  plan) validate; log "would run:"; pin_all ;;
  apply) validate; pin_all; status ;;
  unpin) unpin_all; status ;;
  *) die "usage: $0 status|plan|apply|unpin" ;;
esac
