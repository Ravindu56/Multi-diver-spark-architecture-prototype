#!/usr/bin/env bash
# host-mode.sh — toggle the P5-01 KVM host between desktop and research (headless) modes.
#
# Usage:
#   ./scripts/host-mode.sh research   # stop the GUI session, free ~4 GB for VMs
#   ./scripts/host-mode.sh desktop    # bring the GUI back
#   ./scripts/host-mode.sh status     # show current target + memory headroom
#
# Notes:
#   - The toggle is per-session: a reboot always returns to the graphical desktop.
#   - No packages are removed; ubuntu-desktop stays installed.
#   - Sizing rule: total VM RAM must stay <= (available RAM - 2 GB).

set -euo pipefail

usage() {
    echo "usage: $0 {research|desktop|status}" >&2
    exit 1
}

[ $# -eq 1 ] || usage

case "$1" in
    research)
        echo "[host-mode] switching to headless research session (GUI off)..."
        echo "[host-mode] work continues on the TTY -- use tmux; restore with: $0 desktop"
        sudo systemctl isolate multi-user.target
        ;;
    desktop)
        echo "[host-mode] restoring graphical desktop session..."
        sudo systemctl isolate graphical.target
        ;;
    status)
        echo "default target : $(systemctl get-default)"
        if systemctl is-active --quiet graphical.target; then
            echo "active session : graphical"
        else
            echo "active session : multi-user (headless)"
        fi
        free -g | awk '/Mem/{printf "memory         : %s GB total, %s GB available\n", $2, $7}'
        if [ -r /sys/kernel/mm/ksm/run ]; then
            echo "ksm            : $(cat /sys/kernel/mm/ksm/run) (1=enabled)"
        fi
        ;;
    *)
        usage
        ;;
esac
