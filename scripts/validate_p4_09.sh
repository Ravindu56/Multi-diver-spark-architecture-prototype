#!/usr/bin/env bash
# P4-09 acceptance validation: Docker launcher and documentation checks.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."\ && pwd)"
LAUNCHER="${ROOT_DIR}/scripts/run_docker.sh"
COMPOSE_FILE="${ROOT_DIR}/docker/docker-compose.yml"
WORKFLOW="${ROOT_DIR}/.github/workflows/p4-docker-validation.yml"
README="${ROOT_DIR}/README.md"

PASS=0
FAIL=0

check() {
    local description="$1"
    shift

    if "$@"; then
        printf 'PASS: %s\n' "${description}"
        PASS=$((PASS + 1))
    else
        printf 'FAIL: %s\n' "${description}" >&2
        FAIL=$((FAIL + 1))
    fi
}

echo "=============================================================="
echo "P4-09 Docker Deployment Acceptance Validation"
echo "=============================================================="

check "Docker launcher exists" test -f "${LAUNCHER}"
check "Docker launcher is executable" test -x "${LAUNCHER}"
check "Docker launcher has valid Bash syntax" bash -n "${LAUNCHER}"
check "Docker Compose file exists" test -f "${COMPOSE_FILE}"
check "Docker Compose configuration renders" \
    docker compose -f "${COMPOSE_FILE}" config --quiet
check "P4 Docker CI workflow exists" test -f "${WORKFLOW}"
check "README exists" test -f "${README}"
check "README documents Docker usage" grep -qi "docker" "${README}"
check "README references run_docker.sh" grep -q "run_docker.sh" "${README}"

echo
echo "P4-09 Results: PASS=${PASS}, FAIL=${FAIL}"

if (( FAIL > 0 )); then
    echo "P4-09 ACCEPTANCE CRITERIA NOT MET"
    exit 1
fi

echo "P4-09 ACCEPTANCE CRITERIA MET"
