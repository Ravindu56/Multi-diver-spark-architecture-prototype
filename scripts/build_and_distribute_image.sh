#!/usr/bin/env bash
# =============================================================================
# P5: build the mpj-spark image from the CURRENT checkout, tag it by git
# version, smoke-test it, distribute it to every Swarm VM, and verify it there.
# No registry: docker save | gzip | ssh | docker load, one transfer per node in
# parallel, skipped when the node already holds the same tag with the same ID.
#
# Why it always builds: the previous version only built when mpj-spark:latest
# did not exist, so code changes never reached the VMs (see #29/#30 notes).
#
# Tag = `git describe --tags --always` (e.g. 61e9d3a or p5-campaign-1-3-g61e9d3a).
# A dirty tree is refused unless ALLOW_DIRTY=1 (then the tag gets a -dirty-HHMMSS
# suffix so it can never be mistaken for a committed build).
#
# Outputs (git-ignored):
#   scripts/.image-tag       the version tag, read by bootstrap_stack.sh as IMAGE_TAG
#   scripts/.image-manifest  tag, image ID, commit, time, nodes
#
# Usage:
#   ./scripts/build_and_distribute_image.sh
#   ALLOW_DIRTY=1 NO_CACHE=1 MOVE_LATEST=0 ./scripts/build_and_distribute_image.sh
#   NODE_IPS="ip1 ip2 ip3" SSH_USER=ubuntu ./scripts/build_and_distribute_image.sh
#
# Then deploy: ./scripts/bootstrap_stack.sh   (or: ./scripts/bootstrap_stack.sh 4w)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${NAME:-mpj-spark}"
SSH_USER="${SSH_USER:-ubuntu}"
NODE_IPS="${NODE_IPS:-192.168.122.224 192.168.122.27 192.168.122.7}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
NO_CACHE="${NO_CACHE:-0}"
MOVE_LATEST="${MOVE_LATEST:-1}"

die() { echo "[dist] ERROR: $*" >&2; exit 1; }

COMMIT="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  [ "$ALLOW_DIRTY" = 1 ] || die "uncommitted changes: commit first (or ALLOW_DIRTY=1 for a throwaway build)"
  VERSION="$(git describe --tags --always)-dirty-$(date +%H%M%S)"
else
  VERSION="$(git describe --tags --always)"
fi
IMAGE="$NAME:$VERSION"
echo "[dist] version=$VERSION commit=$COMMIT"
echo "[dist] nodes: $NODE_IPS"

# 1. always build (the layer cache keeps unchanged layers cheap)
CACHE=()
[ "$NO_CACHE" = 1 ] && CACHE=(--no-cache)
docker build "${CACHE[@]}" -t "$IMAGE" \
  --build-arg GIT_COMMIT="$COMMIT" --build-arg BUILD_VERSION="$VERSION" \
  -f docker/Dockerfile .

SAVE_TAGS=("$IMAGE")
if [ "$MOVE_LATEST" = 1 ]; then
  docker tag "$IMAGE" "$NAME:latest"
  SAVE_TAGS+=("$NAME:latest")
fi

# 2. the image must carry this commit, and the code must import inside it
LABEL="$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMAGE")"
[ "$LABEL" = "$COMMIT" ] || die "image revision label '$LABEL' != checkout $COMMIT (is the Dockerfile ARG/LABEL block present?)"
docker run --rm --entrypoint python "$IMAGE" -c \
  "import mpj_spark.applications.kmeans.allreduce, mpj_spark.applications.logreg.cli" \
  || die "smoke import failed inside $IMAGE"
HOST_ID="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
echo "[dist] built $IMAGE id=$HOST_ID"

# 3. distribute in parallel; skip nodes that already hold this exact tag and ID
distribute_one() {
  local ip="$1" rid
  rid="$(ssh -o ConnectTimeout=10 "$SSH_USER@$ip" "docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null" || true)"
  if [ "$rid" = "$HOST_ID" ]; then
    echo "[dist] $ip: up to date"
    return 0
  fi
  echo "[dist] $ip: transferring (remote id: ${rid:-none})"
  docker save "${SAVE_TAGS[@]}" | gzip -1 | ssh "$SSH_USER@$ip" 'gunzip | docker load' >/dev/null
  rid="$(ssh "$SSH_USER@$ip" "docker image inspect -f '{{.Id}}' $IMAGE")"
  [ "$rid" = "$HOST_ID" ] || { echo "[dist] $ip: ID mismatch after load ($rid)"; return 1; }
  echo "[dist] $ip: loaded OK"
}
pids=()
for ip in $NODE_IPS; do
  distribute_one "$ip" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || die "distribution failed on at least one node"

# 4. final per-node verification, then record the result
for ip in $NODE_IPS; do
  got="$(ssh -n "$SSH_USER@$ip" "docker image inspect -f '{{.Id}} {{index .Config.Labels \"org.opencontainers.image.revision\"}}' $IMAGE")"
  echo "[verify] $ip $got"
  [ "$got" = "$HOST_ID $COMMIT" ] || die "$ip does not hold $IMAGE with revision $COMMIT"
done

echo "$VERSION" > scripts/.image-tag
printf 'tag=%s\nid=%s\ncommit=%s\nbuilt=%s\nnodes=%s\n' \
  "$IMAGE" "$HOST_ID" "$COMMIT" "$(date -Is)" "$NODE_IPS" > scripts/.image-manifest
echo "[dist] done: $IMAGE on all nodes. Deploy with ./scripts/bootstrap_stack.sh (IMAGE_TAG=$VERSION)"
