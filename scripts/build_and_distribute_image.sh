#!/usr/bin/env bash
# =============================================================================
# P5-03: build the mpj-spark image on the host (if needed) and distribute it
# to every Swarm VM's docker daemon. No registry: docker save | ssh | docker
# load, idempotent via remote image-ID comparison.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-mpj-spark:latest}"
SSH_USER="${SSH_USER:-ubuntu}"
NODE_IPS="${NODE_IPS:-192.168.122.224 192.168.122.27 192.168.122.7}"

echo "[dist] image: $IMAGE"
echo "[dist] nodes: $NODE_IPS"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[dist] $IMAGE not found locally - building from docker/Dockerfile"
  docker build -t "$IMAGE" -f docker/Dockerfile .
fi
HOST_ID="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
echo "[dist] host image id: $HOST_ID"

for ip in $NODE_IPS; do
  remote_id="$(ssh -o ConnectTimeout=10 "$SSH_USER@$ip" \
    "docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null" || true)"
  if [[ "$remote_id" == "$HOST_ID" ]]; then
    echo "[dist] $ip: up to date - skipping"
    continue
  fi
  echo "[dist] $ip: transferring (remote id: ${remote_id:-none}) - this takes a few minutes"
  docker save "$IMAGE" | ssh "$SSH_USER@$ip" docker load
  new_id="$(ssh "$SSH_USER@$ip" "docker image inspect -f '{{.Id}}' $IMAGE")"
  [[ "$new_id" == "$HOST_ID" ]] || { echo "[dist] ERROR: $ip image id mismatch after load"; exit 1; }
  echo "[dist] $ip: loaded OK"
done

echo "[dist] image distribution complete"
