#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"
IMAGE="${IMAGE:-local/qwen-flash-next:0.5.21-20261020-kern2}"
python3 "$HERE/verify-runtime.py"
MANIFEST_SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$HERE/runtime.json")"
docker build --build-arg "MAX_JOBS=${MAX_JOBS:-8}" --build-arg "RUNTIME_MANIFEST_SHA256=$MANIFEST_SHA" -t "$IMAGE" "$HERE"
docker run --rm --network none --entrypoint python3 "$IMAGE" /opt/flash-next/verify-runtime.py --installed
