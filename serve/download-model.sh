#!/usr/bin/env bash
# Download the exact RedHatAI NVFP4 snapshot (target + native MTP), with no GPU allocation,
# then build the target/draft views the server loads.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"
IMAGE="${IMAGE:-local/qwen-flash-next:0.5.21-20261020-kern2}"
HF_CACHE="${HF_CACHE:-$HOME/models/huggingface}"
mkdir -p "$HF_CACHE"
docker run --rm --security-opt no-new-privileges:true \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  --entrypoint hf "$IMAGE" download RedHatAI/Qwen3.8-Flash-Next-NVFP4 \
  --revision c8f2fb1b9869f686b214782036123b10ff96d14a
HF_CACHE="$HF_CACHE" python3 "$HERE/prepare-model.py"
