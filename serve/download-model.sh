#!/usr/bin/env bash
# Download the exact target/native-MTP snapshot, with no GPU allocation.
set -euo pipefail
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"
IMAGE="${IMAGE:-local/qwen-flash-next:0.5.20-20260919-hotmap}"
HF_CACHE="${HF_CACHE:-$HOME/models/huggingface}"
mkdir -p "$HF_CACHE"
exec docker run --rm --security-opt no-new-privileges:true \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  --entrypoint hf "$IMAGE" download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594
