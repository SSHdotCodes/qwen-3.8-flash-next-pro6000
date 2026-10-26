#!/usr/bin/env bash
# Run explicitly on an idle GPU after building the image and downloading the pinned model.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"
IMAGE="${IMAGE:-local/qwen-flash-next:0.5.21-20261020-kern2}"
HF_CACHE="${HF_CACHE:-$HOME/models/huggingface}"
MODEL_VIEWS="${MODEL_VIEWS:-$HOME/models/qwen38-flash-next-redhat}"
VIEW="$MODEL_VIEWS/c8f2fb1b9869f686b214782036123b10ff96d14a"
DRAFT_A16="${DRAFT_A16:-$VIEW/draft-a16.pt}"
if [[ ! -f "$VIEW/draft/config.json" ]]; then
  echo 'Model views missing. Run ./serve/download-model.sh first.' >&2; exit 1
fi
if [[ -e "$DRAFT_A16" || -e "$DRAFT_A16.json" ]]; then
  echo 'Draft A16 output already exists. Set DRAFT_A16 to a new output path.' >&2; exit 1
fi
python3 "$HERE/verify-runtime.py"
MANIFEST_SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$HERE/runtime.json")"
IMAGE_SHA="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "local.flash-next.manifest-sha256"}}')"
if [[ "$IMAGE_SHA" != "$MANIFEST_SHA" ]]; then
  echo 'Image does not match this checkout. Run ./serve/build-image.sh first.' >&2; exit 1
fi
mkdir -p "$(dirname "$DRAFT_A16")"
docker run --rm --gpus all --network none --security-opt no-new-privileges:true \
  -v "$HF_CACHE:$HF_CACHE:ro" -v "$VIEW/draft:/models/draft:ro" \
  -v "$(dirname "$DRAFT_A16"):/output" \
  --entrypoint python3 "$IMAGE" /opt/flash-next/prepare-draft-a16.py \
  --output "/output/$(basename "$DRAFT_A16")"
