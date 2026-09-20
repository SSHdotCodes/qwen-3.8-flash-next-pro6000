#!/usr/bin/env bash
# Validated recommended-sampling profile: full 262K context, two slots,
# original NVFP4/BF16 weights, FP8 E4M3 KV, exact rejection, draft-only hot map.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"
IMAGE="${IMAGE:-local/qwen-flash-next:0.5.20-20260919-hotmap}"
NAME="${NAME:-qwen38-flash-next-sglang}"
PORT="${PORT:-30010}"
HF_CACHE="${HF_CACHE:-$HOME/models/huggingface}"
RUNTIME_CACHE="${RUNTIME_CACHE:-$HOME/.cache/sglang-flash-next/0.5.20-hotmap}"
SNAPSHOT="hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594"
MODEL="/root/.cache/huggingface/$SNAPSHOT"
if ! [[ "$PORT" =~ ^[1-9][0-9]{0,4}$ ]] || (( PORT > 65535 )); then
  echo 'PORT must be an integer from 1 to 65535' >&2; exit 1
fi

command=(docker run --rm --pull never --name "$NAME"
  --gpus all -p "127.0.0.1:$PORT:$PORT" --ipc host --shm-size 32g
  --security-opt no-new-privileges:true --log-opt max-size=50m --log-opt max-file=3
  -e HF_HOME=/root/.cache/huggingface -e TRANSFORMERS_CACHE=/root/.cache/huggingface
  -e TRITON_CACHE_DIR=/root/.cache/triton
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e QWEN_MTP_SHORTLIST=131072 -e QWEN_MTP_HOTMAP=/opt/qwen-mtp-hotmap.json
  -v "$HF_CACHE:/root/.cache/huggingface"
  -v "$RUNTIME_CACHE/triton:/root/.cache/triton"
  -v "$RUNTIME_CACHE/sglang:/root/.cache/sglang"
  -v "$RUNTIME_CACHE/tilelang:/root/.tilelang"
  "$IMAGE" python3 -m sglang.launch_server
  --model-path "$MODEL" --served-model-name qwen3.8-flash-next
  --tp 1 --trust-remote-code --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass
  --context-length 262144 --max-total-tokens 524288 --kv-cache-dtype fp8_e4m3
  --mem-fraction-static 0.98 --page-size 64 --chunked-prefill-size 4096
  --ple-offload-embedding --linear-attn-prefill-backend triton --linear-attn-decode-backend flashinfer
  --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer
  --mamba-track-interval 64 --max-mamba-cache-size 14
  --max-running-requests 2 --cuda-graph-max-bs-decode 2
  --speculative-algorithm NEXTN --speculative-draft-model-path "$MODEL"
  --speculative-use-rejection-sampling --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}'
  --reasoning-parser auto --tool-call-parser auto --sampling-defaults model
  --stream-interval 4 --host 0.0.0.0 --port "$PORT" --enable-response-store)

# Inspect exact argv without Docker, downloads, or filesystem writes.
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
python3 "$HERE/verify-runtime.py"
MANIFEST_SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$HERE/runtime.json")"
IMAGE_SHA="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "local.flash-next.manifest-sha256"}}')"
if [[ "$IMAGE_SHA" != "$MANIFEST_SHA" ]]; then
  echo 'Image does not match this checkout. Run ./serve/build-image.sh first.' >&2; exit 1
fi
if [[ ! -f "$HF_CACHE/$SNAPSHOT/model.safetensors.index.json" ]]; then
  echo 'Pinned model snapshot missing. Run ./serve/download-model.sh first.' >&2; exit 1
fi
mkdir -p "$RUNTIME_CACHE/triton" "$RUNTIME_CACHE/sglang" "$RUNTIME_CACHE/tilelang"
# Container networking is isolated. Only the HTTP port is published, to loopback.
exec "${command[@]}"
