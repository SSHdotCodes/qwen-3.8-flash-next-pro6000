#!/usr/bin/env bash
# Validated recommended-sampling profile: full 262K context, two slots, RedHatAI NVFP4 routed
# experts run as W4A16 (Marlin for prefill, a persistent decode kernel for 1-8 tokens), BF16 dense
# weights stored exactly or as group-scaled INT8, FP8 E4M3 KV, exact rejection sampling,
# draft hot map, screened target head and GDN replay.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock}"
IMAGE="${IMAGE:-local/qwen-flash-next:0.5.21-20261020-kern2}"
NAME="${NAME:-qwen38-flash-next-sglang}"
PORT="${PORT:-30010}"
HF_CACHE="${HF_CACHE:-$HOME/models/huggingface}"
RUNTIME_CACHE="${RUNTIME_CACHE:-$HOME/.cache/sglang-flash-next/0.5.21}"
MODEL_VIEWS="${MODEL_VIEWS:-$HOME/models/qwen38-flash-next-redhat}"
REVISION="c8f2fb1b9869f686b214782036123b10ff96d14a"
VIEW="$MODEL_VIEWS/$REVISION"
DRAFT_A16="${DRAFT_A16:-$VIEW/draft-a16.pt}"
if ! [[ "$PORT" =~ ^[1-9][0-9]{0,4}$ ]] || (( PORT > 65535 )); then
  echo 'PORT must be an integer from 1 to 65535' >&2; exit 1
fi

command=(docker run --rm --pull never --name "$NAME"
  --gpus all -p "127.0.0.1:$PORT:$PORT" --ipc host --shm-size 32g
  --security-opt no-new-privileges:true --log-opt max-size=50m --log-opt max-file=3
  -e HF_HOME=/root/.cache/huggingface -e TRANSFORMERS_CACHE=/root/.cache/huggingface
  -e TRITON_CACHE_DIR=/root/.cache/triton
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,pinned_max_round_threshold_mb:128
  -e QWEN_MTP_SHORTLIST=131072 -e QWEN_MTP_HOTMAP=/opt/qwen-mtp-hotmap.json
  -e QWEN_MTP_HOT_INT8=1 -e QWEN_MTP_FC_INT8=1 -e QWEN_MTP_HOT_SCREEN=1 -e QWEN_DRAFT_TRUNC=1
  -e QWENFAST_ROUTER=1 -e QWENFAST_PK=1 -e QWENFAST_W8=inproj,qkv,oproj,head,shared,outproj,hc
  -e QWEN_NVFP4_MARLIN=1
  -e QWENFAST_A16MOE=1 -e QWENFAST_W8M=1 -e QWENFAST_W8M_PDL=1 -e QWENFAST_HCCOMB=1
  -e QWEN_MTP_A16=/opt/qwen-draft-a16.pt
  -e QWENFAST_HC_WP=1 -e QWENFAST_ATTN=1 -e QWENFAST_HEAD_SCREEN=1
  -e QWEN_MTP_HOT_SCREEN2=1 -e QWENFAST_GDNOUT_WP=1
  -e QWENFAST_GDNOUT_VARIANT=replay -e QWENFAST_GDN_REPLAY=1
  -v "$DRAFT_A16:/opt/qwen-draft-a16.pt:ro"
  -v "$HF_CACHE:/root/.cache/huggingface"
  -v "$HF_CACHE:$HF_CACHE:ro"
  -v "$VIEW:/opt/redhat-model:ro"
  -v "$RUNTIME_CACHE/triton:/root/.cache/triton"
  -v "$RUNTIME_CACHE/sglang:/root/.cache/sglang"
  -v "$RUNTIME_CACHE/tilelang:/root/.tilelang"
  "$IMAGE" python3 -m sglang.launch_server
  --model-path /opt/redhat-model/target --served-model-name qwen3.8-flash-next
  --tp 1 --trust-remote-code --quantization compressed-tensors --fp4-gemm-backend flashinfer_cutlass
  --context-length 262144 --max-total-tokens 786432 --kv-cache-dtype fp8_e4m3
  --mem-fraction-static 0.98 --page-size 64 --chunked-prefill-size 4096
  --ple-offload-embedding --linear-attn-prefill-backend triton --linear-attn-decode-backend flashinfer
  --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer
  --mamba-track-interval 64 --max-mamba-cache-size 14
  --max-running-requests 2 --cuda-graph-max-bs-decode 2
  --speculative-algorithm NEXTN --speculative-draft-model-path /opt/redhat-model/draft
  --speculative-draft-model-quantization fp8
  --speculative-use-rejection-sampling --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  --model-loader-extra-config '{"enable_multithread_load":false}'
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
if [[ ! -f "$VIEW/target/config.json" || ! -f "$VIEW/draft/config.json" ]]; then
  echo 'Model views missing. Run ./serve/download-model.sh first.' >&2; exit 1
fi
if [[ ! -f "$DRAFT_A16" ]]; then
  echo 'Draft A16 artifact missing. Run ./serve/prepare-draft-a16.sh first.' >&2; exit 1
fi
mkdir -p "$RUNTIME_CACHE/triton" "$RUNTIME_CACHE/sglang" "$RUNTIME_CACHE/tilelang"
# FlashInfer's tuned fused-MoE tactics for this GPU and library stack. Without them a fresh cache uses default
# tactics and prefill is ~6% slower (results/20261026). Existing entries are never overwritten.
mkdir -p "$RUNTIME_CACHE/sglang/flashinfer/autotune"
cp -rn "$HERE/tuning/flashinfer-autotune/." "$RUNTIME_CACHE/sglang/flashinfer/autotune/"
# Container networking is isolated. Only the HTTP port is published, to loopback.
exec "${command[@]}"
