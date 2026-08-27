#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next NVFP4 on a single RTX PRO 6000 Blackwell (SM120, 96 GB)
# at full 262,144-token context with NEXTN/MTP speculative decoding.
#
#   ./run-server.sh            # start (foreground container, loopback only)
#   PORT=30010 ./run-server.sh
#
# Requires: rootless Docker + NVIDIA Container Toolkit, ~130 GiB free disk for
# the checkpoint, and ~50 GiB host RAM free for the offloaded PLE table.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1}"
MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
NAME="${NAME:-qwen38-flash-next-sglang}"
PORT="${PORT:-30010}"
HF_CACHE="${HF_CACHE:-$HOME/models/huggingface}"
QSA_PATCH="${QSA_PATCH:-$HERE/patches/qwen_sparse_attn_backend.py}"

mkdir -p "$HF_CACHE"

if [ ! -f "$QSA_PATCH" ]; then
  cat >&2 <<'MSG'
Missing the SM120 QSA backend file.

The pinned day-0 image dispatches the QSA fallback through a pip flash-attn
CUTE kernel that does not compile for the packed QSA decode shape on SM120.
Build the patched file once:

    ./patches/build-qsa-patch.sh

See patches/qwen_sparse_attn_backend.sm120.patch and upstream
https://github.com/sgl-project/sglang/issues/36531 (fix in PR #36556).
MSG
  exit 1
fi

# No --network host on purpose. With it, SGLang's ZMQ IPC -- which carries
# prompt and generated tokens in plaintext -- binds to the node IP and is
# reachable from anything that can route to this host. Isolating the container
# netns and publishing only to host loopback keeps that traffic internal.
exec docker run --rm --name "$NAME" \
  --gpus all -p 127.0.0.1:$PORT:$PORT --ipc host --shm-size 32g \
  --security-opt no-new-privileges:true \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$QSA_PATCH":/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py:ro \
  "$IMAGE" python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name qwen3.8-flash-next \
    --tp 1 --trust-remote-code \
    --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
    --context-length 262144 \
    --max-total-tokens 262144 \
    --mem-fraction-static 0.98 \
    --page-size 64 \
    --chunked-prefill-size 4096 \
    --ple-offload-embedding \
    --linear-attn-prefill-backend triton \
    --linear-attn-decode-backend flashinfer \
    --mamba-ssm-dtype bfloat16 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-track-interval 64 \
    --max-mamba-cache-size 8 \
    --max-running-requests 1 \
    --cuda-graph-max-bs-decode 1 \
    --disable-flashinfer-autotune \
    --speculative-algorithm NEXTN \
    --speculative-draft-model-path "$MODEL" \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --reasoning-parser auto \
    --sampling-defaults model \
    --stream-interval 4 \
    --host 0.0.0.0 --port "$PORT"
