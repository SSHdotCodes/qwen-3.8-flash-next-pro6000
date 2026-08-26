#!/usr/bin/env bash
# Produce the SM120-patched qwen_sparse_attn_backend.py by applying the shipped
# diff to the file inside the pinned SGLang image.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1}"
SRC=/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py

docker run --rm --entrypoint cat "$IMAGE" "$SRC" > "$HERE/qwen_sparse_attn_backend.py"
patch -p0 "$HERE/qwen_sparse_attn_backend.py" < "$HERE/qwen_sparse_attn_backend.sm120.patch" \
  || patch -l -p0 "$HERE/qwen_sparse_attn_backend.py" < "$HERE/qwen_sparse_attn_backend.sm120.patch"
python3 -m py_compile "$HERE/qwen_sparse_attn_backend.py"
echo "built: $HERE/qwen_sparse_attn_backend.py"
