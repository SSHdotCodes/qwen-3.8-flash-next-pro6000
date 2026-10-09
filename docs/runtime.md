# Runtime notes — 2026-09-19, decode kernels 2026-09-24, RedHatAI and decode work 2026-10-05, W4A16 2026-10-09

## Reproducible image

`serve/Dockerfile` starts from the official SGLang v0.5.21 image, digest
`b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`.
The dependency repairs match the tested workstation image:
- remove the unused `nixl` (generic CUDA 12 shim) and the optional `cosmos-guardrail`;
- keep the CUDA 13 NIXL distribution;
- pin NCCL, protobuf, grpcio-tools, typeguard, Pillow and OpenCV.

`pip check` must pass. The runtime verifier also checks SGLang 0.5.21, PyTorch 2.13.0+cu130, SGLang
kernel 0.4.7 and FlashInfer 0.6.18.

The measured local image digest begins `ed0946e2`; it is **not a pullable public
registry digest**. This repository rebuilds from its public base and puts the
overlays into the image instead of workstation-specific bind mounts.
Build timestamps and attestations can change the resulting image digest. Package
versions and all installed overlay hashes are checked against `serve/runtime.json`.

## What the overlays do

- **SM120 QSA and FP8 KV:** preserve the validated SM120 attention dispatch,
  query-dtype packing and in-register FP8 tile dequantization. These are the
  current cache/attention correctness fixes, rather than the day-zero patch.
- **16-slot QSA pending ring:** update allocation, eager slots, captured metadata,
  fallback indexer calls and pending-state transfer strides together. Compression
  stays at four. Validation requires `draft_tokens + compress_ratio - 1 <= 16`,
  leaving room for a prior partial group. Removing the old width guard alone
  is incorrect. The original four-slot layout failed 248 width-four oracle cases;
  the selected layout passed 4,026 adversarial cases and 48 CUDA graph replays.
- **Proposal lifetime:** clone proposal probabilities before target graph replay
  can overwrite shared logits/probability storage. This fixed observed corruption
  under sampled exact rejection after the 0.5.20 update. Greedy smoke tests had
  missed it. Keep `--speculative-use-rejection-sampling` enabled.
- **Draft-only hot map:** gather/cache a BF16 copy of 65,800 draft-head rows;
  fill the full proposal-logit tensor with negative infinity outside those IDs.
  Target head weights and full-vocabulary target sampling remain unchanged.
  `QWEN_MTP_SHORTLIST=131072` activates the processor; `QWEN_MTP_HOTMAP` selects
  the smaller map and overrides the contiguous shortlist in the supported path.
  The old two-range fallback and some comments/log strings remain in the source
  to preserve the exact validated file hashes. Unsupported head layouts fall
  back to the original full-head implementation.
- **Startup:** indexed expert mapping, indexed native-MTP tensor loading and
  collection of unreachable temporary tensors before sizing the KV pool. These
  change startup work, not checkpoint values. Multithread loading uses 16 threads.

The standalone numerical check of FlashInfer target verification on SM120 was
limited to this model's shapes. Its combined performance experiment did not
justify a deployment change. The launcher retains the original Triton verify
path and no draft temperature multiplier.

## Decode kernels (2026-09-24)

`serve/qwenfast/` holds CUDA kernels for the decode and verify steps, compiled into the image by
`serve/Dockerfile` (`torch.utils.cpp_extension`, the image's nvcc 13.0, `-gencode arch=compute_120a,code=sm_120a`).
`sglang/srt/qwenfast.py` loads `/opt/qwenfast/qwenfast.so` and replaces a handful of SGLang functions. The model
modules import it. Every replacement is shape-gated and falls back to the stock SGLang path otherwise. Prefill and
the checkpoint, quantization, KV format, context and sampling settings are unchanged.

A verify step (4 tokens) took 14.3 ms, spread over about 1,400 kernels. It is already about 78% bandwidth-efficient,
so the kernels mostly remove latency between launches:

- **Small-M BF16 GEMM** (1–8 rows) for the GDN input projection (16,480 × 2,560), the MoE router, the LM head and the
  draft's hot head.
- **NVFP4 decode MoE** for 1–8 tokens. Two kernels read each selected expert once for all tokens routed to it. They
  use the same W4A4 recipe as FlashInfer's CUTLASS path: FP4 activations with the same global and block scales, and
  BF16 rounding of the first GEMM's output and of SwiGLU. The top-k-ordered reduction is deterministic, and the
  kernels use programmatic dependent launch.
- **Hyper-connections.** The per-branch Gemma RMSNorm, the low-rank gated mix and the combine's inject products run in
  one tensor-core kernel (a grid barrier between the down and up projections), followed by a one-pass combine:
  97 calls per step, 16.6 → about 11 µs each.
- **GDN output.** The gated RMSNorm is fused into `out_proj`. The kernel is released when the recurrent-state kernel
  starts and pulls its weight slice into L2 while the recurrence runs (up to 4 tokens).
- **Verify sampling.** Top-k candidates per row, then one kernel applies temperature, top-k (ties kept), top-p
  (FlashInfer's pivot rule), renormalization and the chain rejection sampling of SGLang's Triton kernel over the
  sparse set. It falls back to the dense path unless every request has 1 ≤ top_k ≤ 56. With identical random coins it
  gave identical accept counts, accepted tokens and final tokens to the dense path in 12,000 of 12,000 randomized
  cases (random temperatures, top-k 1–56, top-p on and off, tied logits).
- **Also:** a multi-block softmax for the few 248K-wide rows (the dense fallback and the draft proposal), and the QSA
  graph row metadata launched as one block over the page table (13–18 → 3–5 µs).

Switches, read by the server process (pass them with `-e` in `serve/run-server.sh`): `QWENFAST=0` disables every
replacement of an SGLang function. The draft head's small-row GEMM (`hot_head`, called from the MTP model file) stays
on; it computes draft proposals only, and the target still verifies every token. `QWENFAST_MOE=0`, `QWENFAST_HC=0`, `QWENFAST_GDN=0`, `QWENFAST_SAMPLE=0` and `QWENFAST_QSAMETA=0` disable
one each. The server log lists the active replacements at startup.

Measured and not used: a fused router GEMM plus top-k ahead of the shared expert (the MoE block is bandwidth-bound:
72.9 → 73–74 µs), small-M kernels for QKV, `index_qk` and the shared expert (they starve the concurrent GEMM on the
other stream), CUTLASS and Marlin MoE backends (unsupported for NVFP4, or out of memory), and truncated draft
proposals (lower acceptance).

Kernel tests (GPU, inside the image; the MoE, router and GDN tests read one layer of the pinned checkpoint from the
HF cache mount, or from `QWEN_SNAPSHOT`):

```bash
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
docker run --rm --gpus all --network none -v "$HOME/models/huggingface:/root/.cache/huggingface:ro" \
  --entrypoint python3 local/qwen-flash-next:0.5.20-20260924-qwenfast /opt/qwenfast/src/test_moe.py
```

`test_sample.py` (sampler equivalence), `test_hc2.py`, `test_router.py` and `test_gdnout.py` run the same way. Each
writes its measurements to `/tmp` (`QWENFAST_TEST_OUT`). The published results are in
[results/20260924/kernel-tests](../results/20260924/kernel-tests).

## RedHatAI checkpoint (2026-10-05)

`serve/prepare-model.py` builds `target/` and `draft/` views of the pinned snapshot from symlinks plus their own
`config.json` and index. The raw checkpoint is not modified.
- **Layer labels:** SGLang 0.5.21 calls the QSA blocks `full_attention`.
- **Target scope:** the target view keeps the BF16 PLE table (`ple_embedding_dtype`). It adds an ignore rule
  limiting quantization to `mlp.experts`, because Qwen4's runtime flattens module prefixes that the checkpoint's
  own ignore list names. It also sets `redhat_streamed_load`, which `model_loader/loader.py` reads to stage the
  checkpoint in bounded memory.
- **Draft view:** keeps only the `mtp.*` tensors and drops the quantization config. FP8 is applied online.
- **Per-expert activation scales:** RedHatAI stores one GEMM2 activation scale per expert, and the W4A4 MoE decode
  kernel accepts scalar and per-expert scales.
- **Pinned host memory:** `pinned_max_round_threshold_mb:128` in `PYTORCH_CUDA_ALLOC_CONF` stops the 102.4 GB
  pinned PLE allocation from being rounded up to 137.4 GB.

## Decode work (2026-10-05)

Environment switches:
- `QWENFAST_PK=1`: exact 12-bit storage of the BF16 GDN in_proj, attention qkv/o_proj and lm_head (`qf_pk.cu`).
  Each weight is one byte plus a 4-bit exponent offset from a per-row base, with a per-row exception list holding
  the rare outliers exactly. Every matrix is unpacked and compared bit for bit at load; any mismatch stops startup.
- `QWENFAST_ROUTER=1`: fused router GEMM and softmax top-10.
- `QWEN_MTP_HOT_INT8=1`, `QWEN_MTP_FC_INT8=1` and `--speculative-draft-model-quantization fp8`: draft-side INT8 / FP8
  weights. Draft numerics change acceptance only.

The draft hot map grew from 65,800 to 97,800 IDs (CJK coverage; see the token-map section).

## October 9 changes

Target side:
- `QWENFAST_W8=inproj,qkv,oproj,head,shared,outproj,hc`: group-scaled INT8 dense weights (`qf_w8g.cu`,
  `qf_hc8.cu`, `qf_gdnout.cu`). There is one BF16 scale per 128 weights, rounded before quantizing so that q * s is
  the exact dequantized value. Prefill unpacks to BF16 and runs the stock GEMM on the same dequantized weights.
- `QWEN_NVFP4_MARLIN=1` (compressed-tensors NVFP4 MoE overlay): the routed experts run as W4A16 through SGLang's
  Marlin NVFP4 MoE kernels. Gate and up are loaded in `[gate; up]` order. Each expert is repacked into its own
  storage, because the Marlin layout has the same byte count, which avoids a second copy of a layer's experts at
  load. The checkpoint's per-expert input scales are unused in this mode.

Draft side:
- `QWEN_DRAFT_TRUNC=1`: proposals are drawn from the draft's own top-k / top-p truncation, built from its top-64
  logits. Rejection sampling stays exact because the verify step sees the same truncated q. The draft CUDA graph
  carries `top_ps` in a device buffer (`speculative/eagle_draft_cuda_graph_runner.py`).
- `QWEN_MTP_HOT_SCREEN=1`: an INT4 copy of the hot head screens 64 candidates per row, and only those get exact
  INT8 logits (`qf_w4s.cu`).
- `qf_topk.cu`: a radix-select top-k matching `torch.topk`, and the fused truncated-proposal sampler.

Turn off one feature by removing its variable from `serve/run-server.sh`. `QWENFAST=0` turns off all qwenfast
replacements.

## Token-map provenance

The original 65,536-ID map was downloaded from:

https://raw.githubusercontent.com/gabrielolympie/sglang-flashnext-sm120/main/hot_tokens_64k.pt

Source size: 209,175 bytes. SHA-256:
`f03551e4709e50e5842b24912845c243bfe628b74bf185617fae7ebaaa0da86c`.
The source URL names a mutable branch; the checksum identifies the exact
artifact used. It was loaded with `torch.load(..., weights_only=True)` and
validated as a list/tensor of in-range integer IDs. The shipped JSON is:

```python
sorted(set(source_ids) | set(range(248044, 248320)))
```

That September map has 65,800 unique IDs and SHA-256
`8a4f4c7c5905b76861a62befa1e060e16362e1775087fbb78e907fd85c4c9c12`. It missed almost all
CJK tokens: 8.67% of Chinese output tokens fell outside it. The October 5 map adds the 16,000
lowest-ID missing CJK tokens and the 16,000 lowest-ID other missing tokens. It has 97,800 unique IDs,
including all 276 special IDs, and SHA-256
`2123b2db7f4e1ecd99879db17fcbdd58a887fbb315a45f61c6c83b6ffb04f679`.
No map download or pickle loading occurs at server startup. Changing this map
requires new sampling/correctness/performance checks; a smaller proposal space
can hurt acceptance on languages or tasks poorly represented in it.

## Verification and recovery

`python3 serve/verify-runtime.py` checks the exported sources without a GPU.
`./serve/build-image.sh` also checks installed files and package versions inside
the rebuilt image. `python3 -m unittest discover -s tests -v` checks launch
arguments, response parsing, sampling and the published aggregation locally.

To rerun the ring oracle and CUDA graph checks without loading model weights,
stop inference and run on an available GPU:

```bash
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
docker run --rm --gpus all --network none \
  -v "$PWD/bench/test_qsa_ring.py:/test_qsa_ring.py:ro" \
  local/qwen-flash-next:0.5.20-20260924-qwenfast python3 /test_qsa_ring.py
```

The ring test verifies buffer indexing, not full-model bitwise equivalence.
The original long-context and two-request checks can also be reproduced against
an idle loaded server with `python3 bench/context_checks.py context --flush-cache`
and `python3 bench/concurrent_check.py concurrent --flush-cache`. These preserve
the synthetic prompts and seeds and write results into `results/local/`.
For an upgrade, retain your previous checkout, image and cache until your own
checks pass. To roll back, stop this container/user unit and restore that prior
launcher/image. This repository does not install or overwrite workstation model
catalogs, inference proxies, credentials, drivers or system packages. The
historical August checkout changes precision/capacity and is not an equivalent
rollback for the September baseline.
