# Runtime notes — 2026-09-19, decode kernels 2026-09-24, RedHatAI and decode work 2026-10-05, W4A16 2026-10-09, decode kernels 2026-10-10, kern2 2026-10-20

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

## October 10 changes

Decode-only kernels for batches of 1-8 rows. Prefill, weights, formats and sampling are unchanged. The
[October 10 report](../results/20261010/REPORT.md) has the measurements.

- `QWENFAST_A16MOE=1` (`qf_a16moe.cu`, needs `QWEN_NVFP4_MARLIN=1` and `QWENFAST_ROUTER=1`): one persistent
  kernel runs the routed experts on the Marlin-packed NVFP4 weights, with Marlin's dequantization, BF16 activations
  and FP32 accumulation. It covers gate/up, SiLU-mul, down, the top-k weighting, the sum over the 10 experts and the sigmoid-gated shared
  expert. It is shape-gated: it declines (and logs `a16 MoE decode declined`) unless the experts carry the Marlin
  layout and the router selects 10 experts. Without `QWEN_NVFP4_MARLIN=1` it never runs, and without
  `QWENFAST_ROUTER=1` it is not reached.
- `QWENFAST_A16SHARED` (default `1` when `QWENFAST_A16MOE=1`): the shared expert's group-scaled INT8 weights are
  computed inside the persistent kernel instead of on a side stream. `QWENFAST_A16SHARED=0` keeps the side-stream
  shared expert and passes its output to the kernel. The in-kernel path needs `shared` in `QWENFAST_W8`; otherwise
  the side stream is used.
- `QWENFAST_W8M=1` (`qf_w8g.cu`): the group-scaled INT8 dense GEMMs (K = 2,560 and 6,144) run on tensor cores
  (`mma.m16n8k16`), with the same weights and scales as w8g.
- `QWENFAST_W8M_PDL=1`: w8m is launched with programmatic dependent launch, so its first weight requests overlap
  the preceding hyper-connection mix. It only affects w8m.
- `QWENFAST_HCCOMB=1` (`qf_hc8.cu` `hc_combine_mix_w8`, needs `hc` in `QWENFAST_W8`): each hyper-connection combine
  runs inside the next mix kernel, within a layer and across layers except around the PLE layer. It is used for up
  to 4 rows (`QWENFAST_HCCOMB_MAXROWS`) and is bit-identical to combine followed by mix.
- Router (`qf_router.cu`, part of `QWENFAST_ROUTER=1`): token staging is sized to the batch and the top-k uses
  `redux.sync`. This change has no switch; its outputs are bit-identical to the October 9 router.

The server log lists the active paths at startup: `a16 MoE decode active`, `a16 shared expert in-kernel` and
`hc combine->mix fusion installed`.

To turn the October 10 kernels off, remove `QWENFAST_A16MOE=1`, `QWENFAST_W8M=1`, `QWENFAST_W8M_PDL=1` and
`QWENFAST_HCCOMB=1` from `serve/run-server.sh`. Decode then runs the October 9 kernels; only the router change
remains, with identical outputs. Each variable can also be removed on its own.

Kernel tests for these paths (GPU, inside the image):
- `test_a16moe.py`: compares the persistent kernel with SGLang's Marlin MoE path on 12 layers. It reads the target
  view at `/models/target`. Mount `$MODEL_VIEWS/<revision>/target` there, and `$HF_CACHE` at its own host path,
  where the view's symlinks point. It builds the kernel with `qfa16_bind.cpp`.
- `test_w8m.py`: compares w8m with w8g and an FP32 reference.
- `test_hc8c.py`: checks that the fused combine-mix is bit-identical to combine followed by mix.
- `test_router3.py`: compares the router and HC mix outputs of two libraries in separate processes.

`test_w8m.py`, `test_hc8c.py` and `test_router3.py` import `build_qf` from `/ext`, so mount `serve/qwenfast` there.
`test_hc8c.py` and `test_router3.py` also load the previous build's library from `/prod/qwenfast.so`.
`test_router3.py` runs twice: `prod` first, which writes `/ext/rt3-prod.pt`, then `new`.

## October 20 kern2

The [October 20 report](../results/20261020/REPORT.md) records 303.8 tok/s and 125.2 verify steps/s.
**Decode-path KLD was not re-measured.** The public image build and the regenerated draft artifact were
checked on October 26 ([report](../results/20261026/REPORT.md)); the ported GPU kernel suites were not run.

Of the 27 Python overlays and hot map, **26 files match the measured installed manifest byte for byte**.
Two overlay files have the intentional public/installed differences listed below. The public target
view and public loader both use `redhat_streamed_load`, so bounded checkpoint loading remains enabled.
All 26 kernel compilation units are pinned
separately from headers, the build script and optional support/test sources. The standalone module
in `head_run.cpp` is disabled unless `WP_HEAD_STANDALONE` is defined; the shared build has only the
`qf_bind.cpp` module. `gdn_replay_update.cuh` is pinned as an included dependency.

### Intentional public/installed differences

This follows the earlier public-release convention. These are the complete source/config differences
introduced when removing workstation-specific naming:

| Public file | Intentional difference from the measured installation |
|---|---|
| `serve/overlays/sglang/srt/model_loader/loader.py` | Reads `redhat_streamed_load` instead of the installed workstation-prefixed config key. No loading algorithm changes. |
| `serve/overlays/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` | Dated comment starts `2026-10-09:`; the workstation name is omitted. Executable code is unchanged. |
| `serve/prepare-model.py` | Writes `redhat_streamed_load` in the target view to match the public loader; the installed target config uses a workstation-prefixed spelling. |

`tests/test_publication.py` checks the coordinated public key, and `docs/runtime.md` describes it. Those
support files do not have mounted counterparts in the installed runtime. The draft config is unchanged.

`serve/runtime.json` uses `sha256` for each public file. The two changed overlays also retain
`installed_sha256` and a `public_difference` explanation. The measured manifest digest stays unchanged.
`verify-runtime.py` checks the public hashes locally and in the rebuilt public image (`--installed`);
that flag does not mean the original workstation overlays are byte-identical. To compare with a saved
measured manifest without accessing the workstation, run:

```bash
python3 serve/verify-runtime.py --installed-manifest /path/to/saved/manifest.json
```

The comparison verifies the saved manifest digest and reports 26 identical file hashes plus two
intentional differences with pinned installed hashes. An unrecorded difference or changed installed
reference is rejected. Kernel sources/bindings and the token map still match the installed bytes.

The launcher reproduces the installed unit's environment and SGLang flags. New explicit variables:

| Variable | Selected behavior |
|---|---|
| `QWEN_MTP_A16=/opt/qwen-draft-a16.pt` | Draft NVFP4 A16 routed experts, INT8 shared expert; frees unused FP8 copies |
| `QWENFAST_HC_WP=1` | Module-specific fast HC mix/combine scheduling |
| `QWENFAST_ATTN=1` | Fused QSA norm/RoPE/store and count packing |
| `QWENFAST_HEAD_SCREEN=1` | Target head screen for top_k 1–56 without logprobs, grammar, penalties, min_p or logit bias |
| `QWEN_MTP_HOT_SCREEN2=1` | Two-stage INT2/INT4 draft hot-head screen at M=1 |
| `QWENFAST_GDN_REPLAY=1` | Replay accepted GDN deltas on commit; snapshot fallback for unsupported paths |
| `QWENFAST_GDNOUT_WP=1` | New GDN output projection |
| `QWENFAST_GDNOUT_VARIANT=replay` | Output preset paired with replay |

Candidate-only draft sampling and speculative relay/QSA/sequence/stack fusions default on in the
shipped overlays, as on the installed unit; they do not need additional launcher env flags. Optional
index prep/MQA/linear and PLE extras remain off; cached top-k row starts default on. Each kernel retains its supported-shape/dtype gate and fallback.
GDN replay and output must be assessed together: each was flat alone. Merely removing the October 10
flags does not reconstruct an earlier runtime while these newer overlays/artifacts remain active.
Use the earlier release checkout/image for a controlled rollback.

The target head's candidate values are exact W8M results, but the sigma screen's inclusion guarantee
is empirical (602 captured inputs passed), not a proof over every input. Ineligible requests use a
dense fallback; the recorded dense-head overhead was about 56 µs per call versus the old dense head.
Draft-screen misses change only proposal/acceptance under correct rejection sampling.

### Draft artifact preparation

After model download, run `./serve/prepare-draft-a16.sh` explicitly on an idle GPU. The script uses the
same pinned image and model view, no network, and saves a tensor-only torch archive plus provenance.
`DRAFT_A16` selects the host output/mount path. It refuses an existing output. The recipe is adapted
from the October 20 converter, with seeded synthetic validation replacing private activation captures.
It validates M=1/4 A16 outputs against dequantized FP32 arithmetic before saving; it has not yet been
executed during public packaging.

The measured artifact hash is `f397eaae4919d1b6d8e2553c1fae3bed5f3a17081baac03f449fcb90a4919c11`.
Serialization can change archive bytes; a fresh conversion needs tensor/layout and end-to-end GPU
validation even if its file hash differs only because of serialization. To check a copied measured
artifact's exact bytes without loading torch: `python3 serve/verify-runtime.py --draft-a16 PATH`.
The default verifier checks sources and preparation hashes; `--installed` additionally checks packages
and library presence. It does not claim rebuilt binaries equal the measured binary or validate an
external draft artifact automatically.

### Ported work-package GPU tests

The image includes these scripts in `/opt/qwenfast/src`. They load the integrated extension and
shipped Python overlays. Run on a dedicated idle GPU inside the built image, for example:

```bash
docker run --rm --gpus all --network none --entrypoint python3 \
  local/qwen-flash-next:0.5.21-20261020-kern2 /opt/qwenfast/src/test_hc_kern2.py
```

| Script | Public coverage |
|---|---|
| `test_hc_kern2.py` | WP-HC M=1–8, mix/combine, inject/no-inject, both selected variants, bitwise outputs |
| `test_attn_cached_topk.py` | WP-Attn cached row starts vs original top-k sets |
| `test_attn_prepare.py` | WP-Attn synthetic norm/RoPE/KV-store and FP8 edge values |
| `test_head_kern2.py` | WP-Head candidate/top-k, dense/overflow fallback and mutable graph inputs |
| `test_drafthead_edges.py` | WP-DraftHead full rescore, ties, histogram saturation and strided input |
| `test_spec_fused.py` | WP-Spec sequence/bonus/extend helpers and mrope expressions |
| `test_spec_qsa.py` | WP-Spec fused metadata vs retained unfused path, including untouched buffers |
| `test_spec_relay.py` | WP-Spec scheduler relay scenarios vs elision disabled |
| `test_gdn_pool_activation.py` | WP-GDN actual pool allocation and layer views, on/off and BF16/FP32 |
| `test_gdn_replay.py` | WP-GDN synthetic verify/commit, tree/snapshot fallback, tracking and sequential acceptance |
| `test_gdnout_kern2.py` | WP-GDNout both presets/gates, M=1–8 and mutable-graph bitwise outputs |

`test_head_kern2.py` additionally needs the pinned target at `/models/target` (`QWEN_TARGET` overrides it).
Mount its view there and the backing `HF_CACHE` at the same host path so symlinks resolve. Run
`test_gdn_replay.py` twice, passing `WPGDN_STATE_DTYPE=bfloat16` and `WPGDN_STATE_DTYPE=float32`.
Standalone FP32-state kernel coverage does not imply the current FlashInfer model backend supports
FP32 state. The public ports use synthetic inputs/weights where the originals used activation
captures. They do not reproduce the original captured-input cohorts or cold-neighbor timing tables,
and have only been syntax-checked during packaging. No GPU tests were executed for this release draft.

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
