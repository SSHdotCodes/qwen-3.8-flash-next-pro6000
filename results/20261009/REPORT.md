# October 9: 244 tok/s decode and W4A16 routed experts

Two changes landed on October 9, each validated before the next:

1. **Decode kernels and draft changes:** 202.6 to **244.4 tok/s**.
2. **Routed experts run as W4A16:** KLD against BF16 falls from 0.0178 to **0.0111**. The checkpoint's NVFP4
   weights are unchanged and run with BF16 activations through SGLang's Marlin kernels.

Context stays at 262,144 tokens, with two concurrent requests, a KV pool of 786,432 tokens and recommended
sampling.

## Decode

Benchmark: `bench/decode_bench.py`, run as on October 5 (4,096 tokens, 4 profiles, 3 runs, xhigh, recommended
sampling). Steps/s and accept length come from `bench/accept_window.py`.

| Build | code | agent | prose | chinese | mean tok/s | steps/s | accept len |
|---|---:|---:|---:|---:|---:|---:|---:|
| October 5 | 219.6 | 203.5 | 193.4 | 193.9 | 202.6 | 87.1 | 2.284 |
| + group-scaled INT8 dense target weights | 247.0 | 238.4 | 227.2 | 218.5 | 232.7 | 100.4 | 2.286 |
| + truncated draft proposals, fused top-k / sampling | 258.2 | 235.8 | 234.7 | 227.1 | 238.9 | 101.5 | 2.319 |
| + INT4-screened draft hot head | 248.9 | 248.2 | 243.2 | 237.3 | **244.4** | **103.1** | **2.329** |
| + W4A16 routed experts (current) | 249.9 | 250.0 | 237.5 | 236.3 | **243.4** | 100.8 | 2.381 |

Overall that is +20.6% tok/s and +18.4% steps/s, with 0 degenerate outputs. Six sustained 1,536-token generations
went from 280-295 to **326-348 tok/s**. Accept length varies by about 2% between benchmark runs because the
sampled text differs.

### What changed

1. **w8g: group-scaled INT8 dense target weights** (`QWENFAST_W8=inproj,qkv,oproj,head,shared,outproj,hc`).
   - Coverage: every non-expert BF16 linear the verify pass reads. That is GDN in_proj and out_proj, attention qkv
     and o_proj, lm_head, the shared expert, and the hyper-connection mix.
   - Format: INT8 with one BF16 scale per 128 weights of a row (64 for the 320-wide mix weights). The scale is
     rounded before quantizing, so q * s is exactly the dequantized value.
   - Not touched: the MoE router, the HC inject weights, the routed experts and the embeddings.
   - Decode reads 4.26 GB instead of 8.38 GB.
   - Kernels (`qf_w8g.cu`, `qf_hc8.cu`, `qf_gdnout.cu`): the INT8 HC mix and the fused GDN norm + out_proj are
     bit-exact against the BF16 kernels run on the dequantized weights.
   - Prefill unpacks to BF16 and runs the stock GEMM on the same dequantized weights.
2. **Truncated draft proposals** (`QWEN_DRAFT_TRUNC=1`, draft side only).
   - Problem: SGLang's rejection sampling proposes from the untruncated draft softmax, while the target samples with
     top-k 20 and top-p 0.95, so draft probability outside the target's support is wasted.
   - Change: the proposal is now the draft's own top-k / top-p truncation, computed from its top-64 logits.
   - Exactness: rejection sampling is exact for any proposal q, as long as the token is drawn from q and the verify
     step sees that q.
   - Measured on the same 2,500 decode steps: per-level acceptance 0.695 / 0.617 / 0.567 rose to
     0.712 / 0.635 / 0.584.
   - The draft CUDA graph carries `top_ps` as a device buffer (`eagle_draft_cuda_graph_runner.py`).
3. **Fused top-k and sampling** (`qf_topk.cu`).
   - A two-stage radix-select top-k: 12.5 us vs 48-55 us for `torch.topk` on a 248K row, with identical results.
   - A one-launch truncated-proposal sampler, checked against a torch reference.
4. **INT4-screened draft hot head** (`QWEN_MTP_HOT_SCREEN=1`, `qf_w4s.cu`, draft side only).
   - An INT4 copy of the 97,800-row INT8 hot head picks 64 candidates per row.
   - Exact INT8 logits are computed only for those candidates: 91 us vs 147 us per call.
   - Draft numerics change acceptance only, never the output distribution.

FP8 E4M3 (+29% KLD) and per-row INT8 (+7% KLD) dense weights were measured and rejected; group-scaled INT8 is the
format that stays within noise. Tree drafting was measured in simulation and rejected: +3.8% tokens per step, but
about 7% more time per extra verify row on this MoE.

## W4A16 routed experts

Most of the remaining KLD came from the 4-bit **activations**, not the 4-bit weights. The checkpoint's experts are
W4A4: FlashInfer CUTLASS quantizes every expert input to e2m1 with e4m3 block scales. The table runs the same NVFP4
weights with BF16 activations, scored as full-vocabulary KLD against true BF16 logits over 45,610 positions
(`kld/`):

| Configuration | KLD | Top-1 agreement | PPL |
|---|---:|---:|---:|
| W4A4 experts, BF16 dense (October 5) | 0.01756 | 0.9608 | 1.4067 |
| W4A4 experts + INT8 dense (first October 9 build) | 0.01784 | 0.9595 | 1.4078 |
| W4A16 experts (Marlin), BF16 dense | 0.01064 | 0.9707 | 1.3965 |
| **W4A16 experts (Marlin) + INT8 dense (current)** | **0.01109** | **0.9689** | **1.3973** |

The W4A16 path is SGLang's Marlin NVFP4 MoE kernel. The overlaid compressed-tensors scheme enables it with
`QWEN_NVFP4_MARLIN=1`. SGLang's Marlin repack kept a second copy of each layer's experts and ran out of memory at
load with these settings, so the overlay repacks each expert into its own storage, which has the same byte count.

Validation against the W4A4 build (same seeds and harnesses):

| Check | W4A4 + INT8 dense | W4A16 + INT8 dense |
|---|---|---|
| Agent tool-use bench, 60 tasks x thinking off/on | 105/120 | **112/120** |
| Code-copying probe, 96 rows: log P(full code) drift vs BF16, copy failures | 0.139, 2 | **0.116, 0** |
| GSM8K chat, thinking off (1,314) / thinking on (200) | 0.968 / 0.985 | 0.968 / 0.990 |
| Retrieval at 8K / 120K / 240K, 2 x 24K, 2 x 240K, degeneration probe | pass, 0/36 | pass, 0/36 |
| Decode (tok/s) | 244.4 | 243.4 |
| Prefill 8K / 32K / 128K (s) | 0.51 / 2.04 / 8.77 | **0.59 / 2.37 / 10.13** |

The trade is prefill: about 15% slower, because Marlin's large-batch GEMMs run BF16 tensor cores where CUTLASS ran
FP4. Marlin's block size, larger prefill chunks (6,144 and 8,192) and a Triton W4A16 kernel were tried and did not
help. To return to W4A4, remove `QWEN_NVFP4_MARLIN=1` from `serve/run-server.sh`.

### Measured and not adopted

- **BF16 KV cache:** KLD 0.01064 to 0.01038, at the cost of half the KV pool.
- **nvidia/Qwen3.8-Flash-Next-NVFP4 routed experts** (same NVFP4 layout, MSE-calibrated scales):
  - KLD 0.01109 to 0.00941, same speed.
  - The behaviour tests did not improve: agent bench 110 vs 112/120; the code-copying probe went back to the W4A4
    level (drift 0.202, 3 failures).
  - Below about 0.011, KLD stopped predicting the behaviour tests.

## Public build check

The published files were checked on the measuring workstation. `serve/build-image.sh` built the image from the pinned
SGLang v0.5.21 base, compiling the kernels in 135 s with `MAX_JOBS=4`. Its overlays, kernel sources and draft token
map are byte-identical to the measured runtime, apart from a renamed loader key and a comment.
`serve/download-model.sh` rebuilt the target and draft views from the cached snapshot. The only difference from the
measured views is that renamed key.

`serve/run-server.sh` served from that image with a fresh runtime cache:
- A chat request answered correctly.
- A tool call returned well-formed arguments.
- 32K-token prefill took 2.43 s.

`decode_bench.py`, run with the decode protocol above (3 runs of each profile):

| Run | code | agent | prose | chinese | mean tok/s | steps/s | accept len |
|---|---:|---:|---:|---:|---:|---:|---:|
| first server start, empty runtime cache | 238.3 | 236.7 | 226.2 | 227.3 | 232.1 | 96.5 | 2.367 |
| restart, same cache | 252.8 | 239.6 | 239.6 | 231.7 | 240.9 | 100.7 | 2.361 |
| measured workstation cache | 248.9 | 245.6 | 233.8 | 231.8 | 240.0 | 100.8 | 2.348 |

After the first start, verify steps/s match the release measurement (100.8). The tok/s gap to 243.4 tracks accept
length, which moves about 2% between runs with sampled text.
