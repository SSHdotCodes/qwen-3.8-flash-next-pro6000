# October 5: RedHatAI checkpoint, SGLang 0.5.21, faster decode and a larger KV pool

Two changes landed on October 5, one after the other:

1. The checkpoint moved from RadixArk to **RedHatAI/Qwen3.8-Flash-Next-NVFP4** (revision
   `c8f2fb1b9869f686b214782036123b10ff96d14a`) on **SGLang 0.5.21**.
2. Exact decode work on top of it: 183.4 to **199.5 tok/s**, and a KV pool of 524,288 to **786,432 tokens**.

Context stays at 262,144 tokens, two concurrent requests, FP8 E4M3 KV, and recommended sampling
(temperature 1.0, top-p 0.95, top-k 20).

## Checkpoint switch

The RedHatAI checkpoint keeps the original BF16 PLE (n-gram embedding) table, dense weights and MTP weights. Only
the routed experts are NVFP4. Raw checkpoint files are not modified: `serve/prepare-model.py` builds a target view
and a draft view next to the snapshot, from symlinks plus their own `config.json` and index.

- **Target view.** SGLang 0.5.21 calls the QSA blocks `full_attention`, so the view relabels them. The PLE table
  stays BF16. The quantization scope is written as "routed experts only", because Qwen4's runtime flattens module
  prefixes that the checkpoint's ignore list names.
- **Draft view.** The native MTP tensors, with no quantization config. FP8 is applied online by
  `--speculative-draft-model-quantization fp8`.

Matched against RadixArk on the same runtime (3 runs of 4,096 tokens each for code, agent and prose; xhigh;
recommended sampling):

| Profile | RadixArk tok/s | RedHatAI tok/s | Change |
|---|---:|---:|---:|
| code | 187.5 | 192.9 | +2.9% |
| agent | 185.9 | 190.3 | +2.3% |
| prose | 184.4 | 182.5 | -1.0% |
| prefill, 24K tokens (tok/s) | 16,015 | 15,892 | -0.8% |
| prefill, 96K tokens (tok/s) | 15,030 | 15,026 | 0.0% |

Three compatibility fixes were needed:

- **Per-expert activation scales.** RedHatAI stores one GEMM2 activation scale per expert. The custom MoE decode
  kernel now accepts scalar and per-expert scales. It was checked on 80 synthetic cases and on real layers 0 and 47
  at 1, 2, 4 and 8 tokens, against stock FlashInfer (about 0.28% relative L2 difference).
- **Pinned host memory.** A large pinned allocation rounded the 102.4 GB PLE table up to 137.4 GB, more than
  physical RAM. `PYTORCH_CUDA_ALLOC_CONF=...,pinned_max_round_threshold_mb:128` keeps the requested size.
- **Loader.** A bounded loader limits staging memory while streaming the checkpoint (`redhat_streamed_load` in the
  target view's config).

## Decode work

Benchmark: `bench/decode_bench.py`. Each request generates 4,096 tokens. There are 4 profiles (code, agent, prose,
Chinese), 3 runs each, xhigh reasoning and recommended sampling. Steps/s and accept length come from the server log
over the same window (`bench/accept_window.py`).

| Build | code | agent | prose | chinese | mean tok/s | steps/s | accept len |
|---|---:|---:|---:|---:|---:|---:|---:|
| RedHatAI checkpoint, before | 193.8 | 187.4 | 185.1 | 167.3 | 183.4 | 79.4 | 2.275 |
| **October 5** | 213.3 | 202.1 | 194.2 | 188.2 | **199.5** | **87.0** | 2.262 |

That is +8.8% tok/s and +9.6% verify steps/s. Six sustained 1,536-token generations run at 279-294 tok/s. The
4,096-token xhigh profiles are slower because acceptance falls on long reasoning text.

What changed (all in `qwenfast` and the MTP module; every path is shape-gated and falls back to stock):

1. **pk12: exact 12-bit storage of BF16 target weights** (`QWENFAST_PK=1`, `qf_pk.cu`).
   - Format: each weight is one byte (sign and 7 mantissa bits) plus a 4-bit exponent offset from a per-row base.
     The 2-3 weights per 10,000 outside the row's window are kept exactly in a per-row exception list.
   - Coverage: 61 matrices (GDN in_proj, attention qkv and o_proj, lm_head), 5.50 to 4.14 GB.
   - Every matrix is unpacked and compared bit for bit against the original at load. The server refuses to start
     on any mismatch.
   - Decode GEMMs read the packed form directly. Prefill unpacks into a BF16 scratch and runs the stock GEMM.
2. **Fused router** (`QWENFAST_ROUTER=1`): router GEMM and softmax top-10 in one launch.
   - Saves 8 us per MoE block.
   - Same top-10 expert set as stock in 100% of 3,600 rows.
3. **Draft (MTP) side.** This affects only acceptance, never the output distribution, because rejection sampling is
   exact:
   - FP8 draft weights;
   - weight-only FP8 GEMMs for 1-8 rows (`qf_w8.cu`);
   - the 97,800-row hot head stored INT8 per row;
   - INT8 copies of the draft `fc` projections.
4. **Draft hot map: 65,800 to 97,800 IDs.** The old map missed almost all CJK tokens (8.67% of Chinese output fell
   outside it). The new map adds the 16,000 lowest-ID missing CJK tokens and 16,000 other missing tokens. Chinese
   rose 12.5%.
5. **KV pool: 524,288 to 786,432 tokens** (three full 262K contexts). This uses the 3.9 GB freed by FP8 draft
   weights and pk12.

## Quality

| Check | Before | October 5 |
|---|---|---|
| GSM8K chat, thinking off, greedy (1,314 scored) | 0.9673 | **0.9696** |
| GSM8K chat, thinking xhigh, 200 questions, recommended sampling | 0.980 | **0.985** |
| Ledger retrieval at 8,336 / 120,144 / 240,144 prompt tokens | | pass / pass / pass |
| Two concurrent 24K requests; two simultaneous 240K requests | | pass |
| Degeneration probe (36 generations) | | 0 degenerate |
| Smoke: 131,072-token output cap, forced and automatic tool calls, vision, 120K retrieval, 4 requests at concurrency 2 | | pass |

Prefill is unchanged: 24K tokens in 1.51 s, 96K in 6.37 s. Prefill is power-limited at 580 W, so a GPU that has
just been under load measures about 3% slower.
