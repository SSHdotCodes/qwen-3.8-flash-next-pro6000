# Qwen3.8-Flash-Next on one RTX PRO 6000

Single-stream decode at **180–226 tok/s** for `Qwen3.8-Flash-Next` on **one**
RTX PRO 6000 Blackwell (SM120, 96 GB), at the **full 262,144-token context**,
with the **quantization unchanged** (NVFP4 W4A4 experts, BF16 KV cache).

Baseline for the same checkpoint and card is ~105 tok/s. Everything here is
config plus a 14-line upstream compatibility patch — no retraining, no
re-quantization, no second GPU.

| workload | baseline | this repo |
|---|---|---|
| coding (1024 tok out) | 105 | **216.9** |
| agent tool-call (1.7K prompt) | 99 | **225.8** |
| agent + 9 tools (2.4K prompt) | 99 | **210.1** |
| plain prose | 105 | **180.5** |
| reasoning / thinking mode | 104 | **136.6** |

Median decode tok/s over 3 runs, `temperature=0`, single request, measured with
`bench/bench.py`. Raw output in [`results/`](results/). A 216,685-token prompt
answers coherently in 8.4 s.

## Quick start

```bash
git clone https://github.com/SSHdotCodes/qwen-3.8-flash-next-pro6000
cd qwen-3.8-flash-next-pro6000

./serve/patches/build-qsa-patch.sh    # one-time, needs docker
./serve/run-server.sh                 # first run downloads ~126 GiB

# in another shell
python3 bench/rate.py  --iters 6      # correctness gate — must report 0
python3 bench/bench.py --tag mine     # speed
```

Requirements: one RTX PRO 6000 Blackwell (or another SM120 96 GB card), rootless
Docker with the NVIDIA Container Toolkit, ~130 GiB disk, and ~50 GiB free host
RAM (the PLE n-gram embedding table is offloaded to the host).

`serve/qwen38-flash-next-sglang.service` is a systemd **user** unit for
run-forever operation. It binds to loopback only; put an authenticated proxy in
front of it before exposing it anywhere.

## What actually produces the speedup

### 1. NEXTN / MTP speculative decoding (the whole win)

The checkpoint ships an MTP module. Two things about it are widely assumed and
both are wrong:

- **It does not cost 4.37 GB.** SGLang quantizes the MTP module to NVFP4 on
  load. It costs **0.51 GB**:

  ```
  Load weight end. type=Qwen4ExpForCausalLMMTP, quant_algo=NVFP4, mem usage=0.51 GB
  ```

- **It does not force you to shrink the context.** With MTP on and
  `--max-total-tokens` left unset, the auto-sized KV pool is *larger* than the
  pinned-262K baseline (330,944 tokens), with 3.39 GB still free.

Measured accept length is **2.3–3.9 out of a maximum 4**, accept rate 0.86–0.96.

**QSA hard-caps the draft chain at 4 tokens.** `num_draft_tokens` must be `<=`
the indexer compress ratio; `--speculative-num-steps 5 --speculative-num-draft-tokens 6`
refuses to start with:

```
NotImplementedError: Qwen QSA requires speculative_num_draft_tokens <= the QSA
compress ratio (4): the pending index-key ring holds one group; got 6
```

So `steps 3 / topk 1 / draft-tokens 4` is the deepest usable linear chain.

`--enable-linear-replayssm-spec` was tested and **rejected**: 186.1 vs 185.7
tok/s (noise), and it warns about recurrent-state drift with
`--mamba-ssm-dtype bfloat16`.

### 2. `--disable-flashinfer-autotune` (correctness, not speed)

This one **costs** ~8% raw decode and is here anyway, because without it the
model silently produces garbage.

FlashInfer autotunes `trtllm::fused_moe::gemm1` / `gemm2` by **latency only —
it never checks numerics**. The selected tactic ids are **numbered per problem
shape** (gemm2 ids are offset by that shape's gemm1 tactic count), while the
autotune cache is keyed on the tuned shape and replayed on neighbouring shapes.
An id tuned for one shape can therefore address a different kernel config.

The visible failure is decode collapsing into a run of token id 0, which this
tokenizer renders as `!`:

```
To understand how a modern Mixture-of-Experts (Mo!!!!!!!!!!!!!!!!!!!!!!!!!!!!...
```

Controlled A/B on the same build, 1024-token greedy generations:

| config | corrupted generations | tok/s |
|---|---|---|
| autotune on (picked gemm1=17, gemm2=57) | **36/36** and **18/18** | 104.9 |
| skip `trtllm::fused_moe::gemm1` only | 0/15 | 98.4 |
| `--disable-flashinfer-autotune` | **0/36** | 96.6 |

That the id space is genuinely shape-dependent is easy to confirm: pin
`gemm2=57` across shapes by hand and SGLang refuses to boot with

```
Check failed: (id2 >= mGemm1TacticCount && id2 < mGemm1TacticCount + mGemm2TacticCount)
    is false: Invalid gemm2 profile id: 57
```

The silent corruption is that same defect without the bounds check.

There is a **second, smaller effect** that is not fully explained: with the
fused-MoE ops skipped but autotune otherwise enabled, corruption drops to 3/30
and is always in the *first* iteration after startup, then never recurs.
`--disable-flashinfer-autotune` also skips the dummy warmup forward and is the
only setting measured at 0. Threading the attention backend's
`on_after_cuda_graph_warmup` into the target-side autotune path (the
speculative-draft path already does this as `post_warmup_hook`) did **not** fix
it. Root cause still open — see [Open questions](#open-questions).

Verified clean across 78+ long generations after the fix.

### 3. The SM120 QSA compatibility patch

`serve/patches/qwen_sparse_attn_backend.sm120.patch` — 14 lines. The pinned
day-0 image routes the QSA fallback through the pip `flash-attn` CUTE SM100
kernel, which fails to compile for the packed QSA decode shape on SM120; the
patch routes it to SGLang's own SM120 FA4 dispatcher in the same image.

Upstream: [issue #36531](https://github.com/sgl-project/sglang/issues/36531),
fix in [PR #36556](https://github.com/sgl-project/sglang/pull/36556) (which
takes a better approach — TRTLLM sparse decode first, FA4 as fallback). Once
that ships in a released image, drop the patch and the `-v` mount.

## Why 200+ tok/s needs speculative decoding and nothing else

At the ~9.5 ms/token baseline the model is already running at **~69% of the
card's HBM roofline**, so there is no kernel tuning left that gets to 200.

Per decoded token the card must read roughly:

- ~1.5 GB of NVFP4 expert weights (10 routed + 1 shared of 512 experts, 48 layers)
- ~10 GB of BF16 dense weights — attention and linear-attention projections,
  hyper-connection mixers, and the 248,320 x 2560 LM head

≈ 11.8 GB per token. At ~1.79 TB/s that is a ~6.6 ms floor, against 9.5 ms
measured. The dense BF16 weights dominate, not the quantized experts.

The only way past that at batch 1 is to **amortize one weight read over several
tokens**, which is exactly what MTP does. Anything else — better MoE kernels,
attention backends, CUDA graph tweaks — is competing for the remaining ~30%.

## Measuring this honestly

`bench/rate.py` is a **correctness gate**, and it exists because of two traps
that make this model look fine when it is not.

**Short generations hide the corruption.** On a server that was 36/36 corrupt at
1024 output tokens, 256-token requests were 0/48 clean. Any probe under a few
hundred tokens is not evidence of anything. Always gate on long generations.

**Do not call `/flush_cache` twice around a generation.** That sequence on its
own corrupts the mamba extra-buffer state and causes the same `!` collapse
(2/8 in a controlled run), independently of the autotune bug. `bench.py`
defeats the prefix cache with a per-request nonce instead.

Also note **greedy is not bit-deterministic here** — repeated identical
`temperature=0` requests return differently-worded answers, so exact-match
comparison between configurations is not a usable quality metric. Use the
corruption rate plus per-workload throughput.

## Repository layout

```
serve/run-server.sh          one-command launcher (loopback only)
serve/*.service              systemd user unit for run-forever
serve/patches/               SM120 QSA patch + builder
bench/bench.py               per-workload speed + captured output
bench/rate.py                corruption-rate gate over N long generations
bench/compare.py             diff two bench result files
results/*.json               raw measurements behind the tables above
```

Useful `bench/bench.py` profiles: `speed`, `coding`, `thinking`, `agent`
(OpenCode-shaped: system prompt + 9 tool schemas + file context), `agent_tool`.

## Open questions

- Root cause of the residual first-iteration corruption when the autotune
  warmup forward runs at all.
- Whether a *correct* fused-MoE tactic exists that is also faster than the
  heuristic fallback, which would return the ~8%. The tactic id space being
  shape-dependent makes hand-pinning unsafe.
- Whether upstream PR #36556's TRTLLM-first QSA decode path changes any of the
  above.

Issue reports and measurements welcome. Please include `bench/rate.py` output
with any performance claim.

## Credits and licence

Built on [SGLang](https://github.com/sgl-project/sglang) and
[FlashInfer](https://github.com/flashinfer-ai/flashinfer). The patch in
`serve/patches/` is a derivative of SGLang source (Apache-2.0). This repository
is Apache-2.0; see [LICENSE](LICENSE).

Model weights are governed by their own licences:
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and
the NVFP4 checkpoint
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).
