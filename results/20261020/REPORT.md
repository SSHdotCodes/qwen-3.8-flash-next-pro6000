# October 20: kern2, 303.8 tok/s and 125.2 verify steps/s

The second decode-kernel round improves the installed October 10 control from **259.4 to 303.8 tok/s**
(+17.1%) and **111.1 to 125.2 verify steps/s** (+12.7%), measured in the same round on one RTX PRO 6000
Blackwell, SM120, 96 GB. The selected configuration is `i12`.

**Decode-path KLD was not re-measured for kern2.** The October 10 value, 0.01112655 against BF16 over
45,610 positions, belongs to the previous release. It is not a kern2 quality result. Copy probes were
also not run because their reference data was unavailable.

This release packages the installed runtime's source files and launch settings. The measurements below
are the recorded workstation results, not a new test of an image built from this public checkout.
That public build and its GPU checks remain pending in [RELEASE_NOTES_DRAFT.md](../../RELEASE_NOTES_DRAFT.md).

## Measurement method

The benchmark runs one streaming chat request at a time, four public synthetic profiles (code, agent,
prose, Chinese), two seeds per profile, and a 4,096-completion-token limit. It includes reasoning tokens,
uses `xhigh` reasoning, enables and preserves thinking, and explicitly sets temperature 1.0, top_p 0.95,
top_k 20, min_p 0, presence penalty 0 and repetition penalty 1. Two short READY requests warm the server;
prefix state is flushed before each measured request. The candidate uses production server arguments
on a separate localhost port, with the selected source/env overrides.

`bench/decode_bench.py` contains the same prompts, seeds and client timing protocol:

```bash
python3 bench/decode_bench.py kern2 --port 30010 --runs 2 --tokens 4096 \
  --profiles code,agent,prose,chinese --efforts xhigh --prefill "" --out results/local
```

Run only against a dedicated idle server; the harness flushes its prefix cache.

The first timestamp is the first nonempty content or reasoning chunk. The last is stream completion:

```text
decode_tok_s = (completion_tokens - 1) / (end_time - first_nonempty_chunk_time)
prefill_tok_s = prompt_tokens / (first_nonempty_chunk_time - request_start)
```

The headline is the mean of the two per-request decode rates in each profile, then the equal-weight
mean of the four profiles. Streaming interval 4 makes this a chunk-timed client measurement.
All eight responses ended at the token cap. This measures sustained decode, not task completion,
natural response length or total task latency.

Verify steps/s is the median of single-request server-log throughput divided by acceptance length,
using `bench/step_median.py` over the matching benchmark window. The final window contained 335 eligible
log lines. Repeat noise reported for this statistic was about ±0.5 steps/s. Mean logged acceptance length
was 2.405; its aggregation differs from both the median step rate and the HTTP mean, so multiplying them
need not recover the headline.

## Results

| Profile | Prompt tokens | Run 1 tok/s | Run 2 tok/s | Mean tok/s |
|---|---:|---:|---:|---:|
| Code: thread-safe TTL LRU cache | 80 | 320.97 | 311.66 | 316.31 |
| Agent: durable SQLite task queue | 92 | 278.20 | 293.20 | 285.70 |
| Prose: speculative decoding and rejection sampling | 90 | 294.24 | 284.13 | 289.19 |
| Chinese: transaction isolation | 91 | 285.32 | 362.48 | 323.90 |
| **Equal-weight mean** | | | | **303.78** |

The exact per-run timings are in [decode-summary.json](decode-summary.json), with generated text removed.
The two Chinese runs illustrate acceptance/workload variation. These are eight observations, not a
confidence interval or a general speed guarantee.

The earlier [October 10 public build check](../20261010/REPORT.md#public-build-check) reported 264.5 tok/s
and 111.0 steps/s. Its public image used three runs per profile after a restart. The same-round control
for the improvements claimed here is **259.4 tok/s / 111.1 steps/s**, not that earlier public-image result.

| Configuration | Increment | Mean tok/s | Median steps/s |
|---|---|---:|---:|
| October 10 installed control | Previous release | 259.4 | 111.1 |
| dA | Draft NVFP4 A16 experts | 261.1 | 112.4 |
| i1 | HC mix/combine tuning | 264.8 | 114.0 |
| i2 | QSA store/pack and candidate-only draft sampling | 273.5 | 117.6 |
| i4 | Initial screened target head | 278.8 | 119.4 |
| i6 | Final target-head kernels | 281.5 | 120.2 |
| i7same | Free unused draft FP8 expert copies | 283.9 | 120.3 |
| i8 | Speculative bookkeeping fusion | 287.5 | 121.0 |
| i9 | Candidate sampling on draft-extend graph | 286.0 | 120.9 |
| i10 | Two-stage draft hot-head screen | 289.0 | 122.0 |
| i11 | GDN output stream preset, without replay | 281.2 | 121.8 |
| **i12** | **i10 + GDN replay and matching output preset** | **303.8** | **125.2** |

These are incremental experiments, not independent additive effects. GDN replay alone and the new
GDN output alone were flat end to end. Their paired configuration improved 122.0 to 125.2 steps/s (+2.6%).
A single-launch router/MoE experiment was slower (106.6 vs 65.8 µs); tail weight prefetch added 1–2 µs.
Extra attention-index changes and draft HC8 mixes were not selected.

## Runtime and kernel changes

The image and checkpoint pins remain SGLang 0.5.21 and
`RedHatAI/Qwen3.8-Flash-Next-NVFP4@c8f2fb1b9869f686b214782036123b10ff96d14a`.
Target routed experts remain W4A16, dense weights remain grouped INT8 where selected, PLE remains BF16
in pinned host RAM, KV remains FP8 E4M3, and recurrent state remains BF16. NEXTN uses three sequential
draft steps and four verification tokens with exact rejection sampling. Context is 262,144, concurrency
two, and the installed KV pool remains **786,432 tokens**.

- **HC:** `qf_hc_fast.cu` and `qf_hc_g96fast.cu` tune mix/combine scheduling for the real neighboring kernels,
  preserving rounding and reduction order. The installed Python dispatcher chooses the variant by module.
- **QSA:** `norm_store.py`, `attn_kernels.py` and the attention overlays fuse norm/RoPE/KV store and count packing, and cache immutable top-k row starts.
  Unselected optional index prep/MQA/linear helpers are included because they are in the installed runtime; their flags remain off.
- **Target head:** a high-nibble INT4 screen with a per-row sigma bound (z=8) selects candidates for exact W8M
  rescoring. Noncandidates receive negative infinity. It applies only when all requests have top_k 1–56 and
  no logprobs, grammar, penalties, min_p or logit bias. Other modes use the dense-head fallback.
- **Draft experts:** BF16 MTP experts are converted offline to NVFP4, Marlin-packed, and used by the persistent
  A16 kernel for small calls and Marlin for larger calls. Shared experts use grouped INT8. Freeing unused FP8
  expert copies saves approximately 2.5 GB. The command line still requests FP8 draft quantization; the overlay
  replaces the routed expert path at load time.
- **Draft sampling/head:** sample directly from the 64 screened candidates, including draft-extend replay.
  At M=1, an INT2 screen retains a superset of top-4608, followed by INT4 rescoring and exact INT8 row logits.
  M=2–8 retains the existing INT4 screen. The hot map is unchanged at 97,800 IDs.
- **Speculative bookkeeping:** fuse sequence/bonus/extend preparation, elide the scheduler relay when safe,
  fuse QSA graph metadata and retain stacked draft probabilities outside the shared graph pool.
- **GDN:** record per-token deltas during verification and replay only the accepted prefix at commit, keeping
  the original BF16 state. Tree/unsupported cases keep snapshot fallback. The paired output kernel computes
  gated RMSNorm once and tunes projection occupancy and dependent release for replay.

The target screen reproduced exact top-k sets and bitwise candidate logits on 602/602 captured inputs.
That empirical sigma-bound coverage is not a proof against omitted top-k entries on all inputs. The draft
screens are also empirical; misses there affect proposal/acceptance, while correct rejection sampling
preserves the chosen target distribution. Target-screen omissions require a fresh audit because they
would affect the target distribution itself.

## Quality and capacity checks

These are recorded October 20 checks. The `fin12` text-quality suite used the same kernels with an
experimental **1,048,576-token** KV pool. Those observations are not fresh validation at the installed
786,432-token pool. The vision tests below used the retained 786,432-token pool.

| Check | Recorded kern2 result | Limit |
|---|---|---|
| GSM8K chat, thinking off, all 1,314 | 0.9695586 | `fin12`, 1M pool |
| GSM8K chat, thinking on, 200 | 0.985 | Subset, 1M pool |
| Retrieval | 8K / 120K / 240K pass | Synthetic ledger, 1M pool |
| Concurrency | Two 24K and two 240K requests pass | Correctness gate, 1M pool |
| Degeneration | 0/36; median decode 372.31 tok/s | Different workload from headline, 1M pool |
| Two streams | 410.0 aggregate tok/s; 209.7 mean per stream | Different aggregation/workload, 1M pool |
| Paired tool tasks | Control 83/120; kern2 88/120 | 20 definition tasks broken for both |
| Corrupt values / malformed tool arguments / errors | 0 / 0 / 0 for both | Paired tool suite |
| Vision | Kern2 8/8; control 7/8 | Retained 786K pool |
| Decode-path KLD / copy probes | **Not run** | Previous release values cannot establish kern2 quality |

The tool suite's 20 definition tasks depended on an unavailable reference source tree. The useful
paired comparison is **88 vs 83 on the other 100 tasks**. It is not comparable to the earlier 112/120
or October 10 109/120 results.

Cold-prefill client observations from `fin12`:

| Actual input tokens | TTFT | Prompt tokens / TTFT |
|---:|---:|---:|
| 8,336 | 0.702 s | 11,883 tok/s |
| 120,144 | 9.579 s | 12,543 tok/s |
| 240,144 | 21.075 s | 11,395 tok/s |

A 1M KV pool passed the long text checks but a 4096×4096 image exhausted ViT memory and killed the
scheduler. Capacity therefore stays at 786K. At that size, the recorded tests passed single 128²,
2048² and 4096² images, four and eight 2048² images, and two concurrent 4096² image requests. The
control returned an OOM for eight 2048² images but survived; kern2 passed. Reported free memory after
graphs rose from 6.18 to 7.46 GB. These specific cases do not establish arbitrary multimodal admission.

## Correctness evidence and public test ports

Work-package reports record byte-identical HC outputs on 288 real cases across M=1–8; attention output
on six real captures and short synthetic pipelines; exact head top-k/candidate logits on 602 captures;
ordered INT8 draft top-20/proposals on 602 captures; BF16/FP32 GDN replay states and outputs; and
4,423,680 BF16 GDN-output comparisons with zero bit differences. Longer attention selections keep the
same sets but can inherit the original indexer's nondeterministic atomic ordering.

Repeated unchanged-control greedy generations differed on all 12 tested prompts. Therefore greedy
string equality was not used as a full-model gate. It does not replace the per-kernel tests or KLD.

Public GPU test ports are documented in [runtime notes](../../docs/runtime.md#october-20-kern2).
They retain the synthetic work-package checks and adapt loading to the integrated extension. Private
activation captures and raw generated text are not included. The ports were syntax-checked locally;
they have **not** been run on a GPU as part of this packaging work.

## Reproduction and outstanding checks

`serve/runtime.json` pins all shipped overlay bytes, kernel compilation units and supporting headers,
the public base image, measured local image/library digests, model revision, measured draft artifact
hash, preparation source, environment and server arguments. Every overlay matches the installed
manifest byte for byte; the kernel sources/bindings match the installed source directory. Rebuilt
extension bytes can differ because build paths and tooling affect them. The measured local image
digest is not a public registry image.

No model weights or compiled library are redistributed. `serve/prepare-draft-a16.sh` explicitly builds
the draft artifact on an idle GPU from the pinned model. Its parameterized recipe uses synthetic
validation rather than private captures. Torch serialization may yield a different file hash; neither
binary equality nor model-level quality of the regenerated artifact is established by this release
preparation. See the draft notes for the required checks before publication.

## TODO: October 26 update

**Placeholder only. No October 26 runtime, benchmark or quality claim is made here.** The orchestrator
will add tonight's measured configuration, methods, results, KLD/copy-probe status and GPU verification
outcome, and replace the TODO release-table row after its work finishes.
