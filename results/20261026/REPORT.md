# October 26: public build check and tuning screens

This release ships the October 20 kern2 runtime unchanged. It adds two things:

- a check that the image built from this checkout reproduces the installed runtime's speed;
- a set of tuning screens run against that runtime. None beat it, so none were adopted.

All runs used one RTX PRO 6000 Blackwell (SM120, 96 GB), single-stream requests, and recommended sampling (`temperature=1.0`, `top_p=0.95`, `top_k=20`, `xhigh`). Each config ran the four profiles (code, agent, prose, Chinese) with 4,096 generated tokens per request, two runs per profile unless noted.

- **tok/s** is the mean decode rate over the profiles.
- **steps/s** is the median verify-step rate over server log lines (`bench/step_median.py`).
- **accept** is the mean accepted tokens per verify step over the same window.

## Public image built from this checkout

The image was built from this checkout with `serve/build-image.sh` (`MAX_JOBS=8`). The in-image `verify-runtime.py --installed` passed: 28 runtime files, 32 kernel sources and 97,800 draft IDs, with installed package versions matching. The two intentional overlay differences are listed in `docs/runtime.md`.

`serve/prepare-draft-a16.sh` regenerated the draft A16 experts with this image. They are **tensor-identical** to the artifact behind the October 20 measurements: all 10 entries are bit-equal, and only the serialized file bytes differ.

### Decode

The image was started with `serve/run-server.sh` and given the same four-profile benchmark as the installed runtime:

| Run | tok/s | steps/s | accept |
|---|---:|---:|---:|
| Installed runtime (control, same night) | 296.0 | 125.0 | 2.361 |
| Public image, first start (empty caches) | 289.6 | 122.9 | 2.351 |
| Public image, after a restart | 291.3 | **125.0** | 2.325 |

After one restart, decode verify steps per second match the installed runtime exactly. The remaining tok/s difference follows each run's sampled acceptance. Chat and tool calls answered correctly, and none of the 16 generations of 4,096 tokens was degenerate.

### Prefill and compile caches

Prefill depends on tuned kernel choices that persist in the runtime cache (`RUNTIME_CACHE`, mounted at `/root/.cache/{sglang,triton}`). Cold prompts, time to first token:

| Cache state | 8K | 32K | 128K |
|---|---:|---:|---:|
| Installed runtime (settled workstation cache) | 0.59 s | 2.37 s | 10.15 s |
| Public image, empty cache, first and second start | 0.63 s | 2.51–2.52 s | 10.61–10.68 s |
| Public image, empty cache plus the shipped FlashInfer tuning, first and second start | 0.61–0.62 s | 2.45 s | 10.43–10.53 s |
| Public image, empty cache plus the shipped FlashInfer tuning, after one full benchmark run and a restart | **0.59 s** | **2.38 s** | **10.15–10.22 s** |
| Public image with a copy of the workstation's cache | 0.59 s | 2.38 s | 10.15 s |

`serve/tuning/flashinfer-autotune/` holds FlashInfer's tuned fused-MoE tactics for this GPU, FlashInfer 0.6.18 and CUDA 13.0. `run-server.sh` copies them into the runtime cache and never overwrites existing entries. From an empty cache, reaching the workstation's prefill takes the shipped tuning plus one full serving run and a restart. The tuning alone gives 2.45 s at 32K on the first and second start. A full serving run (the four-profile decode benchmark, prefill and a tool call) followed by a restart then gives 2.38 s, matching the settled cache. Without the shipped tuning, the image's own cache stayed at 2.51 s after the same full run and restart. Decode on that seeded cache measured 125.0 verify steps/s (289.2 tok/s, no degenerate outputs) and the tool call was correct.

**Before comparing numbers, run one full workload on a new cache and restart.**

## Tuning screens against the installed runtime

| Config | tok/s | steps/s | accept | Outcome |
|---|---:|---:|---:|---|
| Control (installed kern2 runtime) | 296.0 | 125.0 | 2.361 | baseline for this round |
| 4 draft steps, 5 verify tokens | 270.2 | 106.8 | 2.514 | rejected: +0.15 accepted tokens per step for 15% slower steps |
| 2 draft steps, 3 verify tokens | 285.9 | 136.8 | 2.086 | rejected: faster steps, but acceptance falls more |
| Draft temperature x0.8 (3 runs) | 289.7 | 124.8 | 2.313 | rejected |
| Draft temperature x0.9 (3 runs; Chinese profile lost to an unrelated host OOM) | 293.7 (3 profiles) | 125.2 | 2.354 | rejected |
| Draft temperature x1.15 (3 runs) | 292.7 | 124.9 | 2.335 | rejected |

The draft-temperature screens scale only the temperature of the draft's proposal distribution. Rejection sampling still verifies against the exact distribution each proposal was drawn from, so the output distribution is unchanged and only the acceptance rate can move. The model's own temperature (x1.0) was best. This control's 296.0 tok/s, against 303.8 recorded on October 20, comes from its lower sampled acceptance (2.361 vs 2.405). Verify steps per second match (125.0 vs 125.2).

## Prefill

Cold, uncached prompts, two runs per length (`bench.py --prefill`), time to first token:

| Chunked prefill | 8K | 32K | 128K |
|---|---:|---:|---:|
| 4,096 (shipped) | 0.59 s (13.6K tok/s) | 2.37 s (13.5K tok/s) | 10.15 s (12.6K tok/s) |
| 8,192 | 0.60 s | 2.37 s | 10.04 s |
| 16,384 | 0.60 s | 2.38 s | 9.99 s |

Larger chunks save at most 1.6% at 128K and nothing at 8K or 32K. They also need more activation memory beside a full KV pool, so the shipped 4,096 is kept.

## Where the time goes

**Decode step**, profiled on the control at 125 steps/s: 6.74 ms verify plus 1.29 ms draft and sampling. Within the verify:

| Kernel | Time per step |
|---|---:|
| Persistent W4A16 MoE kernel | 2.73 ms |
| Hyper-connection mixes | 1.01 ms |
| GDN input projection (INT8) | 0.74 ms |
| GDN output projection | 0.53 ms |
| Router | 0.46 ms |

The GPU is idle for only 14 µs per verify step. A 4-row verify touches about 28 unique experts per layer (about 85 MB) at roughly 1.4 TB/s. Reaching 350 tok/s at this acceptance would need about 6.9 ms per step, close to the weight-streaming floor. Most of the remaining headroom is in acceptance, that is a better-matched draft, rather than in kernels.

**Prefill**, a 32K prompt in 4,096-token chunks:

| Component | Share of kernel time |
|---|---:|
| Dense BF16 GEMMs (about 55% of tensor peak) | 29% |
| Marlin W4A16 routed experts | 29% |
| Sparse attention | 9% |
| Hyper-connection mixing and elementwise kernels | about 12% |

Faster prefill would need new large-M GEMM kernels. Moving activations to 4 bits would also speed it up, but the October 9 measurements show that raises KLD.

## Limits

- **KLD:** decode-path KLD against BF16 was not re-measured for kern2. The BF16 reference logits used by earlier releases were not available.
- **Not re-run on the public image:** the ported GPU kernel suites (`serve/qwenfast/test_*`); the target-head screen omission audit on fresh workloads; GSM8K, the tool-task bench, long-context retrieval, two-request operation and vision at the shipped 786,432-token pool.
- **Prefill comparisons:** they use the benchmark's synthetic word-list prompts. Real documents tokenize differently.
- **Arithmetic:** the tuning screens change scheduling and proposal settings only; the installed arithmetic is untouched.
