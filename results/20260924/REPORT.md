# Qwen3.8 Flash Next decode kernels — 2026-09-24

The September 24 release adds CUDA kernels for the decode and verify steps ("qwenfast", [runtime notes](../../docs/runtime.md#decode-kernels-2026-09-24)). Everything else is the same as the September 19 runtime: the RadixArk NVFP4 checkpoint at `7b719225242aacd3dbd3f9407468c2ee9a9d2594`, BF16 dense and draft weights, FP8 E4M3 KV cache, 262,144-token context, 524,288 allocated tokens, two request slots, NEXTN three steps with four verify tokens and exact rejection sampling, the 65,800-ID draft map, and the same launcher arguments. Prefill paths are unchanged.

## Same-day sustained measurements

The previous release image (`local/qwen-flash-next:0.5.20-20260919-hotmap`, `sha256:5dc299354cb1e285803c15392a8ba5947051d0a3fced1b183ced4ab264755c51`, built on 2026-09-19; its manifest label matches `serve/runtime.json` of the previous release, `5f4ce09`) and this release's image (`local/qwen-flash-next:0.5.20-20260924-qwenfast`, `sha256:82f053fbd605f9fbfd4d6bd7c3e7b28a216e06ae600ae26e71be8eac58e827ac`, built from this checkout) ran one after the other on the same machine, each on a freshly started server with the same launcher. Protocol: `bench/sustained.py --runs 3 --profiles code,agent,prose,chinese --flush-cache`, 8,192 generated tokens per request, `xhigh` reasoning, temperature 1.0, top-p 0.95, top-k 20, fixed seeds. Rates include reasoning tokens and exclude time to first token.

| Workload | Sept 19 image | Sept 24 image | Change |
|---|---:|---:|---:|
| Python cache implementation (`code`) | 168.1 tok/s | 190.0 tok/s | +13.0% |
| SQLite queue implementation (`agent`) | 164.1 tok/s | 192.7 tok/s | +17.4% |
| Technical writing (`prose`) | 153.8 tok/s | 179.9 tok/s | +17.0% |
| Chinese explanation/code (`chinese`) | 155.9 tok/s | 179.4 tok/s | +15.1% |
| **Mean of the four** | **160.5 tok/s** | **185.5 tok/s** | **+15.6%** |

Per request (tok/s): Sept 19 code 165.8, 165.3, 173.1; agent 166.2, 162.0, 164.2; prose 152.4, 157.0, 151.8; chinese 154.5, 163.2, 150.0; Sept 24 code 188.2, 189.8, 192.1; agent 193.2, 194.7, 190.1; prose 172.1, 177.7, 190.0; chinese 182.7, 178.0, 177.6. No request collapsed.

Each verify step emits the accepted draft tokens plus one, so per-request rates move with how many draft tokens the sampled text accepts. The steady measure is verify steps per second, the logged throughput divided by the logged acceptance length over the benchmark's decode log lines: **68.1 → 79.1 steps/s (+16.2%)**, at acceptance lengths of 2.34 and 2.33. The gain is time per step; the profiler put a four-token verify step at 14.3 ms before and 12.6 ms after.

The development measurements on the workstation's installed runtimes, with the same protocol on an earlier run, gave 165.5 → 187.3 tok/s (+13.2%) and 69.9 → 79.4 steps/s (+13.6%).

## Correctness and quality

Measured on the workstation's installed September 24 runtime, the same sources as this image (`validation-summary.json`):

| Check | Sept 19 runtime | Sept 24 runtime |
|---|---:|---:|
| GSM8K, chat, non-thinking, greedy, all 1,319 | 0.9665 | **0.9688** |
| GSM8K, chat, thinking `xhigh`, T 1.0 / top-p 0.95 / top-k 20, first 200 | 0.990 | **0.990** |
| Retrieval at 8,336 / 120,144 / 240,144 prompt tokens | | pass / pass / pass |
| Two concurrent 24,070-token requests | | both pass |
| Decode after a 24K prompt, 2,048 tokens, two runs | 165.8 / 165.8 tok/s | 189.6 / 195.9 tok/s |
| Decode after a 96K prompt, 2,048 tokens, two runs | 166.1 / 168.3 tok/s | 184.4 / 189.9 tok/s |
| Greedy degeneration probe, 36 generations | | 0 degenerate |

GSM8K ran through `sglang.test.run_eval` against the server (`gsm8k.json`). It is a sanity check that the kernels did not break the model, not a broad quality evaluation.

Kernel tests (`kernel-tests/`, run on the GPU against one layer of the pinned checkpoint):

- **Verify sampler** (`test_sample.json`): with identical random coins, the sparse sampler and SGLang's dense path (FlashInfer top-k/top-p renormalization plus `chain_speculative_sampling_triton`) gave identical accept counts, accepted tokens and final tokens in 12,000 of 12,000 randomized cases (60 configurations × 200 trials). The cases cover random temperatures, top-k 1–56, top-p on and off, and tied logits. Re-run inside the image built from this checkout on the day of the comparison: the same 12,000 of 12,000 (`kernel-tests/test_sample-image.log`).
- **NVFP4 MoE** (`test_moe.json`): relative L2 error against FlashInfer's CUTLASS path about 0.003, the same distance from a BF16-activation reference as FlashInfer, and deterministic. The outputs are not bit-identical to FlashInfer: summation order differs.
- **Hyper-connections, router and GDN output** (`test_hc2.json`, `test_router.json`, `test_gdnout.json`): the router picks the same experts as the stock path in every case; the hyper-connection and GDN output kernels match the stock path to within BF16 rounding (relative error at most 3.3e-4, 98.8–100% of outputs identical after rounding to BF16), and all of them are deterministic.

## Reproducibility

`serve/Dockerfile` compiles the kernels inside the pinned SGLang image (its PyTorch 2.13.0+cu130 and nvcc 13.0, `-gencode arch=compute_120a,code=sm_120a`); no GPU is needed to build. `serve/verify-runtime.py` checks the SHA-256 of the thirteen overlay files and the nine kernel sources, before the build and again inside the image. The compiled library's bytes differ from the workstation's own build because the build path is embedded in it; the measurements above used the library compiled by this Dockerfile.

`QWENFAST=0` in the container environment turns off every replacement of an SGLang function. The draft head's small-row GEMM stays on: it only computes draft proposals, and the target verifies every token. `QWENFAST_MOE`, `QWENFAST_HC`, `QWENFAST_GDN`, `QWENFAST_SAMPLE` and `QWENFAST_QSAMETA` switch off one each.

## Files

- `old-20260919.json`, `new-20260924.json`: every request of the same-day comparison, with settings (`*-settings.json`) and the server's configuration (`*-server-info.json`).
- `summary.json`: the table above, computed from those files and the server logs.
- `validation-summary.json`, `gsm8k.json`, `context-retrieval.json`, `concurrent-24k.json`: the quality and correctness checks.
- `kernel-tests/`: kernel test output.
