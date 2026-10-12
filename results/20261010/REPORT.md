# October 10: decode kernels, 102.0 to 111.0 verify steps/s

One change landed on October 10: new decode kernels on top of the October 9 W4A16 build. Verify steps per second
rose from **102.0 to 111.0** (+8.8%) against the October 9 build measured the same day. Output quality is unchanged:
decode-path KLD against BF16 went from 0.011134 to 0.011127, the agent bench from 106 and 108 to 109 and 109 of 120,
and GSM8K with thinking off stayed at 0.968.

The kernels run only for decode and verify batches of 1-8 rows. Prefill, the checkpoint, quantization, KV format,
context, concurrency, sampling and exact rejection sampling are unchanged.

## Decode

Benchmark: `bench/decode_bench.py` against a server started with the production launch flags: 4,096 tokens for each
of 4 profiles (code, agent, prose, chinese), xhigh, recommended sampling, one request at a time. Each server start
ran two runs per profile; the October 9 baseline ran three.

Steps/s is the median over the server's log lines of generation throughput divided by accept length
(`bench/step_median.py`). It counts verify steps per second, independent of how many draft tokens the sampled text
accepts. Each configuration was started two to four times, interleaved: A, B and C in turn, twice; then D
alternating with a rerun of C; then D and E alternating. Repeats agreed within 0.5.

| Build | steps/s, each repeat |
|---|---|
| October 9 build (W4A16 experts, INT8 dense weights) | 102.0 |
| A: + persistent W4A16 MoE decode kernel, shared expert on a side stream | 104.8 / 104.8 |
| B: + shared expert inside the MoE kernel, tensor-core INT8 dense GEMM (w8m) | 106.6 / 106.8 |
| C: + hyper-connection combine fused into the next mix | 109.1 / 109.0 |
| D: + router staging sized to the batch; the MoE kernel waits lazily, so shared-expert weights load while the router runs | 110.4 / 110.0 / 109.9 / 110.2 |
| **E: + w8m launched with programmatic dependent launch (current)** | **110.9 / 111.0** |

E against the October 9 build: 102.0 to 111.0 verify steps/s, +8.8%.

These step rates use a different statistic from the October 5 and 9 reports, which divide mean throughput by mean
accept length (`bench/accept_window.py`). Compare them with the 102.0 baseline above, not with the 100.8 in the
October 9 report.

The same runs give mean per-request decode rates of 246.0 tok/s for the October 9 build (three runs per profile)
and 255.9 tok/s for E (two runs per profile). Per-request rates move with how many draft tokens the sampled text
accepts, so the three-run release protocol was run on the public image instead: 264.5 tok/s (see
[Public build check](#public-build-check)).
After installation, the production server's smoke check ran six sustained 1,536-token generations at
332-376 tok/s; the same check gave 324-337 tok/s for the October 9 build.

### What changed

1. **Persistent W4A16 MoE decode kernel** (`qf_a16moe.cu`, `QWENFAST_A16MOE=1`).
   - One persistent kernel computes gate/up (split into two K halves), SiLU-mul, down, the top-k weighting, the sum
     over the 10 routed experts and the sigmoid-gated shared expert. It runs 2 CTAs per SM with an ordered work
     queue and per-expert completion counters; the last CTA to finish a tile does its reduction.
   - It reads the same Marlin-packed NVFP4 weights as SGLang's Marlin path, with Marlin's exact FP4 / FP8-scale
     dequantization, BF16 activations and FP32 accumulation.
   - Error against an FP32 reference at 4 rows: 5.5e-3, against 5.8e-3 for the Marlin path.
   - Microbenchmark (12 layers rotated past L2, 4 rows, 21.5 unique experts): 46.8 µs against 54.6 µs.
2. **Shared expert inside the MoE kernel** (`QWENFAST_A16SHARED=1`, on by default with `QWENFAST_A16MOE=1`).
   - It uses the w8g INT8 arithmetic; its 80 gate/up and 40 down work items are queued first.
   - The router no longer shares the GPU with a side-stream shared expert, and the stream join is gone.
   - It matches the w8g path to a maximum absolute difference of 3e-5 at 4 rows.
   - The kernel issues its weight loads before it waits on the router (a lazy `griddepcontrol.wait`). Every path
     waits before it touches counters or scratch, and back-to-back launches were verified equal to single calls.
3. **w8m: tensor-core INT8 dense GEMM** (`qf_w8g.cu`, `QWENFAST_W8M=1`).
   - The group-scaled INT8 (w8g) weights run on `mma.m16n8k16`; INT8 to BF16 conversion is exact. A fixed
     in-block k permutation lets each thread load 16 contiguous bytes.
   - x is read from L2 once per CTA instead of once per weight row from shared memory.
   - At 4 rows: in_proj 30.7 to 27.8 µs, qkv 26.4 to 25.2 µs, lm_head 396 to 387 µs. The error against the FP32
     reference is the same as w8g.
4. **Hyper-connection combine fused into the next mix** (`qf_hc8.cu` `hc_combine_mix_w8`, `QWENFAST_HCCOMB=1`,
   up to 4 rows).
   - The attention combine runs inside the MLP mix, and the MLP combine inside the next layer's attention mix.
     This covers 46 of 48 layers; it does not cross the PLE layer.
   - Bit-identical to combine followed by mix (`test_hc8c.py`).
   - Slower in isolation (10.6 µs against 9.2 µs) but +2.3% end to end, because the mix now chains straight
     after the MoE or attention kernel.
5. **Router** (`qf_router.cu`, part of `QWENFAST_ROUTER=1`).
   - Token staging is sized to the batch (20 KB at 4 rows instead of a static 40 KB), so the router fits beside the
     mix within SM120's 100 KB of shared memory per SM. Top-k uses `redux.sync`.
   - Outputs are bit-identical to the October 9 build on 728 cases, including ties and zeros, compared in separate
     processes (`test_router3.py`).
6. **w8m with programmatic dependent launch** (`QWENFAST_W8M_PDL=1`).
   - w8m (74 registers, 4.6 KB of shared memory) fits beside the HC mix on every SM. The first wave of in_proj and
     qkv weight requests now overlaps the mix's up-projection phase.

### Measured and not adopted

- Prefetching w8g weights before staging x: in_proj went from 31.7 to 42.5 µs. The October 9 kernel was kept.
- Early launch (PDL) of the October 9 w8g kernel: slower, 99.8 against 104.6 steps/s.
- Profiling trap: a profile that charges overlapping kernels by completion order hid the router's ~10 µs per layer
  behind the side-stream shared expert. Raw kernel durations and offsets from the preceding kernel showed it.
- Mean throughput over mean accept length swung 2-3% from run to run; the per-line median reproduced within 0.4
  steps/s. That is why this report uses the median.

## Quality

The October 10 build (E) against the October 9 build, on the same day with the same harnesses:

| Check | October 9 build | October 10 build |
|---|---|---|
| Agent tool-use bench, 60 tasks x thinking off/on, 2 concurrent, 2 runs each | 106, 108 /120 (thinking off 46, 48) | 109, 109 /120 (thinking off 49, 49) |
| GSM8K chat, thinking off, all 1,314 | 0.968 (October 9) | 0.968 |
| GSM8K chat, thinking on, 200 | 0.99 (October 9) | 0.98 (2 questions) |
| Code-copying probes 1 / 2, mean P(full code); mostly the prefill path, whose kernels are unchanged | 0.944 / 0.823 (October 9) | 0.924 / 0.822 |
| Retrieval at 8K / 120K / 240K | pass (decode 318 / 327 / 306 tok/s, October 9) | pass (decode 347 / 356 / 359 tok/s) |
| Two concurrent 24K and two concurrent 240K requests | pass | pass |
| Degeneration probe, 36 cases | 0 degenerate (median 298 tok/s) | 0 degenerate (median 329 tok/s) |
| Two streams (4,096 tokens, xhigh, 2 pairs x 2 runs): per-stream / aggregate tok/s | 178.8 / 337.3 | 208.0 / 396.4 (+16%) |

"(October 9)" marks October 9 build results carried over from the October 9 measurements rather than rerun. The
two-stream check runs the code and agent profiles together, then prose and chinese, at batch size 2.

The agent bench needed a fix. Between the October 9 and October 10 runs, its working directory moved behind a
symlink. Its directory-listing and file-reading tools compared resolved paths against the unresolved root, so every
call failed. The definition tasks scored 0 of 10 with thinking off, and the October 9 build scored 95/120 overall.
The runs above use a copy of the bench with the root resolved. Because the October 9 build scored 106 and 108 with
that copy, these scores compare with each other, not with the 112/120 in the October 9 report.

### Decode-path KLD

The KLD harness scores logits from prefill, which this release does not change. To score the decode kernels, both
builds ran with the same evaluation-only patch, which production never uses. It runs every changed token-wise
operation on 4-row slices of the prefill, the row count of a single-request verify step:
- the MoE block through its decode branch: the router plus either the Marlin experts or the persistent kernel with
  the in-kernel shared expert;
- the INT8 dense GEMMs with K = 2,560 and 6,144 (w8g in the old build, w8m in the new one);
- the HC mix and lm_head.

Attention and GDN stay on the prefill path; they are unchanged. The HC combine-mix fusion only fires at up to 4 rows
within a layer, so it is not exercised here; it is bit-exact against combine followed by mix. GDN out_proj is sliced
through w8m here, while decode uses the unchanged fused GDN norm + out_proj kernel. The new build is therefore tested
on slightly more than it runs.

Measured from 21:08 to 21:25 EDT on October 10. Each run scores 45,610 positions over 22 prompts, as full-vocabulary
KLD against true BF16 logits (`kld/`):

| Run | KLD | p99 | Top-1 agreement with BF16 |
|---|---:|---:|---:|
| October 9 build, prefill path (recorded 0.011089 on October 9) | 0.011087 | 0.1158 | 0.9698 |
| October 9 build, decode path | 0.011134 | 0.1189 | 0.9695 |
| **October 10 build, decode path** | **0.011127** | **0.1153** | **0.9689** |

Paired per position on the decode path, new minus old: KLD -0.000008 (-0.07%), 95% block-bootstrap interval
[-0.00026, +0.00028] (256-position blocks). The two builds pick the same top-1 token at 98.61% of positions. The
October 9 build's own prefill and decode paths agree at 98.59%. So the new kernels move outputs no more than the old
build's two paths differ from each other, and KLD against BF16 is unchanged. No request was skipped in any run.

## Kernel tests

New tests in `serve/qwenfast/` (GPU):
- `test_a16moe.py`: the persistent MoE kernel against SGLang's Marlin MoE path on 12 checkpoint layers, with the
  production repack, plus timing.
- `test_w8m.py`: w8m against w8g and an FP32 reference at 1, 4 and 8 rows, plus timing.
- `test_hc8c.py`: the fused combine-mix kernel against combine followed by mix, bit for bit.
- `test_router3.py`: router and HC mix outputs, the October 9 library against the new one in separate processes.

## Publication

The kernel sources, the `sglang/srt/qwenfast.py` overlay and the launch flags in this repository are byte-identical
to the installed runtime. Of the SGLang overlays, only `qwenfast.py` changed in this release. The installed runtime
runs the same SGLang 0.5.21 image as October 9, with the new library and overlay mounted in.

## Public build check

The published files were checked on the measuring workstation. `serve/build-image.sh` built the image from the pinned
SGLang v0.5.21 base, compiling the kernels in 71 s with `MAX_JOBS=16`, and `verify-runtime.py --installed` passed
inside it. `serve/run-server.sh` served from that image with a fresh runtime cache and the existing target and draft
views of the same checkpoint revision:
- A chat request answered correctly.
- A tool call returned well-formed arguments.
- 32K-token prefill took 2.36 s.

`decode_bench.py`, run with the decode protocol (3 runs of each profile) after one restart, as the README recommends:

| Run | code | agent | prose | chinese | mean tok/s | steps/s (median, `step_median.py`) | steps/s (`accept_window.py`) | accept len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| October 10 public image, restart | 287.0 | 261.7 | 252.8 | 256.4 | 264.5 | 111.0 | 109.6 | 2.372 |
| October 9 public image, restart ([report](../20261009/REPORT.md#public-build-check)) | 252.8 | 239.6 | 239.6 | 231.7 | 240.9 | | 100.7 | 2.361 |

The median step rate matches the release measurement (111.0). By the October 9 statistic, verify steps/s rise
from 100.7 to 109.6 (+8.8%) between the two public images, and decode tok/s from 240.9 to 264.5 (+9.8%) at almost
the same accept length. No request collapsed.
