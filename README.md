# Qwen3.8-Flash-Next on one RTX PRO 6000

SGLang **0.5.21** setup for one RTX PRO 6000 Blackwell (SM120, 96 GB): the
RedHatAI NVFP4 checkpoint, **262,144-token context**, two request slots, a
786,432-token FP8 KV pool and exact speculative rejection sampling.

The October 20 **kern2** runtime records **303.8 tok/s and 125.2 verify steps/s**, up from
259.4 tok/s and 111.1 steps/s for the October 10 installed control measured in the same round.
It adds draft A16 experts, HC/QSA/head/speculative bookkeeping improvements and paired GDN replay/output
kernels. **Decode-path KLD was not re-measured for kern2.** On October 26 the image built from this checkout
reproduced the installed runtime's decode rate after a restart (125.0 verify steps/s) and, with the FlashInfer tuning
now shipped in `serve/tuning/`, its prefill after one full run and a restart; see the [October 26 report](results/20261026/REPORT.md).

The October 10 public image measured 264.5 tok/s and 111.0 verify steps/s, with decode-path KLD
0.011127 against BF16. Those are historical October 10 measurements, not kern2 quality evidence.

The October 9 release decoded at 243 tok/s, up from 199.5 on October 5. It also moved
closer to the original model: full-vocabulary KLD against BF16 fell from 0.0178 to 0.0111,
because the routed experts run with BF16 activations (W4A16) instead of 4-bit
ones. The agent tool-use bench rose from 105 to 112 of 120.

| Release | Decode tok/s | Verify steps/s | KLD vs BF16 | Prefill 32K | Report |
|---|---:|---:|---:|---:|---|
| RedHatAI checkpoint, before tuning | 183.4 | 79.4 | | | [Oct 5](results/20261005/REPORT.md) |
| October 5: exact 12-bit dense weights, FP8 draft, 97,800-ID draft map | 199.5 | 87.0 | 0.0175 | 2.03 s | [Oct 5](results/20261005/REPORT.md) |
| October 9: INT8 dense weights, truncated draft proposals, new kernels | 244.4 | 103.1 | 0.0178 | 2.04 s | [Oct 9](results/20261009/REPORT.md) |
| October 9: W4A16 routed experts | 243.4 | 100.8 | 0.0111 | 2.37 s | [Oct 9](results/20261009/REPORT.md) |
| **October 10: persistent W4A16 MoE, tensor-core INT8 GEMM, fused HC combine** | **264.5** | **111.0** (same-day baseline 102.0) | **0.0111** | | [Oct 10](results/20261010/REPORT.md) |
| **October 20: kern2 (current packaged runtime)** | **303.8** | **125.2** (same-round control 111.1) | **Not re-measured** | Not re-measured at 32K | [Oct 20](results/20261020/REPORT.md) |
| October 26: public image built from this checkout (kern2 runtime unchanged) | 291.3 | 125.0 (installed 125.0) | Not re-measured | 2.38 s (installed 2.37 s) | [Oct 26](results/20261026/REPORT.md) |

Decode protocol: `bench/decode_bench.py`, single requests of 4,096 generated tokens
including reasoning, four profiles (code, agent, prose, Chinese), `xhigh`,
recommended sampling (`temperature=1.0`, `top_p=0.95`, `top_k=20`). Per-request rates move with
how many draft tokens the sampled text accepts; verify steps per second is the steadier
measure. October 20 uses **two** runs per profile and the installed runtime; October 10's public
image check uses three. October 10 KLD uses 45,610 positions against true BF16 logits; kern2 has no new KLD result.

The October 10 row differs in three ways:

- Its step rates are medians over server log lines (`bench/step_median.py`), with the October 9 build
  re-measured the same day at 102.0. The earlier rows use `bench/accept_window.py`, so compare 111.0 with
  102.0, not with 100.8.
- Its tok/s comes from the public image's release check (three runs per profile, after a restart); the
  October 9 public image measured 240.9 tok/s the same way. The workstation build itself ran two runs per
  profile (see the report).
- Its KLD is the decode-path measurement, because the new kernels do not run during prefill. Prefill is unchanged
  and was not re-measured.

Sustained 1,536-token generations on the production server's smoke check ran at 332-376 tok/s with the
October 10 build, against 324-337 tok/s for the October 9 build.

The October 9 W4A16 precision trade remains relevant to prefill. It is about 15% slower (128K tokens: 8.8 s to 10.1 s), because
Marlin's W4A16 expert GEMMs run on BF16 tensor cores where the W4A4 path used FP4. The [runtime notes](docs/runtime.md#october-20-kern2) describe the selected switches and fallback paths.
Do not infer a kern2 32K prefill result from the older precision comparison.

Recorded kern2 quality checks:

- GSM8K chat: 0.96956 with thinking off (1,314), 0.985 with thinking on (200).
- Paired tool tasks: 88 vs control 83 on 100 usable tasks. Another 20 definition tasks were broken for both;
  the nominal 88/120 vs 83/120 is not comparable to the older tool scores.
- Retrieval at 8K, 120K and 240K; two concurrent 24K and 240K requests: pass.
- Degeneration: 0/36; two-stream aggregate throughput 410.0 tok/s on a separate workload.
- Vision at the retained 786K pool: 8/8 vs control 7/8.
- Decode-path KLD and copy probes: **not run for kern2**.

The text-quality checks used the same kernels with an experimental 1M KV pool, which was rejected after
vision OOM. The installed pool stays at 786,432; the vision checks above used that pool. The
[October 20 report](results/20261020/REPORT.md) separates those configurations and their limits.

The September releases (SGLang 0.5.20, RadixArk checkpoint, 185.5 tok/s on a sustained
8,192-token protocol) are documented in [results/20260924](results/20260924/REPORT.md) and
[results/20260919](results/20260919/REPORT.md). They used a different protocol, so do not compare them directly to the table above.

## Quick start

Requirements:
- Linux with one RTX PRO 6000 96 GB.
- Rootless Docker with NVIDIA Container Toolkit, and Python 3.
- About 180 GB of disk for the checkpoint, plus the SGLang image and caches.
- About **110 GB of free host RAM**: the original BF16 n-gram (PLE) table, 102.4 GB, is kept in pinned host memory.

The measured machine used driver 615.71.09. Container CUDA components are 13.0;
the host toolkit version does not replace the container's dependencies.

```bash
git clone https://github.com/SSHdotCodes/qwen-3.8-flash-next-pro6000
cd qwen-3.8-flash-next-pro6000
./serve/build-image.sh       # compiles the decode kernels; no GPU needed
./serve/download-model.sh    # pinned checkpoint, then the target/draft views
./serve/prepare-draft-a16.sh # explicit GPU preparation on an idle machine, no server running
./serve/run-server.sh        # foreground; http://127.0.0.1:30010
```

The image is **built locally** from the pinned official SGLang v0.5.21 image, with pinned dependency
repairs and the exact source overlays. It is not a published Docker Hub image.
Building checks the source hashes, the draft token map and the installed package
versions. The measured workstation image digest and the public base digest are recorded in
[serve/runtime.json](serve/runtime.json).

`download-model.sh` fetches `RedHatAI/Qwen3.8-Flash-Next-NVFP4` at revision
`c8f2fb1b9869f686b214782036123b10ff96d14a`. It then runs `serve/prepare-model.py`, which builds two
directories of symlinks next to the snapshot, so no weight file is copied or changed:
- A target view: QSA blocks relabelled for SGLang 0.5.21, the BF16 PLE table kept, the routed-experts-only quantization scope.
- A draft view: the native MTP weights only.

`prepare-draft-a16.sh` then reads that pinned draft view on the GPU and writes `draft-a16.pt` plus a
provenance JSON beside the views. It builds NVFP4 expert weights and INT8 shared-expert weights for the
draft override; no weights are redistributed here. The public conversion uses synthetic validation
and still needs GPU verification against the measured artifact and full runtime. Its archive bytes may
differ from the installed reference hash recorded in `serve/runtime.json`.

Environment overrides, used the same way for build, download and serve:
- `HF_CACHE` (default `$HOME/models/huggingface`)
- `MODEL_VIEWS` (default `$HOME/models/qwen38-flash-next-redhat`)
- `RUNTIME_CACHE` (default `$HOME/.cache/sglang-flash-next/0.5.21`)
- `IMAGE`
- `DRAFT_A16` (default `$MODEL_VIEWS/<revision>/draft-a16.pt`, used by preparation and serving)

`PORT` and `NAME` are optional launch overrides. `DRY_RUN=1 ./serve/run-server.sh` prints the exact command without
starting Docker.

**The first start is slower.** It compiles and tunes Triton/Inductor kernels into `RUNTIME_CACHE`. Loading took about
6 minutes. In the October 9 release check, that first server run also decoded about 4% slower: 96.5 verify steps/s against
100.7. After a restart with the populated cache it measured 240.9 tok/s and 100.7 steps/s, matching the measured
workstation's 100.8. Prefill settles the same way: in the October 26 check an empty cache measured 2.45 s at 32K
until one full benchmark run and a restart, then 2.38 s. `run-server.sh` also seeds FlashInfer's tuned MoE tactics
from `serve/tuning/`; without them prefill stayed about 6% slower. Run one full workload and restart before comparing numbers.

For systemd, clone to `~/qwen-3.8-flash-next-pro6000`, build, download and prepare the draft artifact first,
then install the [user unit](serve/qwen38-flash-next-sglang.service):

```bash
mkdir -p ~/.config/systemd/user
cp serve/qwen38-flash-next-sglang.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start qwen38-flash-next-sglang
# Stop and release GPU memory when finished:
systemctl --user stop qwen38-flash-next-sglang
```

The unit calls the same launcher, so its flags cannot drift from the shell
example. Docker publishes only the HTTP port to host loopback, with a separate
container network namespace. Internal SGLang IPC is not exposed through host
networking. Use an authenticated proxy before offering inference remotely.

## Current runtime and sampling

| Setting | Value |
|---|---|
| Checkpoint | RedHatAI/Qwen3.8-Flash-Next-NVFP4 |
| Revision | `c8f2fb1b9869f686b214782036123b10ff96d14a` |
| Routed experts | Checkpoint NVFP4 weights, run as **W4A16** through SGLang's Marlin MoE kernels (`QWEN_NVFP4_MARLIN=1`); decode batches of 1-8 tokens use a persistent qwenfast kernel on the same Marlin-packed weights, with the shared expert inside it (`QWENFAST_A16MOE=1`) |
| Dense target weights | BF16 checkpoint weights; decode reads them as **group-scaled INT8** (one BF16 scale per 128 weights; `QWENFAST_W8`) on tensor cores (`QWENFAST_W8M=1`, `QWENFAST_W8M_PDL=1`) |
| PLE / n-gram table | Original BF16, pinned host RAM |
| KV cache | FP8 E4M3, 786,432 allocated tokens |
| Context / concurrency | 262,144 per request / maximum two requests |
| MTP | NEXTN, three steps, top-k 1, four verify tokens; CLI FP8, with routed experts overridden by NVFP4 A16 and INT8 shared experts |
| Draft proposals | The draft's own top-k / top-p truncation (`QWEN_DRAFT_TRUNC=1`); exact rejection against the full target distribution |
| Draft vocabulary | 97,800 IDs (incl. all 276 special IDs), INT8 head with INT2/INT4 two-stage screening at M=1, INT4 at M=2–8 |
| Target vocabulary | 248,320 IDs; screened exact W8M candidate logits for eligible top-k requests, dense fallback otherwise |
| Linear attention | Triton prefill/verification; FlashInfer decode |
| Recurrent state | BF16 with replay-on-commit, `extra_buffer`, track interval 64, cache size 14 |
| Other retained settings | PLE CPU offload, page 64, prefill chunk 4,096, graph batch 2 |
| Decode kernels | `qwenfast` for 1-8-token batches (SM120a): persistent W4A16 MoE, NVFP4 MoE (W4A4 mode), INT8 / exact 12-bit dense GEMMs, fused router, hyper-connections (combine fused into the next mix, `QWENFAST_HCCOMB=1`), GDN replay/output, head screening, top-k and sampling. Use a previous release checkout for a controlled rollback |

The benchmark sends these request fields explicitly:

```json
{
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.0,
  "reasoning_effort": "xhigh",
  "chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true}
}
```

`--sampling-defaults model` does not force clients to send these settings.
Draft-side changes (A16/INT8 draft experts, FP8 draft linears, the hot-token head and truncated proposals) change
how many draft tokens are accepted, never the output distribution. Exact rejection
uses the full target distribution, including IDs absent from the draft map. There is
no relaxed acceptance or draft temperature sharpening. The target-side changes are:
- W4A16 experts: the same weights with more precise activations.
- INT8 dense weights: a small measured KLD cost, 0.01756 to 0.01784 with W4A4 experts and 0.01064 to 0.01109 with W4A16 experts.
- October 10 decode kernels: the same weights and formats in new kernels; decode-path KLD 0.011134 to 0.011127.
- October 20 target head screen and GDN replay/output: kernel evidence is recorded; decode-path KLD was not re-measured.

Floating-point behaviour and sampled reasoning length still vary.

## Patches and validation

The [runtime notes](docs/runtime.md) cover:
- SM120 QSA/FP8 handling, the 16-slot pending ring and the proposal-probability clone fix;
- startup loading and the draft hot-token map;
- the decode kernels and the October changes.

The kernel sources/bindings, unchanged draft token map and 25 of 27 Python overlays match the installed
kern2 files byte for byte. Two overlays intentionally differ: a coordinated public config-key rename
and a comment edit, listed in the [public/installed differences](docs/runtime.md#intentional-publicinstalled-differences). `serve/runtime.json` pins 26 compilation units plus the build script, header
and optional support sources. The image compiles the extension itself, so library bytes may differ
from the recorded workstation binary. The October 26 check built this image, regenerated the draft A16 artifact
(tensor-identical to the measured one) and reran the decode and prefill benchmarks; the
[October 26 report](results/20261026/REPORT.md) lists what was and was not re-verified.

The target screen uses a sigma bound tested on 602 captured inputs. Candidate logits were bitwise equal
and top-k sets matched there; fresh workloads still need an omission audit. Requests with logprobs,
grammar, penalties, min_p, logit bias or unsupported top-k use the dense fallback.

Kernel checks:
- The exact 12-bit dense format is unpacked and compared bit for bit at every load.
- The INT8 hyper-connection and GDN-output kernels are bit-exact against the BF16 kernels run on the
  dequantized weights.
- The radix top-k matches `torch.topk`.
- The persistent W4A16 MoE kernel is checked against SGLang's Marlin MoE path on 12 checkpoint layers.
- The fused hyper-connection combine-mix and the October 10 router are bit-identical to the October 9 kernels.
- The kernel tests in `serve/qwenfast/` can be run inside the image.

These are correctness checks for specific paths; the KLD, agent, GSM8K and retrieval
results in the reports are the model-level evidence.

## Reproduce measurements

On an idle model server, in another shell:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r bench/requirements.txt
# October 20 protocol (two runs per profile):
python3 bench/decode_bench.py mine --port 30010 --runs 2 --tokens 4096 \
  --profiles code,agent,prose,chinese --efforts xhigh --prefill "" --out results/local
# Step rate from the server log (October 10/20 statistic; pass a UTC window to restrict it):
docker logs qwen38-flash-next-sglang > results/local/server.log 2>&1
python3 bench/step_median.py results/local/server.log
# September protocol:
python3 bench/sustained.py mine --runs 3 --profiles code,agent,prose,chinese --flush-cache
python3 bench/repair_tasks.py repairs --flush-cache
python3 bench/summarize_sustained.py results/20260919
```

`--flush-cache` reproduces the measured protocol and clears server prefix state;
use it only on a dedicated idle server. New results go to ignored `results/local/`.
The repair harness runs generated code in rootless containers with no network,
no GPU, a read-only filesystem and CPU/memory/PID limits. It preserves the
sampled output, elapsed time and executable-check results.

`bench/bench.py` and `bench/rate.py` retain the historical greedy protocol.
The `!`-collapse heuristic is only a narrow degeneration check. Inspect
completion status and execute task-specific checks before making speed claims.

## Credits and licence

Built on [SGLang](https://github.com/sgl-project/sglang) and
[FlashInfer](https://github.com/flashinfer-ai/flashinfer). SGLang overlays retain
their upstream licence; this repository is Apache-2.0, see [LICENSE](LICENSE)
and [NOTICE](NOTICE). The hot-token IDs derive from
[gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120),
with the source checksum and special-token union documented in the runtime notes.

Model weights are downloaded separately and governed by their own licences:
[Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and
[RedHatAI NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-Flash-Next-NVFP4)
(the September releases used [RadixArk NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)).
