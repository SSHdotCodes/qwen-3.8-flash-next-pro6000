# Qwen3.8-Flash-Next on one RTX PRO 6000

SGLang **0.5.21** setup for one RTX PRO 6000 Blackwell (SM120, 96 GB): the
RedHatAI NVFP4 checkpoint, **262,144-token context**, two request slots, a
786,432-token FP8 KV pool and exact speculative rejection sampling.

The October 10 release adds decode kernels to the October 9 build: **111.0 verify steps/s, up from
102.0** (+8.8%, both builds measured the same day), with **quality unchanged**: decode-path KLD against
BF16 0.011134 to 0.011127. The public image built from this checkout decodes at **264.5 tok/s** with the
decode protocol below, against 240.9 for the October 9 public image measured the same way.

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
| **October 10: persistent W4A16 MoE, tensor-core INT8 GEMM, fused HC combine (current)** | **264.5** | **111.0** (same-day baseline 102.0) | **0.0111** | | [Oct 10](results/20261010/REPORT.md) |

Decode protocol: `bench/decode_bench.py`, single requests of 4,096 generated tokens
including reasoning, four profiles (code, agent, prose, Chinese) with three runs each, `xhigh`,
recommended sampling (`temperature=1.0`, `top_p=0.95`, `top_k=20`). Per-request rates move with
how many draft tokens the sampled text accepts; verify steps per second is the steadier
measure. KLD is measured over 45,610 positions against true BF16 logits.

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

The trade in the current build is prefill. It is about 15% slower (128K tokens: 8.8 s to 10.1 s), because
Marlin's W4A16 expert GEMMs run on BF16 tensor cores where the W4A4 path used FP4. Removing
`QWEN_NVFP4_MARLIN=1` from `serve/run-server.sh` restores the W4A4 experts. Removing
`QWENFAST_A16MOE=1 QWENFAST_W8M=1 QWENFAST_W8M_PDL=1 QWENFAST_HCCOMB=1` returns decode to the
October 9 kernels; see the [runtime notes](docs/runtime.md#october-10-changes).

Quality checks for the current build:
- Agent tool-use bench: 109 and 109 of 120. The October 9 build scored 106 and 108 the same day, on a
  corrected harness ([why](results/20261010/REPORT.md#quality)).
- GSM8K: 0.968 (thinking off, all 1,314) and 0.980 (thinking on, 200).
- Retrieval at 8K, 120K and 240K tokens: pass.
- Two concurrent 24K and two concurrent 240K requests: pass.
- Degeneration probe: 0 of 36 degenerate.
- Decode-path KLD against BF16: 0.011127; the October 9 build scored 0.011134.

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

Environment overrides, used the same way for build, download and serve:
- `HF_CACHE` (default `$HOME/models/huggingface`)
- `MODEL_VIEWS` (default `$HOME/models/qwen38-flash-next-redhat`)
- `RUNTIME_CACHE` (default `$HOME/.cache/sglang-flash-next/0.5.21`)
- `IMAGE`

`PORT` and `NAME` are optional launch overrides. `DRY_RUN=1 ./serve/run-server.sh` prints the exact command without
starting Docker.

**The first start is slower.** It compiles and tunes Triton/Inductor kernels into `RUNTIME_CACHE`. Loading took about
6 minutes. In the October 9 release check, that first server run also decoded about 4% slower: 96.5 verify steps/s against
100.7. After a restart with the populated cache it measured 240.9 tok/s and 100.7 steps/s, matching the measured
workstation's 100.8. Restart once after the first start before comparing numbers.

For systemd, clone to `~/qwen-3.8-flash-next-pro6000`, build and download first,
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
| MTP | NEXTN, three steps, top-k 1, four verify tokens; FP8 draft weights |
| Draft proposals | The draft's own top-k / top-p truncation (`QWEN_DRAFT_TRUNC=1`); exact rejection against the full target distribution |
| Draft vocabulary | 97,800 IDs (incl. all 276 special IDs), INT8 head screened by an INT4 copy |
| Target vocabulary | Full 248,320 IDs |
| Linear attention | Triton prefill/verification; FlashInfer decode |
| Recurrent state | BF16, `extra_buffer`, track interval 64, cache size 14 |
| Other retained settings | PLE CPU offload, page 64, prefill chunk 4,096, graph batch 2 |
| Decode kernels | `qwenfast` for 1-8-token batches (SM120a): persistent W4A16 MoE, NVFP4 MoE (W4A4 mode), INT8 / exact 12-bit dense GEMMs, fused router, hyper-connections (combine fused into the next mix, `QWENFAST_HCCOMB=1`), GDN output, top-k and sampling. `QWENFAST=0` restores the stock SGLang paths |

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
Draft-side changes (FP8 draft weights, the hot-token head, truncated proposals) change
how many draft tokens are accepted, never the output distribution. Exact rejection
uses the full target distribution, including IDs absent from the draft map. There is
no relaxed acceptance or draft temperature sharpening. The target-side changes are:
- W4A16 experts: the same weights with more precise activations.
- INT8 dense weights: a small measured KLD cost, 0.01756 to 0.01784 with W4A4 experts and 0.01064 to 0.01109 with W4A16 experts.
- October 10 decode kernels: the same weights and formats in new kernels; decode-path KLD 0.011134 to 0.011127.

Floating-point behaviour and sampled reasoning length still vary.

## Patches and validation

The [runtime notes](docs/runtime.md) cover:
- SM120 QSA/FP8 handling, the 16-slot pending ring and the proposal-probability clone fix;
- startup loading and the draft hot-token map;
- the decode kernels and the October changes.

The 13 overlay files, the draft token map and the 17 kernel source files match the workstation's production
sources, with two exceptions: one config key is renamed and one comment is edited. The image compiles the kernels itself, so the
library's bytes differ from the workstation build (the build path is embedded). The October 10 image was rebuilt
and served from this checkout on the measuring workstation ([public build check](results/20261010/REPORT.md#public-build-check)).

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
# October protocol (decode table above):
python3 bench/decode_bench.py mine --port 30010 --runs 3 --tokens 4096 \
  --profiles code,agent,prose,chinese --efforts xhigh --prefill "" --out results/local
# Step rate from the server log (October 10 statistic; pass a UTC window to restrict it):
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
