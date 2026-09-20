# Qwen3.8-Flash-Next on one RTX PRO 6000

SGLang **0.5.20** setup for one RTX PRO 6000 Blackwell (SM120, 96 GB), with
**262,144-token context**, two request slots, and exact speculative rejection
sampling. The September 19 tuning improved repeated sustained throughput by
**3.6%** on three fixed English workloads. **300–400 tok/s was not reached.**

| Sustained workload | Before tuning | Current | Change |
|---|---:|---:|---:|
| Python cache implementation | 171.4 tok/s | 177.2 tok/s | +3.4% |
| SQLite queue implementation | 166.9 tok/s | 172.0 tok/s | +3.0% |
| Technical writing | 149.3 tok/s | 155.9 tok/s | +4.4% |
| Chinese explanation/code | 157.6 tok/s | 164.1 tok/s | +4.2% |

These are single-request means over **8,192 generated tokens**, including
reasoning and excluding time to first token. Both sides used SGLang 0.5.20,
`temperature=1.0`, `top_p=0.95`, `top_k=20`, and `xhigh` thinking. English cells
have three observations per configuration; Chinese has one control and three
candidate observations. This is a small fixed workload sample.

Two completed code repairs passed all nine executable checks. The selected
configuration generated at **178–180 tok/s**, but both candidate wall times
were longer in those samples because reasoning lengths changed. Token rate
alone does not establish faster task completion. See the
[full report and raw evidence](results/20260919/REPORT.md).

The previous **180–226 tok/s** headline used short, mostly non-thinking,
**greedy** requests and a different runtime configuration. Those historical
results remain in [results/RESULTS.md](results/RESULTS.md) and the
[archived README](docs/legacy-20260826.md); they are not comparable to this table.

## Quick start

Requirements: Linux, one RTX PRO 6000 96 GB, rootless Docker with NVIDIA
Container Toolkit, Python 3, space for the model (~130 GiB) plus the SGLang
image and caches, and roughly 50 GiB free host RAM for the offloaded PLE table.
The measured machine used driver 615.71.09. Container CUDA components are 13.0;
the host toolkit version does not replace the container's dependencies.

```bash
git clone https://github.com/SSHdotCodes/qwen-3.8-flash-next-pro6000
cd qwen-3.8-flash-next-pro6000
./serve/build-image.sh
./serve/download-model.sh   # pinned checkpoint; resumes existing HF cache
./serve/run-server.sh       # foreground; http://127.0.0.1:30010
```

The image is **built locally** from a pinned official SGLang release, with
pinned dependency repairs and the exact validated source overlays. It is not a
published Docker Hub image. Building checks the source hashes, token map and
installed package versions. The measured workstation image digest and public
base digest are recorded separately in [serve/runtime.json](serve/runtime.json).

`HF_CACHE` defaults to `$HOME/models/huggingface`; `RUNTIME_CACHE` defaults to
`$HOME/.cache/sglang-flash-next/0.5.20-hotmap`. Use the same `IMAGE` and `HF_CACHE`
overrides for build/download/serve. `PORT` and `NAME` are optional launch overrides.
`DRY_RUN=1 ./serve/run-server.sh` prints the exact command without starting Docker.

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
| Checkpoint | RadixArk/Qwen3.8-Flash-Next-NVFP4 |
| Revision | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| Weights | Original NVFP4 routed experts and BF16 dense components; native MTP |
| KV cache | FP8 E4M3, 524,288 allocated tokens |
| Context / concurrency | 262,144 per request / maximum two requests |
| MTP | NEXTN, three steps, top-k 1, four verify tokens |
| Draft vocabulary | 65,800 IDs, including all 276 special IDs |
| Target vocabulary | Full 248,320 IDs |
| Linear attention | Triton prefill/verification; FlashInfer decode |
| Recurrent state | BF16, `extra_buffer`, track interval 64, cache size 14 |
| Other retained settings | PLE CPU offload, page 64, prefill chunk 4,096, graph batch 2 |

The September tuning preserved all precision and capacity settings on both
sides of the comparison. Relative to the original August repo, the current
profile uses FP8 KV instead of BF16 KV and two slots instead of one; the
historical speed figures must not be used as an unchanged-configuration baseline.

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
The smaller head changes draft proposals only. Exact rejection still uses the
full target distribution, including IDs absent from the draft map. There is no
relaxed acceptance, draft temperature sharpening, or additional quantization.
Floating-point behavior and sampled reasoning length can still vary.

## Patches and validation

The [runtime notes](docs/runtime.md) describe the SM120 QSA/FP8 handling,
**16-slot pending ring**, **proposal-probability clone fix**, indexed startup
loading, and draft hot-token map. All eleven installed files match the
workstation's verified production source hashes. Autotune stays enabled;
the old claim that it must always be disabled is superseded.

The selected configuration passed retrieval checks at 8K, 120K and **240K**
input tokens, two concurrent 24K requests, 4,026 adversarial ring cases and 48
CUDA graph replays. Short retrieval responses are correctness evidence, not
sustained throughput evidence. These checks are not a broad model-quality eval.

## Reproduce measurements

On an idle model server, in another shell:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r bench/requirements.txt
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
[RadixArk NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).
