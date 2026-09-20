# Qwen3.8 Flash Next throughput experiments — 2026-09-19

Installed the smaller draft vocabulary and the validated QSA buffer change; production smoke checks passed.

The selected experiment has a modest fixed-length throughput gain. Completed code-repair timings were mixed, and reasoning length varied substantially. These results do not establish a broad improvement in whole-task completion time.

## Preserved configuration

One RTX PRO 6000 96 GB; SGLang 0.5.20. The original NVFP4 checkpoint, BF16 dense/draft components, FP8 E4M3 KV cache, 262,144-token context, 524,288-token allocated cache, and two-request limit were retained. Target sampling stayed at temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1, and xhigh reasoning. Exact rejection sampling and the proposal-probability clone fix remained enabled.

Model revision: `7b719225242aacd3dbd3f9407468c2ee9a9d2594`. Image: `sha256:a2198845e5f69cf887d3bcccf6dacb5962a77c493cb3eae68a61357a1c67ef9b`.

## Repeated sustained measurements

Single requests generated 8,192 tokens each. Rates include reasoning tokens and exclude time to first token. English cells have three observations for each configuration across separate starts; Chinese has one control and three candidate observations.

| Workload | Original tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|
| code | 171.4 | 177.2 | +3.4% |
| agent | 166.9 | 172.0 | +3.0% |
| prose | 149.3 | 155.9 | +4.4% |
| chinese | 157.6 | 164.1 | +4.2% |

English geometric-mean gain: **3.6%**. Exploratory bootstrap interval: 2.3% to 5.1% for these workloads only. This small sample is not a general performance guarantee.

## Experiments

| Candidate | Coding | Queue implementation | Writing |
|---|---:|---:|---:|
| 16-slot QSA buffer, original three-step MTP | 173.5 | 162.0 | 151.9 |
| Five-step MTP with six-token verification | 154.4 | 152.2 | 141.7 |
| 65,800-token BF16 draft head, first screen | 183.7 | 170.3 | 156.4 |
| Smaller draft head + FlashInfer verify + draft sharpening | 179.5 | 175.3 | 156.6 |
| Smaller draft head, confirmation runs | 173.9 | 172.8 | 155.7 |

Longer drafts lost throughput. The combined FlashInfer/draft-sharpening experiment was essentially tied with the simpler head change and was not selected. Its draft temperature multiplier affected proposals only; target sampling stayed unchanged. No relaxed acceptance or additional quantization was used.

The hot-token map scores 65,800 draft IDs, including all 276 special IDs. The full 248,320-token target head is retained. Excluded draft IDs have zero proposal probability; the unchanged exact rejection verifier can still emit them. Map source and SHA-256 are recorded in `hotmap-source.json` and [runtime notes](../../docs/runtime.md).

## Completed tasks and correctness

| Repair | Original tok/s | Candidate tok/s | Original total time | Candidate total time | Checks |
|---|---:|---:|---:|---:|---|
| retry | 184.8 | 179.9 | 19.4s | 55.3s | Passed |
| migrate | 171.3 | 178.0 | 125.5s | 138.5s | Passed |

Both configurations completed the rate-limiter clock-rollback repair and SQLite transaction repair and passed all nine independent checks. Candidate outputs used 9,924 and 24,631 tokens; the control used 3,559 and 21,489. Different reasoning lengths make whole-task timing variable. Generated code was tested in rootless containers with no network, no GPU, read-only filesystems, no Linux capabilities, and CPU/memory/PID limits.

Two earlier from-scratch implementation prompts each exhausted 32,768 tokens in reasoning without returning code. Both are retained as failed completions, not successes.

The candidate returned the correct values at 8,336, 120,144, and 240,144 input tokens. Those short retrieval responses are correctness checks, not sustained-speed measurements. Two concurrent requests also passed separate account-code retrieval checks.

The 16-slot QSA pending buffer passed 4,026 adversarial state/rejection cases and 48 CUDA graph replays. The original four-slot layout failed 248 of the corresponding width-four cases by overwriting a prior partial compression group. Compression ratio and context capacity were not changed.

Direct FlashInfer verification checks on the actual 16 key heads / 48 value heads / 128 head dimension passed four batch/window combinations after correcting the test to use SGLang’s batch-indexed scratch layout. Maximum output difference was about 3.1e-5 and intermediate-state difference about 0.00195. This is limited numerical evidence; the extra kernel override was not selected.

## Observed runtime costs

The GPU remained fully busy at roughly 400 W, below its 580 W limit. A fresh profile showed about 1,865 kernels in each target verification graph, with a roughly 11.6–11.9 ms GPU span, plus about 1.6 ms for drafting and 0.9 ms for draft extension. BF16 matrix operations were prominent. Longer drafts increased work enough to erase their acceptance benefit. These measurements identify current costs; they do not prove that further optimization is impossible.

## Evidence and reproduction

The JSON files in this directory preserve the original request outputs, reasoning,
token counts, finish reasons and timings for the synthetic prompts. No workstation
catalogs, private deployment paths, credentials or unrelated model data are included.

- Control: `baseline.json` (two runs per English workload) + `control-final.json`
  (one run per English workload and one Chinese run).
- Candidate: `hotmap-results.json` (one English run), `hotmap-chinese.json`
  (one Chinese run), `selected-final.json` (two runs per workload).
- Screens: `wide3.json`, `wide5-results.json`, `combined-results.json`.
- Repairs: `control-repair-real.json`, `hotmap-repair-real.json`, and
  `completed-repairs/` containing both generated modules and executable checks.
- Failed from-scratch completions: `baseline-real.json`.
- Retrieval/concurrency: `hotmap-context.json`, `selected-final-concurrent.json`.
- Buffer and numerical checks: `qsa-wide-ring-tests.json`, `gdn-verify-tests.json`.
- Aggregation and original qualification checks: `final-comparison.json`.
  Its qualification is for this narrow tuning decision, not a broad quality claim.

Run `python3 bench/summarize_sustained.py results/20260919` from the repository
root to recompute the means and geometric-mean gain from raw request records.
[bench/sustained.py](../../bench/sustained.py) preserves the prompts, sampling,
seeds and streaming timing calculation; [bench/repair_tasks.py](../../bench/repair_tasks.py)
contains the focused repair prompts. Use `--flush-cache` to match these runs.

The rate is `(completion_tokens - 1) / (end - first_output_chunk)`. Stream interval
was four, so this is a streaming approximation to decode speed. All fixed-length
comparison outputs reached 8,192 tokens and ended at the length cap. These
implementation prompts measure sustained generation, not completed coding quality.
Natural completion was evaluated separately in the repair tasks. Request seeds
are fixed, but reuse of workloads and limited repetitions constrain generalization.

The current launcher reproduces the selected settings. Installed source hashes
are in [serve/runtime.json](../../serve/runtime.json); runtime changes and the
public image build are explained in [docs/runtime.md](../../docs/runtime.md).
Actual production endpoint checks on September 19 verified SGLang 0.5.20, all
installed source hashes, 262,144 context and 524,288 allocated cache tokens, and
correct answers to two short smoke prompts. The GPU was offloaded afterward.

Prepared seven-step and torch.compile configurations were never executed and are
not counted as tested experiments. No model weights, drivers or system packages
were changed during the throughput tuning. The earlier SGLang update is a
separate change; the controlled comparison here used 0.5.20 on both sides.
