> **Historical greedy measurements (August 2026).** Current recommended-sampling
> results and their limitations are in [20260919/REPORT.md](20260919/REPORT.md).
> The original autotune causal diagnosis below was superseded by later QSA and
> proposal-buffer findings; these files remain an experiment record.

# Raw measurements

All runs: single request, `temperature=0`, RTX PRO 6000 Blackwell (SM120, 96 GB),
SGLang image `sha256:12d3392b...`, checkpoint `RadixArk/Qwen3.8-Flash-Next-NVFP4`.

`corrupt` = the captured generation collapsed into a run of `!` (token id 0).

| result file | profile | decode tok/s | TTFT s | prompt tok | corrupt |
|---|---|---:|---:|---:|---|
| `FINAL-mtp-noautotune.json` | speed | 174.25 | 0.0755 | 59 | no |
| `FINAL-mtp-noautotune.json` | coding | 211.77 | 0.0745 | 51 | no |
| `FINAL-mtp-noautotune.json` | thinking | 145.03 | 0.0952 | 98 | no |
| `FINAL-mtp-noautotune.json` | agent | 198.25 | 0.1143 | 2407 | no |
| `FINAL-mtp-noautotune.json` | agent_tool | 215.73 | 0.0875 | 1692 | no |
| `PRODUCTION-bench.json` | speed | 180.49 | 0.0771 | 59 | no |
| `PRODUCTION-bench.json` | coding | 216.9 | 0.0754 | 51 | no |
| `PRODUCTION-bench.json` | thinking | 136.55 | 0.0993 | 98 | no |
| `PRODUCTION-bench.json` | agent | 210.09 | 0.1156 | 2407 | no |
| `PRODUCTION-bench.json` | agent_tool | 225.78 | 0.0878 | 1692 | no |
| `baseline2.json` | speed | 103.17 | 0.057 | 53 | no |
| `baseline2.json` | coding | 103.17 | 0.0816 | 45 | no |
| `baseline2.json` | thinking | 102.09 | 0.0633 | 92 | no |
| `baseline2.json` | agent | 99.33 | 0.1794 | 2401 | **yes** |
| `baseline2.json` | agent_tool | 98.58 | 0.1362 | 1686 | no |
| `mtp-s3-t1-d4-clean.json` | speed | 190.06 | 0.0762 | 59 | no |
| `mtp-s3-t1-d4-clean.json` | coding | 227.49 | 0.0746 | 51 | no |
| `mtp-s3-t1-d4-clean.json` | thinking | 265.03 | 0.092 | 98 | **yes** |
| `mtp-s3-t1-d4-clean.json` | agent | 273.65 | 0.1178 | 2407 | **yes** |
| `mtp-s3-t1-d4-clean.json` | agent_tool | 246.71 | 0.1482 | 1692 | no |

## Reading these files

- `baseline2.json` — the pre-change configuration (no MTP, FlashInfer autotune
  enabled). Note the `corrupt` column: this is the config that measured 36/36
  corrupt on the `rate.py` gate. Its throughput numbers are therefore an *upper*
  bound that partly reflects degenerate output being cheap to predict.
- `mtp-s3-t1-d4-clean.json` — MTP enabled, autotune still on.
- `FINAL-mtp-noautotune.json`, `PRODUCTION-bench.json` — the shipped
  configuration: MTP + `--disable-flashinfer-autotune`. Two independent runs on
  separate server starts.

Corruption-rate gate for the shipped configuration: 0/24, 0/30, 0/24 and 0/18
across separate server starts (`python3 bench/rate.py --iters N`).

## Falsified hypotheses

Two effects that looked like separate defects during investigation and did not
survive re-testing on the fixed configuration. Recorded so nobody re-derives
them from this repo's intermediate numbers.

| hypothesis | original signal | re-test on fixed config | verdict |
|---|---|---|---|
| `/flush_cache` twice around a generation corrupts mamba state | 2/8 corrupt vs 0/8 for three other orderings | **0/12 for all five orderings**, including the suspect | noise against a ~12% corrupting background |
| autotune's dummy warmup forward corrupts the first requests | 3/24 and 3/30, always iteration 0, on patched builds | **0/30** with `--flashinfer-autotune-skip-ops trtllm::fused_moe::gemm1 trtllm::fused_moe::gemm2` (zero fused-MoE entries tuned) | 6/54 vs 0/30, Fisher exact p≈0.08 — start-to-start variance |

The lesson generalising from both: a corruption rate measured against a
non-zero background rate attributes nothing. Get the background to zero first,
then test one variable.
