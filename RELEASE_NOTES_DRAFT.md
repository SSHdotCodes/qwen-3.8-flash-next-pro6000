# Release preparation draft: October 20 kern2

Local packaging only. No push, PR, publication, model download, GPU execution or workstation change
was performed during this preparation. The October 10 release remains the comparison baseline.

The packaged runtime is the installed October 20 `i12` configuration: recorded **303.8 tok/s** and
**125.2 verify steps/s**. [The public report](results/20261020/REPORT.md) explains the measurement and
quality limits. **Decode-path KLD was not re-measured for kern2.** Copy probes were not run either.
Do not carry the October 10 KLD into the October 20 release-table cell.

## Verified locally

- Installed overlay files match the source manifest, with the unchanged hot map included.
- Kernel sources and bindings match independent read-only checksums of the installed source directory.
- Launcher environment and normalized SGLang arguments match the installed unit; container paths retain
  the public layout and source files are baked into the image rather than mounted from workstation paths.
- `python3 -m unittest discover -s tests -v`: **13 tests passed**, including all launcher flags/env,
  model-view preparation, source integrity and benchmark aggregation checks.
- `python3 serve/verify-runtime.py`: **28 runtime files, 32 kernel/support sources and 97,800 draft IDs verified**.
- Parsed all **82 Python** and **56 JSON** files; `bash -n` passed for all **5 shell scripts**.
  `git diff --cached --check`, relative release-document links and a staged sensitive-data scan passed.
- Target/draft configs generated from the pinned public checkpoint config match the installed config
  objects exactly. The live i12 benchmark download matches the recorded raw-results digest, and the
  public harness's four prompts match the recorded benchmark.
- `decode-summary.json` reproduces the eight-run, four-profile 303.7750964 tok/s mean using timings only.
- Third-party notices and original source headers are retained; no weights, private activation captures,
  owner sessions, private prompts, raw generated benchmark text or workstation unit are included.

## Needs GPU verification

- **Public image build:** build from this checkout on Linux/CUDA 13/SM120a and run `pip check` and
  `verify-runtime.py --installed`. Compilation itself needs the CUDA toolchain, not a GPU, but it was
  not available/used on the packaging Mac. The workstation image digest is a measured reference,
  not a claim that this checkout has already built or a public pullable digest.
- **Regenerated draft A16:** run the public preparation recipe on the pinned MTP view and validate
  tensor/layout/scales against the measured artifact or its outputs. Record artifact and tensor hashes,
  validate the persistent A16 and Marlin prefill paths and confirm unused FP8 expert copies are freed.
  The recipe is adapted from the recorded conversion, with synthetic inputs replacing private captures.
  Torch archive bytes may differ after serialization; equality to the installed artifact is unverified.
- **GPU tests:** execute the ported work-package suites and previous kernel tests, including mutable
  CUDA graphs, replay/fallback state tracking, candidate sampling and head dense/overflow fallback.
  The new public ports were syntax-checked only. They do not reproduce private captured-input cohorts.
- **Head omission audit:** verify candidate inclusion and top-k exactness on fresh eligible workloads;
  the target screen uses an empirical sigma bound. Exercise ineligible modes (logprobs, grammar,
  penalties, min_p, logit bias, top_k outside 1–56) and the dense fixup path.
- **Public end-to-end check:** restart after cache warmup, compare against the matched October 10
  control with the recorded two-run four-profile protocol and median verify-step statistic. Check
  candidate-only sampling, draft-extend graphs, overlap scheduling and two-request operation.
- **Quality at installed capacity:** rerun GSM8K, valid tool tasks, retrieval, concurrency, degeneration
  and vision at the shipped 786,432-token pool. Recorded `fin12` text checks used an experimental 1M
  pool; the installed 786K vision observations are recorded separately. Repair the broken definition
  task fixture before using a /120 tool score as a fresh comparable result.
- **Decode-path KLD and copy probes:** locate/rebuild the BF16 reference harness and data, then measure
  the kern2 paths. The October 10 value is historical, and the old harness path was unavailable.
- FP32 recurrent state with the selected FlashInfer decode backend is unsupported in the recorded
  stack. Changing backend or state precision needs separate validation and is not part of this release.

## TODO: October 26 update

**Unfilled placeholder for the orchestrator after tonight's GPU work.**

| Date | Runtime / pins | Decode tok/s | Verify steps/s | Decode-path KLD | GPU / quality status |
|---|---|---|---|---|---|
| October 26 | TODO | TODO | TODO | TODO: explicitly state measured or not measured | TODO |

Add the final measured configuration and results, link its public evidence/report, replace the README
TODO row, and record any remaining verification limits before rewriting the final release commit.
