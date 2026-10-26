# Release preparation draft: October 20 kern2

Local packaging plus an interrupted CPU-only image-build attempt. No push, PR, publication, model
weight download or GPU execution was performed. The October 10 release remains the comparison baseline.

The packaged runtime is the installed October 20 `i12` configuration: recorded **303.8 tok/s** and
**125.2 verify steps/s**. [The public report](results/20261020/REPORT.md) explains the measurement and
quality limits. **Decode-path KLD was not re-measured for kern2.** Copy probes were not run either.
Do not carry the October 10 KLD into the October 20 release-table cell.

## Verified locally

- Public overlay/map hashes verify against `runtime.json`. Against the measured installed manifest,
  26 file hashes match and two overlays intentionally differ (public loader key and dated comment).
  [Every public/installed source/config difference is listed here](docs/runtime.md#intentional-publicinstalled-differences).
- Kernel sources and bindings match independent read-only checksums of the installed source directory.
- Launcher environment and normalized SGLang arguments match the installed unit; container paths retain
  the public layout and source files are baked into the image rather than mounted from workstation paths.
- `python3 -m unittest discover -s tests -v`: **14 tests passed**, including all launcher flags/env,
  model-view preparation, source integrity and benchmark aggregation checks.
- `python3 serve/verify-runtime.py`: **28 runtime files, 32 kernel/support sources and 97,800 draft IDs verified**.
- Parsed all **82 Python** and **56 JSON** files; `bash -n` passed for all **5 shell scripts**.
  `git diff --cached --check`, relative release-document links and a staged sensitive-data scan passed.
- Target/draft configs generated from the pinned public checkpoint config match the installed config
  objects after the documented target loading-key rename; the draft config is unchanged. The live i12 benchmark download matches the recorded raw-results digest, and the
  public harness's four prompts match the recorded benchmark.
- `decode-summary.json` reproduces the eight-run, four-profile 303.7750964 tok/s mean using timings only.
- Third-party notices and original source headers are retained; no weights, private activation captures,
  owner sessions, private prompts, raw generated benchmark text or workstation unit are included.

## CPU image-build attempt stopped on October 26

**Do not restart automatically. Never build while a Qwen server is running; it pins ~102 GB host RAM.**

The owner stopped the build at 04:53 EDT (08:53 UTC) after parallel compiler processes, reported at
about 2.2 GB RSS each, competed with the running server's pinned host memory. The owner reported that
the kernel OOM killer terminated the server scheduler mid-benchmark. CPU-only execution, nice 19,
idle I/O and MAX_JOBS=8 did not bound host RAM sufficiently. Future parallelism must be selected from
free host RAM, in a window with no Qwen server running.

The build reached **Dockerfile step [6/8], compiling qwenfast via build_qf.py**. Dependency repair and
its `pip check` completed successfully (`No broken requirements found.`); the overlay/hot-map/source
COPY steps completed. Compilation was canceled before copying the verifier/preparation script and
before `verify-runtime.py --installed`, final image export or throwaway final-image verification.
The supervisor recorded exit **-15**, end **08:53:32 UTC**, elapsed **190 s**. No final image digest or
installed-verifier result is available. Minimum observed free disk was **151,583,707,136 bytes**, above
the 120 GB floor. The build directory and BuildKit cache were retained without cleanup.

An earlier attempt stopped before compilation because `.dockerignore` excluded `prepare-draft-a16.py`;
that allowlist is fixed. The Dockerfile now lowers compiler CPU/I/O priority inside the build as well
as in the host wrapper. Neither change makes concurrent model serving safe.

Resume manually from the retained build checkout's `repo` directory, **only after all Qwen servers
are stopped**, with `MAX_JOBS` chosen from available RAM. This reuses BuildKit cache and only the test tag:

```bash
: "${MAX_JOBS:?set MAX_JOBS from free host RAM before resuming}"
IMAGE=qwen38-flash-next-public:oct20-kern2-test MAX_JOBS="$MAX_JOBS" \
  nice -n 19 ionice -c3 bash serve/build-image.sh
```

The retained build supervisor hard-codes eight jobs; do not rerun it unchanged. Recheck the 120 GB
disk floor separately. After a successful build, run the final-image checks in a GPU-invisible,
no-port throwaway container. This Docker CLI rejects literal `--gpus none`; use runtime `runc`, no GPU
device requests, `NVIDIA_VISIBLE_DEVICES=void` and an empty `CUDA_VISIBLE_DEVICES` instead.

Draft-A16 preparation **requires a GPU** (CUDA tensors, Marlin repacking and A16 kernel validation).
It was not run; no new artifact or tensor hashes were produced or compared with the installed artifact.
The known measured artifact digest remains a reference, not a new CPU reproduction.

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
