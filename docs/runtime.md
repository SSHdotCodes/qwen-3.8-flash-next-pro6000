# Runtime notes — 2026-09-19

## Reproducible image

`serve/Dockerfile` starts from official SGLang 0.5.20 digest
`06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f`.
The dependency repairs match the tested workstation image: remove unused
`nixl` (generic CUDA 12 shim) and optional `cosmos-guardrail`, retain the CUDA 13
NIXL distribution, and pin NCCL, protobuf, grpcio-tools, typeguard, Pillow and
OpenCV. `pip check` must pass. The runtime verifier checks SGLang 0.5.20,
PyTorch 2.13.0+cu130, SGLang kernel 0.4.7 and FlashInfer 0.6.18 too.

The measured local image digest begins `a2198845`; it is **not a pullable public
registry digest**. This repository rebuilds from its public base and puts the
validated overlays into the image instead of workstation-specific bind mounts.
Build timestamps/attestations can change the resulting image digest. Package
versions and all installed overlay hashes are checked against `serve/runtime.json`.

## What the overlays do

- **SM120 QSA and FP8 KV:** preserve the validated SM120 attention dispatch,
  query-dtype packing and in-register FP8 tile dequantization. These are the
  current cache/attention correctness fixes, rather than the day-zero patch.
- **16-slot QSA pending ring:** update allocation, eager slots, captured metadata,
  fallback indexer calls and pending-state transfer strides together. Compression
  stays at four. Validation requires `draft_tokens + compress_ratio - 1 <= 16`,
  leaving room for a prior partial group. Removing the old width guard alone
  is incorrect. The original four-slot layout failed 248 width-four oracle cases;
  the selected layout passed 4,026 adversarial cases and 48 CUDA graph replays.
- **Proposal lifetime:** clone proposal probabilities before target graph replay
  can overwrite shared logits/probability storage. This fixed observed corruption
  under sampled exact rejection after the 0.5.20 update. Greedy smoke tests had
  missed it. Keep `--speculative-use-rejection-sampling` enabled.
- **Draft-only hot map:** gather/cache a BF16 copy of 65,800 draft-head rows;
  fill the full proposal-logit tensor with negative infinity outside those IDs.
  Target head weights and full-vocabulary target sampling remain unchanged.
  `QWEN_MTP_SHORTLIST=131072` activates the processor; `QWEN_MTP_HOTMAP` selects
  the smaller map and overrides the contiguous shortlist in the supported path.
  The old two-range fallback and some comments/log strings remain in the source
  to preserve the exact validated file hashes. Unsupported head layouts fall
  back to the original full-head implementation.
- **Startup:** indexed expert mapping, indexed native-MTP tensor loading and
  collection of unreachable temporary tensors before sizing the KV pool. These
  change startup work, not checkpoint values. Multithread loading uses 16 threads.

The standalone numerical check of FlashInfer target verification on SM120 was
limited to this model's shapes. Its combined performance experiment did not
justify a deployment change. The launcher retains the original Triton verify
path and no draft temperature multiplier.

## Token-map provenance

The original 65,536-ID map was downloaded from:

https://raw.githubusercontent.com/gabrielolympie/sglang-flashnext-sm120/main/hot_tokens_64k.pt

Source size: 209,175 bytes. SHA-256:
`f03551e4709e50e5842b24912845c243bfe628b74bf185617fae7ebaaa0da86c`.
The source URL names a mutable branch; the checksum identifies the exact
artifact used. It was loaded with `torch.load(..., weights_only=True)` and
validated as a list/tensor of in-range integer IDs. The shipped JSON is:

```python
sorted(set(source_ids) | set(range(248044, 248320)))
```

It has 65,800 unique IDs and SHA-256
`8a4f4c7c5905b76861a62befa1e060e16362e1775087fbb78e907fd85c4c9c12`.
No map download or pickle loading occurs at server startup. Changing this map
requires new sampling/correctness/performance checks; a smaller proposal space
can hurt acceptance on languages or tasks poorly represented in it.

## Verification and recovery

`python3 serve/verify-runtime.py` checks the exported sources without a GPU.
`./serve/build-image.sh` also checks installed files and package versions inside
the rebuilt image. `python3 -m unittest discover -s tests -v` checks launch
arguments, response parsing, sampling and the published aggregation locally.

To rerun the ring oracle and CUDA graph checks without loading model weights,
stop inference and run on an available GPU:

```bash
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
docker run --rm --gpus all --network none \
  -v "$PWD/bench/test_qsa_ring.py:/test_qsa_ring.py:ro" \
  local/qwen-flash-next:0.5.20-20260919-hotmap python3 /test_qsa_ring.py
```

The ring test verifies buffer indexing, not full-model bitwise equivalence.
The original long-context and two-request checks can also be reproduced against
an idle loaded server with `python3 bench/context_checks.py context --flush-cache`
and `python3 bench/concurrent_check.py concurrent --flush-cache`. These preserve
the synthetic prompts and seeds and write results into `results/local/`.
For an upgrade, retain your previous checkout, image and cache until your own
checks pass. To roll back, stop this container/user unit and restore that prior
launcher/image. This repository does not install or overwrite workstation model
catalogs, inference proxies, credentials, drivers or system packages. The
historical August checkout changes precision/capacity and is not an equivalent
rollback for the September baseline.
