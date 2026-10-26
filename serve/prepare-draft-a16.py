#!/usr/bin/env python3
"""Prepare draft NVFP4 experts and INT8 shared expert from the pinned native MTP.

Adapted from the October 20 conversion recipe: RTN NVFP4, block 16, FP8 E4M3
block scales, one global scale per expert, followed by the production Marlin
repack. Shared expert uses INT8 groups of 128. Requires an idle SM120 GPU and
the public runtime image. No model weights are included in this repository.
The synthetic validation replaces private activation captures; it does not
establish model-level quality or equality to the installed artifact.
"""
import argparse
import hashlib
import importlib.machinery, importlib.util, json, os, sys, time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--draft", default="/models/draft", help="Pinned native-MTP model view")
parser.add_argument("--output", default="/output/draft-a16.pt")
parser.add_argument("--library", default="/opt/qwenfast/qwenfast.so")
args = parser.parse_args()

import torch
from safetensors import safe_open

dev, bf = "cuda", torch.bfloat16
torch.manual_seed(20261020)
D = args.draft
idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
pre = "mtp.layers.0.mlp."
T = {}
for n in [pre + s for s in ("experts.gate_up_proj", "experts.down_proj", "gate.weight", "shared_expert_gate.weight",
                            "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight", "shared_expert.down_proj.weight")]:
    with safe_open(f"{D}/{idx[n]}", framework="pt", device=dev) as h:
        T[n] = h.get_tensor(n)
w13_bf, w2_bf = T[pre + "experts.gate_up_proj"], T[pre + "experts.down_proj"]   # [E,1280,2560], [E,2560,640]
E = w13_bf.shape[0]
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
BOUND = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)


def nvfp4(W):
    """W [N, K] (one expert) -> packed uint8 [N, K/2], fp8 scales [N, K/16], global scale (fp32 scalar), dequant fp32."""
    N, K = W.shape
    Wf = W.float()
    gs = 448.0 * 6.0 / Wf.abs().amax().clamp_min(1e-12)
    blk = Wf.view(N, K // 16, 16)
    sc = (blk.abs().amax(-1) / 6.0 * gs).clamp_(max=448.0).to(torch.float8_e4m3fn)
    scf = (sc.float() / gs).unsqueeze(-1)
    x = blk / scf.clamp_min(1e-30)
    code = torch.bucketize(x.abs(), BOUND).to(torch.uint8)            # 0..7 magnitude index
    deq = (GRID[code.long()] * torch.sign(x) * scf).view(N, K)
    code = (code | ((x < 0).to(torch.uint8) << 3)).view(N, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, sc, gs, deq


def w8q(w):
    N, K = w.shape
    G = 128
    xg = w.float().view(N, K // G, G)
    s16 = (xg.abs().amax(2) / 127).clamp_(min=1e-30).to(bf)
    q = (xg / s16.float().unsqueeze(-1)).round_().clamp_(-127, 127).to(torch.int8).view(N, K)
    return q.contiguous(), s16.contiguous()


t0 = time.time()
p13 = torch.empty((E, 1280, 1280), dtype=torch.uint8, device=dev); s13 = torch.empty((E, 1280, 160), dtype=torch.float8_e4m3fn, device=dev)
p2 = torch.empty((E, 2560, 320), dtype=torch.uint8, device=dev); s2 = torch.empty((E, 2560, 40), dtype=torch.float8_e4m3fn, device=dev)
g13 = torch.empty(E, device=dev); g2 = torch.empty(E, device=dev)
err13 = err2 = 0.0
for e in range(E):
    p13[e], s13[e], g13[e], d13 = nvfp4(w13_bf[e])
    p2[e], s2[e], g2[e], d2 = nvfp4(w2_bf[e])
    if e < 8:
        err13 += ((d13 - w13_bf[e].float()).norm() / w13_bf[e].float().norm()).item() / 8
        err2 += ((d2 - w2_bf[e].float()).norm() / w2_bf[e].float().norm()).item() / 8
print(f"quantized {E} experts in {time.time() - t0:.1f}s; weight rel-RMS error w13 {err13:.4f} w2 {err2:.4f}", flush=True)
canon = dict(p13=p13.clone(), s13=s13.clone(), p2=p2.clone(), s2=s2.clone(), g13=g13.clone(), g2=g2.clone())

# production Marlin repack (compressed_tensors_w4a4_nvfp4_moe._process_weights_marlin)
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.srt.layers.quantization.marlin_utils_fp4 import (_permute_moe_fp4_scales_for_marlin,
    nvfp4_marlin_process_global_scale, nvfp4_marlin_process_scales)
perm = torch.empty(0, dtype=torch.int, device=dev)


def repack(weight, size_n, size_k):
    for i in range(E):
        q = weight[i].view(torch.int32).T.contiguous()
        out = gptq_marlin_repack(b_q_weight=q, perm=perm, size_k=size_k, size_n=size_n, num_bits=4)
        weight[i].view(-1).copy_(out.view(torch.uint8).reshape(-1))
    return weight.view(torch.int32).view(E, size_k // 16, size_n * 2)


def scales(sf, n, k):
    return _permute_moe_fp4_scales_for_marlin(sf.to(bf), num_experts=E, size_n=n, size_k=k, group_size=16,
                                              process_scales=nvfp4_marlin_process_scales)


I, H = 640, 2560
out = dict(
    w13=repack(p13, 2 * I, H), w2=repack(p2, H, I), s13=scales(s13, 2 * I, H), s2=scales(s2, H, I),
    g13=nvfp4_marlin_process_global_scale((1 / g13).to(bf)), g2=nvfp4_marlin_process_global_scale((1 / g2).to(bf)))
gu = torch.cat([T[pre + "shared_expert.gate_proj.weight"], T[pre + "shared_expert.up_proj.weight"]], 0).contiguous()
out["sq1"], out["ss1"] = w8q(gu)
out["sq2"], out["ss2"] = w8q(T[pre + "shared_expert.down_proj.weight"].contiguous())
print({k: (tuple(v.shape), v.dtype) for k, v in out.items()}, flush=True)

# ---- validate the a16 kernel on these weights vs FP32 reference (dequantized canonical weights) ----
P = args.library
ld = importlib.machinery.ExtensionFileLoader("qwenfast", P)
qf = importlib.util.module_from_spec(importlib.util.spec_from_loader("qwenfast", ld, origin=P)); ld.exec_module(qf)
router, wg = T[pre + "gate.weight"], T[pre + "shared_expert_gate.weight"].reshape(-1).contiguous()


def deq(pk, sc, gs):   # canonical -> fp32 [N, K]
    lo, hi = pk & 15, pk >> 4
    code = torch.stack([lo, hi], -1).view(pk.shape[0], -1).long()
    v = GRID[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    bf_inv = (1 / gs).to(bf).float()
    return (v.view(v.shape[0], -1, 16) * (sc.float() * bf_inv).unsqueeze(-1)).view(v.shape[0], -1)


def w8deq(q, s):
    return (q.float().view(q.shape[0], -1, 128) * s.float().unsqueeze(-1)).view(q.shape[0], -1)


# Public validation uses seeded synthetic inputs; no workstation captures are needed.
xs = torch.randn(6, 4, H, device=dev, dtype=bf)
g1, d1 = w8deq(out["sq1"], out["ss1"]), w8deq(out["sq2"], out["ss2"])
buf = dict(hbuf=torch.empty(80 * 8 * 640, dtype=bf, device=dev), ybuf=torch.empty(8 * 10 * 2560, dtype=bf, device=dev),
           gl=torch.empty(8, device=dev), ctrl=torch.zeros(1024, dtype=torch.int32, device=dev),
           pbuf=torch.empty(80 * 10 * 2 * 1024, device=dev), dbg=torch.empty(0, dtype=torch.int64, device=dev),
           shbuf=torch.empty(8 * (640 + 2560), dtype=bf, device=dev), none=torch.empty(0, dtype=bf, device=dev))
worst = 0.0
for xi in range(xs.shape[0]):
    for M in (1, 4):
        x = xs[xi, :M].contiguous()
        logits = x.float() @ router.float().T
        tw, ti = torch.softmax(logits, -1).topk(10, -1)
        tw = tw / tw.sum(-1, keepdim=True)
        y = torch.empty_like(x)
        qf.a16_moe_decode(x, ti.to(torch.int32).contiguous(), tw.float().contiguous(), out["w13"], out["s13"], out["g13"].view(-1),
                          out["w2"], out["s2"], out["g2"].view(-1), wg, buf["none"], buf["hbuf"], buf["ybuf"], buf["gl"], buf["ctrl"],
                          y, 4, buf["dbg"], buf["pbuf"], out["sq1"], out["ss1"], out["sq2"], out["ss2"], buf["shbuf"])
        ref = torch.zeros(M, H, device=dev)
        for m in range(M):
            for j in range(10):
                e = ti[m, j].item()
                a = deq(canon["p13"][e], canon["s13"][e], canon["g13"][e]) @ x[m].float()
                hh = torch.nn.functional.silu(a[:I]) * a[I:]
                ref[m] += tw[m, j] * (deq(canon["p2"][e], canon["s2"][e], canon["g2"][e]) @ hh)
            a = g1 @ x[m].float()
            ref[m] += torch.sigmoid(x[m].float() @ wg.float()) * (d1 @ (torch.nn.functional.silu(a[:I]) * a[I:]))
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        worst = max(worst, rel)
print(f"a16 kernel vs FP32 reference on the NVFP4 draft experts: worst rel err {worst:.2e}", flush=True)
assert worst < 2e-2, "layout/scale mismatch"
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
torch.save({k: v.cpu() for k, v in out.items()}, output)
sha = hashlib.file_digest(output.open("rb"), "sha256").hexdigest()
manifest = json.loads((Path(__file__).parent / "runtime.json").read_text())
receipt = {
    "model": manifest["model"], "model_revision": manifest["model_revision"],
    "sha256": sha, "size_bytes": output.stat().st_size,
    "preparation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "validation": "seeded synthetic a16 vs dequantized FP32, M=1,4",
    "worst_relative_error": worst,
    "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in out.items()},
    "measured_reference_sha256": manifest["draft_a16"]["measured_sha256"],
}
output.with_suffix(output.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
print(f"saved {output}: {output.stat().st_size:,} bytes; sha256 {sha}")
if sha != receipt["measured_reference_sha256"]:
    print("File bytes differ from the measured artifact. Tensor/quality/acceptance validation is still required; "
          "torch serialization can differ between output paths and builds.")
