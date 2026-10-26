"""WP-HC full-row mix/combine checks, ported to the integrated extension.

Uses synthetic weights, both delivered variants, inject/no-inject and M=1..8.
This checks arithmetic; it does not reproduce the cold-neighbor timing fixture.
"""
import torch
from kern2_test_utils import extension

qf = extension()
torch.manual_seed(100)
bf, dev = torch.bfloat16, "cuda"
HC, HS, K, R = 4, 2560, 10240, 320


def quant(w, group):
    n, k = w.shape
    x = w.float().reshape(n, k // group, group)
    s = (x.abs().amax(2) / 127).clamp_min(1e-30).to(bf)
    q = (x / s.float().unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8).reshape(n, k)
    return q.contiguous(), s.contiguous()


dq, ds = quant((torch.randn(R, K, device=dev) * .02).to(bf), 128)
uq, us = quant((torch.randn(K, R, device=dev) * .05).to(bf), 64)
wn = (torch.randn(K, device=dev) * .1).to(bf)
wi = (torch.randn(HC, K, device=dev) * .02).to(bf)
normed = torch.empty(8 * K, device=dev, dtype=bf)
partial = torch.empty(4 * 8 * 324, device=dev)
bar = torch.zeros(2, device=dev, dtype=torch.int32)
for m in range(1, 9):
    r = torch.randn(m, K, device=dev).to(bf)
    y = torch.randn(m, HS, device=dev).to(bf)
    ip = torch.randn(m, HC, device=dev) * 3
    for combine in (False, True):
        for inject in (False, True):
            prefix = "hc_combine_mix_w8" if combine else "hc_mix_fused_w8"
            args = (y, r, ip) if combine else (r,)
            args += (wn, dq, ds, wi if inject else None, uq, us, 1e-6, normed, partial, bar, True)
            ref = getattr(qf, prefix)(*args)
            for variant in ("fast", "g96fast"):
                result = getattr(qf, prefix + "_" + variant)(*args)
                for i, (a, b) in enumerate(zip(ref, result)):
                    if not inject and i == 1:  # undefined no-inject output
                        continue
                    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (m, combine, inject, variant, i)
                assert bar.tolist() == [0, 0]
    print(f"M={m}: HC mix/combine, both variants, inject/no-inject bitwise", flush=True)
