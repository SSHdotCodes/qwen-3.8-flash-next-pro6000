"""WP-GDNout mutable-graph exactness port for the two integrated presets.

Synthetic weights replace the private fixture's checkpoint projection weights.
Checks the same row counts, gates, zero/extreme inputs and BF16 output bits.
"""
import torch
from kern2_test_utils import extension

qf = extension()
torch.manual_seed(194137)
dev, bf = "cuda", torch.bfloat16
w = (torch.randn(2560, 6144, device=dev) * .02).to(bf)
x = w.float().view(2560, 48, 128)
s = (x.abs().amax(2) / 127).clamp_min(1e-30).to(bf)
q = (x / s.float().unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8).reshape(2560, 6144)
wn = torch.randn(128, device=dev).to(bf)
for m in range(1, 9):
    core = torch.randn(m * 48, 128, device=dev).to(bf)
    gate = torch.randn_like(core)
    for swish in (False, True):
        def old():
            return qf.gdn_norm_oproj_w8(core, gate, wn, 1e-6, swish, q, s, True, True)
        for preset in (0, 1):
            def new():
                return qf.wpgdnout_w8(core, gate, wn, 1e-6, swish, q, s, preset)
            new()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = new()
            for trial in range(12):
                core.copy_((torch.randn_like(core, dtype=torch.float32) * 2. ** ((trial % 9) - 4)).to(bf))
                gate.copy_((torch.randn_like(gate, dtype=torch.float32) * (trial % 5 + 1) * 4).to(bf))
                if trial == 0:
                    core.zero_()
                if trial == 1:
                    gate.zero_()
                graph.replay()
                assert torch.equal(actual.view(torch.uint8), old().view(torch.uint8)), (m, swish, preset, trial)
    print(f"M={m}: both gates and presets, 12 mutable graph inputs, bitwise", flush=True)
