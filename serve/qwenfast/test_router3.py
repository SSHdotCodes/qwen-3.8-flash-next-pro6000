"""router_topk + hc_mix_fused_w8: production .so vs new build, in separate processes (same module name).
python3 test_router3.py prod|new   -> prod saves outputs + timings to /ext/rt3-prod.pt; new compares."""
import json, sys, torch, importlib.machinery, importlib.util
mode = sys.argv[1]
if mode == 'prod':
    _l = importlib.machinery.ExtensionFileLoader('qwenfast', '/prod/qwenfast.so')
    _sp = importlib.util.spec_from_loader('qwenfast', _l, origin='/prod/qwenfast.so')
    Q = importlib.util.module_from_spec(_sp); _l.exec_module(Q)
else:
    sys.path.insert(0, '/ext')
    from build_qf import build
    Q = build()
print(mode, getattr(Q, "__file__", "prod-so"), flush=True)
dev, bf = 'cuda', torch.bfloat16
g = torch.Generator(device=dev); g.manual_seed(1234)
NC = 48
RW = [(torch.randn(512, 2560, device=dev, generator=g) * 0.02).to(bf) for _ in range(NC)]
sc, ct = torch.empty(8 * 512, device=dev), torch.zeros(1, dtype=torch.int32, device=dev)
outs = []
for T in (1, 2, 3, 4, 5, 8):
    for j in range(60):
        if j % 3 == 0:
            x = (torch.randn(T, 2560, device=dev, generator=g) * (0.2 + j / 10)).to(bf)
            w = RW[j % NC]
        elif j % 3 == 1:     # coarse values -> many tied logits (and exact zeros)
            x = (torch.randint(-1, 2, (T, 2560), device=dev, generator=g).to(bf)
                 * (torch.rand(T, 2560, device=dev, generator=g) < 0.002))
            w = torch.randint(-2, 3, (512, 2560), device=dev, generator=g).to(bf)
        else:
            x = torch.zeros(T, 2560, device=dev, dtype=bf); x[:, j] = 1.0
            w = RW[j % NC]
        for renorm in (True, False):
            outs.append([t.cpu() for t in Q.router_topk(x, w, 10, renorm, sc, ct, True)])
HC, K, R = 4, 10240, 320
def w8q(w, G):
    N, Kk = w.shape
    xg = w.float().view(N, Kk // G, G)
    s16 = (xg.abs().amax(2) / 127).clamp_(min=1e-30).to(bf)
    return (xg / s16.float().unsqueeze(-1)).round_().clamp_(-127, 127).to(torch.int8).view(N, Kk).contiguous(), s16.contiguous()
HW = []
for _ in range(NC):
    dq, ds = w8q((torch.randn(R, K, device=dev, generator=g) * 0.02).to(bf), 128)
    uq, us = w8q((torch.randn(K, R, device=dev, generator=g) * 0.05).to(bf), 64)
    HW.append((dq, ds, uq, us))
wn = (torch.randn(K, device=dev, generator=g) * 0.1).to(bf)
winj = (torch.randn(4, K, device=dev, generator=g) * 0.02).to(bf)
normed = torch.empty(8 * K, device=dev, dtype=bf)
partial = torch.empty(HC * 8 * 324, device=dev)
bar = torch.zeros(2, dtype=torch.int32, device=dev)
for T in (1, 2, 4, 8):
    xh = torch.randn(T, K, device=dev, generator=g).to(bf)
    for wi in (winj, None):
        o, inj = Q.hc_mix_fused_w8(xh, wn, HW[0][0], HW[0][1], wi, HW[0][2], HW[0][3], 1e-6, normed, partial, bar, True)
        outs.append([o.cpu(), inj.cpu() if wi is not None else torch.zeros(1)])

def timeit(fn, reps=10):
    for i in range(3): fn(i)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for i in range(NC): fn(i)
    gr.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps): gr.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / (reps * NC)
tim = {}
for T in (1, 4, 8):
    xh = torch.randn(T, K, device=dev, generator=g).to(bf)
    def chain(i):
        h = HW[i]
        out, _ = Q.hc_mix_fused_w8(xh, wn, h[0], h[1], winj, h[2], h[3], 1e-6, normed, partial, bar, True)
        return Q.router_topk(out, RW[i], 10, True, sc, ct, True)
    tim[f'T{T}'] = dict(mix=round(timeit(lambda i: Q.hc_mix_fused_w8(xh, wn, HW[i][0], HW[i][1], winj, HW[i][2], HW[i][3], 1e-6, normed, partial, bar, True)), 2),
                        router=round(timeit(lambda i: Q.router_topk(xh[:, :2560].contiguous(), RW[i], 10, True, sc, ct, True)), 2),
                        chain=round(timeit(chain), 2))
print(mode, json.dumps(tim), flush=True)
if mode == 'prod':
    torch.save(dict(outs=outs, tim=tim), '/ext/rt3-prod.pt')
else:
    ref = torch.load('/ext/rt3-prod.pt')
    bad = sum(0 if all(torch.equal(a, b) for a, b in zip(o, r)) else 1 for o, r in zip(outs, ref['outs']))
    print(json.dumps(dict(cases=len(outs), mismatches=bad, prod_tim=ref['tim'], new_tim=tim)))
