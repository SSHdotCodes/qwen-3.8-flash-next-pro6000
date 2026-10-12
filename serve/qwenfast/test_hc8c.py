"""hc_combine_mix_w8 vs hc_combine_apply -> hc_mix_fused_w8 (must be bit-identical) and graph timing:
(a) mix alone, (b) combine + mix, (c) fused, each also behind a w8g in_proj-size GEMM (PDL chain)."""
import json, os, sys, torch
sys.path.insert(0, '/ext')
from build_qf import build
qf = build()
import importlib.machinery, importlib.util
_l = importlib.machinery.ExtensionFileLoader('qwenfast', '/prod/qwenfast.so')
_sp = importlib.util.spec_from_loader('qwenfast', _l, origin='/prod/qwenfast.so')
PROD = importlib.util.module_from_spec(_sp); _l.exec_module(PROD)
torch.manual_seed(0)
HC, HS, R, K = 4, 2560, 320, 10240
dev, bf = 'cuda', torch.bfloat16
eps = 1e-6
copies = 24


def w8q(w, G):
    N, Kk = w.shape
    xg = w.float().view(N, Kk // G, G)
    s16 = (xg.abs().amax(2) / 127).clamp_(min=1e-30).to(bf)
    q = (xg / s16.float().unsqueeze(-1)).round_().clamp_(-127, 127).to(torch.int8).view(N, Kk)
    return q.contiguous(), s16.contiguous()


W = []
for _ in range(copies):
    wd = (torch.randn(R, K, device=dev) * 0.02).to(bf)
    wu = (torch.randn(K, R, device=dev) * 0.05).to(bf)
    dq, ds = w8q(wd, 128)
    uq, us = w8q(wu, 64)
    W.append(dict(wn=(torch.randn(K, device=dev) * 0.1).to(bf), dq=dq, ds=ds, uq=uq, us=us,
                  wi=(torch.randn(HC, K, device=dev) * 0.02).to(bf)))
G1 = [w8q((torch.randn(16480, 2560, device=dev) * 0.02).to(bf), 128) for _ in range(3)]
normed_s = torch.empty(8 * K, device=dev, dtype=bf)
partial = torch.empty(HC * 8 * 324, device=dev, dtype=torch.float32)
bar = torch.zeros(2, device=dev, dtype=torch.int32)


def two(y, r, ip, w):
    x = qf.hc_combine_apply(y, r, ip, True)
    out, inj = qf.hc_mix_fused_w8(x, w['wn'], w['dq'], w['ds'], w['wi'], w['uq'], w['us'], eps, normed_s, partial, bar, True)
    return out, inj, x


def one(y, r, ip, w):
    return qf.hc_combine_mix_w8(y, r, ip, w['wn'], w['dq'], w['ds'], w['wi'], w['uq'], w['us'], eps, normed_s, partial, bar, True)


def two_prod(y, r, ip, w):
    x = PROD.hc_combine_apply(y, r, ip, True)
    out, inj = PROD.hc_mix_fused_w8(x, w['wn'], w['dq'], w['ds'], w['wi'], w['uq'], w['us'], eps, normed_s, partial, bar, True)
    return out, inj, x


def mix_only(y, r, ip, w):
    return qf.hc_mix_fused_w8(r, w['wn'], w['dq'], w['ds'], w['wi'], w['uq'], w['us'], eps, normed_s, partial, bar, True)


def timeit(fn, iters=2400):
    for i in range(3): fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(copies): fn(i)
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters // copies): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


rows = []
for M in (1, 2, 3, 4, 5, 8):
    r = torch.randn(M, K, device=dev).to(bf)
    y = torch.randn(M, HS, device=dev).to(bf)
    ip = torch.randn(M, 4, device=dev) * 3
    a = two(y, r, ip, W[0])
    b = one(y, r, ip, W[0])
    c = one(y, r, ip, W[0])
    exact = all(torch.equal(u, v) for u, v in zip(a, b))
    det = all(torch.equal(u, v) for u, v in zip(b, c))
    xg = torch.randn(M, 2560, device=dev).to(bf)
    gemm = lambda i: qf.w8g_gemm(xg, G1[i % 3][0], G1[i % 3][1], 0x100)
    row = dict(M=M, exact=exact, deterministic=det, bar=bar.tolist(),
               mix_us=round(timeit(lambda i: mix_only(y, r, ip, W[i])), 2),
               comb_mix_us=round(timeit(lambda i: two(y, r, ip, W[i])), 2),
               prod_comb_mix_us=round(timeit(lambda i: two_prod(y, r, ip, W[i])), 2),
               prod_exact=all(torch.equal(u, v) for u, v in zip(two_prod(y, r, ip, W[0]), b)),
               fused_us=round(timeit(lambda i: one(y, r, ip, W[i])), 2),
               gemm_us=round(timeit(gemm), 2),
               gemm_comb_mix_us=round(timeit(lambda i: (gemm(i), two(y, r, ip, W[i]))), 2),
               gemm_prod_us=round(timeit(lambda i: (gemm(i), two_prod(y, r, ip, W[i]))), 2),
               gemm_fused_us=round(timeit(lambda i: (gemm(i), one(y, r, ip, W[i]))), 2))
    if not exact:
        row['maxdiff'] = [((u.float() - v.float()).abs().max().item()) for u, v in zip(a, b)]
    rows.append(row)
    print(json.dumps(row), flush=True)
