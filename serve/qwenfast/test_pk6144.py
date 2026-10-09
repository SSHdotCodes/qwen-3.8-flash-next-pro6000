import os, sys, time, json, struct, subprocess, torch
sys.path.insert(0, '/experiment/ext')
from build_qf import build
import pk
qf = build()
dev = 'cuda'
W = torch.randn(2560, 6144, device=dev).to(torch.bfloat16) * 0.02
NC = 14
Ps = [pk.pack(W) for _ in range(NC)]; Ws = [W.clone() for _ in range(NC)]
def gt(fn, reps=24):
    g = torch.cuda.CUDAGraph(); fn(0); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for i in range(reps): fn(i)
    g.replay(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / 10 / reps * 1e6
mode = os.environ.get('QWENFAST_PK_XB_KB', '48')
for M in (1, 2, 4, 8):
    x = torch.randn(M, 6144, device=dev).to(torch.bfloat16)
    res = {r: gt(lambda i, r=r: qf.pk_gemm(x, Ps[i % NC]['m8'], Ps[i % NC]['e4'], Ps[i % NC]['base'], Ps[i % NC]['ex_ptr'], Ps[i % NC]['ex_k'], Ps[i % NC]['ex_v'], r)) for r in (1, 2, 4)}
    tb = gt(lambda i: torch.matmul(x, Ws[i % NC].T))
    print(f'xb_limit {mode} KB M={M}: ' + ' '.join(f'rpw{r} {t:.1f}' for r, t in res.items()) + f' us | cuBLAS bf16 {tb:.1f} us')
