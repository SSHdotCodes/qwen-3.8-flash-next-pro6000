import sys, time, torch
sys.path.insert(0, '/experiment/ext')
from build_qf import build
qf = build()
torch.manual_seed(0)
dev = 'cuda'
N, K = 66076, 2560
w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
amax = w.float().abs().amax(1).clamp_min(1e-12)
s = (amax / 127).float().contiguous()
q = torch.round(w.float() / s[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
deq = q.float() * s[:, None]
worst = 0
for M in (1, 2, 3, 4):
    x = torch.randn(M, K, device=dev).to(torch.bfloat16)
    for rpw in (1, 2, 4):
        y = qf.w8_gemm(x, q, s, rpw).float()
        ref = x.float() @ deq.T
        err = ((y - ref).abs().max() / ref.abs().max()).item()
        worst = max(worst, err)
    print(f'M={M} rel max err {err:.2e}')
assert worst < 1e-2
# timing: rotate copies so weights stream from DRAM (128 MB L2)
NC = 4
qs = [q.clone() for _ in range(NC)]
ws = [w.clone() for _ in range(NC)]
def bench(fn, it=50):
    for i in range(5): fn(i)
    torch.cuda.synchronize(); t = time.perf_counter()
    for i in range(it): fn(i)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
for M in (1, 2, 4):
    x = torch.randn(M, K, device=dev).to(torch.bfloat16)
    tb = bench(lambda i: qf.smallm_gemm(x, ws[i % NC], 1, 4))
    res = {rpw: bench(lambda i, r=rpw: qf.w8_gemm(x, qs[i % NC], s, r)) for rpw in (1, 2, 4)}
    print(f'M={M} bf16 smallm {tb:.1f} us ({N*K*2/tb/1e6:.0f} GB/s) | int8 ' + ' '.join(f'rpw{r} {t:.1f} us ({N*K/t/1e6:.0f} GB/s)' for r, t in res.items()))
