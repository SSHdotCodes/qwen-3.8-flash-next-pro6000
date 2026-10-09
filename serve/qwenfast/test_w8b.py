import sys, time, torch
sys.path.insert(0, '/experiment/ext')
from build_qf import build
qf = build()
dev = 'cuda'
def gt(fn, reps=24):
    g = torch.cuda.CUDAGraph(); fn(0); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for i in range(reps): fn(i)
    g.replay(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / 10 / reps * 1e6
for (N, K) in ((13312, 2560), (2560, 6144), (640, 2560), (1280, 2560), (2560, 640), (65800, 2560)):
    w = torch.randn(N, K, device=dev) * 0.02
    s = (w.abs().amax(1) / 448).clamp_min(1e-12)
    q8 = (w / s[:, None]).to(torch.float8_e4m3fn).contiguous()
    deq = q8.float() * s[:, None]
    si = (w.abs().amax(1) / 127)
    qi = torch.round(w / si[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
    deqi = qi.float() * si[:, None]
    NC = max(2, 400_000_000 // (N * K))
    qs = [q8.clone() for _ in range(NC)]
    line = []
    for M in (1, 4, 8):
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        y = qf.w8_gemm(x, q8, s.float().contiguous(), 0).float(); ref = x.float() @ deq.T
        e1 = ((y - ref).abs().max() / ref.abs().max()).item()
        y = qf.w8_gemm(x, qi, si.float().contiguous(), 0).float(); ref = x.float() @ deqi.T
        e2 = ((y - ref).abs().max() / ref.abs().max()).item()
        assert e1 < 1e-2 and e2 < 1e-2, (N, K, M, e1, e2)
        t = min(gt(lambda i, r=r: qf.w8_gemm(x, qs[i % NC], s, r)) for r in (1, 2, 4))
        line.append(f'M={M} {t:5.1f} us ({N*K/t/1e3:.2f} TB/s)')
    print(f'{N}x{K}: ' + ' | '.join(line))
