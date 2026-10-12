"""w8m (tensor-core w8g) vs w8g_gemm vs fp32 truth; graph timing with weights rotated past L2."""
import json, sys, torch
sys.path.insert(0, '/ext')
from build_qf import build
qf = build()
torch.manual_seed(0)
dev, bf = 'cuda', torch.bfloat16


def w8q(w):
    N, K = w.shape
    xg = w.float().view(N, K // 128, 128)
    s16 = (xg.abs().amax(2) / 127).clamp_(min=1e-30).to(bf)
    q = (xg / s16.float().unsqueeze(-1)).round_().clamp_(-127, 127).to(torch.int8).view(N, K)
    return q.contiguous(), s16.contiguous()


def gtime(fn, n, reps=8):
    for i in range(3): fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n): fn(i)
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * n)


for (N, K, name) in ((16480, 2560, 'in_proj'), (13312, 2560, 'qkv'), (2560, 6144, 'o_proj'), (248320, 2560, 'lm_head')):
    copies = max(2, int(400e6 // (N * K)))
    Ws = [w8q((torch.randn(N, K, device=dev) * 0.02).to(bf)) for _ in range(min(copies, 12 if N < 200000 else 2))]
    C = len(Ws)
    for M in (1, 4, 8):
        x = torch.randn(M, K, device=dev).to(bf)
        q, s = Ws[0]
        truth = x.float() @ (q.float() * s.float().repeat_interleave(128, 1)).T
        y0 = qf.w8g_gemm(x, q, s, 0x200 | (4 if M > 4 else 2))
        y1 = qf.w8g_gemm(x, q, s, 0x100)
        e0 = ((y0.float() - truth).norm() / truth.norm()).item()
        e1 = ((y1.float() - truth).norm() / truth.norm()).item()
        agree = (y0 == y1).float().mean().item()
        det = torch.equal(y1, qf.w8g_gemm(x, q, s, 0x100))
        t0 = gtime(lambda i: qf.w8g_gemm(x, Ws[i % C][0], Ws[i % C][1], 0x200 | (4 if M > 4 else 2)), 2 * C)
        t1 = gtime(lambda i: qf.w8g_gemm(x, Ws[i % C][0], Ws[i % C][1], 0x100), 2 * C)
        mb = (N * K + N * K // 128 * 2) / 1e6
        print(json.dumps(dict(shape=name, M=M, copies=C, rel_err_w8g=f'{e0:.2e}', rel_err_w8m=f'{e1:.2e}', bf16_agree=round(agree, 4),
                              det=det, w8g_us=round(t0, 1), w8m_us=round(t1, 1), w8g_TBs=round(mb / t0 / 1e3, 2), w8m_TBs=round(mb / t1 / 1e3, 2))), flush=True)
