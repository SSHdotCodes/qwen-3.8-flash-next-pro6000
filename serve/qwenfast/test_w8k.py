"""w8g kernels in the qwenfast build: exactness vs the BF16 kernels on dequantized weights + timing.
python3 test_w8k.py SO_PATH [bench]   (inside the SGLang image)"""
import sys, importlib.machinery, importlib.util
import torch
path = sys.argv[1]
loader = importlib.machinery.ExtensionFileLoader('qwenfast', path)
spec = importlib.util.spec_from_loader('qwenfast', loader, origin=path)
Q = importlib.util.module_from_spec(spec); loader.exec_module(Q)
dev = 'cuda'
torch.manual_seed(0)

def quant(w):
    N, K = w.shape
    G = 128 if K % 128 == 0 else 64
    xg = w.float().view(N, K // G, G)
    s = (xg.abs().amax(2) / 127).clamp_(min=1e-30).to(torch.bfloat16)
    q = (xg / s.float().unsqueeze(-1)).round_().clamp_(-127, 127).to(torch.int8).view(N, K).contiguous()
    return q, s.contiguous()

def deq(q, s):
    N, K = q.shape
    G = K // s.shape[1]
    return (q.float().view(N, K // G, G) * s.float().unsqueeze(-1)).view(N, K).to(torch.bfloat16)

def bench(fn, iters=300):
    for i in range(10): fn(i)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for i in range(iters): fn(i)
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000

BENCH = len(sys.argv) > 2 and sys.argv[2] == 'bench'
# ---------------- HC mix ----------------
normed = torch.empty(8 * 10240, dtype=torch.bfloat16, device=dev)
partial = torch.empty(4 * 8 * 324, dtype=torch.float32, device=dev)
bar = torch.zeros(2, dtype=torch.int32, device=dev)
C = 12 if BENCH else 1
wds = [(torch.randn(320, 10240, device=dev) * 0.01).to(torch.bfloat16) for _ in range(C)]
wus = [(torch.randn(10240, 320, device=dev) * 0.05).to(torch.bfloat16) for _ in range(C)]
winj = (torch.randn(4, 10240, device=dev) * 0.01).to(torch.bfloat16)
wn = (torch.randn(10240, device=dev) * 0.1).to(torch.bfloat16)
qd = [quant(w) for w in wds]; qu = [quant(w) for w in wus]
dd = [deq(*p) for p in qd]; du = [deq(*p) for p in qu]
for M in (1, 2, 3, 4, 8):
    x = torch.randn(M, 10240, device=dev).to(torch.bfloat16)
    for wi in (winj, None):
        o1, i1 = Q.hc_mix_fused(x, wn, dd[0], wi, du[0], 1e-6, normed, partial, bar, True)
        o2, i2 = Q.hc_mix_fused_w8(x, wn, qd[0][0], qd[0][1], wi, qu[0][0], qu[0][1], 1e-6, normed, partial, bar, True)
        assert torch.equal(o1, o2), ('hc out', M, (o1.float() - o2.float()).abs().max())
        if wi is not None:
            assert torch.equal(i1, i2), ('hc inj', M)
    if BENCH:
        t1 = bench(lambda i: Q.hc_mix_fused(x, wn, wds[i % C], winj, wus[i % C], 1e-6, normed, partial, bar, True))
        t2 = bench(lambda i: Q.hc_mix_fused_w8(x, wn, qd[i % C][0], qd[i % C][1], winj, qu[i % C][0], qu[i % C][1], 1e-6, normed, partial, bar, True))
        print(f'hc_mix M={M}: bf16 {t1:6.1f} us  w8 {t2:6.1f} us', flush=True)
print('hc_mix_fused_w8 bit-exact vs hc_mix_fused on dequantized weights', flush=True)
del wds, wus, qd, qu, dd, du
# ---------------- GDN out ----------------
C = 12 if BENCH else 1
ws = [(torch.randn(2560, 6144, device=dev) * 0.01).to(torch.bfloat16) for _ in range(C)]
qs = [quant(w) for w in ws]; ds = [deq(*p) for p in qs]
nw = (torch.randn(128, device=dev) * 0.1 + 1).to(torch.bfloat16)
for T in (1, 2, 3, 4):
    core = torch.randn(T * 48, 128, device=dev).to(torch.bfloat16)
    z = torch.randn(T * 48, 128, device=dev).to(torch.bfloat16)
    for swish in (True, False):
        y1 = Q.gdn_norm_oproj(core, z, nw, 1e-6, swish, ds[0], True, True)
        y2 = Q.gdn_norm_oproj_w8(core, z, nw, 1e-6, swish, qs[0][0], qs[0][1], True, True)
        assert torch.equal(y1, y2), ('gdn', T, (y1.float() - y2.float()).abs().max())
    if BENCH:
        t1 = bench(lambda i: Q.gdn_norm_oproj(core, z, nw, 1e-6, True, ws[i % C], True, True))
        t2 = bench(lambda i: Q.gdn_norm_oproj_w8(core, z, nw, 1e-6, True, qs[i % C][0], qs[i % C][1], True, True))
        print(f'gdn_out T={T}: bf16 {t1:6.1f} us  w8 {t2:6.1f} us', flush=True)
print('gdn_norm_oproj_w8 bit-exact vs gdn_norm_oproj on dequantized weights', flush=True)
del ws, qs, ds
# ---------------- GEMV ----------------
for N, K in [(16480, 2560), (13312, 2560), (2560, 6144), (248320, 2560), (1280, 2560), (2560, 640)]:
    C = max(2, int(500e6 // (N * K)) + 1) if BENCH else 1
    ws = [(torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16) for _ in range(C)]
    qs = [quant(w) for w in ws]
    for M in (1, 2, 4, 8):
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        y = Q.w8g_gemm(x, qs[0][0], qs[0][1], 0).float()
        ref = x.float() @ deq(*qs[0]).float().T
        err = ((y - ref).norm() / ref.norm()).item()
        assert err < 4e-3, (N, K, M, err)
        if BENCH and M in (1, 4, 8):
            tb = bench(lambda i: torch.nn.functional.linear(x, ws[i % C]))
            res = []
            for rpw in (1, 2, 4):
                tw = bench(lambda i: Q.w8g_gemm(x, qs[i % C][0], qs[i % C][1], rpw))
                res.append(f'rpw{rpw} {tw:6.1f} us {N * K * (1 + 1 / 64) / tw / 1e3:5.0f} GB/s')
            print(f'{N:6d}x{K:4d} M={M}: cuBLAS bf16 {tb:6.1f} us {N * K * 2 / tb / 1e3:5.0f} GB/s | ' + ' | '.join(res), flush=True)
    del ws, qs
    torch.cuda.empty_cache()
print('w8g_gemm ok')
