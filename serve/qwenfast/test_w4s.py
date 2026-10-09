import sys, importlib.machinery, importlib.util, torch
path = sys.argv[1]
loader = importlib.machinery.ExtensionFileLoader('qwenfast', path)
spec = importlib.util.spec_from_loader('qwenfast', loader, origin=path)
Q = importlib.util.module_from_spec(spec); loader.exec_module(Q)
torch.manual_seed(0)
N = 98076
hot = (torch.randn(N, 2560, device='cuda') * 0.02).to(torch.bfloat16)
amax = hot.float().abs().amax(1).clamp_min(1e-30); s8 = (amax / 127).contiguous()
q8 = torch.round(hot.float() / s8[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
xg = hot.float().view(N, 40, 64); s4 = (xg.abs().amax(2) / 7).clamp_min(1e-30).to(torch.bfloat16)
q = (xg / s4.float().unsqueeze(-1)).round().clamp(-7, 7).add(8).to(torch.uint8).view(N, 2560)
q4 = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
deq4 = ((q.float() - 8).view(N, 40, 64) * s4.float().unsqueeze(-1)).view(N, 2560)
for M in (1, 2, 4):
    h = torch.randn(M, 2560, device='cuda').to(torch.bfloat16)
    y4 = Q.w4g_gemm(h, q4, s4).float()
    ref4 = h.float() @ deq4.T
    e = ((y4 - ref4).norm() / ref4.norm()).item(); assert e < 5e-3, e
    exact8 = (h.float() @ q8.float().T) * s8
    _, cidx = Q.topk_rows(y4, 64)
    ex = Q.rows_dot_i8(h, q8, s8, cidx)
    assert torch.allclose(ex, exact8.gather(1, cidx), rtol=1e-4, atol=1e-3), (ex - exact8.gather(1, cidx)).abs().max()
    top20 = exact8.topk(20, dim=1).indices
    recall = torch.tensor([len(set(top20[m].tolist()) & set(cidx[m].tolist())) / 20 for m in range(M)])
    print(f'M={M}: w4 rel err {e:.2e}; INT8 top-20 recall in INT4 top-64 {recall.mean():.3f}')
def bench(fn, it=200):
    for _ in range(10): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(it): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / it * 1000
h = torch.randn(1, 2560, device='cuda').to(torch.bfloat16)
copies = [(q4.clone(), s4.clone(), q8.clone(), s8.clone()) for _ in range(3)]
i = [0]
def screen():
    c = copies[i[0] % 3]; i[0] += 1
    _, ci = Q.topk_rows(Q.w4g_gemm(h, c[0], c[1]), 64); return Q.rows_dot_i8(h, c[2], c[3], ci)
def int8():
    c = copies[i[0] % 3]; i[0] += 1
    return Q.w8_gemm(h, c[2], c[3], 0)
print(f'hot head M=1: INT8 w8_gemm {bench(int8):.1f} us | INT4 screen + top-64 + exact refine {bench(screen):.1f} us')
