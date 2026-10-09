import sys, importlib.machinery, importlib.util, torch
path = sys.argv[1]
loader = importlib.machinery.ExtensionFileLoader('qwenfast', path)
spec = importlib.util.spec_from_loader('qwenfast', loader, origin=path)
Q = importlib.util.module_from_spec(spec); loader.exec_module(Q)
torch.manual_seed(0)
V = 248320
def check(x, k):
    v, i = Q.topk_rows(x, k)
    tv, ti = torch.topk(x.float(), k, dim=-1)
    assert torch.equal(v, tv), (k, (v - tv).abs().max())
    # same multiset of values; indices consistent with values
    assert torch.equal(x.float().gather(1, i), v)
    assert (i.sort(1).values.diff(dim=1) != 0).all()
for dt in (torch.float32, torch.bfloat16):
    for rows in (1, 2, 4, 8):
        for k in (1, 20, 28, 64):
            x = (torch.randn(rows, V, device='cuda') * 3).to(dt)
            check(x, k)
            # hot-map style: most entries -inf
            y = torch.full((rows, V), float('-inf'), device='cuda')
            hot = torch.randperm(V, device='cuda')[:97800]
            y[:, hot] = torch.randn(rows, 97800, device='cuda') * 4
            check(y.to(dt), k)
            # heavy ties
            z = torch.randint(-3, 3, (rows, V), device='cuda').float().to(dt)
            v, i = Q.topk_rows(z, k)
            assert torch.equal(v, torch.topk(z.float(), k).values) and torch.equal(z.float().gather(1, i), v)
print('topk_rows matches torch.topk')
def bench(fn, it=200):
    for _ in range(10): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(it): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / it * 1000
for rows in (1, 4):
    x = torch.randn(rows, V, device='cuda')
    print(f'rows {rows}: torch.topk(64) {bench(lambda: torch.topk(x, 64)):.1f} us | topk_rows(64) {bench(lambda: Q.topk_rows(x, 64)):.1f} us | topk_rows(28) {bench(lambda: Q.topk_rows(x, 28)):.1f} us')
# ---- trunc_sample vs a torch reference: identical q, and empirical sampling frequencies
rows, K = 3, 64
logits = torch.randn(rows, V, device='cuda') * 2
vals, idx = Q.topk_rows(logits, K)
temps = torch.tensor([1.0, 0.7, 1.0], device='cuda'); tks = torch.tensor([20, 64, 1], dtype=torch.int32, device='cuda')
tps = torch.tensor([0.95, 0.8, 1.0], device='cuda')
def ref_q():
    z = vals / temps.view(-1, 1); ar = torch.arange(K, device='cuda').view(1, K)
    z = torch.where(ar < tks.view(-1, 1).clamp(1, K), z, float('-inf')); p = torch.softmax(z, -1)
    p = torch.where((p.cumsum(-1) - p) < tps.view(-1, 1), p, 0.0); return p / p.sum(-1, keepdim=True)
pr = ref_q()
cnt = torch.zeros(rows, K, device='cuda')
for t in range(20000):
    expo = torch.empty_like(vals).exponential_(1.0)
    q = torch.zeros(rows, V, device='cuda')
    px, x = Q.trunc_sample(vals, idx, temps, tks, tps, expo, q)
    if t == 0:
        assert torch.allclose(q.gather(1, idx), pr, atol=1e-6), (q.gather(1, idx) - pr).abs().max()
        assert torch.allclose(q.sum(1), torch.ones(rows, device='cuda'), atol=1e-5)
    pos = (idx == x).float().argmax(1)
    assert torch.allclose(px.view(-1), pr.gather(1, pos.view(-1, 1)).view(-1), atol=1e-6)
    cnt[torch.arange(rows), pos] += 1
freq = cnt / 20000
print('trunc_sample q matches reference; max |freq - p| =', (freq - pr).abs().max().item(), '(sampling noise ~', (pr * (1 - pr) / 20000).sqrt().max().item(), ')')
