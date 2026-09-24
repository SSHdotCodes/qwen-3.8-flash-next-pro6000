"""Sparse verify sampling (topk + fused renorm/chain kernel) vs sglang's dense softmax -> top_k_renorm ->
top_p_renorm -> chain_speculative_sampling_triton, same logits / draft probs / coins."""
import os, json, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_qf import build
qf = build()
from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob
from sglang.kernels.ops.speculative.reject_sampling import chain_speculative_sampling_triton

dev = 'cuda'
V, D = 248320, 4
torch.manual_seed(0)


def make(bs, scale, idt):
    logits = torch.randn(bs * D, V, device=dev) * scale
    # a few strong candidates per row (like a real LM head)
    hot = torch.randint(0, V, (bs * D, 30), device=dev)
    logits.scatter_(1, hot, torch.randn(bs * D, 30, device=dev) * 2 + scale * 4)
    dl = logits.view(bs, D, V)[:, :D - 1] + torch.randn(bs, D - 1, V, device=dev) * 0.7
    draft_probs = torch.softmax(dl, -1).contiguous()
    cand = torch.zeros(bs, D, dtype=idt, device=dev)
    cand[:, 0] = torch.randint(0, V, (bs,), device=dev).to(idt)
    for s in range(1, D):
        cand[:, s] = torch.multinomial(draft_probs[:, s - 1], 1).squeeze(1).to(idt)
    ri = torch.arange(bs * D, device=dev, dtype=idt).view(bs, D)
    return logits, draft_probs, cand, ri


def stock(logits, T, tk, tp, use_p, draft_probs, cand, ri, coins, cf):
    bs = cand.shape[0]
    predict = torch.zeros(bs * D, dtype=torch.int32, device=dev)
    ai = torch.full((bs, D), -1, dtype=torch.int32, device=dev)
    an = torch.empty(bs, dtype=torch.int32, device=dev)
    p = torch.softmax(logits / T.repeat_interleave(D, 0), -1)
    p = top_k_renorm_prob(p, tk.repeat_interleave(D, 0))
    if use_p:
        p = top_p_renorm_prob(p, tp.repeat_interleave(D, 0))
    chain_speculative_sampling_triton(predicts=predict, accept_index=ai, accept_token_num=an, candidates=cand,
                                      retrive_index=ri, retrive_next_token=None, retrive_next_sibling=None,
                                      uniform_samples=coins, uniform_samples_for_final_sampling=cf,
                                      target_probs=p.view(bs, D, V), draft_probs=draft_probs,
                                      threshold_single=1.0, threshold_acc=1.0, deterministic=True)
    return predict, ai, an


def mine(logits, T, tk, tp, use_p, draft_probs, cand, ri, coins, cf, km):
    bs = cand.shape[0]
    predict = torch.zeros(bs * D, dtype=torch.int32, device=dev)
    ai = torch.full((bs, D), -1, dtype=torch.int32, device=dev)
    an = torch.empty(bs, dtype=torch.int32, device=dev)
    vals, idx = torch.topk(logits, km, dim=-1)
    qf.spec_topk_chain(vals, idx, T, tk, tp, use_p, draft_probs, cand, ri, coins, cf, predict, ai, an)
    return predict, ai, an


res = []
for bs in (1, 2):
    for idt in (torch.int64, torch.int32):
        for (temp, topk, topp, use_p) in ((1.0, 20, 0.95, True), (0.7, 20, 0.8, True), (1.0, 40, 1.0, True), (1.0, 20, 1.0, False), (1.3, 5, 0.9, True)):
            for scale in (0.5, 1.5, 3.0):
                logits, dp, cand, ri = make(bs, scale, idt)
                T = torch.full((bs, 1), temp, device=dev)
                tk = torch.full((bs,), topk, dtype=torch.int32, device=dev)
                tp = torch.full((bs,), topp, device=dev)
                km = min(64, topk + 8)
                same = 0
                accs = []
                for trial in range(200):
                    coins = torch.rand(bs, D, device=dev)
                    cf = torch.rand(bs, device=dev)
                    a = stock(logits, T, tk, tp, use_p, dp, cand, ri, coins, cf)
                    b = mine(logits, T, tk, tp, use_p, dp, cand, ri, coins, cf, km)
                    ok = all(torch.equal(x, y) for x, y in zip(a, b))
                    same += ok
                    accs.append(a[2].float().mean().item())
                row = dict(bs=bs, idt=str(idt)[6:], T=temp, top_k=topk, top_p=topp, use_p=use_p, scale=scale,
                           exact_match=same / 200, mean_accept=round(sum(accs) / len(accs), 3))
                res.append(row)
                print(json.dumps(row), flush=True)

# timing (bs=1, the common case), graphs
for bs in (1, 2):
    logits, dp, cand, ri = make(bs, 1.5, torch.int64)
    T = torch.full((bs, 1), 1.0, device=dev); tk = torch.full((bs,), 20, dtype=torch.int32, device=dev)
    tp = torch.full((bs,), 0.95, device=dev); coins = torch.rand(bs, D, device=dev); cf = torch.rand(bs, device=dev)

    def timeit(fn, n=20):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n): fn()
        g.replay(); torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(10): g.replay()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) * 1e3 / (10 * n)
    t_stock = timeit(lambda: stock(logits, T, tk, tp, True, dp, cand, ri, coins, cf))
    t_mine = timeit(lambda: mine(logits, T, tk, tp, True, dp, cand, ri, coins, cf, 28))
    t_topk = timeit(lambda: torch.topk(logits, 28, dim=-1))
    row = dict(bs=bs, stock_us=round(t_stock, 1), mine_us=round(t_mine, 1), topk_only_us=round(t_topk, 1))
    res.append(row)
    print(json.dumps(row), flush=True)
json.dump(res, open(os.environ.get('QWENFAST_TEST_OUT', '/tmp') + '/test_sample.json', 'w'), indent=1)
