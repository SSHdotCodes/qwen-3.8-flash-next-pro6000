"""WP-Spec: spec_fused helpers vs the torch expressions they replace (values, dtypes, shapes), bs 1..8, plus
graph-replay timing old vs new. Also the text-only spec mrope position shortcut. Run inside the public runtime image."""
import importlib.util, json, sys
import torch

spec = importlib.util.spec_from_file_location("spec_fused", __import__("sglang.srt.speculative.spec_fused", fromlist=["__file__"]).__file__)
sf = importlib.util.module_from_spec(spec); spec.loader.exec_module(sf)
from sglang.kernels.ops.speculative.eagle import fill_bonus_tokens_func

dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(7)
N, D = 4, 4  # num_draft_tokens, accept_index width (spec_steps + 1)


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def inputs(bs):
    seq = torch.randint(1, 260000, (bs,), generator=g, dtype=torch.int64).to(dev)
    acc = torch.randint(1, D + 1, (bs,), generator=g, dtype=torch.int32).to(dev)
    predict = torch.randint(0, 248320, (bs * N,), generator=g, dtype=torch.int32).to(dev)
    ai = torch.full((bs, D), -1, dtype=torch.int32)
    for b in range(bs):
        a = int(acc[b])
        ai[b, :a] = torch.arange(b * N, b * N + a, dtype=torch.int32)
    return seq, acc, predict, ai.to(dev)


ok = True
for bs in range(1, 9):
    for _ in range(20):
        seq, acc, predict, ai = inputs(bs)
        # post_verify
        new_seq_ref = seq + acc
        accept_tokens = predict[ai]
        bonus_ref = torch.empty_like(acc, dtype=torch.int32)
        fill_bonus_tokens_func(accept_tokens, acc, bonus_ref, ai.shape[1], bs)
        ns, bonus = sf.post_verify(seq, acc, predict, ai)
        ok &= same(ns, new_seq_ref) and same(bonus, bonus_ref)
        # draft_extend_prep
        nc_ref = acc - 1
        sel_ref = torch.arange(0, bs * N, N, device=dev) + acc - 1
        ids_ref = predict.to(torch.int64)
        nc, sel, ids = sf.draft_extend_prep(acc, predict, N)
        ok &= same(nc, nc_ref) and same(sel, sel_ref) and same(ids, ids_ref)
        # extend_lens
        for fo in (0, 2):
            pre_ref = (seq - fo).clamp(min=0).to(torch.int32)
            ext_ref = torch.full((bs,), N + fo, dtype=torch.int32, device=dev)
            plus_ref = seq + N
            pre, ext, plus = sf.extend_lens(seq, fo, N + fo, N)
            ok &= same(pre, pre_ref) and same(ext, ext_ref) and same(plus, plus_ref)
# edge: negative bonus node wraps like torch indexing; tiny seq_lens clamp
seq = torch.tensor([0, 1], dtype=torch.int64, device=dev)
pre, ext, plus = sf.extend_lens(seq, 2, 6, 4)
ok &= same(pre, (seq - 2).clamp(min=0).to(torch.int32))
acc = torch.tensor([1], dtype=torch.int32, device=dev)
predict = torch.arange(4, dtype=torch.int32, device=dev) + 10
ai = torch.tensor([[-1, -1, -1, -1]], dtype=torch.int32, device=dev)
ns, bonus = sf.post_verify(torch.tensor([5], device=dev), acc, predict, ai)
ok &= int(bonus[0]) == int(predict[ai][0, 0])
# mrope shortcut
pos = torch.randint(0, 10**6, (2, 4), generator=g, dtype=torch.int64).to(dev)
ref = (pos + torch.zeros((2, 1), dtype=torch.int64, device=dev)).flatten().unsqueeze(0).repeat(3, 1)
ok &= same(pos.flatten().unsqueeze(0).repeat(3, 1), ref)
assert ok, "speculative fused helpers differ from reference"
print(json.dumps({"equal": bool(ok)}), flush=True)


def gtime(fn, reps=50, iters=20):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        for _ in range(reps):
            fn()
    graph.replay(); torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = 1e9
    for _ in range(iters):
        st.record(); graph.replay(); en.record(); torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en) * 1000 / reps)
    return best


t = {}
for bs in (1, 2):
    seq, acc, predict, ai = inputs(bs)

    def pv_old():
        ns = seq + acc
        at = predict[ai]
        b = torch.empty_like(acc, dtype=torch.int32)
        fill_bonus_tokens_func(at, acc, b, D, bs)

    def de_old():
        nc = acc - 1
        sel = torch.arange(0, bs * N, N, device=dev) + acc - 1
        ids = predict.to(torch.int64)

    def el_old():
        pre = (seq - 0).clamp(min=0).to(torch.int32)
        ext = torch.full((bs,), N, dtype=torch.int32, device=dev)
        plus = seq + N

    t[f"post_verify bs{bs}"] = (round(gtime(pv_old), 2), round(gtime(lambda: sf.post_verify(seq, acc, predict, ai)), 2))
    t[f"draft_extend_prep bs{bs}"] = (round(gtime(de_old), 2), round(gtime(lambda: sf.draft_extend_prep(acc, predict, N)), 2))
    t[f"extend_lens bs{bs}"] = (round(gtime(el_old), 2), round(gtime(lambda: sf.extend_lens(seq, 0, N, N)), 2))
    pos = torch.arange(bs * N, device=dev).view(bs, -1)
    t[f"mrope bs{bs}"] = (round(gtime(lambda: (pos + torch.zeros((bs, 1), dtype=torch.int64, device=dev)).flatten().unsqueeze(0).repeat(3, 1)), 2),
                         round(gtime(lambda: pos.flatten().unsqueeze(0).repeat(3, 1)), 2))
print(json.dumps({"us_old_new": t}), flush=True)
