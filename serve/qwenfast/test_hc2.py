import json, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_qf import build
qf = build()
from sglang.srt.layers.hc_mix_triton import fused_hc_mix
from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm
from sglang.kernels.ops.elementwise.hc_combine import hc_combine_split
torch.manual_seed(0)
HC, HS, R, K = 4, 2560, 320, 10240
dev, bf = 'cuda', torch.bfloat16
eps = 1e-6
copies = 16
W = [dict(wn=(torch.randn(K, device=dev) * 0.1).to(bf), wd=(torch.randn(R, K, device=dev) * 0.02).to(bf),
          wu=(torch.randn(K, R, device=dev) * 0.05).to(bf), wi=(torch.randn(HC, K, device=dev) * 0.02).to(bf)) for _ in range(copies)]
normed_s = torch.empty(8 * K, device=dev, dtype=bf)
partial = torch.empty(HC * 8 * 324, device=dev, dtype=torch.float32)
bar = torch.zeros(2, device=dev, dtype=torch.int32)


def ref(x, y, w):
    normed = grouped_gemma_rmsnorm(x, w['wn'], HS, eps)
    mixed = fused_hc_mix(normed, w['wd'], w['wu'], HC, HS)
    comb = hc_combine_split(y, x, normed, w['wi'], HC, HS)
    return mixed, comb


def mine(x, y, w, pdl=True):
    mixed, inj = qf.hc_mix_fused(x, w['wn'], w['wd'], w['wi'], w['wu'], eps, normed_s, partial, bar, pdl)
    comb = qf.hc_combine_apply(y, x, inj, pdl)
    return mixed, comb


def timeit(fn, iters=1600):
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
    x = torch.randn(M, K, device=dev).to(bf)
    y = torch.randn(M, HS, device=dev).to(bf)
    a_m, a_c = ref(x, y, W[0])
    b_m, b_c = mine(x, y, W[0])
    c_m, c_c = mine(x, y, W[0])
    err_m = ((a_m.float() - b_m.float()).norm() / a_m.float().norm()).item()
    agree_m = (a_m == b_m).float().mean().item()
    agree_c = (a_c == b_c).float().mean().item()
    det = torch.equal(b_m, c_m) and torch.equal(b_c, c_c)
    t_ref = timeit(lambda i: ref(x, y, W[i]))
    t_mine = timeit(lambda i: mine(x, y, W[i]))
    t_nopdl = timeit(lambda i: mine(x, y, W[i], False))
    row = dict(M=M, mix_rel_err=err_m, mix_bf16_agree=round(agree_m, 4), combine_bf16_agree=round(agree_c, 4),
               deterministic=det, ref_us=round(t_ref, 1), mine_us=round(t_mine, 1), mine_nopdl_us=round(t_nopdl, 1), bar=bar.tolist())
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump(rows, open(os.environ.get('QWENFAST_TEST_OUT', '/tmp') + '/test_hc2.json', 'w'), indent=1)
