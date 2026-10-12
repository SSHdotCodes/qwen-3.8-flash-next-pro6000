"""W4A16 decode MoE (qf_a16moe.cu) vs SGLang's Marlin MoE path on real checkpoint layers, production repack."""
import json, os, sys, time
import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

D = os.path.dirname(os.path.abspath(__file__))
qf = load(name='qfa16', sources=[f'{D}/qfa16_bind.cpp', f'{D}/qf_a16moe.cu'], build_directory=os.environ.get('QF_BUILD', f'{D}/build_a16'),
          extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-gencode=arch=compute_120a,code=sm_120a', '--use_fast_math', '-lineinfo'], verbose=False)

from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import (_permute_moe_fp4_scales_for_marlin,
    nvfp4_marlin_process_global_scale, nvfp4_marlin_process_scales)
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add

SNAP = '/models/target'
LAYERS = [int(v) for v in os.environ.get('LAYERS', '0,4,8,13,17,21,26,30,34,39,43,47').split(',')]
E, Hd, I, K = 512, 2560, 640, 10
dev = 'cuda'
idx = json.load(open(f'{SNAP}/model.safetensors.index.json'))['weight_map']

def load_layer(L):
    pre = f'model.language_model.layers.{L}.mlp.'
    names = [f'{pre}experts.{e}.{p}.{s}' for e in range(E) for p in ('gate_proj', 'up_proj', 'down_proj')
             for s in ('weight_packed', 'weight_scale', 'weight_global_scale')]
    names += [pre + 'gate.weight', pre + 'shared_expert_gate.weight'] + [pre + f'shared_expert.{p}.weight' for p in ('gate_proj', 'up_proj', 'down_proj')]
    T, files = {}, {}
    for n in names: files.setdefault(idx[n], []).append(n)
    for f, ns in files.items():
        with safe_open(f'{SNAP}/{f}', framework='pt', device=dev) as h:
            for n in ns: T[n] = h.get_tensor(n)
    st = lambda p, s: torch.stack([T[f'{pre}experts.{e}.{p}.{s}'] for e in range(E)])
    w13 = torch.cat([st('gate_proj', 'weight_packed'), st('up_proj', 'weight_packed')], 1).contiguous()   # [E, 1280, 1280] u8
    s13 = torch.cat([st('gate_proj', 'weight_scale'), st('up_proj', 'weight_scale')], 1).contiguous()
    w2, s2 = st('down_proj', 'weight_packed').contiguous(), st('down_proj', 'weight_scale').contiguous()
    g13 = torch.stack([st('gate_proj', 'weight_global_scale').reshape(E), st('up_proj', 'weight_global_scale').reshape(E)], 1).float()
    g2 = st('down_proj', 'weight_global_scale').reshape(E).float()
    canon = dict(w13=w13.clone(), s13=s13.clone(), w2=w2.clone(), s2=s2.clone(), g13=g13.clone(), g2=g2.clone())
    # production _process_weights_marlin
    perm = torch.empty(0, dtype=torch.int, device=dev)
    def repack(weight, size_n, size_k):
        for i in range(E):
            q = weight[i].view(torch.int32).T.contiguous()
            out = gptq_marlin_repack(b_q_weight=q, perm=perm, size_k=size_k, size_n=size_n, num_bits=4)
            weight[i].view(-1).copy_(out.view(torch.uint8).reshape(-1))
        return weight.view(torch.int32).view(E, size_k // 16, size_n * 2)
    sc = lambda sf, n, k: _permute_moe_fp4_scales_for_marlin(sf.to(torch.bfloat16), num_experts=E, size_n=n, size_k=k,
                                                               group_size=16, process_scales=nvfp4_marlin_process_scales)
    d = dict(
        w13=repack(w13, 2 * I, Hd), w2=repack(w2, Hd, I), s13=sc(s13, 2 * I, Hd), s2=sc(s2, Hd, I),
        gs13=nvfp4_marlin_process_global_scale((1 / g13[:, 0]).to(torch.bfloat16)),
        gs2=nvfp4_marlin_process_global_scale((1 / g2).to(torch.bfloat16)),
        router=T[pre + 'gate.weight'], wg=T[pre + 'shared_expert_gate.weight'].reshape(-1).contiguous(), canon=canon)
    gu = torch.cat([T[pre + 'shared_expert.gate_proj.weight'], T[pre + 'shared_expert.up_proj.weight']], 0).contiguous()
    d['sq1'], d['ss1'] = w8q(gu)
    d['sq2'], d['ss2'] = w8q(T[pre + 'shared_expert.down_proj.weight'].contiguous())
    return d

def w8q(w):          # qwenfast.w8_quant
    N, K = w.shape; G = 128
    xg = w.float().view(N, K // G, G)
    s16 = (xg.abs().amax(2) / 127).clamp_(min=1e-30).to(torch.bfloat16)
    q = (xg / s16.float().unsqueeze(-1)).round_().clamp_(-127, 127).to(torch.int8).view(N, K)
    return q.contiguous(), s16.contiguous()

def shared_ref(d, x):   # w8g dequant, fp32 matmul, bf16 rounding at the same points as w8g_gemm + act_and_mul
    w1 = d['sq1'].float() * d['ss1'].float().repeat_interleave(128, 1)
    w2 = d['sq2'].float() * d['ss2'].float().repeat_interleave(128, 1)
    gu = (x.float() @ w1.T).to(torch.bfloat16).float()
    h = (torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]).to(torch.bfloat16).float()
    return (h @ w2.T).to(torch.bfloat16)

t0 = time.time()
layers = [load_layer(L) for L in LAYERS]
print('loaded layers', LAYERS, 'in', round(time.time() - t0, 1), 's; w13', tuple(layers[0]['w13'].shape), layers[0]['s13'].dtype,
      tuple(layers[0]['s13'].shape), 'gs', layers[0]['gs13'].dtype, flush=True)
ws = marlin_make_workspace(torch.device(dev), 4)

def route(d, x):
    logits = x.float() @ d['router'].float().T
    p = torch.softmax(logits, -1)
    w, ids = torch.topk(p, K, -1)
    w = w / w.sum(-1, keepdim=True)
    return logits, w.float().contiguous(), ids.int().contiguous()

def marlin_path(d, x, logits, w, ids, shared):
    out = fused_marlin_moe(hidden_states=x, w1=d['w13'], w2=d['w2'], w1_scale=d['s13'], w2_scale=d['s2'], gating_output=logits,
                           topk_weights=w, topk_ids=ids, global_num_experts=E, w1_global_scale=d['gs13'], w2_global_scale=d['gs2'],
                           workspace=ws, num_bits=4, is_k_full=True, inplace=False)
    fused_gate_sigmoid_mul_add(x, d['wg'], shared, out)
    return out

hbuf = torch.empty(80 * 8 * I, dtype=torch.bfloat16, device=dev)
ybuf = torch.empty(8 * 10 * Hd, dtype=torch.bfloat16, device=dev)
gl = torch.empty(8, dtype=torch.float32, device=dev)
cnt = torch.zeros(20, dtype=torch.int32, device=dev)
ctrl = torch.zeros(1024, dtype=torch.int32, device=dev)
pbuf = torch.empty(80 * 10 * 2 * 1024, dtype=torch.float32, device=dev)
DBG = [torch.empty(0, dtype=torch.int64, device=dev)]

E0 = torch.empty(0, device=dev)
shbuf = torch.empty(8 * (640 + 2560), dtype=torch.bfloat16, device=dev)
def a16_path(d, x, w, ids, shared, out, phase=3, fused_shared=False):
    if fused_shared:
        return qf.a16_moe_decode(x, ids, w, d['w13'], d['s13'], d['gs13'], d['w2'], d['s2'], d['gs2'], d['wg'], E0, hbuf, ybuf, gl,
                                 ctrl, out, 4, DBG[0], pbuf, d['sq1'], d['ss1'], d['sq2'], d['ss2'], shbuf)
    return qf.a16_moe_decode(x, ids, w, d['w13'], d['s13'], d['gs13'], d['w2'], d['s2'], d['gs2'], d['wg'], shared, hbuf, ybuf, gl,
                             ctrl if phase == 4 else cnt, out, phase, DBG[0], pbuf, E0, E0, E0, E0, E0)

torch.manual_seed(0)
print('--- accuracy (vs Marlin path incl. shared gate) ---')
for M in (1, 2, 4, 8):
    for corr in (0.0, 0.7):
        d = layers[0]
        base = torch.randn(1, Hd, device=dev)
        x = ((corr * base + (1 - corr) * torch.randn(M, Hd, device=dev)) * 0.6).to(torch.bfloat16).contiguous()
        logits, w, ids = route(d, x)
        shared = (torch.randn(M, Hd, device=dev) * 0.05).to(torch.bfloat16)
        ref = marlin_path(d, x, logits, w, ids, shared)
        out = torch.empty(M, Hd, dtype=torch.bfloat16, device=dev)
        got = a16_path(d, x, w, ids, shared, out)
        got4 = a16_path(d, x, w, ids, shared, torch.empty_like(out), 4)
        torch.cuda.synchronize()
        rel4 = ((got4.float() - ref.float()).norm() / ref.float().norm()).item()
        diff = (got.float() - ref.float()).abs()
        rel = diff.norm() / ref.float().norm()
        exact = (got == ref).float().mean().item()
        uniq = len(set(ids.flatten().tolist()))
        print(f'M={M} corr={corr}: unique experts {uniq:2d}  rel L2 {rel:.2e}  max|d| {diff.max().item():.3e}  '
              f'(|ref| max {ref.float().abs().max().item():.3f})  exact-bf16 {exact:.3f}  cnt-after {cnt.abs().sum().item()} ctrl-after {ctrl.abs().sum().item()}  persistent rel L2 {rel4:.2e}')

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
def deq(wp, sf, g):          # canonical packed [N, K/2] u8, scales [N, K/16] fp8, global (stored, = 1/multiplier)
    lo, hi = (wp & 15).long(), (wp >> 4).long()
    v = torch.stack([E2M1[lo], E2M1[hi]], -1).reshape(wp.shape[0], -1)
    return v * sf.float().repeat_interleave(16, 1) / g
def truth(d, x, w, ids, shared):
    c = d['canon']; out = torch.zeros(x.shape[0], Hd, device=dev)
    for t in range(x.shape[0]):
        for kp in range(K):
            e = int(ids[t, kp])
            w13 = deq(c['w13'][e], c['s13'][e], c['g13'][e, 0]); w2 = deq(c['w2'][e], c['s2'][e], c['g2'][e])
            gu = w13 @ x[t].float(); h = torch.nn.functional.silu(gu[:I]) * gu[I:]
            out[t] += w[t, kp] * (w2 @ h)
    gate = torch.sigmoid(x.float() @ d['wg'].float())
    return out + gate[:, None] * shared.float()
print('--- fused shared expert vs precomputed shared (w8g reference) ---')
for M in (1, 2, 4, 8):
    d = layers[0]
    x = (torch.randn(M, Hd, device=dev) * 0.6).to(torch.bfloat16).contiguous()
    logits, w, ids = route(d, x)
    sh = shared_ref(d, x)
    a = a16_path(d, x, w, ids, sh, torch.empty(M, Hd, dtype=torch.bfloat16, device=dev), 4)
    b = a16_path(d, x, w, ids, None, torch.empty(M, Hd, dtype=torch.bfloat16, device=dev), 4, fused_shared=True)
    hs_ref = None
    torch.cuda.synchronize()
    dd = (a.float() - b.float()).abs()
    print(f'M={M}: fused-shared vs precomputed: rel L2 {(dd.norm() / a.float().norm()).item():.2e} max|d| {dd.max().item():.2e} '
          f'exact {(a == b).float().mean().item():.3f}  ctrl-after {ctrl.abs().sum().item()}')
print('--- every row count: fused persistent vs Marlin path + w8g shared, and batch invariance vs single rows ---')
for M in range(1, 9):
    for corr in (0.0, 0.7):
        d = layers[0]
        base = torch.randn(1, Hd, device=dev)
        x = ((corr * base + (1 - corr) * torch.randn(M, Hd, device=dev)) * 0.6).to(torch.bfloat16).contiguous()
        logits, w, ids = route(d, x)
        sh = shared_ref(d, x)
        ref = marlin_path(d, x, logits, w, ids, sh)
        got = a16_path(d, x, w, ids, None, torch.empty(M, Hd, dtype=torch.bfloat16, device=dev), 4, fused_shared=True)
        single = torch.cat([a16_path(d, x[r:r + 1].contiguous(), w[r:r + 1].contiguous(), ids[r:r + 1].contiguous(), None,
                                     torch.empty(1, Hd, dtype=torch.bfloat16, device=dev), 4, fused_shared=True).clone()
                            for r in range(M)])
        torch.cuda.synchronize()
        dd = (got.float() - ref.float())
        rows_rel = (dd.norm(dim=1) / ref.float().norm(dim=1)).max().item()
        print(f'M={M} corr={corr}: worst-row rel L2 vs Marlin {rows_rel:.2e}  batch-invariant {torch.equal(got, single)}  ctrl-after {ctrl.abs().sum().item()}')
print('--- error vs fp32 truth (canonical dequant) ---')
for M in (1, 4):
    d = layers[0]
    x = (torch.randn(M, Hd, device=dev) * 0.6).to(torch.bfloat16).contiguous()
    logits, w, ids = route(d, x)
    shared = (torch.randn(M, Hd, device=dev) * 0.05).to(torch.bfloat16)
    ref = marlin_path(d, x, logits, w, ids, shared).float()
    got = a16_path(d, x, w, ids, shared, torch.empty(M, Hd, dtype=torch.bfloat16, device=dev)).float()
    tr = truth(d, x, w, ids, shared)
    print(f'M={M}: rel L2 vs truth: marlin {((ref - tr).norm() / tr.norm()).item():.2e}  a16 {((got - tr).norm() / tr.norm()).item():.2e}')

print('--- timing: CUDA graph, layers cycled', LAYERS, '---')
def bench(fn, iters=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(10): g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters): g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000

for M in (1, 4, 8):
    for corr in (0.0, 0.7):
        xs, rs, shs, outs = [], [], [], []
        for d in layers:
            base = torch.randn(1, Hd, device=dev)
            x = ((corr * base + (1 - corr) * torch.randn(M, Hd, device=dev)) * 0.6).to(torch.bfloat16).contiguous()
            xs.append(x); rs.append(route(d, x)); shs.append((torch.randn(M, Hd, device=dev) * 0.05).to(torch.bfloat16))
            outs.append(torch.empty(M, Hd, dtype=torch.bfloat16, device=dev))
        uniq = sum(len(set(r[2].flatten().tolist())) for r in rs) / len(rs)
        def run_marlin():
            for d, x, (lg, w, ids), sh in zip(layers, xs, rs, shs): marlin_path(d, x, lg, w, ids, sh)
        def run_a16():
            for d, x, (lg, w, ids), sh, o in zip(layers, xs, rs, shs, outs): a16_path(d, x, w, ids, sh, o)
        def run_g1():
            for d, x, (lg, w, ids), sh, o in zip(layers, xs, rs, shs, outs): a16_path(d, x, w, ids, sh, o, 1)
        def run_g2():
            for d, x, (lg, w, ids), sh, o in zip(layers, xs, rs, shs, outs): a16_path(d, x, w, ids, sh, o, 2)
        def run_p():
            for d, x, (lg, w, ids), sh, o in zip(layers, xs, rs, shs, outs): a16_path(d, x, w, ids, sh, o, 4)
        def run_ps():
            for d, x, (lg, w, ids), sh, o in zip(layers, xs, rs, shs, outs): a16_path(d, x, w, ids, None, o, 4, fused_shared=True)
        tm, ta = bench(run_marlin) / len(layers), bench(run_a16) / len(layers)
        tp, tps = bench(run_p) / len(layers), bench(run_ps) / len(layers)
        refs = []
        for d, x, (lg, w, ids), sh in zip(layers, xs, rs, shs):
            o1 = torch.empty(M, Hd, dtype=torch.bfloat16, device=dev)
            a16_path(d, x, w, ids, None, o1, 4, fused_shared=True); torch.cuda.synchronize(); refs.append(o1.clone())
        for _ in range(5): run_ps()
        torch.cuda.synchronize()
        print('    back-to-back fused calls equal single synced calls:', all(torch.equal(o1, r) for o1, r in zip(outs, refs)))
        print(f'    persistent {tp:6.1f} us ({uniq * 2.7648 / 1e3 / tp * 1e3:.2f} TB/s)   + fused shared expert {tps:6.1f} us')
        t1, t2 = bench(run_g1) / len(layers), bench(run_g2) / len(layers)
        print(f'    gemm1 alone {t1:6.1f} us ({uniq * 1280 * 1440 / 1e3 / t1 / 1e3:.2f} TB/s)   gemm2 alone {t2:6.1f} us ({uniq * 2560 * 360 / 1e3 / t2 / 1e3:.2f} TB/s)')
        gb = uniq * (1280 * 1280 + 1280 * 160 + 2560 * 320 + 2560 * 40) / 1e9 * 1.0
        print(f'M={M} corr={corr}: unique {uniq:5.1f}  marlin+sum+gate {tm:6.1f} us   a16 {ta:6.1f} us   '
              f'({gb / (ta * 1e-6) / 1e3:.2f} TB/s eff.; marlin {gb / (tm * 1e-6) / 1e3:.2f})')

print('--- timeline (persistent, one layer after another layer, M=4) ---')
import collections
for corr in (0.7,):
    xs, rs, shs, outs = [], [], [], []
    for d in layers[:2]:
        base = torch.randn(1, Hd, device=dev)
        x = ((corr * base + (1 - corr) * torch.randn(4, Hd, device=dev)) * 0.6).to(torch.bfloat16).contiguous()
        xs.append(x); rs.append(route(d, x)); shs.append((torch.randn(4, Hd, device=dev) * 0.05).to(torch.bfloat16))
        outs.append(torch.empty(4, Hd, dtype=torch.bfloat16, device=dev))
    for _ in range(3):
        for d, x, (lg, w, ids), sh, o in zip(layers[:2], xs, rs, shs, outs): a16_path(d, x, w, ids, sh, o, 4)
    dbg = torch.zeros(512 * 32, dtype=torch.int64, device=dev)
    torch.cuda.synchronize()
    d, x, (lg, w, ids), sh, o = layers[0], xs[0], rs[0], shs[0], outs[0]
    a16_path(layers[1], xs[1], rs[1][1], rs[1][2], shs[1], outs[1], 4, fused_shared=True)
    DBG[0] = dbg
    a16_path(d, x, w, ids, sh, o, 4, fused_shared=True)
    DBG[0] = torch.empty(0, dtype=torch.int64, device=dev)
    torch.cuda.synchronize()
    v = dbg.view(512, 32).cpu().tolist()
    U = len(set(ids.flatten().tolist()))
    ev = []
    for cta, row in enumerate(v):
        for val in row:
            if val: ev.append((val >> 12, val & 4095, cta))
    t0 = min(e[0] for e in ev)
    ev = [((tm - t0) / 1000, tag, cta) for tm, tag, cta in ev]
    by = collections.defaultdict(list)
    for tm, tag, cta in ev: by[tag].append(tm)
    def rng(tag): return f'{min(by[tag]):6.2f}..{max(by[tag]):6.2f}' if by[tag] else '-'
    print(f'corr={corr} U={U}: n1={U*10} n2={U*20}  CTAs={len(set(e[2] for e in ev))}')
    print('  shared g/u done', rng(4088), ' shared down past wait', rng(4087))
    print('  start', rng(4095), ' past-wait', rng(4094), ' gemm1 done', rng(4093), ' gemm2 h-ready', rng(4092),
          ' gemm2 item done', rng(4091), ' final reduce done', rng(4090), ' exit', rng(4089))
    g1 = sorted(tm for tm, tag, c in ev if 61 <= tag <= 60 + U * 20)
    g2 = sorted(tm for tm, tag, c in ev if 60 + U * 20 < tag <= 60 + U * 40)
    print(f'  gemm1 claims {g1[0]:.2f}..{g1[-1]:.2f} (n={len(g1)})   gemm2 claims {g2[0]:.2f}..{g2[-1]:.2f} (n={len(g2)})')
    # per-CTA sequence for a few CTAs
    seq = collections.defaultdict(list)
    for tm, tag, cta in sorted(ev): seq[cta].append((round(tm, 2), tag))
    for cta in list(seq)[:4]: print('  cta', cta, seq[cta][:12])
