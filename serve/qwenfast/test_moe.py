"""Decode MoE kernel vs FlashInfer CUTLASS (as SGLang calls it) on real layer weights from the checkpoint."""
import json, os, sys, time
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_qf import build
qf = build()

from flashinfer.fused_moe import cutlass_fused_moe
from sglang.srt.layers.quantization.utils import swizzle_blockscale

SNAP = os.environ.get('QWEN_SNAPSHOT', '/root/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594')
LAYER = int(os.environ.get('LAYER', '0'))
E, H, I, K = 512, 2560, 640, 10
dev = 'cuda'
torch.manual_seed(0)

idx = json.load(open(f'{SNAP}/model.safetensors.index.json'))['weight_map']
pre = f'model.language_model.layers.{LAYER}.mlp.'


def load(names):
    out = {}
    files = {}
    for n in names:
        files.setdefault(idx[n], []).append(n)
    for f, ns in files.items():
        with safe_open(f'{SNAP}/{f}', framework='pt', device=dev) as h:
            for n in ns:
                out[n] = h.get_tensor(n)
    return out


t0 = time.time()
names = []
for e in range(E):
    for p in ('gate_proj', 'up_proj', 'down_proj'):
        for s in ('weight', 'weight_scale', 'weight_scale_2', 'input_scale'):
            names.append(f'{pre}experts.{e}.{p}.{s}')
names.append(pre + 'gate.weight')
T_ = load(names)
print('loaded', len(T_), 'tensors in', round(time.time() - t0, 1), 's', flush=True)


def stack(p, s):
    return torch.stack([T_[f'{pre}experts.{e}.{p}.{s}'] for e in range(E)])


gate_w, up_w, down_w = stack('gate_proj', 'weight'), stack('up_proj', 'weight'), stack('down_proj', 'weight')
gate_sf, up_sf, down_sf = stack('gate_proj', 'weight_scale'), stack('up_proj', 'weight_scale'), stack('down_proj', 'weight_scale')
gate_s2, up_s2, down_s2 = [stack(p, 'weight_scale_2').float().reshape(E) for p in ('gate_proj', 'up_proj', 'down_proj')]
gate_in, up_in, down_in = [stack(p, 'input_scale').float().reshape(E) for p in ('gate_proj', 'up_proj', 'down_proj')]
router = T_[pre + 'gate.weight']
print('shapes', gate_w.shape, gate_w.dtype, gate_sf.shape, gate_sf.dtype, down_w.shape, down_sf.shape)
print('gate_s2 == up_s2:', torch.equal(gate_s2, up_s2), 'max |diff|', (gate_s2 - up_s2).abs().max().item())
print('input scales: gate/up max', gate_in.max().item(), up_in.max().item(), 'min', gate_in.min().item(), 'down max', down_in.max().item())

# SGLang layout for FlashInfer CUTLASS: w13 = [up; gate]
w13 = torch.cat([up_w, gate_w], 1).contiguous()
w13_sf_raw = torch.cat([up_sf, gate_sf], 1).contiguous()
w13_sf = swizzle_blockscale(w13_sf_raw).contiguous()
w2 = down_w.contiguous()
w2_sf = swizzle_blockscale(down_sf.contiguous()).contiguous()
w13_input_scale = torch.maximum(gate_in, up_in).max().float()
w2_input_scale = down_in.max().float()
g1_alphas = (w13_input_scale * gate_s2).float().contiguous()
g2_alphas = (w2_input_scale * down_s2).float().contiguous()
a1_gs = (1 / w13_input_scale).float()
a2_gs = (1 / w2_input_scale).float()
del T_


def flashinfer_moe(x, ids, w):
    out = torch.empty_like(x)
    qs = [a1_gs, w13_sf.view(torch.int32), g1_alphas, a2_gs, w2_sf.view(torch.int32), g2_alphas]
    return cutlass_fused_moe(output=out, input=x, token_selected_experts=ids.to(torch.int), token_final_scales=w,
                             fc1_expert_weights=w13.view(torch.long), fc2_expert_weights=w2.view(torch.long),
                             output_dtype=torch.bfloat16, input_sf=None, quant_scales=qs, ep_size=1, ep_rank=0,
                             tp_size=1, tp_rank=0, tune_max_num_tokens=8)[0]


PDL = os.environ.get('PDL', '1') == '1'


def mine(x, ids, w):
    return qf.nvfp4_moe_decode(x, ids, w, w13, w13_sf, g1_alphas, a1_gs, w2, w2_sf, g2_alphas, a2_gs, PDL)


def route(x):
    logits = x.float() @ router.float().T
    p = torch.softmax(logits, -1)
    tw, ids = torch.topk(p, K, -1)
    return ids.int().contiguous(), (tw / tw.sum(-1, keepdim=True)).float().contiguous()


E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)


def deq(wq, sf, s2):
    lo, hi = (wq & 15).long(), (wq >> 4).long()
    v = torch.stack([E2M1[lo], E2M1[hi]], -1).reshape(wq.shape[0], -1)
    return v * sf.float().repeat_interleave(16, -1) * s2


def ref_moe(x, ids, w):
    """unquantized activations, dequantized FP4 weights, fp32 math"""
    out = torch.zeros(x.shape[0], H, device=dev)
    for t in range(x.shape[0]):
        for k in range(K):
            e = ids[t, k].item()
            g = x[t].float() @ deq(gate_w[e], gate_sf[e], gate_s2[e]).T
            u = x[t].float() @ deq(up_w[e], up_sf[e], up_s2[e]).T
            h = torch.nn.functional.silu(g) * u
            out[t] += w[t, k] * (h @ deq(down_w[e], down_sf[e], down_s2[e]).T)
    return out


def timeit(fn, n_inputs, iters=480):
    for i in range(3): fn(i % n_inputs)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n_inputs): fn(i)
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters // n_inputs): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


results = []
for T in (1, 2, 4, 8):
    # realistic-ish inputs: consecutive tokens share structure (x_t = base + noise) so routes overlap
    xs, rs = [], []
    for j in range(24):
        base = torch.randn(1, H, device=dev)
        x = (base + 0.7 * torch.randn(T, H, device=dev)).to(torch.bfloat16).contiguous()
        xs.append(x); rs.append(route(x))
    uniq = sum(len(torch.unique(r[0])) for r in rs) / len(rs)
    errs, errs_ref = [], []
    for j in range(6):
        x, (ids, tw) = xs[j], rs[j]
        a = flashinfer_moe(x, ids, tw).float()
        b = mine(x, ids, tw).float()
        errs.append(((a - b).norm() / a.norm()).item())
        if j < 2:
            r = ref_moe(x, ids, tw)
            errs_ref.append((((a - r).norm() / r.norm()).item(), ((b - r).norm() / r.norm()).item()))
    b1 = mine(xs[0], *rs[0]); b2 = mine(xs[0], *rs[0])
    det = torch.equal(b1, b2)
    t_fi = timeit(lambda i: flashinfer_moe(xs[i], *rs[i]), 24)
    t_me = timeit(lambda i: mine(xs[i], *rs[i]), 24)
    row = dict(T=T, unique_experts=round(uniq, 1), rel_l2_vs_flashinfer=max(errs), deterministic=det,
               vs_bf16_act_ref_flashinfer_mine=errs_ref,
               flashinfer_us=round(t_fi, 1), mine_us=round(t_me, 1))
    results.append(row)
    print(json.dumps(row), flush=True)
json.dump(results, open(os.environ.get('QWENFAST_TEST_OUT', '/tmp') + '/test_moe.json', 'w'), indent=1)

if os.environ.get('PROFILE'):
    from torch.profiler import profile, ProfilerActivity
    T = int(os.environ.get('PROFILE'))
    xs, rs = [], []
    for j in range(24):
        base = torch.randn(1, H, device=dev)
        x = (base + 0.7 * torch.randn(T, H, device=dev)).to(torch.bfloat16).contiguous()
        xs.append(x); rs.append(route(x))
    for i in range(24): mine(xs[i], *rs[i]); flashinfer_moe(xs[i], *rs[i])
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(24): mine(xs[i], *rs[i])
        for i in range(24): flashinfer_moe(xs[i], *rs[i])
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25, max_name_column_width=70))
