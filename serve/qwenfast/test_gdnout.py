"""Fused GDN gated RMSNorm + out_proj vs sglang's RMSNormGated (fla Triton) + BF16 linear, real layer weights."""
import os, json, sys
import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_qf import build
qf = build()
from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated

SNAP = os.environ.get('QWEN_SNAPSHOT', '/root/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594')
idx = json.load(open(f'{SNAP}/model.safetensors.index.json'))['weight_map']
cfg = json.load(open(f'{SNAP}/config.json'))
cfg = cfg.get('text_config', cfg)
eps = cfg['rms_norm_eps']
dev = 'cuda'
torch.manual_seed(0)
res = []
for layer in (0, 1, 46):
    names = [f'model.language_model.layers.{layer}.linear_attn.norm.weight', f'model.language_model.layers.{layer}.linear_attn.out_proj.weight']
    if names[0] not in idx:
        print('skip layer', layer); continue
    tens = {}
    for n in names:
        with safe_open(f'{SNAP}/{idx[n]}', framework='pt', device=dev) as h:
            tens[n] = h.get_tensor(n)
    wn, W = tens[names[0]].to(torch.bfloat16).contiguous(), tens[names[1]].contiguous()
    Ws = [W] + [W.clone() for _ in range(5)]          # 6 x 31.5 MB > 128 MB L2: timing sees DRAM, as in the model
    norm = RMSNormGated(128, eps=eps, group_size=None, norm_before_gate=True, device=dev, dtype=torch.bfloat16, activation='sigmoid')
    norm.weight.data.copy_(wn)

    def ref(core, z, i=0):
        o = norm(core, z)
        return F.linear(o.reshape(-1, 6144), Ws[i % 6])

    def mine(core, z, pdl=True, pf=True, i=0):
        return qf.gdn_norm_oproj(core, z, wn, eps, False, Ws[i % 6], pdl, pf)

    for T in (1, 2, 4, 8):
        errs, agree = [], []
        for j in range(6):
            core = (torch.randn(T * 48, 128, device=dev) * (0.05 + j * 0.3)).to(torch.bfloat16)
            z = (torch.randn(T * 48, 128, device=dev) * 2).to(torch.bfloat16)
            a, b = ref(core, z).float(), mine(core, z).float()
            errs.append(((a - b).norm() / a.norm()).item())
            agree.append((a.to(torch.bfloat16) == b.to(torch.bfloat16)).float().mean().item())
        cores = [(torch.randn(T * 48, 128, device=dev)).to(torch.bfloat16) for _ in range(24)]
        zs = [(torch.randn(T * 48, 128, device=dev)).to(torch.bfloat16) for _ in range(24)]
        det = torch.equal(mine(cores[0], zs[0]), mine(cores[0], zs[0]))

        def timeit(fn):
            for i in range(3): fn(i)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(24): fn(i)
            g.replay(); torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(20): g.replay()
            e.record(); torch.cuda.synchronize()
            return s.elapsed_time(e) * 1e3 / 480
        row = dict(layer=layer, T=T, rel_l2=max(errs), bf16_agree=min(agree), deterministic=det,
                   ref_us=round(timeit(lambda i: ref(cores[i], zs[i], i)), 2),
                   mine_us=round(timeit(lambda i: mine(cores[i], zs[i], True, True, i)), 2),
                   mine_noprefetch_us=round(timeit(lambda i: mine(cores[i], zs[i], True, False, i)), 2),
                   mine_nopdl_us=round(timeit(lambda i: mine(cores[i], zs[i], False, True, i)), 2))
        res.append(row)
        print(json.dumps(row), flush=True)
json.dump(res, open(os.environ.get('QWENFAST_TEST_OUT', '/tmp') + '/test_gdnout.json', 'w'), indent=1)
