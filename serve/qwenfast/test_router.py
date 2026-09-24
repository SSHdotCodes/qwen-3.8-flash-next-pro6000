"""Fused router (gemm + softmax top-10) vs sglang's gate linear + fused_topk on a real router weight."""
import json, os, sys
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_qf import build
qf = build()
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

SNAP = os.environ.get('QWEN_SNAPSHOT', '/root/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594')
idx = json.load(open(f'{SNAP}/model.safetensors.index.json'))['weight_map']
dev = 'cuda'
torch.manual_seed(0)
res = []
for layer in (0, 23, 47):
    n = f'model.language_model.layers.{layer}.mlp.gate.weight'
    with safe_open(f'{SNAP}/{idx[n]}', framework='pt', device=dev) as h:
        w = h.get_tensor(n).contiguous()
    scratch = torch.empty(8 * 512, dtype=torch.float32, device=dev)
    counter = torch.zeros(1, dtype=torch.int32, device=dev)

    def ref(x, renorm):
        logits = (x @ w.T)                      # bf16 out, fp32 accumulate (cuBLAS)
        tw, ti = moe_fused_gate(logits, None, 10, scoring_func='softmax', renormalize=renorm)
        return tw, ti, logits

    def mine(x, renorm, pdl=True):
        return qf.router_topk(x, w, 10, renorm, scratch, counter, pdl)

    for T in (1, 2, 4, 8):
        n_same_ids = n_same_set = n_rows = 0
        max_w_err = 0.0
        logit_mismatch = 0
        for j in range(40):
            x = (torch.randn(T, 2560, device=dev) * (0.5 + j / 20)).to(torch.bfloat16)
            for renorm in (True, False):
                rw, ri, rl = ref(x, renorm)
                mw, mi, ml = mine(x, renorm)
                logit_mismatch += int((rl != ml).sum())
                # compare against the reference top-k computed from MY logits (isolates the top-k from gemm rounding)
                rw2, ri2 = moe_fused_gate(ml, None, 10, scoring_func='softmax', renormalize=renorm)
                n_same_ids += int((ri2 == mi).all(dim=1).sum())
                n_same_set += int((ri.sort(1).values == mi.sort(1).values).all(dim=1).sum())
                n_rows += T
                max_w_err = max(max_w_err, float((rw2 - mw).abs().max()))
        # timing: stock = gemm + moe_fused_gate; mine = fused
        xs = [(torch.randn(T, 2560, device=dev)).to(torch.bfloat16) for _ in range(24)]

        def timeit(fn):
            for i in range(3): fn(xs[i])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(24): fn(xs[i])
            g.replay(); torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(20): g.replay()
            e.record(); torch.cuda.synchronize()
            return s.elapsed_time(e) * 1e3 / 480
        row = dict(layer=layer, T=T, same_ids_given_same_logits=n_same_ids / n_rows, same_set_vs_cublas=n_same_set / n_rows,
                   logit_bf16_mismatches=logit_mismatch, max_weight_err=max_w_err,
                   stock_us=round(timeit(lambda x: ref(x, True)), 2), mine_us=round(timeit(lambda x: mine(x, True)), 2),
                   mine_nopdl_us=round(timeit(lambda x: mine(x, True, False)), 2), counter=int(counter.item()))
        res.append(row)
        print(json.dumps(row), flush=True)
json.dump(res, open(os.environ.get('QWENFAST_TEST_OUT', '/tmp') + '/test_router.json', 'w'), indent=1)
