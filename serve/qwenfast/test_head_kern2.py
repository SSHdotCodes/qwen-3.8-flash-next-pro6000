"""WP-Head smoke/graph checks ported to the integrated extension.

Requires the pinned target at /models/target (QWEN_TARGET overrides it).
Uses seeded synthetic hidden states instead of private activation captures.
This does not reproduce the 602-capture evidence or prove the sigma bound.
"""
import argparse,json,time,os
os.environ.setdefault("QWENFAST_W8M","1")
from pathlib import Path
import torch
from safetensors import safe_open
from kern2_test_utils import extension as build

D=Path(__file__).resolve().parent
def weights(ext):
    base=os.environ.get("QWEN_TARGET", "/models/target")
    idx=json.load(open(base+"/model.safetensors.index.json"))["weight_map"]
    with safe_open(base+"/"+idx["lm_head.weight"],framework="pt",device="cuda") as f:
        w=f.get_tensor("lm_head.weight")
    N,K=w.shape
    q=torch.empty_like(w,dtype=torch.int8);s=torch.empty((N,20),device="cuda",dtype=torch.bfloat16)
    for r in range(0,N,8192):
        x=w[r:r+8192].float().view(-1,20,128)
        sc=(x.abs().amax(2)/127).clamp_min(1e-30).bfloat16()
        q[r:r+8192]=(x/sc.float()[:,:,None]).round().clamp(-127,127).char().flatten(1)
        s[r:r+8192]=sc
    del w
    hi=ext.head_pack_hi(q)
    torch.manual_seed(20261020)
    h=torch.randn(32,2560,device="cuda",dtype=torch.bfloat16)
    return h,q,s,hi

def canonical_ids(x,k):
    key=x.double()-torch.arange(x.shape[1],device=x.device,dtype=torch.float64)[None,:]*1e-10
    return key.topk(k,1).indices.sort(1).values

if __name__ == "__main__":
    ext=build(); H,q,s,hi=weights(ext)
    dense=torch.ones(1,device="cuda",dtype=torch.int32)
    kdev=torch.full((1,),64,device="cuda",dtype=torch.int32)
    def run(h,z=8.,cap=4096):
        return ext.head_run(h,q,s,hi,kdev,dense,z,cap,4 if len(h)>4 else 2)
    for m in range(1,9):
        h=H[:m].contiguous();ref=ext.w8g_gemm(h,q,s,0x100)
        dense.fill_(1);assert torch.equal(run(h)[0],ref)
        for k in (1,20,28,56):
            dense.zero_();kdev.fill_(k);state=run(h);out=state[0];finite=out.isfinite()
            assert torch.equal(out[finite],ref[finite]),("candidate mismatch",m,k)
            assert torch.equal(canonical_ids(out,k),canonical_ids(ref,k)),("top-k mismatch",m,k)
        kdev.fill_(64)
        fallback=run(h,128.,64)
        assert fallback[2].item()==1 and torch.equal(fallback[0],ref),"overflow fallback mismatch"
    h=H[:4].clone();dense.fill_(1);run(h);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=run(h)
    for ri,(flag,k) in enumerate(((1,64),(0,28),(0,56),(1,28),(0,1),(0,20))):
        h.copy_(H[ri*4:ri*4+4]);dense.fill_(flag);kdev.fill_(k);graph.replay()
        ref=ext.w8g_gemm(h,q,s,0x100);mask=captured[0].isfinite()
        assert torch.equal(captured[0][mask],ref[mask])
        if flag:assert torch.equal(captured[0],ref)
        else:assert torch.equal(canonical_ids(captured[0],k),canonical_ids(ref,k))
    print("head: candidate logits/top-k, overflow, dense fallback and mutable graph checks passed",flush=True)
