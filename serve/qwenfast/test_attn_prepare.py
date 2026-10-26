"""Ported from WP-Attn test_prepare.py; synthetic FP8 edge and norm/store checks."""
import torch,json
from kern2_test_utils import bench,err
from sglang.srt.norm_store import prepare_store
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.attention import fused_qk_rmsnorm_rope_gate as stock
res=[]
for m in range(1,9):
    torch.manual_seed(m)
    qkv=torch.randn(m,13312,device='cuda',dtype=torch.bfloat16)*4
    qw=torch.randn(256,device='cuda',dtype=torch.bfloat16)*.2
    kw=torch.randn_like(qw)*.2
    pos=torch.arange(m,device='cuda',dtype=torch.int64).repeat(3,1)+256
    axis=(torch.arange(32,device='cuda',dtype=torch.int32)%3).contiguous()
    phase=torch.randn(512,32,device='cuda');cs=torch.cat([phase.cos(),phase.sin()],-1)
    loc=torch.arange(m,device='cuda',dtype=torch.int64)+1
    if m>1:loc[0]=0
    kc=torch.zeros(32,2,256,device='cuda',dtype=torch.float8_e4m3fn);vc=torch.zeros_like(kc)
    qg,ki,vi=qkv.split([12288,512,512],-1)
    def old(i=0):
        q,k,g=stock.fused_qk_gemma_rmsnorm_rope_gate(qg,ki,qw,kw,cs,pos,1.e-6,24,2,256,64,mrope_axis_map=axis)
        # HybridLinearKVPool passes default unit scales to the full KV pool.
        k.div_(1.0);vi.div_(1.0)
        kk=k.to(torch.float8_e4m3fn);vv=vi.to(torch.float8_e4m3fn)
        store_cache(kk,vv,kc.view(32,512),vc.view(32,512),loc,row_bytes=512)
        return q,k,vi,g
    def new(i=0):return prepare_store(qkv,qw,kw,cs,pos,loc,kc,vc,mrope_axis_map=axis)
    if m==1:
        edges=torch.tensor([0.,1.,-1.,448.,464.,480.,-448.,-464.,-480.,1.e10,-1.e10,
                            float('inf'),-float('inf'),float('nan')],device='cuda',dtype=torch.bfloat16)
        saved=qkv.clone();qkv[0,-512:]=edges.repeat(37)[:512]
        old();edge_ref=vc.clone();vc.zero_();new()
        assert torch.equal(vc.view(torch.uint8),edge_ref.view(torch.uint8)), dict(
             expected=edge_ref[1].flatten()[:14].view(torch.uint8).tolist(),
             actual=vc[1].flatten()[:14].view(torch.uint8).tolist())
        qkv.copy_(saved);kc.zero_();vc.zero_()
    ref=old();rk,rv=kc.clone(),vc.clone()
    kc.zero_();vc.zero_();out=new()
    e=[err(ref[j],out[j].reshape(ref[j].shape)) for j in range(4)]
    assert all(x['bitwise'] for x in e),e
    assert torch.equal(kc.view(torch.uint8),rk.view(torch.uint8))
    assert torch.equal(vc.view(torch.uint8),rv.view(torch.uint8))
    row=dict(M=m,old_us=bench(old),new_us=bench(new),exactness=e)
    print(json.dumps(row),flush=True);res.append(row)
print(json.dumps({'cases': res}), flush=True)
