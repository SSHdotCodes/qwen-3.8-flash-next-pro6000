"""WP-Attn cached row-start top-k set equivalence, synthetic inputs."""
import torch,json
from kern2_test_utils import bench
from sglang.kernels.ops.attention.fast_topk import fast_topk
from sglang.kernels.ops.attention import fast_topk as ft
from sglang.srt.glue_index import install_cached_row_starts
install_cached_row_starts()
results=[]
for m in range(1,9):
    logits=torch.randn(m,65536,device='cuda',dtype=torch.float32)
    lens=torch.full((m,),2049,device='cuda',dtype=torch.int32)

    old=lambda i=0:fast_topk(logits,lens,512,row_starts=None)
    new=lambda i=0:ft.fast_topk(logits,lens,512,row_starts=None)
    a,b=old(),new()
    assert torch.equal(a.sort(-1).values,b.sort(-1).values)
    row=dict(M=m,old_us=bench(old),new_us=bench(new),same_set=True,same_order=bool(torch.equal(a,b)))
    print(json.dumps(row),flush=True);results.append(row)
print(json.dumps({'cases': results}),flush=True)
