"""Full-rescore, all-tie, saturated-histogram and strided-input checks."""
import json, torch, importlib.util, importlib.machinery
from pathlib import Path
from kern2_test_utils import extension
prod=e=extension()
torch.manual_seed(7721)
N=8192
hot=(torch.randn(N,2560,device='cuda')*.02).bfloat16()
xg=hot.float().view(N,40,64)
s4=(xg.abs().amax(2)/7).clamp_min(1e-30).bfloat16()
code=(xg/s4.float().unsqueeze(-1)).round().clamp(-7,7).add(8).byte().view(N,2560)
q4=(code[:,0::2] | (code[:,1::2]<<4)).contiguous()
s2=(xg.square().mean(2).sqrt().clamp_min(1e-30)/2).bfloat16()
code2=(xg/(2*s2.float().unsqueeze(-1))).floor().clamp(-2,1).add(2).byte().view(N,2560)
q2=(code2[:,0::4] | (code2[:,1::4]<<2) | (code2[:,2::4]<<4) | (code2[:,3::4]<<6)).contiguous()
w=dict(q4=q4,s4=s4,q2=q2,s2=s2)
N=len(w['q4'])
torch.manual_seed(7721)
cases=[]
for M in range(1,9):
    hp=torch.randn((M,2576),device='cuda',dtype=torch.bfloat16)
    h=hp[:,:2560]
    hpref=torch.cat((h,h[:1].expand(8-M,-1))) if 5<=M<=7 else h
    y=prod.w4g_gemm(hpref,w['q4'],w['s4'])[:M]
    ov,oi=prod.topk_rows(y,64)
    for C in (64,N):
        score=e.drafthead_q2(h,w['q2'],w['s2'])
        th=e.drafthead_threshold(score,C)
        assert ((score.float()>=th[:,None]).sum(1)>=C).all()
        nv,ni=e.drafthead_rescore(h,w['q4'],w['s4'],score,th)
        assert torch.equal(nv,y.gather(1,ni).float())
        assert ((ni>=0)&(ni<N)).all() and (ni.sort(1).values.diff(dim=1)>0).all()
        assert (nv[:,:-1]>=nv[:,1:]).all()
        if C==N:
            # Full rescore must have exactly the same ordered values; ties may choose different IDs.
            assert torch.equal(nv,ov)
        cases.append(dict(M=M,C=C,int4_bitwise=True))
h=torch.zeros((8,2560),device='cuda',dtype=torch.bfloat16)
for v in (-1000.,0.,1000.):
    score=torch.full((8,N),v,device='cuda',dtype=torch.bfloat16)
    th=e.drafthead_threshold(score,4608)
    assert ((score.float()>=th[:,None]).sum(1)==N).all()
    nv,ni=e.drafthead_rescore(h,w['q4'],w['s4'],score,th)
    assert torch.equal(nv,torch.zeros_like(nv))
    assert ((ni>=0)&(ni<N)).all() and (ni.sort(1).values.diff(dim=1)>0).all()
print(json.dumps(dict(cases=cases,all_ties_and_saturated_bins=True,max_abs=0.,max_rel=0.)),flush=True)
