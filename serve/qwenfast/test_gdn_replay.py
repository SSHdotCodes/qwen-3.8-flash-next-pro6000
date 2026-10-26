"""Exact replay-on-commit validation and DRAM-rotated 36-layer graph timing."""
import argparse,importlib.util,sys,json,torch,os
from pathlib import Path
from kern2_test_utils import extension
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update as recur
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_mamba_state_scatter_with_mask as scatter
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_conv_window_scatter_with_mask as convscatter
newrecur = recur
STATE_DTYPE = torch.bfloat16 if os.getenv("WPGDN_STATE_DTYPE", "float32") == "bfloat16" else torch.float32
REPLAY_TILE=int(os.environ["WPGDN_REPLAY_TILE"]) if "WPGDN_REPLAY_TILE" in os.environ else None

def records(L,R,T):
 kw=dict(device='cuda',dtype=torch.float32)
 return dict(k=torch.empty(L,R,T,16,128,**kw),d=torch.empty(L,R,T,48,128,**kw),decay=torch.empty(L,R,T,48,**kw),active=torch.zeros(L,R,device='cuda',dtype=torch.uint8))

def recordargs(r,l,on=True):
 return dict(replay_k=r['k'][l],replay_d=r['d'][l],replay_decay=r['decay'][l],replay_active=r['active'][l],record_replay=on,replay_value_tile=REPLAY_TILE)

def commit(ext,ss,snap,r,ix,ends,ti=None,ts=None):
 ext.wpgdn_replay_commit(ss,snap,r['k'],r['d'],r['decay'],r['active'],ix,ends,ti,ts)

def bitwise(a,b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    dtype={torch.float32:torch.int32,torch.bfloat16:torch.int16}.get(a.dtype)
    return torch.equal(a.view(dtype),b.view(dtype)) if dtype else torch.equal(a,b)

def error(a,b):
 d=(a-b).abs();return dict(equal=bitwise(a,b),max_abs=d.max().item(),max_rel=(d/a.abs().clamp_min(1e-8)).max().item())

def quality(ext):
 torch.manual_seed(20261020);out=[];L=3
 for T in range(1,9):
  for B in ((1,4) if T in (1,4,8) else (1,)):
   slots=2*B+4;R=B;kw=dict(device='cuda',dtype=STATE_DTYPE)
   initial=torch.randn(L,slots,48,128,128,**kw)*(.03 if T%2 else 1.)
   if T==4 and B==4:
    pitch=786432+256;tmp=torch.empty_strided(initial.shape,(slots*pitch+512,pitch,16384,128,1),**kw);tmp.copy_(initial);initial=tmp
   snap=torch.empty(L,R,T,48,128,128,**kw);newss=torch.empty_strided(initial.shape,initial.stride(),**kw);newss.copy_(initial);newsnap=torch.full_like(snap,1234.5);r=records(L,R,T)
   ix=torch.randperm(B+2,device='cuda',dtype=torch.int64)[:B];ii=torch.arange(B,device='cuda',dtype=torch.int32);cu=torch.arange(0,(B+1)*T,T,device='cuda',dtype=torch.int32)
   A=torch.randn(48,**kw);dt=torch.randn(48,device='cuda',dtype=torch.bfloat16)
   for l in range(L):
    q=torch.randn(1,B*T,16,128,device='cuda',dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn(1,B*T,48,128,device='cuda',dtype=torch.bfloat16)
    a=torch.randn(B*T,48,device='cuda',dtype=torch.bfloat16);b=torch.randn_like(a)
    args=dict(A_log=A,a=a,dt_bias=dt,softplus_beta=1.,softplus_threshold=20.,q=q,k=k,v=v,b=b,initial_state_indices=ix,cu_seqlens=cu,use_qk_l2norm_in_kernel=True,disable_state_update=True,intermediate_state_indices=ii,cache_steps=T)
    y0=recur(initial_state_source=initial[l],intermediate_states_buffer=snap[l],**args)
    y1=newrecur(initial_state_source=newss[l],intermediate_states_buffer=newsnap[l],**args,**recordargs(r,l))
    e=error(y0.float(),y1.float());assert e['equal'],('verify',T,B,l,e)
   assert (r['active']==1).all().item();assert (newsnap==1234.5).all().item()
   worst=0.
   for trial in range(10):
    ends=torch.randint(-1,T,(B,),device='cuda',dtype=torch.int64 if trial%2 else torch.int32)
    if trial==5:
     eb=torch.empty(B,2,device='cuda',dtype=ends.dtype);eb[:,0]=ends;ends=eb[:,0]
    ti=torch.arange(B+2,2*B+2,device='cuda',dtype=torch.int64);ts=torch.randint(-1,T,(B,),device='cuda',dtype=torch.int64)
    if trial%3==0:ti[0]=ix[0]  # Preserve main/track alias ordering.
    ref=initial.clone();newss.copy_(initial)
    scatter(ref,snap,ix,ends);scatter(ref,snap,ti,ts)
    if trial%2:r['active'][0,0]=0 # Mixed per-layer fallback from full snapshots.
    else:r['active'].fill_(1)
    commit(ext,newss,snap,r,ix,ends,ti,ts)
    e=error(ref,newss);worst=max(worst,e['max_abs']);assert e['equal'],('commit',T,B,trial,e)
   row=dict(T=T,B=B,replay_tile=(32 if T==1 else 8) if REPLAY_TILE is None else REPLAY_TILE,layers=L,patterns=10,verify_bitwise=True,committed_bitwise=True,snapshots_untouched=True,max_abs=worst)
   print(json.dumps(row),flush=True);out.append(row)
 # Tree path records active=0, writes old snapshots and commits by copy.
 T=4;B=1;initial=torch.randn(1,4,48,128,128,device='cuda',dtype=STATE_DTYPE);s0=torch.empty(1,1,T,48,128,128,device='cuda',dtype=STATE_DTYPE);s1=torch.empty_like(s0);r=records(1,1,T)
 ix=torch.tensor([1],device='cuda',dtype=torch.int32);ii=torch.tensor([0],device='cuda',dtype=torch.int32);cu=torch.tensor([0,T],device='cuda',dtype=torch.int32);parents=torch.tensor([[0,0,0,1]],device='cuda',dtype=torch.int32)
 args=dict(A_log=torch.zeros(48,device='cuda'),a=torch.randn(T,48,device='cuda',dtype=torch.bfloat16),dt_bias=torch.zeros(48,device='cuda'),softplus_beta=1.,softplus_threshold=20.,q=torch.randn(1,T,16,128,device='cuda',dtype=torch.bfloat16),k=torch.randn(1,T,16,128,device='cuda',dtype=torch.bfloat16),v=torch.randn(1,T,48,128,device='cuda',dtype=torch.bfloat16),b=torch.randn(T,48,device='cuda',dtype=torch.bfloat16),initial_state_indices=ix,cu_seqlens=cu,use_qk_l2norm_in_kernel=True,disable_state_update=True,intermediate_state_indices=ii,cache_steps=T,retrieve_parent_token=parents)
 y0=recur(initial_state_source=initial[0],intermediate_states_buffer=s0[0],**args);y1=newrecur(initial_state_source=initial[0],intermediate_states_buffer=s1[0],**args,**recordargs(r,0,False))
 assert bitwise(y0,y1) and bitwise(s0,s1) and r['active'].item()==0
 ref=initial.clone();cand=initial.clone();ends=torch.tensor([3],device='cuda',dtype=torch.int32);scatter(ref,s0,ix,ends);commit(ext,cand,s1,r,ix,ends)
 assert bitwise(ref,cand);print('tree fallback bitwise',flush=True)
 # Sequential verifies with random acceptance and radix-track steps.
 T=4;B=2;ss0=torch.randn(1,6,48,128,128,device='cuda',dtype=STATE_DTYPE)*.03;ss1=ss0.clone();sn=torch.empty(1,B,T,48,128,128,device='cuda',dtype=STATE_DTYPE);r=records(1,B,T)
 ix=torch.tensor([1,0],device='cuda',dtype=torch.int32);ii=torch.arange(B,device='cuda',dtype=torch.int32);cu=torch.tensor([0,4,8],device='cuda',dtype=torch.int32)
 for step in range(16):
  q=torch.randn(1,8,16,128,device='cuda',dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn(1,8,48,128,device='cuda',dtype=torch.bfloat16);a=torch.randn(8,48,device='cuda',dtype=torch.bfloat16)
  args=dict(A_log=torch.zeros(48,device='cuda'),a=a,dt_bias=torch.zeros(48,device='cuda'),softplus_beta=1.,softplus_threshold=20.,q=q,k=k,v=v,b=a,initial_state_indices=ix,cu_seqlens=cu,use_qk_l2norm_in_kernel=True,disable_state_update=True,intermediate_state_indices=ii,cache_steps=T)
  y0=recur(initial_state_source=ss0[0],intermediate_states_buffer=sn[0],**args);y1=newrecur(initial_state_source=ss1[0],intermediate_states_buffer=sn[0],**args,**recordargs(r,0));assert bitwise(y0,y1)
  ends=torch.randint(0,T,(B,),device='cuda',dtype=torch.int32);ti=torch.tensor([4,5],device='cuda',dtype=torch.int32);ts=torch.randint(-1,T,(B,),device='cuda',dtype=torch.int64)
  scatter(ss0,sn,ix,ends);scatter(ss0,sn,ti,ts);commit(ext,ss1,sn,r,ix,ends,ti,ts);assert bitwise(ss0,ss1)
 print('16 sequential verify/commit rounds, random acceptance and tracking: bitwise',flush=True)
 # Preserve FP32 subnormals despite the extension's --use_fast_math.
 initial=torch.full((1,1,48,128,128),1e-40,device='cuda',dtype=STATE_DTYPE);sn=torch.empty(1,1,2,48,128,128,device='cuda',dtype=STATE_DTYPE);r=records(1,1,2)
 q=torch.zeros(1,2,16,128,device='cuda',dtype=torch.bfloat16);q[:,:,:,0]=1;v=torch.zeros(1,2,48,128,device='cuda',dtype=torch.bfloat16);a=torch.zeros(2,48,device='cuda',dtype=torch.bfloat16);ix=torch.zeros(1,device='cuda',dtype=torch.int32);cu=torch.tensor([0,2],device='cuda',dtype=torch.int32)
 args=dict(A_log=torch.zeros(48,device='cuda'),a=a,dt_bias=torch.zeros(48,device='cuda'),softplus_beta=1.,softplus_threshold=20.,q=q,k=q,v=v,b=a,initial_state_source=initial[0],initial_state_indices=ix,cu_seqlens=cu,use_qk_l2norm_in_kernel=True,disable_state_update=True,intermediate_state_indices=ix,cache_steps=2)
 recur(intermediate_states_buffer=sn[0],**args);newrecur(intermediate_states_buffer=sn[0],**args,**recordargs(r,0));ss=initial.clone();ends=torch.tensor([1],device='cuda',dtype=torch.int32);commit(ext,ss,sn,r,ix,ends)
 assert bitwise(ss[0,0],sn[0,0,1]);print(f'{STATE_DTYPE} subnormal state bitwise',flush=True)

if __name__ == "__main__":
 quality(extension())
