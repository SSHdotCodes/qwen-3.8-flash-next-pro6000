"""Exact-layout QSA gather and experimental direct sparse decode, M=1..8."""
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_pack_counts(K,V,R2T,REQ,IDX,LEN,OK,OV,COUNTS,
                      TOPK:tl.constexpr, STRIDE:tl.constexpr, RS:tl.constexpr,
                      IS:tl.constexpr, H:tl.constexpr,D:tl.constexpr,
                      BT:tl.constexpr, BC:tl.constexpr):
    row,head,block=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    length=tl.load(LEN+row)
    if head==0 and block==0:
        cc=tl.arange(0,BC)
        pi=tl.load(IDX+row*IS+cc,cc<TOPK,other=-1)
        count=tl.sum(((pi>=0)&(pi<length)&(cc<TOPK)).to(tl.int32),0)
        tl.store(COUNTS+row,count)
    cols=block*BT+tl.arange(0,BT)
    dims=tl.arange(0,D)
    pos=tl.load(IDX+row*IS+cols,cols<TOPK,other=-1)
    valid=(cols<TOPK)&(pos>=0)&(pos<length)
    req=tl.load(REQ+row)
    slots=tl.load(R2T+req*RS+tl.where(valid,pos,0),valid,other=0).to(tl.int64)
    src=slots[:,None]*(H*D)+head*D+dims[None,:]
    dst=(row*STRIDE+cols[:,None])*(H*D)+head*D+dims[None,:]
    kval=tl.load(K+src,valid[:,None],other=0.0).to(OK.dtype.element_ty)
    vval=tl.load(V+src,valid[:,None],other=0.0).to(OV.dtype.element_ty)
    tl.store(OK+dst,kval,(cols<STRIDE)[:,None])
    tl.store(OV+dst,vval,(cols<STRIDE)[:,None])


def pack_counts(k,v,req_to_token,req_indices,indices,seq_lens,out_k,out_v,counts,
                stride=None,block=8,warps=8):
    m,topk=indices.shape
    stride=stride or triton.cdiv(topk,64)*64
    _attn_pack_counts[(m,k.shape[1],triton.cdiv(stride,block))](
        k,v,req_to_token,req_indices,indices,seq_lens,out_k,out_v,counts,
        topk,stride,req_to_token.stride(0),indices.stride(0),k.shape[1],k.shape[2],
        block,triton.next_power_of_2(topk),num_warps=warps)


@triton.jit
def _attn_sparse_split(Q,K,V,R2T,REQ,IDX,LEN,COUNTS,PART,MAXS,SUMS,
                       TOPK:tl.constexpr, IS:tl.constexpr,RS:tl.constexpr,
                       QS0:tl.constexpr,QS1:tl.constexpr,
                       H:tl.constexpr,G:tl.constexpr,D:tl.constexpr,
                       NS:tl.constexpr,CHUNK:tl.constexpr,BN:tl.constexpr,
                       SCALE:tl.constexpr,PROB_FP32:tl.constexpr,COUNT_EACH_BLOCK:tl.constexpr,BC:tl.constexpr):
    row,head,split=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    hs=tl.arange(0,16)
    ds=tl.arange(0,D)
    qs=tl.load(Q+row*QS0+(head*G+hs[:,None])*QS1+ds[None,:],(hs<G)[:,None],other=0)
    length=tl.load(LEN+row)
    req=tl.load(REQ+row)
    if COUNT_EACH_BLOCK:
        cc=tl.arange(0,BC)
        pi=tl.load(IDX+row*IS+cc,cc<TOPK,other=-1)
        count=tl.sum(((pi>=0)&(pi<length)&(cc<TOPK)).to(tl.int32),0)
    else:
        count=tl.load(COUNTS+row)
    ns=tl.arange(0,BN)
    mm=tl.full((16,),-float('inf'),tl.float32)
    ll=tl.zeros((16,),tl.float32)
    acc=tl.zeros((16,D),tl.float32)
    for start in range(split*CHUNK,(split+1)*CHUNK,BN):
        cols=start+ns
        pos=tl.load(IDX+row*IS+cols,cols<TOPK,other=-1)
        valid=(cols<count)&(cols<TOPK)&(pos>=0)&(pos<length)
        slots=tl.load(R2T+req*RS+tl.where(valid,pos,0),valid,other=0).to(tl.int64)
        kk=tl.load(K+slots[None,:]*(H*D)+head*D+ds[:,None],valid[None,:],other=0.0).to(qs.dtype)
        vv=tl.load(V+slots[:,None]*(H*D)+head*D+ds[None,:],valid[:,None],other=0.0).to(qs.dtype)
        scores=tl.dot(qs,kk).to(tl.float32)*(SCALE*1.4426950408889634)
        scores=tl.where(((cols<TOPK)&(cols<count))[None,:],scores,-float('inf'))
        newmax=tl.maximum(mm,tl.max(scores,1))
        # Empty rows/splits produce zero and -inf, rather than NaN.
        safe=tl.where(newmax==-float('inf'),0.0,newmax)
        alpha=tl.exp2(mm-safe)
        pp=tl.exp2(scores-safe[:,None])
        acc=acc*alpha[:,None]
        if PROB_FP32:
            # Three BF16 components exactly represent every normal FP32 p.
            # V came from FP8, so each product is exact before FP32 accumulation.
            # Scale by 2**32 so even FP32 subnormal probabilities and their
            # low components lie in BF16's representable exponent range.
            ps=pp*4294967296.0
            ph=ps.to(tl.bfloat16)
            residual=ps-ph.to(tl.float32)
            pl=residual.to(tl.bfloat16)
            pt=(residual-pl.to(tl.float32)).to(tl.bfloat16)
            acc=tl.dot(ph,vv,acc)
            acc=tl.dot(pl,vv,acc)
            acc=tl.dot(pt,vv,acc)
        else:
            acc=tl.dot(pp.to(qs.dtype),vv,acc)
        ll=ll*alpha+tl.sum(pp,1)
        mm=newmax
    if PROB_FP32:
        acc=acc*2.3283064365386963e-10
    base=((row*H+head)*NS+split)*G
    tl.store(PART+(base+hs[:,None])*D+ds[None,:],acc,(hs<G)[:,None])
    tl.store(MAXS+base+hs,mm,hs<G)
    tl.store(SUMS+base+hs,ll,hs<G)


@triton.jit
def _attn_sparse_merge(PART,MAXS,SUMS,OUT,GATE,GR:tl.constexpr,GH:tl.constexpr,FUSE_GATE:tl.constexpr,
                       H:tl.constexpr,G:tl.constexpr,D:tl.constexpr,NS:tl.constexpr,
                       BS:tl.constexpr,BD:tl.constexpr):
    row,head,db=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    group=head//G
    qh=head%G
    ss=tl.arange(0,BS)
    ds=db*BD+tl.arange(0,BD)
    b=((row*H+group)*NS+ss)*G+qh
    mm=tl.load(MAXS+b,ss<NS,other=-float('inf'))
    ll=tl.load(SUMS+b,ss<NS,other=0)
    ma=tl.max(mm,0)
    safe=tl.where(ma==-float('inf'),0.0,ma)
    aa=tl.exp2(mm-safe)
    den=tl.sum(ll*aa,0)
    pp=tl.load(PART+b[:,None]*D+ds[None,:],(ss<NS)[:,None]&(ds<D)[None,:],other=0)
    oo=tl.sum(pp*aa[:,None],0)/tl.maximum(den,1.e-20)
    if FUSE_GATE:
        # The original attention output is BF16 before sigmoid-multiply.
        oo=oo.to(OUT.dtype.element_ty).to(tl.float32)
        gv=tl.load(GATE+row*GR+head*GH+ds,ds<D,other=0.0).to(tl.float32)
        oo=oo*tl.sigmoid(gv)
    tl.store(OUT+(row*(H*G)+head)*D+ds,oo,ds<D)


def sparse_decode(q,k,v,req_to_token,req_indices,indices,seq_lens,scale,
                  chunk=128,block=64,warps=4,scratch=None,prob_fp32=True,count_each_block=True,gate=None):
    m,nq,d=q.shape
    h=k.shape[1];g=nq//h
    ns=triton.cdiv(indices.shape[1],chunk)
    if scratch is None:
        part=torch.empty((m,h,ns,g,d),device=q.device,dtype=torch.float32)
        mm=torch.empty((m,h,ns,g),device=q.device,dtype=torch.float32)
        ll=torch.empty_like(mm)
    else:
        part,mm,ll=scratch
    counts=torch.empty(m,device=q.device,dtype=torch.int32)
    if not count_each_block:
        _attn_sparse_counts[(m,)](indices,seq_lens,counts,indices.shape[1],indices.stride(0),
                                 triton.next_power_of_2(indices.shape[1]),num_warps=4)
    out=torch.empty((m,nq,d),device=q.device,dtype=q.dtype)
    _attn_sparse_split[(m,h,ns)](q,k,v,req_to_token,req_indices,indices,seq_lens,counts,
        part,mm,ll,indices.shape[1],indices.stride(0),req_to_token.stride(0),
        q.stride(0),q.stride(1),h,g,d,ns,chunk,block,scale,prob_fp32,count_each_block,triton.next_power_of_2(indices.shape[1]),num_warps=warps,num_stages=1)
    gr=gate.stride(0) if gate is not None else 0
    gh=(gate.stride(1) if gate.ndim==3 else d) if gate is not None else 0
    _attn_sparse_merge[(m,nq,triton.cdiv(d,64))](part,mm,ll,out,gate if gate is not None else out,
        gr,gh,gate is not None,h,g,d,ns,
        triton.next_power_of_2(ns),64,num_warps=4)
    return out


@triton.jit
def _attn_sparse_counts(IDX,LEN,COUNTS,TOPK:tl.constexpr,IS:tl.constexpr,BC:tl.constexpr):
    row=tl.program_id(0)
    cc=tl.arange(0,BC)
    length=tl.load(LEN+row)
    pi=tl.load(IDX+row*IS+cc,cc<TOPK,other=-1)
    count=tl.sum(((pi>=0)&(pi<length)&(cc<TOPK)).to(tl.int32),0)
    tl.store(COUNTS+row,count)
