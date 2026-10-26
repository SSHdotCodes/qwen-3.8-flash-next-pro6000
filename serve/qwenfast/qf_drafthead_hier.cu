// Two-stage draft-only hot head. INT4 rescore retains production w4g arithmetic.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>
namespace {
constexpr int K=2560, G=64, KC=5, W=8, T=256, NB=1024, HB=64, BINS=512, TOP=64;
__device__ __forceinline__ unsigned bfkey(__nv_bfloat16 v) {
    unsigned b=__bfloat16_as_ushort(v);
    return (b&0x8000) ? ((~b)&65535) : (b|0x8000);
}
__device__ __forceinline__ float keyfloat(unsigned k) {
    unsigned b=(k&0x8000) ? (k&0x7fff) : ((~k)&65535);
    return __uint_as_float(b<<16);
}
__device__ __forceinline__ void nib16(uint2 v,float *f) {
#pragma unroll
    for(int w=0;w<2;++w) {
        unsigned u=w?v.y:v.x,lo=u&0x0f0f0f0f,hi=(u>>4)&0x0f0f0f0f;
#pragma unroll
        for(int b=0;b<4;++b) {
            f[w*8+2*b]=__int_as_float(__byte_perm(lo,0x4b000000u,0x7440+b))-8388616.f;
            f[w*8+2*b+1]=__int_as_float(__byte_perm(hi,0x4b000000u,0x7440+b))-8388616.f;
        }
    }
}
template<int M,int RPW>
__global__ __launch_bounds__(256) void dh_q2_kernel(
    const __nv_bfloat16* __restrict__ x,const uint8_t* __restrict__ q,
    const uint16_t* __restrict__ s,__nv_bfloat16* __restrict__ y,int N,int ldx) {
    extern __shared__ float xs[];
    for(int i=threadIdx.x;i<M*K;i+=blockDim.x)
        xs[i]=__bfloat162float(x[(i/K)*ldx+i%K]);
    __syncthreads();
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,row0=(blockIdx.x*W+warp)*RPW;
    if(row0>=N)return;
    uint32_t wv[RPW][KC]; uint16_t sv[RPW][KC];
#pragma unroll
    for(int r=0;r<RPW;++r) {
        int row=min(row0+r,N-1);
#pragma unroll
        for(int c=0;c<KC;++c) {
            int k0=c*512+lane*16;
            wv[r][c]=__ldcs(reinterpret_cast<const unsigned*>(q+(size_t)row*(K/4)+k0/4));
            sv[r][c]=__ldg(s+(size_t)row*40+k0/64);
        }
    }
    float acc[RPW][M];
#pragma unroll
    for(int r=0;r<RPW;++r)
#pragma unroll
        for(int m=0;m<M;++m)acc[r][m]=0.f;
#pragma unroll
    for(int c=0;c<KC;++c) {
        int k0=c*512+lane*16;
#pragma unroll
        for(int r=0;r<RPW;++r) {
            float f[16];
#pragma unroll
            for(int i=0;i<16;++i)
                f[i]=__uint_as_float(0x4b000000u|(((wv[r][c]>>(2*i))&3)<<1))-8388611.f;
            float sc=__uint_as_float((unsigned)sv[r][c]<<16);
#pragma unroll
            for(int m=0;m<M;++m) {
                float a=0.f;
#pragma unroll
                for(int i=0;i<16;++i)a=fmaf(f[i],xs[m*K+k0+i],a);
                acc[r][m]=fmaf(a,sc,acc[r][m]);
            }
        }
    }
#pragma unroll
    for(int r=0;r<RPW;++r)
#pragma unroll
        for(int m=0;m<M;++m) {
            float a=acc[r][m];
#pragma unroll
            for(int o=16;o;o>>=1)a+=__shfl_xor_sync(0xffffffff,a,o);
            if(lane==0&&row0+r<N)y[(size_t)m*N+row0+r]=__float2bfloat16(a);
        }
}
template<int M> void launch_q2(const at::Tensor& x,const at::Tensor& q,const at::Tensor& s,at::Tensor& out,cudaStream_t st) {
    constexpr int R=2;
    auto fn=dh_q2_kernel<M,R>;
    if(M*K*4>48*1024)cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,M*K*4);
    fn<<<(q.size(0)+W*R-1)/(W*R),256,M*K*4,st>>>(
        (const __nv_bfloat16*)x.data_ptr(),q.data_ptr<uint8_t>(),(const uint16_t*)s.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(),q.size(0),x.stride(0));
}
__device__ __forceinline__ int bin(float v) {
    if(!isfinite(v)) return v>0?BINS-1:0;
    return max(0,min(BINS-1,__float2int_rd(v*4.f)+128));
}
__global__ void dh_hist_kernel(const __nv_bfloat16* scores,int N,int* hist) {
    __shared__ int h[BINS];
    int t=threadIdx.x,m=blockIdx.y,b=blockIdx.x;
    h[t]=h[t+T]=0; __syncthreads();
    for(int i=b*T+t;i<N;i+=HB*T)atomicAdd(h+bin(__bfloat162float(scores[(size_t)m*N+i])),1);
    __syncthreads();
    hist[((m*HB+b)*BINS)+t]=h[t];
    hist[((m*HB+b)*BINS)+t+T]=h[t+T];
}
__global__ void dh_threshold_kernel(const int* hist,float* threshold,int C,int* npart) {
    __shared__ int warp_total[8];
    int t=threadIdx.x,m=blockIdx.x,lane=t&31,warp=t>>5;
    if(t==0&&npart)npart[m]=0;
    int b0=BINS-1-2*t,b1=b0-1,a=0,b=0;
#pragma unroll
    for(int j=0;j<HB;++j) {
        a+=hist[(m*HB+j)*BINS+b0];
        b+=hist[(m*HB+j)*BINS+b1];
    }
    int inc=a+b;
#pragma unroll
    for(int o=1;o<32;o<<=1) {
        int z=__shfl_up_sync(0xffffffff,inc,o);
        if(lane>=o)inc+=z;
    }
    if(lane==31)warp_total[warp]=inc;
    __syncthreads();
    int above=inc-a-b;
#pragma unroll
    for(int w=0;w<8;++w)if(w<warp)above+=warp_total[w];
    int selected=-1;
    if(above<C&&C<=above+a)selected=b0;
    if(above+a<C&&C<=above+a+b)selected=b1;
    if(selected>=0)threshold[m]=selected==0?-INFINITY:(selected-128)*.25f;
}
// Select a BF16 key threshold from keys held in registers, ignoring zero padding.
template<int E> __device__ void radix(const unsigned (&keys)[E],int k,unsigned& th,int& eq,int* hist,int* bucket) {
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    unsigned prefix=0,mask=0;
#pragma unroll
    for(int shift=8;shift>=0;shift-=8) {
        hist[tid]=0;__syncthreads();
#pragma unroll
        for(int e=0;e<E;++e) {
            bool valid=keys[e]&&(keys[e]&mask)==prefix;
            unsigned active=__ballot_sync(0xffffffff,valid);
            if(valid) {
                unsigned b=(keys[e]>>shift)&255;
                unsigned peers=__match_any_sync(active,b);
                if(lane==__ffs(peers)-1)atomicAdd(hist+b,__popc(peers));
            }
        }
        __syncthreads();
        if(warp==0) {
            int c[8],sum=0;
#pragma unroll
            for(int j=0;j<8;++j){c[j]=hist[255-8*lane-j];sum+=c[j];}
            int inc=sum;
#pragma unroll
            for(int o=1;o<32;o<<=1){int z=__shfl_up_sync(0xffffffff,inc,o);if(lane>=o)inc+=z;}
            int above=inc-sum;
            if(above<k&&k<=inc) {
#pragma unroll
                for(int j=0;j<8;++j) {
                    if(above<k&&k<=above+c[j]){bucket[0]=255-8*lane-j;bucket[1]=k-above;}
                    above+=c[j];
                }
            }
        }
        __syncthreads();
        prefix|=(unsigned)bucket[0]<<shift;mask|=255u<<shift;k=bucket[1];
        __syncthreads();
    }
    th=prefix;eq=k;
}
__global__ __launch_bounds__(256) void dh_rescore_kernel(
    const __nv_bfloat16* __restrict__ x,int ldx,const uint8_t* __restrict__ q,
    const uint16_t* __restrict__ s,const __nv_bfloat16* __restrict__ scores,
    const float* threshold,int N,unsigned* pkey,int* pidx,int* npart) {
    __shared__ float xs[K];
    __shared__ int ids[512],count,counts[2],hist[256],bucket[2],base_out;
    __shared__ unsigned skey[512];
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,m=blockIdx.y,b=blockIdx.x;
    int seg=(N+gridDim.x-1)/gridDim.x,lo=b*seg,hi=min(N,lo+seg);
    for(int i=tid;i<K;i+=T)xs[i]=__bfloat162float(x[(size_t)m*ldx+i]);
    if(tid==0)count=0;
    __syncthreads();
#pragma unroll
    for(int e=0;e<2;++e) {
        int i=lo+e*T+tid;
        bool keep=i<hi&&__bfloat162float(scores[(size_t)m*N+i])>=threshold[m];
        unsigned mask=__ballot_sync(0xffffffff,keep);
        int base=0;if(lane==0)base=atomicAdd(&count,__popc(mask));
        base=__shfl_sync(0xffffffff,base,0);
        if(keep)ids[base+__popc(mask&((1u<<lane)-1))]=i;
    }
    __syncthreads();
    int n=count;
    // Prefetch both rows and all five K partitions before their FMA chains.
    for(int j0=warp*2;j0<n;j0+=W*2) {
        uint2 wv[2][KC];uint16_t sv[2][KC];
#pragma unroll
        for(int r=0;r<2;++r) {
            int row=ids[min(j0+r,n-1)];
#pragma unroll
            for(int c=0;c<KC;++c) {
                int k0=c*512+lane*16;
                wv[r][c]=__ldcs((const uint2*)(q+(size_t)row*(K/2)+k0/2));
                sv[r][c]=__ldg(s+(size_t)row*40+k0/64);
            }
        }
        float acc[2]={0.f,0.f};
#pragma unroll
        for(int c=0;c<KC;++c) {
            int k0=c*512+lane*16;
#pragma unroll
            for(int r=0;r<2;++r) {
                float wf[16];nib16(wv[r][c],wf);
                float sc=__uint_as_float((unsigned)sv[r][c]<<16),a=0.f;
#pragma unroll
                for(int i=0;i<16;++i)a=fmaf(wf[i],xs[k0+i],a);
                acc[r]=fmaf(a,sc,acc[r]);
            }
        }
#pragma unroll
        for(int r=0;r<2;++r) {
            float a=acc[r];
#pragma unroll
            for(int o=16;o;o>>=1)a+=__shfl_xor_sync(0xffffffff,a,o);
            if(lane==0&&j0+r<n)skey[j0+r]=bfkey(__float2bfloat16(a));
        }
    }
    if(tid==0)base_out=atomicAdd(npart+m,min(n,TOP));
    __syncthreads();
    unsigned* ok=pkey+(size_t)m*gridDim.x*TOP+base_out;
    int* oi=pidx+(size_t)m*gridDim.x*TOP+base_out;
    if(n<=TOP) {
        if(tid<n){ok[tid]=skey[tid];oi[tid]=ids[tid];}
        return;
    }
    unsigned keys[2];
#pragma unroll
    for(int e=0;e<2;++e)keys[e]=e*T+tid<n?skey[e*T+tid]:0;
    unsigned th;int eq;radix(keys,TOP,th,eq,hist,bucket);
    if(tid<2)counts[tid]=0;__syncthreads();
#pragma unroll
    for(int e=0;e<2;++e) {
        unsigned key=keys[e];int j=e*T+tid;
        if(key>th){int a=atomicAdd(counts,1);ok[a]=key;oi[a]=ids[j];}
        else if(key==th&&key){int a=atomicAdd(counts+1,1);if(a<eq){ok[TOP-eq+a]=key;oi[TOP-eq+a]=ids[j];}}
    }
}
constexpr int MERGE=8;
template<int E> __device__ __forceinline__ void dh_hier_select(
    const unsigned* pk,const int* pi,const int* npart,int pitch,unsigned* outk,int* outi,int* hist,int* bucket,int* counts) {
    int tid=threadIdx.x,m=blockIdx.y,b=blockIdx.x,n=npart[m];
    int seg=(n+MERGE-1)/MERGE,lo=b*seg,hi=min(n,lo+seg),len=max(0,hi-lo);
    unsigned* ok=outk+((size_t)m*MERGE+b)*TOP;
    int* oi=outi+((size_t)m*MERGE+b)*TOP;
    if(len<=TOP) {
        if(tid<TOP){ok[tid]=tid<len?pk[(size_t)m*pitch+lo+tid]:0;oi[tid]=tid<len?pi[(size_t)m*pitch+lo+tid]:-1;}
        return;
    }
    unsigned keys[E];int ids[E];
#pragma unroll
    for(int e=0;e<E;++e) {
        int j=lo+e*T+tid;
        keys[e]=j<hi?pk[(size_t)m*pitch+j]:0;
        ids[e]=j<hi?pi[(size_t)m*pitch+j]:-1;
    }
    unsigned th;int eq;radix(keys,TOP,th,eq,hist,bucket);
    if(tid<2)counts[tid]=0;__syncthreads();
#pragma unroll
    for(int e=0;e<E;++e) {
        unsigned key=keys[e];
        if(key>th){int a=atomicAdd(counts,1);ok[a]=key;oi[a]=ids[e];}
        else if(key==th&&key){int a=atomicAdd(counts+1,1);if(a<eq){ok[TOP-eq+a]=key;oi[TOP-eq+a]=ids[e];}}
    }
}
__global__ __launch_bounds__(256) void dh_hier_stage1(
    const unsigned* pk,const int* pi,const int* npart,int pitch,unsigned* outk,int* outi) {
    __shared__ int hist[256],bucket[2],counts[2];
    int seg=(npart[blockIdx.y]+MERGE-1)/MERGE;
    if(seg<=1024)dh_hier_select<4>(pk,pi,npart,pitch,outk,outi,hist,bucket,counts);
    else if(seg<=4096)dh_hier_select<16>(pk,pi,npart,pitch,outk,outi,hist,bucket,counts);
    else dh_hier_select<64>(pk,pi,npart,pitch,outk,outi,hist,bucket,counts);
}
__global__ __launch_bounds__(256) void dh_hier_finish(
    const unsigned* pk,const int* pi,float* vals,int64_t* idx) {
    constexpr int E=MERGE*TOP/T;
    __shared__ int hist[256],bucket[2],counts[2],si[64];
    __shared__ unsigned sk[64];
    int tid=threadIdx.x,m=blockIdx.x;
    unsigned keys[E];int ids[E];
#pragma unroll
    for(int e=0;e<E;++e){int j=e*T+tid;keys[e]=pk[(size_t)m*MERGE*TOP+j];ids[e]=pi[(size_t)m*MERGE*TOP+j];}
    unsigned th;int eq;radix(keys,TOP,th,eq,hist,bucket);
    if(tid<2)counts[tid]=0;__syncthreads();
#pragma unroll
    for(int e=0;e<E;++e) {
        unsigned key=keys[e];
        if(key>th){int a=atomicAdd(counts,1);sk[a]=key;si[a]=ids[e];}
        else if(key==th&&key){int a=atomicAdd(counts+1,1);if(a<eq){sk[TOP-eq+a]=key;si[TOP-eq+a]=ids[e];}}
    }
    __syncthreads();
    if(tid<TOP) {
        unsigned key=sk[tid];int id=si[tid],r=0;
#pragma unroll
        for(int j=0;j<TOP;++j)r+=(sk[j]>key)||(sk[j]==key&&si[j]<id);
        vals[(size_t)m*TOP+r]=keyfloat(key);idx[(size_t)m*TOP+r]=id;
    }
}
void check(const at::Tensor& x,const at::Tensor& q,const at::Tensor& s,int div) {
    TORCH_CHECK(x.is_cuda()&&q.is_cuda()&&s.is_cuda()&&q.device()==x.device()&&s.device()==x.device());
    TORCH_CHECK(x.scalar_type()==at::kBFloat16&&x.dim()==2&&x.size(1)==K&&x.stride(1)==1&&x.size(0)>=1&&x.size(0)<=8);
    TORCH_CHECK(q.scalar_type()==at::kByte&&q.dim()==2&&q.size(1)==K/div&&q.is_contiguous());
    TORCH_CHECK(s.scalar_type()==at::kBFloat16&&s.dim()==2&&s.size(0)==q.size(0)&&s.size(1)==40&&s.is_contiguous());
}
at::Tensor q2_impl(const at::Tensor& x,const at::Tensor& q,const at::Tensor& s,cudaStream_t st) {
    auto out=at::empty({x.size(0),q.size(0)},x.options());
#define CASE(M) case M:launch_q2<M>(x,q,s,out,st);break
    switch(x.size(0)){CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);}
#undef CASE
    return out;
}
at::Tensor threshold_impl(const at::Tensor& score,int C,cudaStream_t st,at::Tensor npart=at::Tensor()) {
    auto hist=at::empty({score.size(0),HB,BINS},score.options().dtype(at::kInt));
    auto th=at::empty({score.size(0)},score.options().dtype(at::kFloat));
    dh_hist_kernel<<<dim3(HB,score.size(0)),T,0,st>>>((const __nv_bfloat16*)score.data_ptr(),score.size(1),hist.data_ptr<int>());
    dh_threshold_kernel<<<score.size(0),T,0,st>>>(hist.data_ptr<int>(),th.data_ptr<float>(),C,npart.defined()?npart.data_ptr<int>():nullptr);
    return th;
}
std::vector<at::Tensor> rescore_impl(const at::Tensor& x,const at::Tensor& q,const at::Tensor& s,
    const at::Tensor& score,const at::Tensor& th,cudaStream_t st,at::Tensor npart=at::Tensor(),int blocks=NB) {
    auto pk=at::empty({x.size(0),blocks,TOP},x.options().dtype(at::kInt)),pi=at::empty_like(pk);
    if(!npart.defined())npart=at::zeros({x.size(0)},x.options().dtype(at::kInt));
    auto vals=at::empty({x.size(0),TOP},x.options().dtype(at::kFloat));
    auto ids=at::empty({x.size(0),TOP},x.options().dtype(at::kLong));
    dh_rescore_kernel<<<dim3(blocks,x.size(0)),T,0,st>>>((const __nv_bfloat16*)x.data_ptr(),x.stride(0),
        q.data_ptr<uint8_t>(),(const uint16_t*)s.data_ptr(),(const __nv_bfloat16*)score.data_ptr(),
        th.data_ptr<float>(),q.size(0),(unsigned*)pk.data_ptr<int>(),pi.data_ptr<int>(),npart.data_ptr<int>());
    auto pk2=at::empty({x.size(0),MERGE,TOP},x.options().dtype(at::kInt)),pi2=at::empty_like(pk2);
    dh_hier_stage1<<<dim3(MERGE,x.size(0)),T,0,st>>>((const unsigned*)pk.data_ptr<int>(),pi.data_ptr<int>(),npart.data_ptr<int>(),blocks*TOP,(unsigned*)pk2.data_ptr<int>(),pi2.data_ptr<int>());
    dh_hier_finish<<<x.size(0),T,0,st>>>((const unsigned*)pk2.data_ptr<int>(),pi2.data_ptr<int>(),vals.data_ptr<float>(),ids.data_ptr<int64_t>());
    return {vals,ids};
}
}
at::Tensor drafthead_q2(const at::Tensor& x,const at::Tensor& q,const at::Tensor& s) {
    check(x,q,s,4);const c10::cuda::CUDAGuard guard(x.device());
    auto out=q2_impl(x,q,s,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
at::Tensor drafthead_threshold(const at::Tensor& score,int64_t C) {
    TORCH_CHECK(score.is_cuda()&&score.is_contiguous()&&score.dim()==2&&score.scalar_type()==at::kBFloat16);
    TORCH_CHECK(C>=64&&C<=score.size(1));
    const c10::cuda::CUDAGuard guard(score.device());
    auto out=threshold_impl(score,C,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<at::Tensor> drafthead_rescore(const at::Tensor& x,const at::Tensor& q,const at::Tensor& s,
    const at::Tensor& score,const at::Tensor& th) {
    check(x,q,s,2);TORCH_CHECK(q.size(0)>=64&&q.size(0)<=131072);
    TORCH_CHECK(score.sizes()==at::IntArrayRef({x.size(0),q.size(0)})&&score.is_contiguous()&&score.scalar_type()==at::kBFloat16&&score.device()==x.device());
    TORCH_CHECK(th.is_contiguous()&&th.numel()==x.size(0)&&th.scalar_type()==at::kFloat&&th.device()==x.device());
    const c10::cuda::CUDAGuard guard(x.device());
    auto out=rescore_impl(x,q,s,score,th,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<at::Tensor> drafthead_screen(const at::Tensor& x,const at::Tensor& q2,const at::Tensor& s2,
    const at::Tensor& q4,const at::Tensor& s4,int64_t C) {
    check(x,q2,s2,4);check(x,q4,s4,2);
    TORCH_CHECK(q2.size(0)==q4.size(0)&&q4.size(0)>=64&&q4.size(0)<=131072&&C>=64&&C<=q4.size(0));
    const c10::cuda::CUDAGuard guard(x.device());auto st=at::cuda::getCurrentCUDAStream();
    auto npart=at::empty({x.size(0)},x.options().dtype(at::kInt));
    auto score=q2_impl(x,q2,s2,st);auto th=threshold_impl(score,C,st,npart);
    auto out=rescore_impl(x,q4,s4,score,th,st,npart);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// Diagnostic/tuning entry point; the production six-argument API uses NB=1024.
std::vector<at::Tensor> drafthead_screen_blocks(const at::Tensor& x,const at::Tensor& q2,const at::Tensor& s2,
    const at::Tensor& q4,const at::Tensor& s4,int64_t C,int64_t blocks) {
    check(x,q2,s2,4);check(x,q4,s4,2);
    TORCH_CHECK(q2.size(0)==q4.size(0)&&q4.size(0)>=64&&q4.size(0)<=131072&&C>=64&&C<=q4.size(0));
    TORCH_CHECK(blocks>=256&&blocks<=4096&&(q4.size(0)+blocks-1)/blocks<=512);
    const c10::cuda::CUDAGuard guard(x.device());auto st=at::cuda::getCurrentCUDAStream();
    auto npart=at::empty({x.size(0)},x.options().dtype(at::kInt));
    auto score=q2_impl(x,q2,s2,st);auto th=threshold_impl(score,C,st,npart);
    auto out=rescore_impl(x,q4,s4,score,th,st,npart,blocks);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
