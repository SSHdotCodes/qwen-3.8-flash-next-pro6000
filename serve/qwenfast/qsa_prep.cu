// One-CTA decode indexer prep: Q norm/MRoPE, raw K/position store, compressed K norm/MRoPE.
// Arithmetic and explicit BF16 rounding follow the frozen production qsa_indexer.cuh.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

namespace {
using B=__nv_bfloat16;
__device__ __forceinline__ float rnd(float x) {
    unsigned short bits;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(x));
    return __bfloat162float(__ushort_as_bfloat16(bits));
}
__device__ __forceinline__ float bf(B x) { return __bfloat162float(x); }
__device__ __forceinline__ float red(float x) {
#pragma unroll
    for(int s=16;s>0;s>>=1) x+=__shfl_xor_sync(0xffffffff,x,s);
    return x;
}
__device__ __forceinline__ void norm(const B* x,const B* weight,B* buf,float eps) {
    const int lane=threadIdx.x&31;
    float ss=0.f;
    if(lane<16) {
#pragma unroll
        for(int i=0;i<8;++i) {float f=bf(x[lane*8+i]);ss+=f*f;}
    }
    ss=red(ss);float nf=rsqrtf(ss/128.f+eps);
#pragma unroll
    for(int i=0;i<4;++i) {int d=lane*4+i;buf[d]=__float2bfloat16(bf(x[d])*nf*(1.f+bf(weight[d])));}
    __syncwarp();
}
__device__ __forceinline__ void rope(const B* buf,B* out,const float* cs,const int* axis,
                                    const int64_t* pos,int rotary) {
    const int lane=threadIdx.x&31,half=rotary/2;
#pragma unroll
    for(int i=0;i<4;++i) {
        int d=lane*4+i;float outval=bf(buf[d]);
        if(d<rotary) {
            int pair=d<half?d:d-half,partner=d<half?d+half:d-half;
            const float* r=cs+pos[axis[pair]]*rotary;
            float c=rnd(r[pair]),s=rnd(r[half+pair]);
            float a=rnd(bf(buf[d])*c),b=rnd(bf(buf[partner])*s);
            outval=d<half?a-b:a+b;
        }
        out[d]=__float2bfloat16(outval);
    }
}

__global__ __launch_bounds__(256) void wp_attn_index_prep_kernel(
        const B* qk,B* qo,const B* qw,const B* kw,const float* cs,const int* axis,
        const int64_t* pos,int pos_axes,int pos_stride,const int64_t* slots,
        B* state,int64_t* state_pos,const int* groups,const int* writes,B* cache,
        int M,int NG,int rotary,float qeps,float keps) {
    const int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    __shared__ B buf[8][128];
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // Commit every current token before any compression group reads the ring.
    for(int t=warp;t<M;t+=8) {
        int64_t loc=slots[t];
        *reinterpret_cast<uint2*>(state+loc*128+lane*4)=
            *reinterpret_cast<const uint2*>(qk+t*640+512+lane*4);
        if(lane<3)state_pos[loc*3+lane]=pos[(lane<pos_axes?lane:0)*pos_stride+t];
    }
    __syncthreads();
    for(int task=warp;task<M*4;task+=8) {
        const int t=task/4,h=task%4;
        int64_t pp[3];
#pragma unroll
        for(int a=0;a<3;++a)pp[a]=pos[(a<pos_axes?a:0)*pos_stride+t];
        norm(qk+t*640+h*128,qw,buf[warp],qeps);
        rope(buf[warp],qo+(t*8+h)*128,cs,axis,pp,rotary);
        *reinterpret_cast<uint2*>(qo+(t*8+h+4)*128+lane*4)=make_uint2(0,0);
    }
    for(int group=warp;group<NG;group+=8) {
        const int* gg=groups+group*4;
        float mf[4];
#pragma unroll
        for(int i=0;i<4;++i) {
            int d=lane*4+i;float sum=bf(state[(int64_t)gg[0]*128+d]);
#pragma unroll
            for(int r=1;r<4;++r)sum+=bf(state[(int64_t)gg[r]*128+d]);
            B v=__float2bfloat16(sum*.25f);mf[i]=bf(v);buf[warp][d]=v;
        }
        __syncwarp();
        float ss=0.f;
        if(lane<16) {
#pragma unroll
            for(int i=0;i<8;++i) {float f=bf(buf[warp][lane*8+i]);ss+=f*f;}
        }
        ss=red(ss);float nf=rsqrtf(ss/128.f+keps);
#pragma unroll
        for(int i=0;i<4;++i) {int d=lane*4+i;buf[warp][d]=__float2bfloat16(mf[i]*nf*(1.f+bf(kw[d])));}
        __syncwarp();
        int64_t pp[3];
#pragma unroll
        for(int a=0;a<3;++a)pp[a]=state_pos[(int64_t)gg[0]*3+a];
        rope(buf[warp],cache+(int64_t)writes[group]*128,cs,axis,pp,rotary);
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// Independent CTAs: compression reads this forward's raw keys directly from qk,
// avoiding a cross-CTA dependency on the state-ring stores.
__global__ __launch_bounds__(128) void wp_attn_index_prep_parallel_kernel(
        const B* qk,B* qo,const B* qw,const B* kw,const float* cs,const int* axis,
        const int64_t* pos,int pos_axes,int pos_stride,const int64_t* slots,
        B* state,int64_t* state_pos,const int* groups,const int* writes,B* cache,
        int M,int NG,int rotary,float qeps,float keps) {
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    __shared__ B buf[4][128];
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if((int)blockIdx.x<M) {
        int t=blockIdx.x;int64_t loc=slots[t];int64_t pp[3];
#pragma unroll
        for(int a=0;a<3;++a)pp[a]=pos[(a<pos_axes?a:0)*pos_stride+t];
        if(warp==0)*reinterpret_cast<uint2*>(state+loc*128+lane*4)=
             *reinterpret_cast<const uint2*>(qk+t*640+512+lane*4);
        if(warp==1&&lane<3)state_pos[loc*3+lane]=pp[lane];
        norm(qk+t*640+warp*128,qw,buf[warp],qeps);
        rope(buf[warp],qo+(t*8+warp)*128,cs,axis,pp,rotary);
        *reinterpret_cast<uint2*>(qo+(t*8+warp+4)*128+lane*4)=make_uint2(0,0);
    } else {
        int group=((int)blockIdx.x-M)*4+warp;
        if(group>=NG)return;
        const int* gg=groups+group*4;
        const int64_t own_slot=lane<M?slots[lane]:-1;
        float acc[4]={0.f,0.f,0.f,0.f};int token0=-1;
#pragma unroll
        for(int r=0;r<4;++r) {
            int loc=gg[r];unsigned mask=__ballot_sync(0xffffffff,lane<M&&own_slot==loc);
            int token=mask?31-__clz(mask):-1;
            if(r==0)token0=token;
            const B* src=token>=0?qk+token*640+512:state+(int64_t)loc*128;
#pragma unroll
            for(int i=0;i<4;++i) {float f=bf(src[lane*4+i]);acc[i]=(r==0)?f:acc[i]+f;}
        }
        float mf[4];
#pragma unroll
        for(int i=0;i<4;++i) {B b=__float2bfloat16(acc[i]*.25f);mf[i]=bf(b);buf[warp][lane*4+i]=b;}
        __syncwarp();float ss=0.f;
        if(lane<16) {
#pragma unroll
            for(int i=0;i<8;++i) {float f=bf(buf[warp][lane*8+i]);ss+=f*f;}
        }
        ss=red(ss);float nf=rsqrtf(ss/128.f+keps);
#pragma unroll
        for(int i=0;i<4;++i) {int d=lane*4+i;buf[warp][d]=__float2bfloat16(mf[i]*nf*(1.f+bf(kw[d])));}
        __syncwarp();int64_t pp[3];
#pragma unroll
        for(int a=0;a<3;++a)pp[a]=token0>=0?pos[(a<pos_axes?a:0)*pos_stride+token0]:state_pos[(int64_t)gg[0]*3+a];
        rope(buf[warp],cache+(int64_t)writes[group]*128,cs,axis,pp,rotary);
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // namespace

at::Tensor attn_index_prep_impl(const at::Tensor& qk,const at::Tensor& qw,const at::Tensor& kw,
        const at::Tensor& cs,const at::Tensor& axis,const at::Tensor& positions,
        const at::Tensor& slots,const at::Tensor& state,const at::Tensor& state_pos,
        const at::Tensor& groups,const at::Tensor& writes,const at::Tensor& cache,
        int64_t rotary,double qeps,double keps,bool parallel) {
    TORCH_CHECK(qk.is_cuda()&&qk.scalar_type()==at::kBFloat16&&qk.dim()==2&&qk.size(1)==640&&qk.is_contiguous());
    TORCH_CHECK(qk.size(0)>=1&&qk.size(0)<=8&&qw.numel()==128&&kw.numel()==128);
    TORCH_CHECK(state.scalar_type()==at::kBFloat16&&cache.scalar_type()==at::kBFloat16);
    TORCH_CHECK(groups.scalar_type()==at::kInt&&groups.dim()==2&&groups.size(1)==4&&groups.is_contiguous());
    TORCH_CHECK(writes.scalar_type()==at::kInt&&slots.scalar_type()==at::kLong&&slots.is_contiguous());
    TORCH_CHECK(positions.scalar_type()==at::kLong&&(positions.dim()==1||positions.dim()==2)&&positions.stride(-1)==1);
    TORCH_CHECK(state_pos.scalar_type()==at::kLong&&axis.scalar_type()==at::kInt&&cs.scalar_type()==at::kFloat);
    TORCH_CHECK(rotary==64&&groups.size(0)==writes.numel());
    const c10::cuda::CUDAGuard guard(qk.device());
    int M=qk.size(0),axes=positions.dim()==1?1:positions.size(0),ps=positions.dim()==1?M:positions.stride(0);
    auto qo=at::empty({M,8,128},qk.options());
    cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(parallel?M+(groups.size(0)+3)/4:1);cfg.blockDim=dim3(parallel?128:256);cfg.stream=at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];attr[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed=1;cfg.attrs=attr;cfg.numAttrs=1;
    auto kernel=parallel?wp_attn_index_prep_parallel_kernel:wp_attn_index_prep_kernel;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,kernel,
        (const B*)qk.data_ptr(),(B*)qo.data_ptr(),(const B*)qw.data_ptr(),(const B*)kw.data_ptr(),
        (const float*)cs.data_ptr(),(const int*)axis.data_ptr(),(const int64_t*)positions.data_ptr(),axes,ps,
        (const int64_t*)slots.data_ptr(),(B*)state.data_ptr(),(int64_t*)state_pos.data_ptr(),
        (const int*)groups.data_ptr(),(const int*)writes.data_ptr(),(B*)cache.data_ptr(),
        M,(int)groups.size(0),(int)rotary,(float)qeps,(float)keps));
    return qo;
}

at::Tensor attn_index_prep(const at::Tensor& qk,const at::Tensor& qw,const at::Tensor& kw,
        const at::Tensor& cs,const at::Tensor& axis,const at::Tensor& positions,
        const at::Tensor& slots,const at::Tensor& state,const at::Tensor& state_pos,
        const at::Tensor& groups,const at::Tensor& writes,const at::Tensor& cache,
        int64_t rotary,double qeps,double keps) {
 return attn_index_prep_impl(qk,qw,kw,cs,axis,positions,slots,state,state_pos,groups,writes,cache,rotary,qeps,keps,false);
}

at::Tensor attn_index_prep_parallel(const at::Tensor& qk,const at::Tensor& qw,const at::Tensor& kw,
        const at::Tensor& cs,const at::Tensor& axis,const at::Tensor& positions,
        const at::Tensor& slots,const at::Tensor& state,const at::Tensor& state_pos,
        const at::Tensor& groups,const at::Tensor& writes,const at::Tensor& cache,
        int64_t rotary,double qeps,double keps) {
 return attn_index_prep_impl(qk,qw,kw,cs,axis,positions,slots,state,state_pos,groups,writes,cache,rotary,qeps,keps,true);
}
