// WP-GDN output: exact production gated norm followed by the original eight K partials.
// No weight/dtype/rounding changes. The normalized BF16 intermediate is computed once.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
namespace {
constexpr int HEADS=48,HD=128,K=6144,N=2560,LOGICAL_WARPS=8,KW=768;
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 i8x8_to_bf16(uint32_t lo, uint32_t hi, float sc) {
    float f[8];
    lo ^= 0x80808080u; hi ^= 0x80808080u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[i] = __int_as_float(__byte_perm(lo, 0x4B000000u, 0x7440 + i)) - 8388736.f;
        f[4 + i] = __int_as_float(__byte_perm(hi, 0x4B000000u, 0x7440 + i)) - 8388736.f;
    }
    uint4 o;
    __nv_bfloat162 * op = reinterpret_cast<__nv_bfloat162 *>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) op[i] = __floats2bfloat162_rn(f[2 * i] * sc, f[2 * i + 1] * sc);
    return o;
}

__global__ void wpgdnout_norm_kernel(const __nv_bfloat16* core,const __nv_bfloat16* z,const __nv_bfloat16* wn,
                              __nv_bfloat16* out,int pairs,float eps,int swish) {
    int lane=threadIdx.x&31,sl=lane&7;
    int p=(blockIdx.x*blockDim.x+threadIdx.x)/8;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if(p>=pairs)return;
    uint4 cv[2],zv[2],wv[2];float xv[16],zf[16],wf[16];
#pragma unroll
    for(int h=0;h<2;++h) {
        cv[h]=__ldcg(reinterpret_cast<const uint4*>(core+p*128)+sl*2+h);
        zv[h]=__ldcg(reinterpret_cast<const uint4*>(z+p*128)+sl*2+h);
        wv[h]=__ldg(reinterpret_cast<const uint4*>(wn)+sl*2+h);
    }
#pragma unroll
    for(int h=0;h<2;++h) {
        auto c2=reinterpret_cast<const __nv_bfloat162*>(&cv[h]);
        auto z2=reinterpret_cast<const __nv_bfloat162*>(&zv[h]);
        auto w2=reinterpret_cast<const __nv_bfloat162*>(&wv[h]);
#pragma unroll
        for(int i=0;i<4;++i) {
            float2 c=__bfloat1622float2(c2[i]),zz=__bfloat1622float2(z2[i]),w=__bfloat1622float2(w2[i]);
            xv[h*8+2*i]=c.x;xv[h*8+2*i+1]=c.y;
            zf[h*8+2*i]=zz.x;zf[h*8+2*i+1]=zz.y;
            wf[h*8+2*i]=w.x;wf[h*8+2*i+1]=w.y;
        }
    }
    float ss=0.f;
#pragma unroll
    for(int i=0;i<16;++i)ss+=xv[i]*xv[i];
    ss+=__shfl_xor_sync(0xffffffff,ss,4);
    ss+=__shfl_xor_sync(0xffffffff,ss,2);
    ss+=__shfl_xor_sync(0xffffffff,ss,1);
    float rstd=rsqrtf(ss/128+eps);uint4 o[2];auto o2=reinterpret_cast<__nv_bfloat162*>(o);
#pragma unroll
    for(int i=0;i<8;++i) {
        float r[2];
#pragma unroll
        for(int u=0;u<2;++u) {
            int e=2*i+u;float sg=1.f/(1.f+__expf(-zf[e]));
            r[u]=xv[e]*rstd*wf[e]*(swish ? zf[e]*sg : sg);
        }
        o2[i]=__floats2bfloat162_rn(r[0],r[1]);
    }
    auto dst=reinterpret_cast<uint4*>(out+p*128)+sl*2;dst[0]=o[0];dst[1]=o[1];
}


template<int M,int PHYSICAL_WARPS>
__global__ __launch_bounds__(PHYSICAL_WARPS*32) void wpgdnout_project_kernel(
 const __nv_bfloat16* __restrict__ xn,const int8_t* __restrict__ q,const uint16_t* __restrict__ scales,
 __nv_bfloat16* __restrict__ y,int prefetch,int release) {
    extern __shared__ float red[];
    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,grp=lane>>2,tq=lane&3;
    int n0=(int)((long)blockIdx.x*N/gridDim.x),n1=(int)((long)(blockIdx.x+1)*N/gridDim.x),nr=n1-n0;
    if(release==0)asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if(prefetch==1 && tid==0) {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(q+(size_t)n0*K),"r"((unsigned)(nr*K)) : "memory");
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(scales+(size_t)n0*48),"r"((unsigned)(nr*96)) : "memory");
    }
    if(prefetch==3) {
        for(int c=tid*16;c<nr*K;c+=PHYSICAL_WARPS*32*16) {
            uint4 v;
            asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(q+(size_t)n0*K+c) : "memory");
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if(release==2)asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#pragma unroll 1
    for(int pass=0;pass<LOGICAL_WARPS/PHYSICAL_WARPS;++pass) {
        int lw=warp+pass*PHYSICAL_WARPS;
        int r0=grp,r1=grp+8;bool v0=r0<nr,v1=r1<nr;
        int ra=n0+(v0?r0:0),rb=n0+(v1?r1:0),kb=lw*KW+tq*8;
        uint4 zero=make_uint4(0,0,0,0);
        float d[4]={0.f,0.f,0.f,0.f};
        constexpr int BATCH=4;
#pragma unroll 1
        for(int kc0=0;kc0<24;kc0+=BATCH) {
            uint2 w0[BATCH],w1[BATCH];uint32_t s0[BATCH],s1[BATCH];
            uint4 a0[BATCH],a1[BATCH],bv[BATCH];
#pragma unroll
            for(int i=0;i<BATCH;++i) {
                int k=kb+(kc0+i)*32,gk=(lw*KW+(kc0+i)*32)/128;
                w0[i]=v0?__ldcs(reinterpret_cast<const uint2*>(q+(size_t)ra*K+k)):make_uint2(0,0);
                w1[i]=v1?__ldcs(reinterpret_cast<const uint2*>(q+(size_t)rb*K+k)):make_uint2(0,0);
                s0[i]=v0?__ldg(scales+(size_t)ra*48+gk):0u;
                s1[i]=v1?__ldg(scales+(size_t)rb*48+gk):0u;
            }
#pragma unroll
            for(int i=0;i<BATCH;++i) {
                a0[i]=v0?i8x8_to_bf16(w0[i].x,w0[i].y,__uint_as_float(s0[i]<<16)):zero;
                a1[i]=v1?i8x8_to_bf16(w1[i].x,w1[i].y,__uint_as_float(s1[i]<<16)):zero;
            }
            // Read after GDC, bypassing L1 for the scratch written by the norm producer.
#pragma unroll
            for(int i=0;i<BATCH;++i)bv[i]=grp<M?__ldcg(reinterpret_cast<const uint4*>(xn+(size_t)grp*K+kb+(kc0+i)*32)):zero;
#pragma unroll
            for(int i=0;i<BATCH;++i) {
                mma16816(d,a0[i].x,a1[i].x,a0[i].y,a1[i].y,bv[i].x,bv[i].y);
                mma16816(d,a0[i].z,a1[i].z,a0[i].w,a1[i].w,bv[i].z,bv[i].w);
            }
        }
        float (*rp)[16][8]=reinterpret_cast<float (*)[16][8]>(red);
        rp[lw][r0][2*tq]=d[0];rp[lw][r0][2*tq+1]=d[1];
        rp[lw][r1][2*tq]=d[2];rp[lw][r1][2*tq+1]=d[3];
    }
    __syncthreads();
    if(tid<16*M) {
        int r=tid/M,t=tid%M;
        if(r<nr) {
            float (*rp)[16][8]=reinterpret_cast<float (*)[16][8]>(red);
            float value=0.f;
#pragma unroll
            for(int w=0;w<LOGICAL_WARPS;++w)value+=rp[w][r][t];
            y[(size_t)t*N+n0+r]=__float2bfloat16(value);
        }
    }
}

template<int M,int PW>
void project(const at::Tensor& x,const at::Tensor& q,const at::Tensor& sc,at::Tensor& y,
 int grid,int prefetch,int release) {
    cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(PW*32);cfg.dynamicSmemBytes=4096;
    cfg.stream=at::cuda::getCurrentCUDAStream();cudaLaunchAttribute attr[1];
    attr[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;attr[0].val.programmaticStreamSerializationAllowed=1;
    cfg.attrs=attr;cfg.numAttrs=1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,wpgdnout_project_kernel<M,PW>,(const __nv_bfloat16*)x.data_ptr(),
      (const int8_t*)q.data_ptr(),(const uint16_t*)sc.data_ptr(),(__nv_bfloat16*)y.data_ptr(),prefetch,release));
}
}

at::Tensor wpgdnout_trial(const at::Tensor& core,const at::Tensor& z,const at::Tensor& wn,double eps,bool swish,
 const at::Tensor& q,const at::Tensor& sc,int64_t grid,int64_t prefetch,int64_t release,int64_t physical_warps) {
    TORCH_CHECK(core.is_cuda()&&core.scalar_type()==at::kBFloat16&&core.dim()==2&&core.size(1)==128&&core.is_contiguous());
    int m=core.size(0)/48;TORCH_CHECK(m>=1&&m<=8&&core.size(0)==m*48);
    TORCH_CHECK(z.is_cuda()&&z.device()==core.device()&&z.sizes()==core.sizes()&&z.scalar_type()==at::kBFloat16&&z.is_contiguous());
    TORCH_CHECK(wn.is_cuda()&&wn.device()==core.device()&&wn.scalar_type()==at::kBFloat16&&wn.numel()==128&&wn.is_contiguous());
    TORCH_CHECK(q.is_cuda()&&q.device()==core.device()&&q.scalar_type()==at::kChar&&q.sizes()==at::IntArrayRef({2560,6144})&&q.is_contiguous());
    TORCH_CHECK(sc.is_cuda()&&sc.device()==core.device()&&sc.scalar_type()==at::kBFloat16&&sc.sizes()==at::IntArrayRef({2560,48})&&sc.is_contiguous());
    TORCH_CHECK(grid>=160&&grid<=640&&(prefetch==0||prefetch==1||prefetch==3)&&(release==0||release==2)&&(physical_warps==4||physical_warps==8));
    const c10::cuda::CUDAGuard guard(core.device());
    auto x=at::empty_like(core),y=at::empty({m,N},core.options());
    cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((m*48+15)/16);cfg.blockDim=dim3(128);cfg.stream=at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];attr[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed=1;cfg.attrs=attr;cfg.numAttrs=1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,wpgdnout_norm_kernel,(const __nv_bfloat16*)core.data_ptr(),(const __nv_bfloat16*)z.data_ptr(),
        (const __nv_bfloat16*)wn.data_ptr(),(__nv_bfloat16*)x.data_ptr(),m*48,(float)eps,swish?1:0));
#define CALL(T) if(physical_warps==4)project<T,4>(x,q,sc,y,grid,prefetch,release);else project<T,8>(x,q,sc,y,grid,prefetch,release)
    switch(m) {case 1:CALL(1);break;case 2:CALL(2);break;case 3:CALL(3);break;case 4:CALL(4);break;
               case 5:CALL(5);break;case 6:CALL(6);break;case 7:CALL(7);break;case 8:CALL(8);break;}
#undef CALL
    return y;
}

// variant 0: common streaming configuration; variant 1: short-predecessor configuration.
at::Tensor wpgdnout_w8(const at::Tensor& core,const at::Tensor& z,const at::Tensor& wn,double eps,bool swish,
 const at::Tensor& q,const at::Tensor& sc,int64_t variant) {
    TORCH_CHECK(variant==0||variant==1);
    int m=core.size(0)/48;
    if(m==4)return wpgdnout_trial(core,z,wn,eps,swish,q,sc,variant?320:256,variant?0:3,variant?0:2,8);
    return wpgdnout_trial(core,z,wn,eps,swish,q,sc,m>4?188:320,0,0,8);
}
