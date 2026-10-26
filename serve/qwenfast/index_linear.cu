// Exact BF16 weight GEMM for the indexer [M,2560] x [640,2560]^T.
// Twenty 128-wide FP32 partials, followed by FP32 summation and one BF16 rounding.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
namespace {
using B=__nv_bfloat16;
__device__ __forceinline__ void mma(float (&d)[4],uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__global__ __launch_bounds__(32) void wp_attn_index_linear_parts(const B* x,const B* w,float* part,int M) {
    int split=blockIdx.y,lane=threadIdx.x,g=lane>>2,t=lane&3;
    int n=blockIdx.x*16,k0=split*128;
    uint32_t aa[8][4],bb[8][2];
#pragma unroll
    for(int i=0;i<8;++i) {
        int k=k0+i*16+t*2;
        aa[i][0]=__ldcs(reinterpret_cast<const uint32_t*>(w+(n+g)*2560+k));
        aa[i][1]=__ldcs(reinterpret_cast<const uint32_t*>(w+(n+g+8)*2560+k));
        aa[i][2]=__ldcs(reinterpret_cast<const uint32_t*>(w+(n+g)*2560+k+8));
        aa[i][3]=__ldcs(reinterpret_cast<const uint32_t*>(w+(n+g+8)*2560+k+8));
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
#pragma unroll
    for(int i=0;i<8;++i) {
        int k=k0+i*16+t*2;
        bb[i][0]=g<M?__ldg(reinterpret_cast<const uint32_t*>(x+g*2560+k)):0;
        bb[i][1]=g<M?__ldg(reinterpret_cast<const uint32_t*>(x+g*2560+k+8)):0;
    }
    float d[4]={0.f,0.f,0.f,0.f};
#pragma unroll
    for(int i=0;i<8;++i)mma(d,aa[i][0],aa[i][1],aa[i][2],aa[i][3],bb[i][0],bb[i][1]);
    if(2*t<M) {
        part[(split*M+2*t)*640+n+g]=d[0];
        part[(split*M+2*t)*640+n+g+8]=d[2];
    }
    if(2*t+1<M) {
        part[(split*M+2*t+1)*640+n+g]=d[1];
        part[(split*M+2*t+1)*640+n+g+8]=d[3];
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__global__ __launch_bounds__(128) void wp_attn_index_linear_reduce(const float* part,B* out,int M) {
    int col=blockIdx.x*128+threadIdx.x;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if(col<M*640) {
        float v=0.f;
#pragma unroll
        for(int i=0;i<20;++i)v+=part[i*M*640+col];
        out[col]=__float2bfloat16(v);
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
}
at::Tensor attn_index_linear(const at::Tensor& x,const at::Tensor& w) {
    TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&x.scalar_type()==at::kBFloat16&&w.scalar_type()==at::kBFloat16);
    TORCH_CHECK(x.is_contiguous()&&w.is_contiguous()&&x.dim()==2&&w.sizes()==at::IntArrayRef({640,2560}));
    TORCH_CHECK(x.size(0)>=1&&x.size(0)<=8&&x.size(1)==2560);
    const c10::cuda::CUDAGuard guard(x.device());int M=x.size(0);
    auto part=at::empty({20,M,640},x.options().dtype(at::kFloat));auto out=at::empty({M,640},x.options());
    cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(40,20);cfg.blockDim=dim3(32);cfg.stream=at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];attr[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed=1;cfg.attrs=attr;cfg.numAttrs=1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,wp_attn_index_linear_parts,(const B*)x.data_ptr(),(const B*)w.data_ptr(),(float*)part.data_ptr(),M));
    cfg.gridDim=dim3((M*640+127)/128);cfg.blockDim=dim3(128);
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,wp_attn_index_linear_reduce,(const float*)part.data_ptr(),(B*)out.data_ptr(),M));
    return out;
}
