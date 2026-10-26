// Decode-size paged indexer MQA. Identical BF16 tensor-core dot products, FP32 ReLU/head sum.
// A bounded grid traverses active groups; the same CTAs initialize the full logits tail.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math_constants.h>

namespace {
using B=__nv_bfloat16;
__device__ __forceinline__ void mma(float (&d)[4],uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
         : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__global__ __launch_bounds__(128) void wp_attn_mqa_kernel(const B* q,const B* cache,
         const int* table,const int* lens,float* out,int table_stride,int maxlen,float scale) {
    int row=blockIdx.y,ctas=gridDim.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int g=lane>>2,t=lane&3;
    // Ownership matches the active 64-token groups below, avoiding cross-CTA fill races.
    for(int col=blockIdx.x*64+(threadIdx.x&63)+(threadIdx.x>>6)*ctas*64;
        col<maxlen;col+=ctas*128) out[(size_t)row*maxlen+col]=-CUDART_INF_F;
    __syncthreads();
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int length=lens[row];
    if((int)blockIdx.x*64>=length)return;
    uint32_t qb0[8],qb1[8];
#pragma unroll
    for(int i=0;i<8;++i) {
        qb0[i]=*reinterpret_cast<const uint32_t*>(q+(row*8+g)*128+i*16+t*2);
        qb1[i]=*reinterpret_cast<const uint32_t*>(q+(row*8+g)*128+i*16+8+t*2);
    }
    for(int start=blockIdx.x*64;start<length;start+=ctas*64) {
        int first=start+warp*16;
        if(first<length) {
            int page=table[(size_t)row*table_stride+first/16];
            const B* kk=cache+(int64_t)page*16*128;
            uint32_t aa[8][4];
#pragma unroll
            for(int i=0;i<8;++i) {
                aa[i][0]=*reinterpret_cast<const uint32_t*>(kk+g*128+i*16+t*2);
                aa[i][1]=*reinterpret_cast<const uint32_t*>(kk+(g+8)*128+i*16+t*2);
                aa[i][2]=*reinterpret_cast<const uint32_t*>(kk+g*128+i*16+8+t*2);
                aa[i][3]=*reinterpret_cast<const uint32_t*>(kk+(g+8)*128+i*16+8+t*2);
            }
            float d[4]={0.f,0.f,0.f,0.f};
#pragma unroll
            for(int i=0;i<8;++i)mma(d,aa[i][0],aa[i][1],aa[i][2],aa[i][3],qb0[i],qb1[i]);
            float a=fmaxf(d[0],0.f)+fmaxf(d[1],0.f);
            float b=fmaxf(d[2],0.f)+fmaxf(d[3],0.f);
            a+=__shfl_xor_sync(0xffffffff,a,1);
            b+=__shfl_xor_sync(0xffffffff,b,1);
            a+=__shfl_xor_sync(0xffffffff,a,2);
            b+=__shfl_xor_sync(0xffffffff,b,2);
            if(t==0) {
                if(first+g<length)out[(size_t)row*maxlen+first+g]=a/scale;
                if(first+g+8<length)out[(size_t)row*maxlen+first+g+8]=b/scale;
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
}

at::Tensor attn_index_mqa(const at::Tensor& q,const at::Tensor& cache,const at::Tensor& table,
                         const at::Tensor& lens,int64_t maxlen,int64_t ctas,double scale) {
    TORCH_CHECK(q.is_cuda()&&q.scalar_type()==at::kBFloat16&&q.is_contiguous()&&q.dim()==3&&q.size(1)==8&&q.size(2)==128);
    TORCH_CHECK(q.size(0)>=1&&q.size(0)<=8&&cache.is_cuda()&&cache.scalar_type()==at::kBFloat16&&cache.is_contiguous());
    TORCH_CHECK(cache.dim()==4&&cache.size(1)==16&&cache.size(2)==1&&cache.size(3)==128);
    TORCH_CHECK(table.scalar_type()==at::kInt&&table.dim()==2&&table.size(0)==q.size(0)&&table.stride(1)==1);
    TORCH_CHECK(lens.scalar_type()==at::kInt&&lens.numel()==q.size(0)&&lens.is_contiguous());
    TORCH_CHECK(ctas>=1&&ctas<=1024&&maxlen>=1&&maxlen<=table.size(1)*16);
    const c10::cuda::CUDAGuard guard(q.device());
    auto out=at::empty({q.size(0),maxlen},q.options().dtype(at::kFloat));
    cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(ctas,q.size(0));cfg.blockDim=dim3(128);cfg.stream=at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];attr[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed=1;cfg.attrs=attr;cfg.numAttrs=1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,wp_attn_mqa_kernel,(const B*)q.data_ptr(),(const B*)cache.data_ptr(),
        (const int*)table.data_ptr(),(const int*)lens.data_ptr(),(float*)out.data_ptr(),(int)table.stride(0),(int)maxlen,(float)scale));
    return out;
}
