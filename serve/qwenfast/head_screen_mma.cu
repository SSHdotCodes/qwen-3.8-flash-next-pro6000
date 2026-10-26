#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
namespace {
__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                               uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t u4w(const uint4 & v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

// NW warps x NB blocks of 64 = K. Within a block, thread (g = lane / 4, t = lane % 4) holds bytes [16t, 16t + 16)
// of weight rows g and g + 8 and elements [16t, 16t + 16) of token g; mma step j consumes its bytes / elements
// 4j..4j+3. That is a fixed permutation of k inside the block, applied to weights and x alike, so the dot
// product is the same sum (the mma's own fragment layout would need 2-byte gathers).
__device__ __forceinline__ float head_bf(uint16_t v){return __uint_as_float(uint32_t(v)<<16);}
__device__ __forceinline__ uint32_t head_u2w(uint2 v,int i){return i?v.y:v.x;}
__device__ __forceinline__ void head_i4_to_bf16(uint32_t u,uint32_t& lo,uint32_t& hi){
 float f[4];
 #pragma unroll
 for(int i=0;i<4;i++)f[i]=__uint_as_float(0x47000000u|(((u>>(4*i))&15)<<12))-32888.5f;
 lo=__byte_perm(__float_as_uint(f[0]),__float_as_uint(f[1]),0x7632);
 hi=__byte_perm(__float_as_uint(f[2]),__float_as_uint(f[3]),0x7632);
}
template <int NB, int NW>
__global__ void __launch_bounds__(NW * 32) head_screen_mma_kernel(
        const __nv_bfloat16 * __restrict__ x, const uint8_t * __restrict__ q, const uint16_t * __restrict__ s,
        const float* en, float* screen,float* sigma,float* lower,float* upper,int N,int M,int ldx,int ldy,float z,const int* dense) {
    if(*dense)return;
    constexpr int K = NW * NB * 64, NG = K / 128;
    __shared__ float red[NW][16][9],varred[NW][16][9];
    __shared__ float energy[160];
    if(threadIdx.x<M*20)energy[threadIdx.x]=en[threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * 16;
    const size_t ra = (size_t) min(row0 + g, N - 1), rb = (size_t) min(row0 + g + 8, N - 1);
    const int kw = warp * NB * 64 + t * 16;
    uint2 wa[NB], wb[NB];
    uint16_t sa[NB], sb[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        wa[b] = __ldcs(reinterpret_cast<const uint2 *>(q + ra * (K/2) + (kw + b * 64)/2));
        wb[b] = __ldcs(reinterpret_cast<const uint2 *>(q + rb * (K/2) + (kw + b * 64)/2));
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        const int grp = (warp * NB * 64 + b * 64) / 128;
        sa[b] = __ldg(s + ra * NG + grp);
        sb[b] = __ldg(s + rb * NG + grp);
    }
    uint4 xv[NB][2];
    const bool tok = g < M;
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        const uint4 * xp = reinterpret_cast<const uint4 *>(x + (size_t) (tok ? g : 0) * ldx + kw + b * 64);
        xv[b][0] = tok ? __ldg(xp) : make_uint4(0, 0, 0, 0);
        xv[b][1] = tok ? __ldg(xp + 1) : make_uint4(0, 0, 0, 0);
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, var[4]={0.f,0.f,0.f,0.f};
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t a0, a1, a2, a3;
            head_i4_to_bf16(head_u2w(wa[b],j>>1)>>((j&1)*16),a0,a2);
            head_i4_to_bf16(head_u2w(wb[b],j>>1)>>((j&1)*16),a1,a3);
            const uint4 & xh = xv[b][j >> 1];
            mma_bf16_16816(d, a0, a1, a2, a3, u4w(xh, (j & 1) * 2), u4w(xh, (j & 1) * 2 + 1));
        }
        const float sca = head_bf(sa[b]), scb = head_bf(sb[b]);
        acc[0] = fmaf(d[0], sca, acc[0]);
        acc[1] = fmaf(d[1], sca, acc[1]);
        acc[2] = fmaf(d[2], scb, acc[2]);
        acc[3] = fmaf(d[3], scb, acc[3]);
        const int grp=(warp*NB*64+b*64)/128;
        float ea=2*t<M?energy[(2*t)*20+grp]:0.f, eb=2*t+1<M?energy[(2*t+1)*20+grp]:0.f;
        var[0]=fmaf(sca*sca,ea,var[0]);var[1]=fmaf(sca*sca,eb,var[1]);
        var[2]=fmaf(scb*scb,ea,var[2]);var[3]=fmaf(scb*scb,eb,var[3]);
    }
    // d layout: [0] (row g, token 2t) [1] (g, 2t + 1) [2] (g + 8, 2t) [3] (g + 8, 2t + 1)
    red[warp][g][2 * t] = acc[0];
    red[warp][g][2 * t + 1] = acc[1];
    red[warp][g + 8][2 * t] = acc[2];
    red[warp][g + 8][2 * t + 1] = acc[3];
    varred[warp][g][2*t]=var[0];varred[warp][g][2*t+1]=var[1];
    varred[warp][g+8][2*t]=var[2];varred[warp][g+8][2*t+1]=var[3];
    __syncthreads();
    if (threadIdx.x < 16 * M) {
        const int r = threadIdx.x & 15, m = threadIdx.x >> 4;
        float v = 0.f, vv=0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) {v += red[w][r][m];vv+=varred[w][r][m];}
        if(row0+r<N){size_t i=size_t(m)*N+row0+r;float sig=sqrtf(vv*(21.25f/2.f));
         screen[i]=v;sigma[i]=sig;lower[i]=v-z*sig;upper[i]=v+z*sig;}
    }
}


}
namespace {bool head_mma_configured=false;}
void head_set_carveout(int64_t percent){
 TORCH_CHECK(percent>=-1&&percent<=100);
 C10_CUDA_CHECK(cudaFuncSetAttribute(head_screen_mma_kernel<5,8>,cudaFuncAttributePreferredSharedMemoryCarveout,(int)percent));
 head_mma_configured=true;
}
void head_screen_mma(const at::Tensor& h,const at::Tensor& hi,const at::Tensor& s,const at::Tensor& en,
 at::Tensor& scr,at::Tensor& sig,at::Tensor& low,at::Tensor& up,const at::Tensor& dense,double z){
 if(!head_mma_configured){const char* e=getenv("QWENFAST_HEAD_CARVEOUT");head_set_carveout(e?atoi(e):-1);}
 head_screen_mma_kernel<5,8><<<(hi.size(0)+15)/16,256,0,at::cuda::getCurrentCUDAStream()>>>(
 reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),hi.data_ptr<uint8_t>(),reinterpret_cast<const uint16_t*>(s.data_ptr()),en.data_ptr<float>(),
 scr.data_ptr<float>(),sig.data_ptr<float>(),low.data_ptr<float>(),up.data_ptr<float>(),hi.size(0),h.size(0),h.stride(0),hi.size(0),z,dense.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
