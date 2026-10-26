#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int WARPS = 8;

__device__ __forceinline__ void i8x4_to_f32(uint32_t u, float * f) {
    u ^= 0x80808080u;   // exact: (2^23 + biased byte) - (2^23 + 128)
    f[0] = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7440)) - 8388736.f;
    f[1] = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7441)) - 8388736.f;
    f[2] = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7442)) - 8388736.f;
    f[3] = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7443)) - 8388736.f;
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) { return __uint_as_float(((uint32_t) b) << 16); }

// x row m, elements idx..idx+3 as floats, from fp32 or bf16 staging
template <bool XB>
__device__ __forceinline__ float4 x4(const float * xs, int idx) {
    if constexpr (XB) {
        const uint2 v = *reinterpret_cast<const uint2 *>(reinterpret_cast<const __nv_bfloat16 *>(xs) + idx);
        return make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xFFFF0000u),
                           __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xFFFF0000u));
    } else {
        return *reinterpret_cast<const float4 *>(xs + idx);
    }
}

// One warp per RPW rows. VEC weights (bytes) per lane per chunk of 32 * VEC; U chunks per row in flight.
// x staged in shared memory as fp32 [M][K] (bf16 when XB, for large M * K).
template <int M, int RPW, int VEC, int KC, int G, bool XB>
__global__ void __launch_bounds__(WARPS * 32) head_dense_kernel(
        const __nv_bfloat16 * __restrict__ x, const int8_t * __restrict__ q, const uint16_t * __restrict__ s,
        __nv_bfloat16 * __restrict__ y, int N, int ldx, int ldy, const int* dense, const int* overflow) {
    if (!*dense && !*overflow) return;
    constexpr int K = KC * 32 * VEC, NG = K / G;
    static_assert((G % VEC == 0 && (32 * VEC) % G == 0) || G % (32 * VEC) == 0, "group layout");
    extern __shared__ __align__(16) float xs[];   // [M][K]
    for (int c = threadIdx.x; c < M * (K / 8); c += blockDim.x) {
        const int m = c / (K / 8), kc = c % (K / 8);
        const uint4 v = *reinterpret_cast<const uint4 *>(x + (size_t) m * ldx + kc * 8);
        if constexpr (XB) {
            reinterpret_cast<uint4 *>(xs)[c] = v;
        } else {
            const __nv_bfloat162 * h = reinterpret_cast<const __nv_bfloat162 *>(&v);
            float4 * d = reinterpret_cast<float4 *>(xs + m * K + kc * 8);
            const float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
            const float2 c2 = __bfloat1622float2(h[2]), d2 = __bfloat1622float2(h[3]);
            d[0] = make_float4(a.x, a.y, b.x, b.y);
            d[1] = make_float4(c2.x, c2.y, d2.x, d2.y);
        }
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int row0 = (blockIdx.x * WARPS + warp) * RPW; row0 < N; row0 += gridDim.x * WARPS * RPW) {
    using V = typename std::conditional<VEC == 16, uint4, uint32_t>::type;
    float acc[RPW][M];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    constexpr int U = KC <= 6 ? KC : 6;
#pragma unroll 1
    for (int c0 = 0; c0 < KC; c0 += U) {
        V wv[RPW][U];
        uint16_t sv[RPW][U];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const size_t row = min(row0 + r, N - 1);
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (c0 + u < KC) {
                    const int k0 = (c0 + u) * 32 * VEC + lane * VEC;
                    wv[r][u] = __ldcs(reinterpret_cast<const V *>(q + row * K + k0));
                    sv[r][u] = __ldg(s + row * NG + k0 / G);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (c0 + u >= KC) break;
            const int k0 = (c0 + u) * 32 * VEC + lane * VEC;
#pragma unroll
            for (int r = 0; r < RPW; ++r) {
                float wf[VEC];
                const uint32_t * ww = reinterpret_cast<const uint32_t *>(&wv[r][u]);
#pragma unroll
                for (int i = 0; i < VEC / 4; ++i) i8x4_to_f32(ww[i], wf + 4 * i);
                const float sc = bf16_bits_to_f32(sv[r][u]);
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    float p = 0.f;
#pragma unroll
                    for (int i = 0; i < VEC / 4; ++i) {
                        const float4 xv = x4<XB>(xs, m * K + k0 + 4 * i);
                        p = fmaf(wf[4 * i], xv.x, p);
                        p = fmaf(wf[4 * i + 1], xv.y, p);
                        p = fmaf(wf[4 * i + 2], xv.z, p);
                        p = fmaf(wf[4 * i + 3], xv.w, p);
                    }
                    acc[r][m] = fmaf(p, sc, acc[r][m]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float v = acc[r][m];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            acc[r][m] = v;
        }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            if (row0 + r >= N) break;
#pragma unroll
            for (int m = 0; m < M; ++m) y[(size_t) m * ldy + row0 + r] = __float2bfloat16(acc[r][m]);
        }
    }
}
}

template<int M>void head_dense_launch(const at::Tensor& h,const at::Tensor& q,const at::Tensor& s,at::Tensor& out,const at::Tensor& dense,const at::Tensor& overflow){
 constexpr int RPW=M>4?4:2;constexpr bool XB=M>4;
 int N=q.size(0);int blocks=std::min(752,(N+8*RPW-1)/(8*RPW));
 head_dense_kernel<M,RPW,16,5,128,XB><<<blocks,256,M*2560*(XB?2:4),at::cuda::getCurrentCUDAStream()>>>(
 reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),q.data_ptr<int8_t>(),reinterpret_cast<const uint16_t*>(s.data_ptr()),
 reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),N,h.stride(0),N,dense.data_ptr<int>(),overflow.data_ptr<int>());
}
__global__ __launch_bounds__(256) void head_rescore_kernel(const __nv_bfloat16* h,const int8_t* q,const uint16_t* s,
 const int* ids,const int* counts,const int* overflow,const int* dense,__nv_bfloat16* out,int N,int stride,int cap){
 int m=blockIdx.y,c0=blockIdx.x*8;
 if(*dense||*overflow||c0>=min(counts[m],cap))return;
 __shared__ __align__(16) float xs[2560];
 for(int c=threadIdx.x;c<320;c+=256){
  uint4 v=*reinterpret_cast<const uint4*>(h+size_t(m)*stride+c*8);
  const __nv_bfloat162* p=reinterpret_cast<const __nv_bfloat162*>(&v);
  float4* dst=reinterpret_cast<float4*>(xs+c*8);
  float2 a=__bfloat1622float2(p[0]),b=__bfloat1622float2(p[1]),d=__bfloat1622float2(p[2]),e=__bfloat1622float2(p[3]);
  dst[0]=make_float4(a.x,a.y,b.x,b.y);dst[1]=make_float4(d.x,d.y,e.x,e.y);
 }
 __syncthreads();
 int ci=c0+(threadIdx.x>>5),lane=threadIdx.x&31;
 if(ci>=min(counts[m],cap))return;
 int row=ids[m*cap+ci];
 uint4 weights[5];uint16_t scale[5];
 #pragma unroll
 for(int c=0;c<5;c++){
  int k=c*512+lane*16;
  weights[c]=__ldcs(reinterpret_cast<const uint4*>(q+size_t(row)*2560+k));
  scale[c]=__ldg(s+row*20+k/128);
 }
 float acc=0;
 #pragma unroll
 for(int c=0;c<5;c++){
  int k=c*512+lane*16;float wf[16];const uint32_t* w=reinterpret_cast<const uint32_t*>(&weights[c]);
  #pragma unroll
  for(int j=0;j<4;j++)i8x4_to_f32(w[j],wf+j*4);
  float p=0;
  #pragma unroll
  for(int j=0;j<4;j++){
   float4 v=x4<false>(xs,k+j*4);
   p=fmaf(wf[j*4],v.x,p);p=fmaf(wf[j*4+1],v.y,p);p=fmaf(wf[j*4+2],v.z,p);p=fmaf(wf[j*4+3],v.w,p);
  }
  acc=fmaf(p,bf16_bits_to_f32(scale[c]),acc);
 }
 #pragma unroll
 for(int o=16;o;o>>=1)acc+=__shfl_xor_sync(0xffffffff,acc,o);
 if(!lane)out[size_t(m)*N+row]=__float2bfloat16(acc);
}
// 4 int8 -> two bf16x2 words (bytes 0,1 -> lo; 2,3 -> hi): float(q) is exact and has <= 7 mantissa bits,
// so its upper 16 bits are the exact bf16
__device__ __forceinline__ void i8x4_to_bf16x2(uint32_t u, uint32_t & lo, uint32_t & hi) {
    u ^= 0x80808080u;
    const float f0 = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7440)) - 8388736.f;
    const float f1 = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7441)) - 8388736.f;
    const float f2 = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7442)) - 8388736.f;
    const float f3 = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7443)) - 8388736.f;
    lo = __byte_perm(__float_as_uint(f0), __float_as_uint(f1), 0x7632);
    hi = __byte_perm(__float_as_uint(f2), __float_as_uint(f3), 0x7632);
}

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
template <int NB, int NW>
__global__ void __launch_bounds__(NW * 32) head_dense_mma_kernel(
        const __nv_bfloat16 * __restrict__ x, const int8_t * __restrict__ q, const uint16_t * __restrict__ s,
        __nv_bfloat16 * __restrict__ y, int N, int M, int ldx, int ldy, const int* dense, const int* overflow) {
    if (!*dense && !*overflow) return;
    constexpr int K = NW * NB * 64, NG = K / 128;
    __shared__ float red[NW][16][9];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    for (int row0=blockIdx.x*16;row0<N;row0+=gridDim.x*16) {
    const size_t ra = (size_t) min(row0 + g, N - 1), rb = (size_t) min(row0 + g + 8, N - 1);
    const int kw = warp * NB * 64 + t * 16;
    uint4 wa[NB], wb[NB];
    uint16_t sa[NB], sb[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        wa[b] = __ldcs(reinterpret_cast<const uint4 *>(q + ra * K + kw + b * 64));
        wb[b] = __ldcs(reinterpret_cast<const uint4 *>(q + rb * K + kw + b * 64));
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
        xv[b][0] = tok ? __ldcg(xp) : make_uint4(0, 0, 0, 0);
        xv[b][1] = tok ? __ldcg(xp + 1) : make_uint4(0, 0, 0, 0);
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t a0, a1, a2, a3;
            i8x4_to_bf16x2(u4w(wa[b], j), a0, a2);
            i8x4_to_bf16x2(u4w(wb[b], j), a1, a3);
            const uint4 & xh = xv[b][j >> 1];
            mma_bf16_16816(d, a0, a1, a2, a3, u4w(xh, (j & 1) * 2), u4w(xh, (j & 1) * 2 + 1));
        }
        const float sca = bf16_bits_to_f32(sa[b]), scb = bf16_bits_to_f32(sb[b]);
        acc[0] = fmaf(d[0], sca, acc[0]);
        acc[1] = fmaf(d[1], sca, acc[1]);
        acc[2] = fmaf(d[2], scb, acc[2]);
        acc[3] = fmaf(d[3], scb, acc[3]);
    }
    // d layout: [0] (row g, token 2t) [1] (g, 2t + 1) [2] (g + 8, 2t) [3] (g + 8, 2t + 1)
    red[warp][g][2 * t] = acc[0];
    red[warp][g][2 * t + 1] = acc[1];
    red[warp][g + 8][2 * t] = acc[2];
    red[warp][g + 8][2 * t + 1] = acc[3];
    __syncthreads();
    if (threadIdx.x < 16 * M) {
        const int r = threadIdx.x & 15, m = threadIdx.x >> 4;
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) v += red[w][r][m];
        if (row0 + r < N) y[(size_t) m * ldy + row0 + r] = __float2bfloat16(v);
    }
    __syncthreads();
}
}
template <int NB, int NW>
__global__ void __launch_bounds__(NW * 32) head_rescore_mma_kernel(
        const __nv_bfloat16 * __restrict__ x, const int8_t * __restrict__ q, const uint16_t * __restrict__ s,
        __nv_bfloat16 * __restrict__ y, int N, int M, int ldx, int ldy, const int* ids, const int* counts, int cap, const int* dense, const int* overflow) {
    const int m=blockIdx.y, count=min(counts[m],cap);
    if (*dense || *overflow || blockIdx.x*16>=count) return;
    x += (size_t)m*ldx;
    constexpr int K = NW * NB * 64, NG = K / 128;
    __shared__ float red[NW][16][9];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * 16;
    const size_t ra = ids[m*cap+min(row0+g,count-1)], rb = ids[m*cap+min(row0+g+8,count-1)];
    const int kw = warp * NB * 64 + t * 16;
    uint4 wa[NB], wb[NB];
    uint16_t sa[NB], sb[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        wa[b] = __ldcs(reinterpret_cast<const uint4 *>(q + ra * K + kw + b * 64));
        wb[b] = __ldcs(reinterpret_cast<const uint4 *>(q + rb * K + kw + b * 64));
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        const int grp = (warp * NB * 64 + b * 64) / 128;
        sa[b] = __ldg(s + ra * NG + grp);
        sb[b] = __ldg(s + rb * NG + grp);
    }
    uint4 xv[NB][2];
    const bool tok = g == 0;
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        const uint4 * xp = reinterpret_cast<const uint4 *>(x + (size_t) (tok ? g : 0) * ldx + kw + b * 64);
        xv[b][0] = tok ? __ldcg(xp) : make_uint4(0, 0, 0, 0);
        xv[b][1] = tok ? __ldcg(xp + 1) : make_uint4(0, 0, 0, 0);
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t a0, a1, a2, a3;
            i8x4_to_bf16x2(u4w(wa[b], j), a0, a2);
            i8x4_to_bf16x2(u4w(wb[b], j), a1, a3);
            const uint4 & xh = xv[b][j >> 1];
            mma_bf16_16816(d, a0, a1, a2, a3, u4w(xh, (j & 1) * 2), u4w(xh, (j & 1) * 2 + 1));
        }
        const float sca = bf16_bits_to_f32(sa[b]), scb = bf16_bits_to_f32(sb[b]);
        acc[0] = fmaf(d[0], sca, acc[0]);
        acc[1] = fmaf(d[1], sca, acc[1]);
        acc[2] = fmaf(d[2], scb, acc[2]);
        acc[3] = fmaf(d[3], scb, acc[3]);
    }
    // d layout: [0] (row g, token 2t) [1] (g, 2t + 1) [2] (g + 8, 2t) [3] (g + 8, 2t + 1)
    red[warp][g][2 * t] = acc[0];
    red[warp][g][2 * t + 1] = acc[1];
    red[warp][g + 8][2 * t] = acc[2];
    red[warp][g + 8][2 * t + 1] = acc[3];
    __syncthreads();
    if (threadIdx.x < 16) {
        const int r = threadIdx.x;
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) v += red[w][r][0];
        if (row0+r<count) y[(size_t)m*ldy+ids[m*cap+row0+r]]=__float2bfloat16(v);
    }
}

} // namespace

bool head_use_mma(){ static const bool on=[] { const char* e=getenv("QWENFAST_W8M");return e&&atoi(e)!=0;}();return on;}
void head_rescore(const at::Tensor& h,const at::Tensor& q,const at::Tensor& s,const at::Tensor& ids,
 const at::Tensor& counts,const at::Tensor& overflow,const at::Tensor& dense,at::Tensor& out){
 if(head_use_mma()){
  head_rescore_mma_kernel<5,8><<<dim3((ids.size(1)+15)/16,h.size(0)),256,0,at::cuda::getCurrentCUDAStream()>>>(
   reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),q.data_ptr<int8_t>(),reinterpret_cast<const uint16_t*>(s.data_ptr()),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
   q.size(0),h.size(0),h.stride(0),q.size(0),ids.data_ptr<int>(),counts.data_ptr<int>(),ids.size(1),dense.data_ptr<int>(),overflow.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return;
 }
 head_rescore_kernel<<<dim3((ids.size(1)+7)/8,h.size(0)),256,0,at::cuda::getCurrentCUDAStream()>>>(
 reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),q.data_ptr<int8_t>(),reinterpret_cast<const uint16_t*>(s.data_ptr()),ids.data_ptr<int>(),
 counts.data_ptr<int>(),overflow.data_ptr<int>(),dense.data_ptr<int>(),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),q.size(0),h.stride(0),ids.size(1));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void head_dense_fallback(const at::Tensor& h,const at::Tensor& q,const at::Tensor& s,at::Tensor& out,
 const at::Tensor& dense,const at::Tensor& overflow){
 if(head_use_mma()){
  static const int blocks=[] {const char* e=getenv("QWENFAST_HEAD_DENSE_BLOCKS");return std::max(1,std::min(15520,e?atoi(e):3008));}();
  head_dense_mma_kernel<5,8><<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(
   reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),q.data_ptr<int8_t>(),reinterpret_cast<const uint16_t*>(s.data_ptr()),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
   q.size(0),h.size(0),h.stride(0),q.size(0),dense.data_ptr<int>(),overflow.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return;
 }
 #define HDCASE(M) case M:head_dense_launch<M>(h,q,s,out,dense,overflow);break;
 switch(h.size(0)){HDCASE(1) HDCASE(2) HDCASE(3) HDCASE(4) HDCASE(5) HDCASE(6) HDCASE(7) HDCASE(8) default:TORCH_CHECK(false,"head M must be 1..8");}
 #undef HDCASE
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Standalone tuning hook. Production uses head_dense_fallback above.
void head_dense_grid(const at::Tensor& h,const at::Tensor& q,const at::Tensor& s,at::Tensor& out,
 const at::Tensor& dense,const at::Tensor& overflow,int64_t blocks){
 TORCH_CHECK(blocks>=1&&blocks<=15520);
 head_dense_mma_kernel<5,8><<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(
  reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),q.data_ptr<int8_t>(),reinterpret_cast<const uint16_t*>(s.data_ptr()),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
  q.size(0),h.size(0),h.stride(0),q.size(0),dense.data_ptr<int>(),overflow.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
