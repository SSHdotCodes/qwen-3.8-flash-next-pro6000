// Group-scaled INT8 weights ("w8g") for the Qwen3.8-Flash-Next target's dense BF16 linears.
//
//   q[n][k]       int8, symmetric, |q| <= 127
//   s[n][k / G]   bf16 scale of group k / G (G = 128; 64 when K % 128 != 0), w ~= q * s
// The scale is rounded to BF16 *before* quantizing, so q * s is the exact dequantized value.
//
// w8g_gemm:   y[M, N] = x[M, K] @ (q * s)^T for 1..8 rows (decode / verify), fp32 accumulation: each lane sums
//             q * x over its 16 (or 4) weights of one group, then scales the partial once.
// w8g_unpack: BF16 copy of rows (bf16(q * s)) for the large-M path, which then runs the stock cuBLAS GEMM.
// w8m (QWENFAST_W8M=1, K = 2560 / 6144): the same product on tensor cores. A CTA owns 16 weight rows; each warp
// one K slice. Weights go to mma.m16n8k16 as A (int8 -> bf16 is exact: |q| <= 127), the tokens as B (n = 8, rows
// >= M zero), fp32 accumulation per 64-weight block, then * group scale. x is read straight from L2 once per CTA
// instead of once per weight row from shared memory.
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
__global__ void __launch_bounds__(WARPS * 32) w8g_gemm_kernel(
        const __nv_bfloat16 * __restrict__ x, const int8_t * __restrict__ q, const uint16_t * __restrict__ s,
        __nv_bfloat16 * __restrict__ y, int N, int ldx, int ldy) {
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
    const int row0 = (blockIdx.x * WARPS + warp) * RPW;
    if (row0 >= N) return;
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

template <int M, int RPW, int VEC, int KC, int G, bool XB>
void launch_xb(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, at::Tensor & y, cudaStream_t st) {
    const int N = q.size(0);
    const int grid = (N + WARPS * RPW - 1) / (WARPS * RPW);
    const int smem = M * KC * 32 * VEC * (XB ? 2 : 4);
    auto kern = w8g_gemm_kernel<M, RPW, VEC, KC, G, XB>;
    if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kern<<<grid, WARPS * 32, smem, st>>>(reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()),
        reinterpret_cast<const int8_t *>(q.data_ptr()), reinterpret_cast<const uint16_t *>(s.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), N, (int) x.stride(0), (int) y.stride(0));
}

// fp32 x staging while it stays <= 48 KB (QWENFAST_W8G_XB_KB), bf16 above
template <int M, int RPW, int VEC, int KC, int G>
void launch(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, at::Tensor & y, cudaStream_t st) {
    static const int xb_limit = [] { const char * e = getenv("QWENFAST_W8G_XB_KB"); return (e ? atoi(e) : 48) * 1024; }();
    if (M * KC * 32 * VEC * 4 > xb_limit) launch_xb<M, RPW, VEC, KC, G, true>(x, q, s, y, st);
    else launch_xb<M, RPW, VEC, KC, G, false>(x, q, s, y, st);
}

template <int M, int RPW>
void by_k(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, at::Tensor & y, cudaStream_t st) {
    switch (q.size(1)) {
        case 640: launch<M, RPW, 4, 5, 128>(x, q, s, y, st); break;
        case 2560: launch<M, RPW, 16, 5, 128>(x, q, s, y, st); break;
        case 6144: launch<M, RPW, 16, 12, 128>(x, q, s, y, st); break;
        default: TORCH_CHECK(false, "w8g_gemm: unsupported K ", q.size(1));
    }
}

template <int M>
void by_rpw(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, at::Tensor & y, int rpw, cudaStream_t st) {
    if (rpw == 1) by_k<M, 1>(x, q, s, y, st); else if (rpw == 2) by_k<M, 2>(x, q, s, y, st); else by_k<M, 4>(x, q, s, y, st);
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
__global__ void __launch_bounds__(NW * 32) w8m_kernel(
        const __nv_bfloat16 * __restrict__ x, const int8_t * __restrict__ q, const uint16_t * __restrict__ s,
        __nv_bfloat16 * __restrict__ y, int N, int M, int ldx, int ldy) {
    constexpr int K = NW * NB * 64, NG = K / 128;
    __shared__ float red[NW][16][9];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * 16;
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
    asm volatile("griddepcontrol.wait;" ::: "memory");
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
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
}

template <int NB, int NW>
void launch_w8m(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, at::Tensor & y, cudaStream_t st) {
    static const bool pdl = [] { const char * e = getenv("QWENFAST_W8M_PDL"); return e && atoi(e) != 0; }();
    const int N = q.size(0);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((N + 15) / 16); cfg.blockDim = dim3(NW * 32); cfg.dynamicSmemBytes = 0; cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr; cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, w8m_kernel<NB, NW>, reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()),
        reinterpret_cast<const int8_t *>(q.data_ptr()), reinterpret_cast<const uint16_t *>(s.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), N, (int) x.size(0), (int) x.stride(0), (int) y.stride(0)));
}

__global__ void w8g_unpack_kernel(const int8_t * __restrict__ q, const uint16_t * __restrict__ s,
                                  __nv_bfloat16 * __restrict__ w, long total8, int K, int G) {
    const long i = (long) blockIdx.x * blockDim.x + threadIdx.x;   // 8 weights per thread
    if (i >= total8) return;
    const long e = i * 8;
    const long row = e / K;
    const int k = (int) (e % K);
    const float sc = bf16_bits_to_f32(s[row * (K / G) + k / G]);
    const uint2 v = *reinterpret_cast<const uint2 *>(q + e);
    float f[8];
    i8x4_to_f32(v.x, f);
    i8x4_to_f32(v.y, f + 4);
    uint4 o;
    __nv_bfloat162 * op = reinterpret_cast<__nv_bfloat162 *>(&o);
#pragma unroll
    for (int j = 0; j < 4; ++j) op[j] = __floats2bfloat162_rn(f[2 * j] * sc, f[2 * j + 1] * sc);
    *reinterpret_cast<uint4 *>(w + e) = o;
}

}  // namespace

// x [M, K] bf16 (1 <= M <= 8, row stride % 8 == 0); q [N, K] int8; s [N, K / 128] bf16. Returns y [M, N] bf16.
at::Tensor w8g_gemm(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, int64_t rpw) {
    TORCH_CHECK(x.is_cuda() && q.is_cuda() && s.is_cuda() && x.scalar_type() == at::kBFloat16);
    TORCH_CHECK(q.scalar_type() == at::kChar && s.scalar_type() == at::kBFloat16);
    TORCH_CHECK(x.dim() == 2 && q.dim() == 2 && x.size(1) == q.size(1) && q.is_contiguous() && s.is_contiguous());
    TORCH_CHECK(s.dim() == 2 && s.size(0) == q.size(0) && s.size(1) * 128 == q.size(1));
    TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8);
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({x.size(0), q.size(0)}, x.options());
    if (rpw <= 0) rpw = x.size(0) == 1 ? 1 : 4;
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    // rpw 0x100 forces the tensor-core kernel, 0x200 forbids it (tests); otherwise QWENFAST_W8M decides
    static const bool w8m_on = [] { const char * e = getenv("QWENFAST_W8M"); return e && atoi(e) != 0; }();
    const bool use_m = (rpw & 0x100) || (w8m_on && !(rpw & 0x200));
    rpw &= 0xff;
    if (use_m && (q.size(1) == 2560 || q.size(1) == 6144)) {
        if (q.size(1) == 2560) launch_w8m<5, 8>(x, q, s, y, st);
        else launch_w8m<6, 16>(x, q, s, y, st);
        return y;
    }
    if (rpw <= 0) rpw = x.size(0) == 1 ? 1 : 4;
    switch (x.size(0)) {
        case 1: by_rpw<1>(x, q, s, y, (int) rpw, st); break;
        case 2: by_rpw<2>(x, q, s, y, (int) rpw, st); break;
        case 3: by_rpw<3>(x, q, s, y, (int) rpw, st); break;
        case 4: by_rpw<4>(x, q, s, y, (int) rpw, st); break;
        case 5: by_rpw<5>(x, q, s, y, (int) rpw, st); break;
        case 6: by_rpw<6>(x, q, s, y, (int) rpw, st); break;
        case 7: by_rpw<7>(x, q, s, y, (int) rpw, st); break;
        default: by_rpw<8>(x, q, s, y, (int) rpw, st); break;
    }
    return y;
}

// w [rows, K] bf16 <- bf16(q * s) for the given rows (q, s already sliced to them)
void w8g_unpack(const at::Tensor & q, const at::Tensor & s, at::Tensor & w) {
    TORCH_CHECK(q.scalar_type() == at::kChar && s.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
    TORCH_CHECK(q.dim() == 2 && q.is_contiguous() && s.is_contiguous() && w.is_contiguous() && w.sizes() == q.sizes());
    const int K = q.size(1);
    const int G = K / s.size(1);
    TORCH_CHECK(G * s.size(1) == K && G % 8 == 0 && K % 8 == 0);
    const c10::cuda::CUDAGuard guard(q.device());
    const long total8 = q.numel() / 8;
    if (total8 == 0) return;
    w8g_unpack_kernel<<<(unsigned) ((total8 + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const int8_t *>(q.data_ptr()), reinterpret_cast<const uint16_t *>(s.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(w.data_ptr()), total8, K, G);
}
