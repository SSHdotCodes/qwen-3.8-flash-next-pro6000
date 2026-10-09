// Exact 12-bit packing of BF16 weights ("pk12") and its small-M GEMM.
//
// A BF16 weight row is stored as
//   m8[n][k]   uint8: sign (bit 7) + 7 mantissa bits                      (bf16 bits 15, 6..0)
//   e4[n][k/2] uint8: 4-bit exponent offset nib = exp - base[n] in 1..15, low nibble = even k
//   base[n]    uint8: max exponent of the row - 15
// Values with exponent <= base[n] (|w| < 2^-14 of the row max, ~1-3 per 10^4 weights) are stored with
// nib = 0 and m8 = 0, which decodes to d = +2^(base-127); their true values are kept in a per-row
// exception list (CSR) and the GEMM adds (w - d) * x after the main loop. The decoded weights are bit-identical to the BF16
// originals; only the fp32 accumulation order differs from cuBLAS (as with smallm_gemm).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int WARPS = 8;

// 8 weights: m8 word pair (8 bytes) + one e4 word (8 nibbles) -> 8 floats (exact bf16 values as fp32).
__device__ __forceinline__ void decode8(uint32_t m0, uint32_t m1, uint32_t q, uint32_t base2, float * f) {
    const uint32_t qe = q & 0x0F0F0F0Fu, qo = (q >> 4) & 0x0F0F0F0Fu;   // nibbles of even / odd weights
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t m = p < 2 ? m0 : m1;
        const int j = p & 1;
        // bytes 2j, 2j+1 of m -> 16-bit lanes: 0x00bb per lane
        const uint32_t u = __byte_perm(m, 0, j == 0 ? 0x4140 : 0x4342);
        // exponent nibbles of weights 2p (even, byte p of qe) and 2p+1 (odd, byte p of qo) -> lanes
        const uint32_t e = __byte_perm(qe, qo, 0x4040 | p | ((4 + p) << 8)) & 0x000F000Fu;
        const uint32_t ex = (e + base2) << 7;                               // exponent field per lane
        const uint32_t w = ((u & 0x00800080u) << 8) | (u & 0x007F007Fu) | ex; // bf16x2
        f[2 * p] = __uint_as_float(w << 16);
        f[2 * p + 1] = __uint_as_float(w & 0xFFFF0000u);
    }
}

// grid: one warp per RPW rows; each lane handles 16 consecutive weights per 512-wide K chunk.
// x row m, elements k0..k0+3 as floats, from fp32 or bf16 staging
template <bool XB>
__device__ __forceinline__ float4 x4(const void * xs, int idx) {
    if constexpr (XB) {
        const uint2 v = *reinterpret_cast<const uint2 *>(reinterpret_cast<const __nv_bfloat16 *>(xs) + idx);
        return make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xFFFF0000u),
                           __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xFFFF0000u));
    } else {
        return *reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(xs) + idx);
    }
}

template <bool XB>
__device__ __forceinline__ float x1(const void * xs, int idx) {
    if constexpr (XB) return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(xs)[idx]);
    else return reinterpret_cast<const float *>(xs)[idx];
}

template <int M, int RPW, int KC, bool XB>
__global__ void __launch_bounds__(WARPS * 32) pk_gemm_kernel(
        const __nv_bfloat16 * __restrict__ x, const uint8_t * __restrict__ m8, const uint8_t * __restrict__ e4,
        const uint8_t * __restrict__ base, const int * __restrict__ ex_ptr, const int * __restrict__ ex_k,
        const float * __restrict__ ex_v, __nv_bfloat16 * __restrict__ y, int N, int K, int ldx, int ldy) {
    extern __shared__ __align__(16) float xs_raw[];   // [M][K] fp32, or bf16 when XB
    void * xs = xs_raw;
    for (int c = threadIdx.x; c < M * (K / 8); c += blockDim.x) {
        const int m = c / (K / 8), kc = c % (K / 8);
        const uint4 v = *reinterpret_cast<const uint4 *>(x + (size_t) m * ldx + kc * 8);
        if constexpr (XB) {
            reinterpret_cast<uint4 *>(xs_raw)[c] = v;
        } else {
            const __nv_bfloat162 * h = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 t = __bfloat1622float2(h[i]);
                xs_raw[m * K + kc * 8 + 2 * i] = t.x; xs_raw[m * K + kc * 8 + 2 * i + 1] = t.y;
            }
        }
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.x * WARPS + warp) * RPW;
    if (row0 >= N) return;
    float acc[RPW][M];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    uint32_t b2[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const uint32_t b = base[min(row0 + r, N - 1)];
        b2[r] = b | (b << 16);
    }
    constexpr int nck = KC;
    constexpr int U = KC <= 6 ? KC : 6;   // 512-wide chunks in flight per row
#pragma unroll 1
    for (int c0 = 0; c0 < nck; c0 += U) {
        uint4 mv[RPW][U];
        uint2 ev[RPW][U];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const size_t row = min(row0 + r, N - 1);
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (c0 + u < nck) {
                    const int k0 = (c0 + u) * 512 + lane * 16;
                    mv[r][u] = __ldcs(reinterpret_cast<const uint4 *>(m8 + row * K + k0));
                    ev[r][u] = __ldcs(reinterpret_cast<const uint2 *>(e4 + row * (K / 2) + k0 / 2));
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (c0 + u >= nck) break;
            const int k0 = (c0 + u) * 512 + lane * 16;
#pragma unroll
            for (int r = 0; r < RPW; ++r) {
                float wf[16];
                decode8(mv[r][u].x, mv[r][u].y, ev[r][u].x, b2[r], wf);
                decode8(mv[r][u].z, mv[r][u].w, ev[r][u].y, b2[r], wf + 8);
#pragma unroll
                for (int m = 0; m < M; ++m) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float4 xv = x4<XB>(xs, m * K + k0 + 4 * i);
                        acc[r][m] = fmaf(wf[4 * i], xv.x, acc[r][m]);
                        acc[r][m] = fmaf(wf[4 * i + 1], xv.y, acc[r][m]);
                        acc[r][m] = fmaf(wf[4 * i + 2], xv.z, acc[r][m]);
                        acc[r][m] = fmaf(wf[4 * i + 3], xv.w, acc[r][m]);
                    }
                }
            }
        }
    }
    // exceptions: lanes split the row's list
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = row0 + r;
        if (row >= N) break;
        const int a = ex_ptr[row], b = ex_ptr[row + 1];
        const float d = __uint_as_float(((uint32_t) base[row]) << 23);
        for (int i = a + lane; i < b; i += 32) {
            const int k = ex_k[i];
            const float v = ex_v[i] - d;
#pragma unroll
            for (int m = 0; m < M; ++m) acc[r][m] = fmaf(v, x1<XB>(xs, m * K + k), acc[r][m]);
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

template <int M, int RPW, int KC, bool XB>
void launch_xb(const at::Tensor & x, const at::Tensor & m8, const at::Tensor & e4, const at::Tensor & base,
               const at::Tensor & ex_ptr, const at::Tensor & ex_k, const at::Tensor & ex_v, at::Tensor & y, cudaStream_t st) {
    const int N = m8.size(0), K = m8.size(1);
    const int grid = (N + WARPS * RPW - 1) / (WARPS * RPW);
    const size_t smem = (size_t) M * K * (XB ? 2 : 4);
    auto kern = pk_gemm_kernel<M, RPW, KC, XB>;
    if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    kern<<<grid, WARPS * 32, smem, st>>>(reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), m8.data_ptr<uint8_t>(),
        e4.data_ptr<uint8_t>(), base.data_ptr<uint8_t>(), ex_ptr.data_ptr<int>(), ex_k.data_ptr<int>(), ex_v.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), N, K, (int) x.stride(0), (int) y.stride(0));
}

template <int M, int RPW, int KC>
void launch(const at::Tensor & x, const at::Tensor & m8, const at::Tensor & e4, const at::Tensor & base,
            const at::Tensor & ex_ptr, const at::Tensor & ex_k, const at::Tensor & ex_v, at::Tensor & y, cudaStream_t st) {
    const int N = m8.size(0), K = m8.size(1);
    const int grid = (N + WARPS * RPW - 1) / (WARPS * RPW);
    // fp32 x staging while it stays <= the threshold (default 48 KB: two CTAs per SM), bf16 above
    static const int xb_limit = [] { const char * e = getenv("QWENFAST_PK_XB_KB"); return (e ? atoi(e) : 48) * 1024; }();
    if ((size_t) M * KC * 512 * 4 > (size_t) xb_limit) { launch_xb<M, RPW, KC, true>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, st); return; }
    constexpr bool XB = false;
    const size_t smem = (size_t) M * K * (XB ? 2 : 4);
    auto kern = pk_gemm_kernel<M, RPW, KC, XB>;
    if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    kern<<<grid, WARPS * 32, smem, st>>>(reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), m8.data_ptr<uint8_t>(),
        e4.data_ptr<uint8_t>(), base.data_ptr<uint8_t>(), ex_ptr.data_ptr<int>(), ex_k.data_ptr<int>(), ex_v.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), N, K, (int) x.stride(0), (int) y.stride(0));
}

template <int M, int KC>
void dispatch_k(const at::Tensor & x, const at::Tensor & m8, const at::Tensor & e4, const at::Tensor & base,
                const at::Tensor & ex_ptr, const at::Tensor & ex_k, const at::Tensor & ex_v, at::Tensor & y, int rpw, cudaStream_t st) {
    if (rpw == 1) launch<M, 1, KC>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, st);
    else if (rpw == 2) launch<M, 2, KC>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, st);
    else launch<M, 4, KC>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, st);
}

template <int M>
void dispatch(const at::Tensor & x, const at::Tensor & m8, const at::Tensor & e4, const at::Tensor & base,
              const at::Tensor & ex_ptr, const at::Tensor & ex_k, const at::Tensor & ex_v, at::Tensor & y, int rpw, cudaStream_t st) {
    const int K = m8.size(1);
    if (K == 2560) dispatch_k<M, 5>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, rpw, st);
    else if (K == 6144) dispatch_k<M, 12>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, rpw, st);
    else TORCH_CHECK(false, "pk_gemm: unsupported K ", K);
}

// decompress to bf16 (prefill path): one thread per 16 weights
__global__ void pk_unpack_kernel(const uint8_t * __restrict__ m8, const uint8_t * __restrict__ e4, const uint8_t * __restrict__ base,
                                 const int * __restrict__ ex_ptr, const int * __restrict__ ex_k, const float * __restrict__ ex_v,
                                 __nv_bfloat16 * __restrict__ w, int N, int K) {
    const size_t g = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) N * K / 16;
    if (g >= total) return;
    const int row = (int) (g * 16 / K);
    const uint32_t b = base[row];
    const uint4 mv = __ldcs(reinterpret_cast<const uint4 *>(m8) + g);
    const uint2 ev = __ldcs(reinterpret_cast<const uint2 *>(e4) + g);
    float f[16];
    decode8(mv.x, mv.y, ev.x, b | (b << 16), f);
    decode8(mv.z, mv.w, ev.y, b | (b << 16), f + 8);
    uint4 o[2];
    uint32_t * ow = reinterpret_cast<uint32_t *>(o);
#pragma unroll
    for (int i = 0; i < 8; ++i) ow[i] = (__float_as_uint(f[2 * i]) >> 16) | (__float_as_uint(f[2 * i + 1]) & 0xFFFF0000u);
    reinterpret_cast<uint4 *>(w)[2 * g] = o[0];
    reinterpret_cast<uint4 *>(w)[2 * g + 1] = o[1];
}

__global__ void pk_unpack_fix_kernel(const uint8_t * __restrict__ base, const int * __restrict__ ex_ptr, const int * __restrict__ ex_k,
                                     const float * __restrict__ ex_v, __nv_bfloat16 * __restrict__ w, int N, int K) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    for (int i = ex_ptr[row]; i < ex_ptr[row + 1]; ++i)
        w[(size_t) row * K + ex_k[i]] = __float2bfloat16(ex_v[i]);   // exact: ex_v holds the bf16 value
}

}  // namespace

at::Tensor pk_gemm(const at::Tensor & x, const at::Tensor & m8, const at::Tensor & e4, const at::Tensor & base,
                   const at::Tensor & ex_ptr, const at::Tensor & ex_k, const at::Tensor & ex_v, int64_t rpw) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1);
    TORCH_CHECK(m8.scalar_type() == at::kByte && e4.scalar_type() == at::kByte && base.scalar_type() == at::kByte);
    TORCH_CHECK(m8.is_contiguous() && e4.is_contiguous() && x.size(1) == m8.size(1) && m8.size(1) % 512 == 0);
    TORCH_CHECK(x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
    const int M = x.size(0);
    TORCH_CHECK(M >= 1 && M <= 8 && (size_t) M * m8.size(1) * 2 <= 99 * 1024, "pk_gemm: x does not fit in shared memory");
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({M, m8.size(0)}, x.options());
    if (rpw <= 0) rpw = 2;
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    switch (M) {
        case 1: dispatch<1>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, rpw, st); break;
        case 2: dispatch<2>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, rpw, st); break;
        case 3: dispatch<3>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, rpw, st); break;
        case 4: dispatch<4>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, rpw, st); break;
        case 5: dispatch<5>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, rpw, st); break;
        case 6: dispatch<6>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, rpw, st); break;
        case 7: dispatch<7>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, rpw, st); break;
        default: dispatch<8>(x, m8, e4, base, ex_ptr, ex_k, ex_v, y, rpw, st); break;
    }
    return y;
}

void pk_unpack(const at::Tensor & m8, const at::Tensor & e4, const at::Tensor & base, const at::Tensor & ex_ptr,
               const at::Tensor & ex_k, const at::Tensor & ex_v, at::Tensor & w) {
    TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.numel() == m8.numel() && m8.size(1) % 16 == 0);
    const c10::cuda::CUDAGuard guard(m8.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const int N = m8.size(0), K = m8.size(1);
    const size_t total = (size_t) N * K / 16;
    pk_unpack_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, st>>>(m8.data_ptr<uint8_t>(), e4.data_ptr<uint8_t>(), base.data_ptr<uint8_t>(),
        ex_ptr.data_ptr<int>(), ex_k.data_ptr<int>(), ex_v.data_ptr<float>(), reinterpret_cast<__nv_bfloat16 *>(w.data_ptr()), N, K);
    pk_unpack_fix_kernel<<<(N + 255) / 256, 256, 0, st>>>(base.data_ptr<uint8_t>(), ex_ptr.data_ptr<int>(), ex_k.data_ptr<int>(),
        ex_v.data_ptr<float>(), reinterpret_cast<__nv_bfloat16 *>(w.data_ptr()), N, K);
}
