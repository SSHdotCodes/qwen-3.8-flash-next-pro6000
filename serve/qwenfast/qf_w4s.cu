// Draft hot-head screening (MTP draft only): an INT4 copy of the 97,800-row hot head finds each row's candidate
// tokens, then their exact INT8 logits are recomputed, so the truncated draft proposal (top-k / top-p of the
// draft logits) is built from the same INT8 logits as before whenever the INT8 top-k lies inside the INT4
// candidates. Draft numerics only change acceptance under exact rejection sampling, never the output distribution.
//
// w4g_gemm:    y[M, N] (bf16) = x[M, K] @ (q4 * s4)^T, q4 = 4-bit (stored +8, low nibble = even k), s4 bf16 per 64.
// rows_dot_i8: out[m, c] = (sum_k q8[idx[m, c], k] * x[m, k]) * s8[idx[m, c]]   (one warp per (m, c))
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int WARPS = 8, K = 2560, G = 64, KC = K / 512;

// 8 bytes = 16 nibbles -> 16 floats (q - 8), weight order k, k+1, ... (low nibble first)
__device__ __forceinline__ void nib16_to_f32(uint2 v, float * f) {
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        const uint32_t u = w ? v.y : v.x;
        const uint32_t lo = u & 0x0F0F0F0Fu, hi = (u >> 4) & 0x0F0F0F0Fu;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            f[w * 8 + 2 * b] = __int_as_float(__byte_perm(lo, 0x4B000000u, 0x7440 + b)) - 8388616.f;
            f[w * 8 + 2 * b + 1] = __int_as_float(__byte_perm(hi, 0x4B000000u, 0x7440 + b)) - 8388616.f;
        }
    }
}

template <int M, int RPW>
__global__ void __launch_bounds__(WARPS * 32) w4g_kernel(const __nv_bfloat16 * __restrict__ x, const uint8_t * __restrict__ q,
                                                         const uint16_t * __restrict__ s, __nv_bfloat16 * __restrict__ y,
                                                         int N, int ldx, int ldy) {
    extern __shared__ __align__(16) float xs[];   // [M][K] fp32
    for (int c = threadIdx.x; c < M * (K / 8); c += blockDim.x) {
        const int m = c / (K / 8), kc = c % (K / 8);
        const uint4 v = *reinterpret_cast<const uint4 *>(x + (size_t) m * ldx + kc * 8);
        const __nv_bfloat162 * h = reinterpret_cast<const __nv_bfloat162 *>(&v);
        float4 * d = reinterpret_cast<float4 *>(xs + m * K + kc * 8);
        const float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
        const float2 c2 = __bfloat1622float2(h[2]), d2 = __bfloat1622float2(h[3]);
        d[0] = make_float4(a.x, a.y, b.x, b.y);
        d[1] = make_float4(c2.x, c2.y, d2.x, d2.y);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.x * WARPS + warp) * RPW;
    if (row0 >= N) return;
    uint2 wv[RPW][KC];
    uint16_t sv[RPW][KC];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const size_t row = min(row0 + r, N - 1);
#pragma unroll
        for (int c = 0; c < KC; ++c) {
            const int k0 = c * 512 + lane * 16;
            wv[r][c] = __ldcs(reinterpret_cast<const uint2 *>(q + row * (K / 2) + k0 / 2));
            sv[r][c] = __ldg(s + row * (K / G) + k0 / G);
        }
    }
    float acc[RPW][M];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
#pragma unroll
    for (int c = 0; c < KC; ++c) {
        const int k0 = c * 512 + lane * 16;
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            float wf[16];
            nib16_to_f32(wv[r][c], wf);
            const float sc = __uint_as_float(((uint32_t) sv[r][c]) << 16);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float p = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 xv = *reinterpret_cast<const float4 *>(xs + m * K + k0 + 4 * i);
                    p = fmaf(wf[4 * i], xv.x, p);
                    p = fmaf(wf[4 * i + 1], xv.y, p);
                    p = fmaf(wf[4 * i + 2], xv.z, p);
                    p = fmaf(wf[4 * i + 3], xv.w, p);
                }
                acc[r][m] = fmaf(p, sc, acc[r][m]);
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

template <int M>
void launch_w4(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, at::Tensor & y, cudaStream_t st) {
    constexpr int RPW = 2;
    const int N = q.size(0);
    const int grid = (N + WARPS * RPW - 1) / (WARPS * RPW);
    const int smem = M * K * 4;
    auto kern = w4g_kernel<M, RPW>;
    if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kern<<<grid, WARPS * 32, smem, st>>>(reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), q.data_ptr<uint8_t>(),
        reinterpret_cast<const uint16_t *>(s.data_ptr()), reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), N,
        (int) x.stride(0), (int) y.stride(0));
}

__global__ void rows_dot_i8_kernel(const __nv_bfloat16 * __restrict__ x, int ldx, const int8_t * __restrict__ q,
                                   const float * __restrict__ s, const int64_t * __restrict__ idx, int C,
                                   float * __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int item = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);   // m * C + c
    const int m = item / C;
    const int64_t row = idx[item];
    const int8_t * qr = q + row * K;
    const __nv_bfloat16 * xr = x + (size_t) m * ldx;
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < KC; ++c) {
        const int k0 = c * 512 + lane * 16;
        const uint4 w = __ldg(reinterpret_cast<const uint4 *>(qr + k0));
        const uint4 xa = __ldg(reinterpret_cast<const uint4 *>(xr + k0));
        const uint4 xb = __ldg(reinterpret_cast<const uint4 *>(xr + k0 + 8));
        const int8_t * wb = reinterpret_cast<const int8_t *>(&w);
        const __nv_bfloat16 * x0 = reinterpret_cast<const __nv_bfloat16 *>(&xa);
        const __nv_bfloat16 * x1 = reinterpret_cast<const __nv_bfloat16 *>(&xb);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc = fmaf((float) wb[i], __bfloat162float(x0[i]), acc);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc = fmaf((float) wb[8 + i], __bfloat162float(x1[i]), acc);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) out[item] = acc * s[row];
}

}  // namespace

// x [M, 2560] bf16 (M 1..8); q [N, 1280] uint8; s [N, 40] bf16. Returns y [M, N] bf16.
at::Tensor w4g_gemm(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == K);
    TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
    TORCH_CHECK(q.scalar_type() == at::kByte && q.dim() == 2 && q.size(1) == K / 2 && q.is_contiguous());
    TORCH_CHECK(s.scalar_type() == at::kBFloat16 && s.size(0) == q.size(0) && s.size(1) == K / G && s.is_contiguous());
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8);
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({x.size(0), q.size(0)}, x.options());
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    switch (x.size(0)) {
        case 1: launch_w4<1>(x, q, s, y, st); break;
        case 2: launch_w4<2>(x, q, s, y, st); break;
        case 3: launch_w4<3>(x, q, s, y, st); break;
        case 4: launch_w4<4>(x, q, s, y, st); break;
        default: launch_w4<8>(x, q, s, y, st); break;
    }
    return y;
}

// x [M, 2560] bf16; q [N, 2560] int8; s [N] fp32; idx [M, C] int64 rows of q. Returns [M, C] fp32.
at::Tensor rows_dot_i8(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, const at::Tensor & idx) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == K && x.stride(1) == 1 && x.stride(0) % 8 == 0);
    TORCH_CHECK(q.scalar_type() == at::kChar && q.size(1) == K && q.is_contiguous());
    TORCH_CHECK(s.scalar_type() == at::kFloat && s.numel() == q.size(0) && s.is_contiguous());
    TORCH_CHECK(idx.scalar_type() == at::kLong && idx.dim() == 2 && idx.size(0) == x.size(0) && idx.is_contiguous());
    const c10::cuda::CUDAGuard guard(x.device());
    const int M = x.size(0), C = idx.size(1);
    auto out = at::empty({M, C}, x.options().dtype(at::kFloat));
    const int items = M * C;
    if (items == 0) return out;
    TORCH_CHECK(items % 8 == 0, "rows_dot_i8: M * C must be a multiple of 8");
    rows_dot_i8_kernel<<<items / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), (int) x.stride(0), q.data_ptr<int8_t>(), s.data_ptr<float>(),
        idx.data_ptr<int64_t>(), C, out.data_ptr<float>());
    return out;
}
