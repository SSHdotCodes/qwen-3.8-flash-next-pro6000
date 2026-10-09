// Small-M 8-bit weight-only GEMM (W8A16) for the MTP draft: y[M,N] = x[M,K] @ (q[N,K] * s[N])^T, fp32 accumulation.
// q is int8 (symmetric per row) or FP8 e4m3 (SGLang's online per-channel FP8 of the draft), activations stay BF16.
// Draft-only: under exact rejection sampling the draft's numerics change acceptance, never the output distribution.
// 1 <= M <= 8; K % 128 == 0 (16-byte lanes when K % 512 == 0, else 4-byte lanes); x staged as BF16 in shared memory.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr int WARPS = 8;

// 4 packed 8-bit weights -> 4 floats
template <bool FP8>
__device__ __forceinline__ void w8x4_to_f32(uint32_t u, float * f) {
    if constexpr (FP8) {
        const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (u & 0xFFFF), __NV_E4M3);
        const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (u >> 16), __NV_E4M3);
        const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&lo));
        const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&hi));
        f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
    } else {
        u ^= 0x80808080u;   // exact: (2^23 + biased byte) - (2^23 + 128)
        f[0] = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7440)) - 8388736.f;
        f[1] = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7441)) - 8388736.f;
        f[2] = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7442)) - 8388736.f;
        f[3] = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7443)) - 8388736.f;
    }
}

__device__ __forceinline__ void bf16x4_to_f32(uint2 v, float * f) {
    f[0] = __uint_as_float(v.x << 16); f[1] = __uint_as_float(v.x & 0xFFFF0000u);
    f[2] = __uint_as_float(v.y << 16); f[3] = __uint_as_float(v.y & 0xFFFF0000u);
}

// VEC bytes (weights) per lane per chunk; a warp-chunk covers 32*VEC weights; KC chunks per row (all in flight).
template <bool FP8, int M, int RPW, int VEC, int KC>
__global__ void __launch_bounds__(WARPS * 32) w8_kernel(
        const __nv_bfloat16 * __restrict__ x, const uint8_t * __restrict__ q, const float * __restrict__ s, int s_stride,
        __nv_bfloat16 * __restrict__ y, int N, int ldx, int ldy) {
    constexpr int K = KC * 32 * VEC;
    extern __shared__ __align__(16) __nv_bfloat16 xs[];   // [M][K]
    for (int c = threadIdx.x; c < M * (K / 8); c += blockDim.x) {
        const int m = c / (K / 8), kc = c % (K / 8);
        reinterpret_cast<uint4 *>(xs)[c] = *reinterpret_cast<const uint4 *>(x + (size_t) m * ldx + kc * 8);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.x * WARPS + warp) * RPW;
    if (row0 >= N) return;
    using V = typename std::conditional<VEC == 16, uint4, uint32_t>::type;
    V wv[RPW][KC];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = min(row0 + r, N - 1);
#pragma unroll
        for (int c = 0; c < KC; ++c)
            wv[r][c] = __ldcs(reinterpret_cast<const V *>(q + (size_t) row * K + c * 32 * VEC + lane * VEC));
    }
    float acc[RPW][M];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
#pragma unroll
    for (int c = 0; c < KC; ++c) {
        const int k0 = c * 32 * VEC + lane * VEC;
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            float wf[VEC];
            const uint32_t * ww = reinterpret_cast<const uint32_t *>(&wv[r][c]);
#pragma unroll
            for (int i = 0; i < VEC / 4; ++i) w8x4_to_f32<FP8>(ww[i], wf + 4 * i);
#pragma unroll
            for (int m = 0; m < M; ++m) {
#pragma unroll
                for (int i = 0; i < VEC / 4; ++i) {
                    float xf[4];
                    bf16x4_to_f32(*reinterpret_cast<const uint2 *>(xs + m * K + k0 + 4 * i), xf);
                    acc[r][m] = fmaf(wf[4 * i], xf[0], acc[r][m]);
                    acc[r][m] = fmaf(wf[4 * i + 1], xf[1], acc[r][m]);
                    acc[r][m] = fmaf(wf[4 * i + 2], xf[2], acc[r][m]);
                    acc[r][m] = fmaf(wf[4 * i + 3], xf[3], acc[r][m]);
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
    if (lane < RPW && row0 + lane < N) {
        const float sc = s[(size_t) (row0 + lane) * s_stride];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float v = 0.f;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) if (rr == lane) v = acc[rr][m];
            y[(size_t) m * ldy + row0 + lane] = __float2bfloat16(v * sc);
        }
    }
}

template <bool FP8, int M, int RPW, int VEC, int KC>
void launch(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, at::Tensor & y, cudaStream_t st) {
    const int N = q.size(0);
    const int grid = (N + WARPS * RPW - 1) / (WARPS * RPW);
    const int smem = M * KC * 32 * VEC * 2;
    auto kern = w8_kernel<FP8, M, RPW, VEC, KC>;
    if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kern<<<grid, WARPS * 32, smem, st>>>(reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()),
        reinterpret_cast<const uint8_t *>(q.data_ptr()), s.data_ptr<float>(), s.numel() == 1 ? 0 : 1,
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), N, (int) x.stride(0), (int) y.stride(0));
}

template <bool FP8, int M, int RPW>
void by_k(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, at::Tensor & y, cudaStream_t st) {
    switch (q.size(1)) {
        case 640: launch<FP8, M, RPW, 4, 5>(x, q, s, y, st); break;
        case 2560: launch<FP8, M, RPW, 16, 5>(x, q, s, y, st); break;
        case 6144: launch<FP8, M, RPW, 16, 12>(x, q, s, y, st); break;
        default: TORCH_CHECK(false, "w8_gemm: unsupported K ", q.size(1));
    }
}

template <bool FP8, int M>
void by_rpw(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, at::Tensor & y, int rpw, cudaStream_t st) {
    if (rpw == 1) by_k<FP8, M, 1>(x, q, s, y, st); else if (rpw == 2) by_k<FP8, M, 2>(x, q, s, y, st); else by_k<FP8, M, 4>(x, q, s, y, st);
}

template <bool FP8>
void by_m(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, at::Tensor & y, int rpw, cudaStream_t st) {
    switch (x.size(0)) {
        case 1: by_rpw<FP8, 1>(x, q, s, y, rpw, st); break;
        case 2: by_rpw<FP8, 2>(x, q, s, y, rpw, st); break;
        case 3: by_rpw<FP8, 3>(x, q, s, y, rpw, st); break;
        case 4: by_rpw<FP8, 4>(x, q, s, y, rpw, st); break;
        case 5: by_rpw<FP8, 5>(x, q, s, y, rpw, st); break;
        case 6: by_rpw<FP8, 6>(x, q, s, y, rpw, st); break;
        case 7: by_rpw<FP8, 7>(x, q, s, y, rpw, st); break;
        default: by_rpw<FP8, 8>(x, q, s, y, rpw, st); break;
    }
}

}  // namespace

// q: [N, K] int8 or float8_e4m3fn, row-major; s: fp32 [N] (per row) or [1] (per tensor)
at::Tensor w8_gemm(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, int64_t rpw) {
    TORCH_CHECK(x.is_cuda() && q.is_cuda() && s.is_cuda() && x.scalar_type() == at::kBFloat16 && s.scalar_type() == at::kFloat);
    const bool fp8 = q.scalar_type() == at::kFloat8_e4m3fn;
    TORCH_CHECK(fp8 || q.scalar_type() == at::kChar);
    TORCH_CHECK(x.dim() == 2 && q.dim() == 2 && x.size(1) == q.size(1) && q.is_contiguous() && s.is_contiguous());
    TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
    TORCH_CHECK(s.numel() == q.size(0) || s.numel() == 1);
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8);
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({x.size(0), q.size(0)}, x.options());
    if (rpw <= 0) rpw = x.size(0) == 1 ? 2 : 4;
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    if (fp8) by_m<true>(x, q, s, y, (int) rpw, st); else by_m<false>(x, q, s, y, (int) rpw, st);
    return y;
}
