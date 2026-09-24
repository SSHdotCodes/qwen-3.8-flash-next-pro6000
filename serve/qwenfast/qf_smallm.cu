// Small-M BF16 GEMM for decode/verify batches: y[M,N] = x[M,K] @ w[N,K]^T, 1 <= M <= 8.
// Weight-bandwidth bound: each warp streams RPW weight rows with 16-byte loads (all loads of a
// chunk issued before the FMAs), x is staged once per block in shared memory, FP32 accumulation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int WARPS = 8;

__device__ __forceinline__ void bf16x8_to_f32(const uint4 & v, float (&f)[8]) {
    const __nv_bfloat162 * h = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x; f[2 * i + 1] = t.y;
    }
}

// UNROLL k-chunks of 256 elements per warp per iteration; RPW rows per warp.
template <int M, int RPW, int UNROLL>
__global__ void __launch_bounds__(WARPS * 32) smallm_kernel(
        const __nv_bfloat16 * __restrict__ x, const __nv_bfloat16 * __restrict__ w,
        __nv_bfloat16 * __restrict__ y, int N, int K, int ldx, int ldy, bool x_in_smem) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16 * xs = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const __nv_bfloat16 * xp = x;
    int ldxs = ldx;
    if (x_in_smem) {
        // stage x: M rows of K elements, 16-byte chunks
        const int chunks = M * (K / 8);
        for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
            const int m = c / (K / 8), kc = c % (K / 8);
            reinterpret_cast<uint4 *>(xs)[c] = *reinterpret_cast<const uint4 *>(x + (size_t) m * ldx + kc * 8);
        }
        __syncthreads();
        xp = xs; ldxs = K;
    }
    const int row0 = (blockIdx.x * WARPS + warp) * RPW;
    if (row0 >= N) return;
    float acc[RPW][M];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

    const int step = 32 * 8;  // elements per warp-wide chunk
    int k0 = lane * 8;
    for (; k0 + (UNROLL - 1) * step < K; k0 += UNROLL * step) {
        uint4 wv[RPW][UNROLL];
#pragma unroll
        for (int r = 0; r < RPW; ++r)
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int row = min(row0 + r, N - 1);
                wv[r][u] = __ldcs(reinterpret_cast<const uint4 *>(w + (size_t) row * K + k0 + u * step));
            }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float xf[8];
                bf16x8_to_f32(*reinterpret_cast<const uint4 *>(xp + (size_t) m * ldxs + k0 + u * step), xf);
#pragma unroll
                for (int r = 0; r < RPW; ++r) {
                    float wf[8];
                    bf16x8_to_f32(wv[r][u], wf);
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[r][m] = fmaf(wf[i], xf[i], acc[r][m]);
                }
            }
        }
    }
    for (; k0 < K; k0 += step) {  // tail (K not a multiple of UNROLL*256)
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = min(row0 + r, N - 1);
            const uint4 wv = __ldcs(reinterpret_cast<const uint4 *>(w + (size_t) row * K + k0));
            float wf[8];
            bf16x8_to_f32(wv, wf);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float xf[8];
                bf16x8_to_f32(*reinterpret_cast<const uint4 *>(xp + (size_t) m * ldxs + k0), xf);
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[r][m] = fmaf(wf[i], xf[i], acc[r][m]);
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

template <int M, int RPW, int UNROLL>
void launch(const at::Tensor & x, const at::Tensor & w, at::Tensor & y, cudaStream_t stream) {
    const int N = w.size(0), K = w.size(1);
    const int rows_per_block = WARPS * RPW;
    const int grid = (N + rows_per_block - 1) / rows_per_block;
    const size_t xbytes = (size_t) M * K * sizeof(__nv_bfloat16);
    const bool in_smem = xbytes <= 64 * 1024;
    const size_t smem = in_smem ? xbytes : 0;
    auto kern = smallm_kernel<M, RPW, UNROLL>;
    if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    kern<<<grid, WARPS * 32, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16 *>(w.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), N, K, (int) x.stride(0), (int) y.stride(0), in_smem);
}

template <int M>
void dispatch_rpw(const at::Tensor & x, const at::Tensor & w, at::Tensor & y, int rpw, int unroll, cudaStream_t s) {
    if (unroll == 2) {
        if (rpw == 1) launch<M, 1, 2>(x, w, y, s); else if (rpw == 2) launch<M, 2, 2>(x, w, y, s); else launch<M, 4, 2>(x, w, y, s);
    } else {
        if (rpw == 1) launch<M, 1, 4>(x, w, y, s); else if (rpw == 2) launch<M, 2, 4>(x, w, y, s); else launch<M, 4, 4>(x, w, y, s);
    }
}

}  // namespace

// rpw/unroll 0 = pick automatically
at::Tensor smallm_gemm(const at::Tensor & x, const at::Tensor & w, int64_t rpw, int64_t unroll) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1) && w.is_contiguous() && x.stride(1) == 1);
    const int M = x.size(0), N = w.size(0), K = w.size(1);
    TORCH_CHECK(M >= 1 && M <= 8 && K % 8 == 0 && x.stride(0) % 8 == 0);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0);
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({M, N}, x.options());
    if (rpw <= 0) {
        // enough warps to cover all SMs several times over
        const long warps_needed = (long) N;
        rpw = warps_needed >= 188L * WARPS * 8 ? 2 : 1;
    }
    if (unroll <= 0) unroll = K >= 2048 ? 4 : 2;
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    switch (M) {
        case 1: dispatch_rpw<1>(x, w, y, rpw, unroll, s); break;
        case 2: dispatch_rpw<2>(x, w, y, rpw, unroll, s); break;
        case 3: dispatch_rpw<3>(x, w, y, rpw, unroll, s); break;
        case 4: dispatch_rpw<4>(x, w, y, rpw, unroll, s); break;
        case 5: dispatch_rpw<5>(x, w, y, rpw, unroll, s); break;
        case 6: dispatch_rpw<6>(x, w, y, rpw, unroll, s); break;
        case 7: dispatch_rpw<7>(x, w, y, rpw, unroll, s); break;
        default: dispatch_rpw<8>(x, w, y, rpw, unroll, s); break;
    }
    return y;
}
