// Hyper-connection low-rank mix for decode-size batches (1..8 rows), one persistent kernel:
//   t   = bf16(silu((x @ Wd^T) / hc))             x [M, hc*hs], Wd [R, hc*hs]
//   g   = sigmoid(t @ Wu^T)                       Wu [hc*hs, R]
//   out = mean_over_hc(g * x)                     out [M, hs]
// Same math as sglang's hc_mix_triton kernel (fp32 accumulation, t rounded to bf16).
// Every CTA requests its whole share of both weights at kernel start: its Wu slice goes to
// registers before the down projection runs, so the two weight streams overlap and the grid
// barrier between the phases costs no extra DRAM time.
// Phase 1: CTA group q (one per hc branch) computes partial[q][m][n] = x[m, q-branch] . Wd[n, q-branch]
//          for its rows n, with the q-branch of x staged in shared memory.
// Phase 2: each CTA sums the hc partials for all (m, r), then produces its slice of output columns.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int HC = 4, HS = 2560, KQ = HS, R = 320, THREADS = 256, WARPS = THREADS / 32;
constexpr int KQ_VEC = KQ / 8;            // 320 uint4 per branch row
constexpr int R_VEC = R / 8;              // 40 uint4 per Wu row
constexpr int J_PER_WARP = 2;             // up to 16 output columns per CTA
constexpr int WU_VEC_PER_J = HC * R_VEC;  // 160 uint4 (4 rows) per output column
constexpr int WU_PER_LANE = WU_VEC_PER_J / 32;  // 5

__device__ __forceinline__ void bf16x8_to_f32(const uint4 & v, float (&f)[8]) {
    const __nv_bfloat162 * h = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x; f[2 * i + 1] = t.y;
    }
}

__device__ __forceinline__ uint4 ld_stream(const void * p) {
    return __ldcs(reinterpret_cast<const uint4 *>(p));
}

__device__ __forceinline__ unsigned ld_acquire(const unsigned * p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

template <int M>
__global__ void __launch_bounds__(THREADS, 1) hc_mix_kernel(
        const __nv_bfloat16 * __restrict__ x, const __nv_bfloat16 * __restrict__ wd,
        const __nv_bfloat16 * __restrict__ wu, __nv_bfloat16 * __restrict__ out,
        float * __restrict__ partial, unsigned * __restrict__ bar, int rows, float inv_hc) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int G = gridDim.x, cta = blockIdx.x;

    // ---- phase-2 weight prefetch (cp.async into shared memory, issued first) ----
    const int j0 = (int) ((long) cta * HS / G), j1 = (int) ((long) (cta + 1) * HS / G);
    const int nj = j1 - j0;
    uint4 * wus = reinterpret_cast<uint4 *>(smem_raw + (size_t) M * KQ * 2);  // [16 j][HC][R_VEC]
    for (int c = threadIdx.x; c < nj * WU_VEC_PER_J; c += THREADS) {
        const int jl = c / WU_VEC_PER_J, v = c % WU_VEC_PER_J, g = v / R_VEC, k = v % R_VEC;
        const void * src = wu + ((size_t) (g * HS + j0 + jl)) * R + k * 8;
        const unsigned dst = (unsigned) __cvta_generic_to_shared(wus + c);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");

    // ---- phase 1: down projection partials for branch q ----
    const int per_q = G / HC;
    const int q = cta / per_q, qi = cta % per_q;
    uint4 * xs = reinterpret_cast<uint4 *>(smem_raw);  // [M][KQ_VEC]
    if (q < HC) {
        const int n = qi + warp * per_q;
        uint4 wv[KQ_VEC / 32];
        if (n < R) {
#pragma unroll
            for (int u = 0; u < KQ_VEC / 32; ++u)
                wv[u] = ld_stream(wd + (size_t) n * (HC * HS) + q * KQ + (lane + 32 * u) * 8);
        }
        constexpr int XV = (M * KQ_VEC + THREADS - 1) / THREADS;
        uint4 xv[XV];
#pragma unroll
        for (int i = 0; i < XV; ++i) {
            const int c = threadIdx.x + i * THREADS, m = c / KQ_VEC, k = c % KQ_VEC;
            xv[i] = (c < M * KQ_VEC && m < rows)
                        ? __ldcg(reinterpret_cast<const uint4 *>(x + (size_t) m * (HC * HS) + q * KQ + k * 8))
                        : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < XV; ++i) {
            const int c = threadIdx.x + i * THREADS;
            if (c < M * KQ_VEC) xs[c] = xv[i];
        }
        __syncthreads();
        if (n < R) {
            float acc[M];
#pragma unroll
            for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll
            for (int u = 0; u < KQ_VEC / 32; ++u) {
                float wf[8];
                bf16x8_to_f32(wv[u], wf);
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    float xf[8];
                    bf16x8_to_f32(xs[m * KQ_VEC + lane + 32 * u], xf);
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[m] = fmaf(wf[e], xf[e], acc[m]);
                }
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float v = acc[m];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
                if (lane == m % 32) partial[((size_t) q * M + m) * R + n] = v;
            }
        }
    }

    // ---- grid barrier ----
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(bar, 1u);
        while (ld_acquire(bar) < (unsigned) G) { }
    }
    __syncthreads();

    asm volatile("cp.async.wait_all;" ::: "memory");
    // ---- t = bf16(silu(sum_q partial / hc)) for all rows into shared memory (reuses xs) ----
    float * ts = reinterpret_cast<float *>(smem_raw);  // [M][R]
    {
        constexpr int TV = (M * R + THREADS - 1) / THREADS;
        float pv[TV][HC];
#pragma unroll
        for (int i = 0; i < TV; ++i) {
            const int c = threadIdx.x + i * THREADS;
#pragma unroll
            for (int g = 0; g < HC; ++g) pv[i][g] = c < M * R ? __ldcg(partial + (size_t) g * M * R + c) : 0.f;
        }
#pragma unroll
        for (int i = 0; i < TV; ++i) {
            const int c = threadIdx.x + i * THREADS;
            float a = ((pv[i][0] + pv[i][1]) + pv[i][2]) + pv[i][3];
            a *= inv_hc;
            if (c < M * R) ts[c] = __bfloat162float(__float2bfloat16(a / (1.f + __expf(-a))));
        }
    }
    __syncthreads();

    // ---- phase 2: gates and the hc mean for this CTA's output columns ----
#pragma unroll
    for (int jj = 0; jj < J_PER_WARP; ++jj) {
        const int j = j0 + warp + jj * WARPS;
        if (j >= j1) break;
        float acc[HC][M];
#pragma unroll
        for (int g = 0; g < HC; ++g)
#pragma unroll
            for (int m = 0; m < M; ++m) acc[g][m] = 0.f;
#pragma unroll
        for (int i = 0; i < WU_PER_LANE; ++i) {
            const int v = lane + 32 * i, row = v / R_VEC, c = v % R_VEC;
            float wf[8];
            bf16x8_to_f32(wus[(warp + jj * WARPS) * WU_VEC_PER_J + v], wf);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const float4 t0 = *reinterpret_cast<const float4 *>(ts + m * R + c * 8);
                const float4 t1 = *reinterpret_cast<const float4 *>(ts + m * R + c * 8 + 4);
                float s = wf[0] * t0.x;
                s = fmaf(wf[1], t0.y, s); s = fmaf(wf[2], t0.z, s); s = fmaf(wf[3], t0.w, s);
                s = fmaf(wf[4], t1.x, s); s = fmaf(wf[5], t1.y, s); s = fmaf(wf[6], t1.z, s); s = fmaf(wf[7], t1.w, s);
#pragma unroll
                for (int g = 0; g < HC; ++g) acc[g][m] += row == g ? s : 0.f;
            }
        }
#pragma unroll
        for (int g = 0; g < HC; ++g)
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float v = acc[g][m];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
                acc[g][m] = v;
            }
        if (lane < M && lane < rows) {
            float o = 0.f;
#pragma unroll
            for (int m = 0; m < M; ++m) {
                if (m != lane) continue;
#pragma unroll
                for (int g = 0; g < HC; ++g) {
                    const float gate = 1.f / (1.f + __expf(-acc[g][m]));
                    o = fmaf(gate, __bfloat162float(x[(size_t) m * (HC * HS) + g * HS + j]), o);
                }
            }
            out[(size_t) lane * HS + j] = __float2bfloat16(o * inv_hc);
        }
    }

    // ---- last CTA out resets the barrier for the next launch ----
    if (threadIdx.x == 0) {
        const unsigned ticket = atomicAdd(bar + 1, 1u);
        if (ticket == (unsigned) G - 1) {
            bar[0] = 0;
            bar[1] = 0;
            __threadfence();
        }
    }
}

template <int M>
void launch(const at::Tensor & x, const at::Tensor & wd, const at::Tensor & wu, at::Tensor & out,
            at::Tensor & partial, at::Tensor & bar, int rows, int grid, cudaStream_t s) {
    const size_t smem = (size_t) M * KQ * 2 + (size_t) J_PER_WARP * WARPS * WU_VEC_PER_J * 16;
    auto k = hc_mix_kernel<M>;
    if (smem > 48 * 1024) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    k<<<grid, THREADS, smem, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16 *>(wd.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(wu.data_ptr()), reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()),
        partial.data_ptr<float>(), reinterpret_cast<unsigned *>(bar.data_ptr<int>()), rows, 1.f / HC);
}

}  // namespace

// partial: fp32 [HC*16*R] scratch; bar: int32 [2] zero-initialised, owned by one call site.
at::Tensor hc_mix(const at::Tensor & x, const at::Tensor & wd, const at::Tensor & wu,
                  at::Tensor & partial, at::Tensor & bar) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && wd.scalar_type() == at::kBFloat16 &&
                wu.scalar_type() == at::kBFloat16);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == HC * HS && x.is_contiguous());
    TORCH_CHECK(wd.size(0) == R && wd.size(1) == HC * HS && wd.is_contiguous());
    TORCH_CHECK(wu.size(0) == HC * HS && wu.size(1) == R && wu.is_contiguous());
    TORCH_CHECK(partial.scalar_type() == at::kFloat && partial.numel() >= HC * 16 * R);
    TORCH_CHECK(bar.scalar_type() == at::kInt && bar.numel() >= 2);
    const int rows = x.size(0);
    TORCH_CHECK(rows >= 1 && rows <= 8);
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty({rows, HS}, x.options());
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int grid = (sms / HC) * HC;
    TORCH_CHECK(grid * J_PER_WARP * WARPS >= HS && (grid / HC) * WARPS >= R, "too few SMs for hc_mix");
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    if (rows == 1) launch<1>(x, wd, wu, out, partial, bar, rows, grid, s);
    else if (rows == 2) launch<2>(x, wd, wu, out, partial, bar, rows, grid, s);
    else if (rows <= 4) launch<4>(x, wd, wu, out, partial, bar, rows, grid, s);
    else launch<8>(x, wd, wu, out, partial, bar, rows, grid, s);
    return out;
}
