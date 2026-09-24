// Hyper-connection residual for Qwen3.8-Flash-Next decode batches (1..8 rows), two kernels:
//
// hc_mix_fused (one persistent kernel, replaces grouped Gemma RMSNorm + the low-rank mix):
//   normed = bf16(x * rsqrt(mean_g(x^2) + eps) * (1 + w_norm))      per hc branch g (2560 wide)
//   t      = bf16(silu((normed @ Wd^T) / hc));  gate = sigmoid(t @ Wu^T)
//   out    = bf16(mean_g(gate * normed))                              [rows, 2560]
//   inj    = normed @ W_inject^T  (fp32 [rows, 4])  -- the dot products the later combine needs
// Every CTA requests its weights (Wu slice by cp.async, its Wd/W_inject rows into registers) before
// waiting on the previous kernel (programmatic dependent launch); one grid barrier between the down
// projection (CTA group = hc branch) and the up projection (CTA = output columns).
//
// hc_combine_apply: out = bf16(residual + 2 * sigmoid(inj / hc) * block_out), replacing the gate + apply pair.
// Same math as sglang's grouped_gemma_rmsnorm, hc_mix_triton and hc_combine kernels (fp32 accumulation).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int HC = 4, HS = 2560, K = HC * HS, R = 320, NINJ = 4, NROWS = R + NINJ;
constexpr int THREADS = 256, WARPS = THREADS / 32, MAXM = 8;
constexpr int KQ_VEC = HS / 8;               // 320 uint4 per branch row
constexpr int R_VEC = R / 8;                 // 40 uint4 per Wu row
constexpr int WU_VEC_PER_J = HC * R_VEC;     // 160 uint4 (4 rows) per output column
constexpr int MAXJ = 16;                     // output columns per CTA (needs >= 160 CTAs)

__device__ __forceinline__ void bf16x8_to_f32(const uint4 & v, float (&f)[8]) {
    const __nv_bfloat162 * h = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x; f[2 * i + 1] = t.y;
    }
}

__device__ __forceinline__ unsigned ld_acquire(const unsigned * p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// bf16 m16n8k16 tensor-core MMA, fp32 accumulate. The K order inside each 32-element chunk is permuted
// identically for A and B (lane t holds elements 8t..8t+7 of the chunk for two consecutive k-steps), which
// leaves every dot product unchanged and lets each lane use one 16-byte load per chunk.
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

constexpr int XS_STRIDE = HS + 32;        // bf16 elements per token row in shared memory (5184 B, conflict-free)
constexpr int WU_STRIDE = R_VEC + 4;      // uint4 per Wu row in shared memory (704 B)
constexpr int TS_STRIDE = R + 32;         // bf16 elements per token row of t (704 B)

template <int M>
__global__ void __launch_bounds__(THREADS, 1) hc_mix_fused_kernel(
        const __nv_bfloat16 * __restrict__ x, const __nv_bfloat16 * __restrict__ w_norm,
        const __nv_bfloat16 * __restrict__ wd, const __nv_bfloat16 * __restrict__ winj,
        const __nv_bfloat16 * __restrict__ wu, __nv_bfloat16 * __restrict__ out, float * __restrict__ inj_out,
        __nv_bfloat16 * __restrict__ normed, float * __restrict__ partial, unsigned * __restrict__ bar,
        int rows, float eps) {
    extern __shared__ __align__(16) uint4 smem[];
    uint4 * wus = smem;                                                   // [MAXJ*4 rows][WU_STRIDE]
    __nv_bfloat16 * xs = reinterpret_cast<__nv_bfloat16 *>(smem + MAXJ * HC * WU_STRIDE);   // [8][XS_STRIDE]
    __shared__ float s_red[WARPS][M];
    __shared__ float s_rstd[M];
    __shared__ float s_c[WARPS][16][8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, grp = lane >> 2, tq = lane & 3;
    const int G = gridDim.x, cta = blockIdx.x;
    const int nrows = winj != nullptr ? NROWS : R;

    // ---- weights first (independent of the previous kernel); phase-1 rows before the phase-2 slice ----
    // phase 1: A row grp (< 7) = weight row qi + grp*per_q, this warp's K range [warp*320, +320) of branch q
    const int per_q = G / HC, q = cta / per_q, qi = cta % per_q;
    const int n_row = qi + grp * per_q;
    const bool has_row = grp < 7 && n_row < nrows;
    uint4 wa[10];
    {
        const __nv_bfloat16 * wrow = n_row < R ? wd + (size_t) n_row * K : winj + (size_t) (n_row - R) * K;
        wrow += q * HS + warp * 320 + tq * 8;
#pragma unroll
        for (int kc = 0; kc < 10; ++kc)
            wa[kc] = has_row ? __ldcs(reinterpret_cast<const uint4 *>(wrow + kc * 32)) : make_uint4(0, 0, 0, 0);
    }
    const int j0 = (int) ((long) cta * HS / G), j1 = (int) ((long) (cta + 1) * HS / G), nj = j1 - j0;
    for (int c = tid; c < nj * WU_VEC_PER_J; c += THREADS) {
        const int jl = c / WU_VEC_PER_J, v = c % WU_VEC_PER_J, g = v / R_VEC, k = v % R_VEC;
        const void * src = wu + ((size_t) (g * HS + j0 + jl)) * R + k * 8;
        const unsigned dst = (unsigned) __cvta_generic_to_shared(wus + (jl * HC + g) * WU_STRIDE + k);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    uint4 wn[2];
    wn[0] = __ldg(reinterpret_cast<const uint4 *>(w_norm + q * HS) + tid);
    wn[1] = tid < KQ_VEC - THREADS ? __ldg(reinterpret_cast<const uint4 *>(w_norm + q * HS) + tid + THREADS) : make_uint4(0, 0, 0, 0);
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- branch q of x: sum of squares per row, normalize into shared memory ----
    {
        uint4 xv[M][2];
        float ss[M];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4 * src = reinterpret_cast<const uint4 *>(x + (size_t) m * K + q * HS);
            xv[m][0] = m < rows ? __ldcg(src + tid) : make_uint4(0, 0, 0, 0);
            xv[m][1] = (m < rows && tid < KQ_VEC - THREADS) ? __ldcg(src + tid + THREADS) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float s = 0.f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float f[8];
                bf16x8_to_f32(xv[m][h], f);
#pragma unroll
                for (int e = 0; e < 8; ++e) s += f[e] * f[e];
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
            ss[m] = s;
        }
        if (lane == 0) {
#pragma unroll
            for (int m = 0; m < M; ++m) s_red[warp][m] = ss[m];
        }
        // tokens M..7 of the MMA B operand are zero
        for (int c = tid; c < (8 - M) * KQ_VEC; c += THREADS)
            reinterpret_cast<uint4 *>(xs + (M + c / KQ_VEC) * XS_STRIDE)[c % KQ_VEC] = make_uint4(0, 0, 0, 0);
        __syncthreads();
        if (tid < M) {
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) s += s_red[w][tid];
            s_rstd[tid] = rsqrtf(s / HS + eps);
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int k = tid + h * THREADS;
            if (k >= KQ_VEC) break;
            float wf[8];
            bf16x8_to_f32(wn[h], wf);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float f[8];
                bf16x8_to_f32(xv[m][h], f);
                const float r = s_rstd[m];
                uint4 o;
                __nv_bfloat162 * op = reinterpret_cast<__nv_bfloat162 *>(&o);
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    op[e] = __floats2bfloat162_rn(f[2 * e] * r * (1.0f + wf[2 * e]), f[2 * e + 1] * r * (1.0f + wf[2 * e + 1]));
                reinterpret_cast<uint4 *>(xs + m * XS_STRIDE)[k] = o;
                if (k % per_q == qi && m < rows)
                    reinterpret_cast<uint4 *>(normed + (size_t) m * K + q * HS)[k] = o;
            }
        }
        __syncthreads();
    }
    // ---- down projection (+ inject rows): [7 rows x this warp's 320 K] x [8 tokens] on tensor cores ----
    {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        const __nv_bfloat16 * xb = xs + grp * XS_STRIDE + warp * 320 + tq * 8;   // B column = token grp
#pragma unroll
        for (int kc = 0; kc < 10; ++kc) {
            const uint4 bv = *reinterpret_cast<const uint4 *>(xb + kc * 32);
            mma16816(d, wa[kc].x, 0u, wa[kc].y, 0u, bv.x, bv.y);
            mma16816(d, wa[kc].z, 0u, wa[kc].w, 0u, bv.z, bv.w);
        }
        // d[0], d[1]: row grp, tokens 2tq, 2tq+1 (rows grp+8 are padding)
        s_c[warp][grp][2 * tq] = d[0];
        s_c[warp][grp][2 * tq + 1] = d[1];
        __syncthreads();
        if (tid < 7 * M) {
            const int r = tid / M, m = tid % M, n = qi + r * per_q;
            if (n < nrows) {
                float v = 0.f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) v += s_c[w][r][m];
                partial[((size_t) q * MAXM + m) * NROWS + n] = v;
            }
        }
    }

    // ---- grid barrier ----
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        atomicAdd(bar, 1u);
        while (ld_acquire(bar) < (unsigned) G) { }
    }
    __syncthreads();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // ---- t = bf16(silu(sum_q partial / hc)) -> shared [8 tokens][TS_STRIDE] (reuses xs; tokens >= M zero) ----
    __nv_bfloat16 * ts = xs;
    {
        constexpr int TV = (8 * R + THREADS - 1) / THREADS;
        float pv[TV][HC];
#pragma unroll
        for (int i = 0; i < TV; ++i) {
            const int c = tid + i * THREADS, m = c / R, r = c % R;
#pragma unroll
            for (int g = 0; g < HC; ++g)
                pv[i][g] = (c < 8 * R && m < M) ? __ldcg(partial + ((size_t) g * MAXM + m) * NROWS + r) : 0.f;
        }
#pragma unroll
        for (int i = 0; i < TV; ++i) {
            const int c = tid + i * THREADS, m = c / R, r = c % R;
            float a = (pv[i][0] + pv[i][1]) + (pv[i][2] + pv[i][3]);
            a *= (1.0f / HC);
            if (c < 8 * R) ts[m * TS_STRIDE + r] = m < M ? __float2bfloat16(a * (1.0f / (1.0f + __expf(-a)))) : __float2bfloat16(0.f);
        }
    }
    if (cta == 0 && winj != nullptr && tid < M * NINJ) {
        const int m = tid / NINJ, g = tid % NINJ;
        if (m < rows) {
            float a = 0.f;
#pragma unroll
            for (int qq = 0; qq < HC; ++qq) a += __ldcg(partial + ((size_t) qq * MAXM + m) * NROWS + R + g);
            inj_out[m * NINJ + g] = a;
        }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();

    // ---- up projection on tensor cores: rows (jl, g) of Wu x [8 tokens]; warp w < 4 owns rows 16w..16w+15 ----
    if (warp < 4 && warp * 4 < nj) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        const int r0 = warp * 16 + grp, r1 = r0 + 8;              // row = jl * 4 + g
        const bool v0 = r0 < nj * HC, v1 = r1 < nj * HC;
        const uint4 * a0p = wus + r0 * WU_STRIDE + tq;
        const uint4 * a1p = wus + r1 * WU_STRIDE + tq;
        const __nv_bfloat16 * bp = ts + grp * TS_STRIDE + tq * 8;
#pragma unroll
        for (int kc = 0; kc < R / 32; ++kc) {
            const uint4 av0 = v0 ? a0p[kc * 4] : make_uint4(0, 0, 0, 0);
            const uint4 av1 = v1 ? a1p[kc * 4] : make_uint4(0, 0, 0, 0);
            const uint4 bv = *reinterpret_cast<const uint4 *>(bp + kc * 32);
            mma16816(d, av0.x, av1.x, av0.y, av1.y, bv.x, bv.y);
            mma16816(d, av0.z, av1.z, av0.w, av1.w, bv.z, bv.w);
        }
        // d[0..1]: row r0, tokens 2tq, 2tq+1; d[2..3]: row r1. g = row & 3 = grp & 3; mean over g via lanes xor 4, 8
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rr = h ? r1 : r0, jl = rr >> 2, g = rr & 3, j = j0 + jl;
            const bool valid = h ? v1 : v0;
            float o[2];
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int m = 2 * tq + c;
                float v = 0.f;
                if (valid && m < rows) {
                    const float gate = 1.f / (1.f + __expf(-d[2 * h + c]));
                    v = gate * __bfloat162float(__ldcg(normed + (size_t) m * K + g * HS + j));
                }
                v += __shfl_xor_sync(0xffffffff, v, 4);
                v += __shfl_xor_sync(0xffffffff, v, 8);
                o[c] = v;
            }
            if (valid && g == 0) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int m = 2 * tq + c;
                    if (m < rows) out[(size_t) m * HS + j] = __float2bfloat16(o[c] * (1.0f / HC));
                }
            }
        }
    }

    // ---- last CTA out re-arms the barrier ----
    if (tid == 0) {
        const unsigned ticket = atomicAdd(bar + 1, 1u);
        if (ticket == (unsigned) G - 1) {
            bar[0] = 0;
            bar[1] = 0;
            __threadfence();
        }
    }
}

// out[m, g*HS + k] = bf16(r + a[m, g] * y[m, k]),  a = 2 / (1 + exp(-inj / hc))
__global__ void __launch_bounds__(256) hc_combine_apply_kernel(
        const __nv_bfloat16 * __restrict__ y, const __nv_bfloat16 * __restrict__ r, const float * __restrict__ inj,
        __nv_bfloat16 * __restrict__ out) {
    const int m = blockIdx.x, v = blockIdx.y * 256 + threadIdx.x;     // uint4 index within the row (1280)
    const int g = v / KQ_VEC, k = v % KQ_VEC;
    const uint4 rv = __ldg(reinterpret_cast<const uint4 *>(r + (size_t) m * K) + v);
    const float total = __ldg(inj + m * NINJ + g);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const uint4 yv = __ldg(reinterpret_cast<const uint4 *>(y + (size_t) m * HS) + k);
    const float a = 2.0f / (1.0f + expf(-total / HC));
    float rf[8], yf[8];
    bf16x8_to_f32(rv, rf);
    bf16x8_to_f32(yv, yf);
    uint4 o;
    __nv_bfloat162 * op = reinterpret_cast<__nv_bfloat162 *>(&o);
#pragma unroll
    for (int e = 0; e < 4; ++e) op[e] = __floats2bfloat162_rn(rf[2 * e] + a * yf[2 * e], rf[2 * e + 1] + a * yf[2 * e + 1]);
    reinterpret_cast<uint4 *>(out + (size_t) m * K)[v] = o;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

template <int M>
void launch_mix(const at::Tensor & x, const at::Tensor & w_norm, const at::Tensor & wd, const at::Tensor * winj,
                const at::Tensor & wu, at::Tensor & out, at::Tensor & inj, at::Tensor & normed, at::Tensor & partial,
                at::Tensor & bar, int rows, float eps, int grid, cudaStream_t s, bool pdl) {
    const int smem = MAXJ * HC * WU_STRIDE * 16 + 8 * XS_STRIDE * 2;
    auto k = hc_mix_fused_kernel<M>;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(THREADS);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = s;
    cudaLaunchAttribute lattr[1];
    lattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    lattr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = lattr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, (const __nv_bfloat16 *) x.data_ptr(), (const __nv_bfloat16 *) w_norm.data_ptr(),
                                      (const __nv_bfloat16 *) wd.data_ptr(),
                                      winj ? (const __nv_bfloat16 *) winj->data_ptr() : (const __nv_bfloat16 *) nullptr,
                                      (const __nv_bfloat16 *) wu.data_ptr(), (__nv_bfloat16 *) out.data_ptr(),
                                      winj ? inj.data_ptr<float>() : (float *) nullptr, (__nv_bfloat16 *) normed.data_ptr(),
                                      partial.data_ptr<float>(), reinterpret_cast<unsigned *>(bar.data_ptr<int>()), rows, eps));
}

}  // namespace

// x [rows, 10240] bf16 contiguous (rows 1..8); w_norm [10240]; wd [320, 10240]; winj [4, 10240] or None;
// wu [10240, 320]. Scratch (shared by all call sites, which run in stream order): normed [8, 10240] bf16,
// partial fp32 >= 4*8*324, bar int32 [2] zeroed. Returns (out [rows, 2560] bf16, inj [rows, 4] fp32).
std::vector<at::Tensor> hc_mix_fused(const at::Tensor & x, const at::Tensor & w_norm, const at::Tensor & wd,
                                     const c10::optional<at::Tensor> & winj, const at::Tensor & wu, double eps,
                                     at::Tensor & normed, at::Tensor & partial, at::Tensor & bar, bool pdl) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == K && x.is_contiguous());
    TORCH_CHECK(w_norm.scalar_type() == at::kBFloat16 && w_norm.numel() == K && w_norm.is_contiguous());
    TORCH_CHECK(wd.scalar_type() == at::kBFloat16 && wd.size(0) == R && wd.size(1) == K && wd.is_contiguous());
    TORCH_CHECK(wu.scalar_type() == at::kBFloat16 && wu.size(0) == K && wu.size(1) == R && wu.is_contiguous());
    if (winj.has_value())
        TORCH_CHECK(winj->scalar_type() == at::kBFloat16 && winj->size(0) == NINJ && winj->size(1) == K && winj->is_contiguous());
    TORCH_CHECK(normed.scalar_type() == at::kBFloat16 && normed.numel() >= MAXM * K);
    TORCH_CHECK(partial.scalar_type() == at::kFloat && partial.numel() >= HC * MAXM * NROWS);
    TORCH_CHECK(bar.scalar_type() == at::kInt && bar.numel() >= 2);
    const int rows = x.size(0);
    TORCH_CHECK(rows >= 1 && rows <= MAXM);
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty({rows, HS}, x.options());
    auto inj = at::empty({rows, NINJ}, x.options().dtype(at::kFloat));
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int grid = (sms / HC) * HC;
    TORCH_CHECK(grid * MAXJ >= HS && (grid / HC) * 7 >= NROWS, "too few SMs for hc_mix_fused");
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    const at::Tensor * wi = winj.has_value() ? &winj.value() : nullptr;
    const float e = (float) eps;
    if (rows == 1) launch_mix<1>(x, w_norm, wd, wi, wu, out, inj, normed, partial, bar, rows, e, grid, s, pdl);
    else if (rows == 2) launch_mix<2>(x, w_norm, wd, wi, wu, out, inj, normed, partial, bar, rows, e, grid, s, pdl);
    else if (rows <= 4) launch_mix<4>(x, w_norm, wd, wi, wu, out, inj, normed, partial, bar, rows, e, grid, s, pdl);
    else launch_mix<8>(x, w_norm, wd, wi, wu, out, inj, normed, partial, bar, rows, e, grid, s, pdl);
    return {out, inj};
}

at::Tensor hc_combine_apply(const at::Tensor & y, const at::Tensor & r, const at::Tensor & inj, bool pdl) {
    TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.dim() == 2 && y.size(1) == HS && y.is_contiguous());
    TORCH_CHECK(r.scalar_type() == at::kBFloat16 && r.dim() == 2 && r.size(1) == K && r.is_contiguous() && r.size(0) == y.size(0));
    TORCH_CHECK(inj.scalar_type() == at::kFloat && inj.size(0) == y.size(0) && inj.size(1) == NINJ && inj.is_contiguous());
    const c10::cuda::CUDAGuard guard(y.device());
    auto out = at::empty_like(r);
    const int rows = y.size(0);
    if (rows == 0) return out;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(rows, K / 8 / 256);
    cfg.blockDim = dim3(256);
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute lattr[1];
    lattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    lattr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = lattr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, hc_combine_apply_kernel, (const __nv_bfloat16 *) y.data_ptr(),
                                      (const __nv_bfloat16 *) r.data_ptr(), (const float *) inj.data_ptr<float>(),
                                      (__nv_bfloat16 *) out.data_ptr()));
    return out;
}
