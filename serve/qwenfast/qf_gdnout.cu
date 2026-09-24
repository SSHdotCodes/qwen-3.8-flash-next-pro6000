// Gated-DeltaNet output for Qwen3.8-Flash-Next decode batches (1..8 tokens): gated RMSNorm + out_proj in one kernel.
//
//   xn[t, h*128 + d] = bf16( core[t*48 + h, d] * rsqrt(mean_d(core^2) + eps) * w[d] * gate(z[t*48 + h, d]) )
//   y[t, n]          = bf16( sum_k xn[t, k] * W[n, k] )          W: out_proj weight [2560, 6144] bf16
// gate = sigmoid(z) or z * sigmoid(z) (swish). Same math as sglang's fla layernorm_gated (RMS, norm before gate)
// followed by the BF16 linear.
//
// The recurrent-state kernel before this one releases its dependents as soon as it starts, so every CTA
// asks for its (contiguous) slice of W to be pulled into L2 (bulk prefetch) while the recurrence runs, and
// only then waits (programmatic dependent launch). After the wait: normalize the tokens into shared memory,
// multiply on tensor cores reading W from L2 (K split across warps), reduce the warp partials.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int HEADS = 48, HD = 128, K = HEADS * HD, N = 2560, MAXT = 8;
constexpr int THREADS = 256, WARPS = THREADS / 32, KW = K / WARPS;   // 768 K per warp
constexpr int STRIDE = K + 32;                                       // bf16 per smem row (conflict-free 16 B loads)

__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int T>
__global__ void __launch_bounds__(THREADS, 1) gdn_norm_oproj_kernel(
        const __nv_bfloat16 * __restrict__ core, const __nv_bfloat16 * __restrict__ z,
        const __nv_bfloat16 * __restrict__ w_norm, const __nv_bfloat16 * __restrict__ w,
        __nv_bfloat16 * __restrict__ y, float eps, int swish, int prefetch) {
    extern __shared__ __align__(16) uint4 smem[];
    __nv_bfloat16 * xs = reinterpret_cast<__nv_bfloat16 *>(smem);              // [T][STRIDE]
    float (*red)[16][8] = reinterpret_cast<float (*)[16][8]>(smem);           // reuses xs after the multiply
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, grp = lane >> 2, tq = lane & 3;
    const int G = gridDim.x, n0 = (int) ((long) blockIdx.x * N / G), n1 = (int) ((long) (blockIdx.x + 1) * N / G);
    const int nr = n1 - n0;

    // ---- weights first: this CTA's rows are one contiguous range; pull it into L2 ----
    if (prefetch && tid == 0) {
        const __nv_bfloat16 * src = w + (size_t) n0 * K;
        const unsigned bytes = (unsigned) nr * K * 2;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src), "r"(bytes) : "memory");
    }
    // lanes work in groups of 8 on one (token, head) row: 16 elements per lane
    const int sub = lane >> 3, sl = lane & 7;
    float wn[16];
    {
        const uint4 * wp = reinterpret_cast<const uint4 *>(w_norm) + sl * 2;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 v = __ldg(wp + h);
            const __nv_bfloat162 * b2 = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __bfloat1622float2(b2[i]);
                wn[h * 8 + 2 * i] = f.x; wn[h * 8 + 2 * i + 1] = f.y;
            }
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // ---- gated RMSNorm: 4 (token, head) rows per warp iteration, loads batched ----
    constexpr int PAIRS = T * HEADS, ITERS = (PAIRS + WARPS * 4 - 1) / (WARPS * 4), BATCH_N = 4;
#pragma unroll
    for (int i0 = 0; i0 < ITERS; i0 += BATCH_N) {
        uint4 cv[BATCH_N][2], zv[BATCH_N][2];
#pragma unroll
        for (int j = 0; j < BATCH_N; ++j) {
            const int p = ((i0 + j) * WARPS + warp) * 4 + sub;
            if (i0 + j < ITERS && p < PAIRS) {
                const uint4 * cp = reinterpret_cast<const uint4 *>(core + (size_t) p * HD) + sl * 2;
                const uint4 * zp = reinterpret_cast<const uint4 *>(z + (size_t) p * HD) + sl * 2;
                cv[j][0] = __ldcg(cp); cv[j][1] = __ldcg(cp + 1);
                zv[j][0] = __ldcg(zp); zv[j][1] = __ldcg(zp + 1);
            }
        }
#pragma unroll
        for (int j = 0; j < BATCH_N; ++j) {
            const int p = ((i0 + j) * WARPS + warp) * 4 + sub;
            if (i0 + j < ITERS && p < PAIRS) {
                float xv[16], zf[16];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const __nv_bfloat162 * c2 = reinterpret_cast<const __nv_bfloat162 *>(&cv[j][h]);
                    const __nv_bfloat162 * z2 = reinterpret_cast<const __nv_bfloat162 *>(&zv[j][h]);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 cf = __bfloat1622float2(c2[i]), zz = __bfloat1622float2(z2[i]);
                        xv[h * 8 + 2 * i] = cf.x; xv[h * 8 + 2 * i + 1] = cf.y;
                        zf[h * 8 + 2 * i] = zz.x; zf[h * 8 + 2 * i + 1] = zz.y;
                    }
                }
                float ss = 0.f;
#pragma unroll
                for (int i = 0; i < 16; ++i) ss += xv[i] * xv[i];
                ss += __shfl_xor_sync(0xffffffff, ss, 4);
                ss += __shfl_xor_sync(0xffffffff, ss, 2);
                ss += __shfl_xor_sync(0xffffffff, ss, 1);
                const float rstd = rsqrtf(ss / HD + eps);
                uint4 o[2];
                __nv_bfloat162 * o2 = reinterpret_cast<__nv_bfloat162 *>(o);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float r[2];
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        const int e = 2 * i + u;
                        const float sg = 1.0f / (1.0f + __expf(-zf[e]));
                        r[u] = xv[e] * rstd * wn[e] * (swish ? zf[e] * sg : sg);
                    }
                    o2[i] = __floats2bfloat162_rn(r[0], r[1]);
                }
                const int t = p / HEADS, hh = p % HEADS;
                uint4 * dst = reinterpret_cast<uint4 *>(xs + (size_t) t * STRIDE + hh * HD) + sl * 2;
                dst[0] = o[0];
                dst[1] = o[1];
            }
        }
    }
    __syncthreads();

    // ---- y = xn @ W^T on tensor cores: A = up to 16 weight rows, B = 8 token columns, K split over warps ----
    // (K inside each 32-chunk is permuted identically for A and B: lane tq holds elements 8tq..8tq+7)
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    const int r0 = grp, r1 = grp + 8;
    const bool v0 = r0 < nr, v1 = r1 < nr;
    const int kb = warp * KW + tq * 8;
    const uint4 zero = make_uint4(0, 0, 0, 0);
    const uint4 * a0p = reinterpret_cast<const uint4 *>(w + (size_t) (n0 + (v0 ? r0 : 0)) * K + kb);
    const uint4 * a1p = reinterpret_cast<const uint4 *>(w + (size_t) (n0 + (v1 ? r1 : 0)) * K + kb);
    constexpr int KC = KW / 32, BATCH = 12;
#pragma unroll
    for (int kc0 = 0; kc0 < KC; kc0 += BATCH) {
        uint4 a0[BATCH], a1[BATCH];
#pragma unroll
        for (int i = 0; i < BATCH; ++i) {
            a0[i] = v0 ? __ldcs(a0p + (kc0 + i) * 4) : zero;
            a1[i] = v1 ? __ldcs(a1p + (kc0 + i) * 4) : zero;
        }
#pragma unroll
        for (int i = 0; i < BATCH; ++i) {
            const uint4 b = grp < T ? *reinterpret_cast<const uint4 *>(xs + (size_t) grp * STRIDE + kb + (kc0 + i) * 32) : zero;
            mma16816(d, a0[i].x, a1[i].x, a0[i].y, a1[i].y, b.x, b.y);
            mma16816(d, a0[i].z, a1[i].z, a0[i].w, a1[i].w, b.z, b.w);
        }
    }
    __syncthreads();                                                          // xs is dead from here on
    red[warp][r0][2 * tq] = d[0];
    red[warp][r0][2 * tq + 1] = d[1];
    red[warp][r1][2 * tq] = d[2];
    red[warp][r1][2 * tq + 1] = d[3];
    __syncthreads();
    if (tid < 16 * T) {
        const int r = tid / T, t = tid % T;
        if (r < nr) {
            float s = 0.f;
#pragma unroll
            for (int ww = 0; ww < WARPS; ++ww) s += red[ww][r][t];
            y[(size_t) t * N + n0 + r] = __float2bfloat16(s);
        }
    }
}

template <int T>
void launch(const at::Tensor & core, const at::Tensor & z, const at::Tensor & w_norm, const at::Tensor & w,
            at::Tensor & y, float eps, bool swish, int grid, cudaStream_t s, bool pdl, bool prefetch) {
    const int smem = std::max(T * STRIDE * 2, WARPS * 16 * 8 * 4);
    static bool attr = false;
    if (!attr) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_norm_oproj_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
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
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, gdn_norm_oproj_kernel<T>, (const __nv_bfloat16 *) core.data_ptr(),
                                      (const __nv_bfloat16 *) z.data_ptr(), (const __nv_bfloat16 *) w_norm.data_ptr(),
                                      (const __nv_bfloat16 *) w.data_ptr(), (__nv_bfloat16 *) y.data_ptr(), eps,
                                      swish ? 1 : 0, prefetch ? 1 : 0));
}

}  // namespace

// core, z: [T*48, 128] bf16 contiguous (T 1..8); w_norm [128] bf16; w [2560, 6144] bf16. Returns y [T, 2560] bf16.
at::Tensor gdn_norm_oproj(const at::Tensor & core, const at::Tensor & z, const at::Tensor & w_norm, double eps,
                          bool swish, const at::Tensor & w, bool pdl, bool prefetch) {
    TORCH_CHECK(core.is_cuda() && core.scalar_type() == at::kBFloat16 && core.dim() == 2 && core.size(1) == HD && core.is_contiguous());
    TORCH_CHECK(z.scalar_type() == at::kBFloat16 && z.sizes() == core.sizes() && z.is_contiguous());
    TORCH_CHECK(w_norm.scalar_type() == at::kBFloat16 && w_norm.numel() == HD && w_norm.is_contiguous());
    TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.dim() == 2 && w.size(0) == N && w.size(1) == K && w.is_contiguous());
    TORCH_CHECK(core.size(0) % HEADS == 0);
    const int T = core.size(0) / HEADS;
    TORCH_CHECK(T >= 1 && T <= MAXT);
    const c10::cuda::CUDAGuard guard(core.device());
    auto y = at::empty({T, N}, core.options());
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int grid = std::min(sms, N / 8);
    TORCH_CHECK((N + grid - 1) / grid <= 16, "too few SMs for gdn_norm_oproj");
    TORCH_CHECK(w.data_ptr() != nullptr && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0);
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    const float e = (float) eps;
    switch (T) {
        case 1: launch<1>(core, z, w_norm, w, y, e, swish, grid, s, pdl, prefetch); break;
        case 2: launch<2>(core, z, w_norm, w, y, e, swish, grid, s, pdl, prefetch); break;
        case 3: launch<3>(core, z, w_norm, w, y, e, swish, grid, s, pdl, prefetch); break;
        case 4: launch<4>(core, z, w_norm, w, y, e, swish, grid, s, pdl, prefetch); break;
        case 5: launch<5>(core, z, w_norm, w, y, e, swish, grid, s, pdl, prefetch); break;
        case 6: launch<6>(core, z, w_norm, w, y, e, swish, grid, s, pdl, prefetch); break;
        case 7: launch<7>(core, z, w_norm, w, y, e, swish, grid, s, pdl, prefetch); break;
        default: launch<8>(core, z, w_norm, w, y, e, swish, grid, s, pdl, prefetch); break;
    }
    return y;
}
