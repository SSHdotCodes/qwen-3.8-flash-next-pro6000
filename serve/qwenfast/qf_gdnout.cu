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

// pk12 (exact 12-bit packed bf16, see qf_pk.cu): 8 weights from 8 m8 bytes + 8 nibbles -> 4 bf16x2 words
__device__ __forceinline__ uint4 pk_decode8_bf16(uint2 m, uint32_t q, uint32_t base2) {
    const uint32_t qe = q & 0x0F0F0F0Fu, qo = (q >> 4) & 0x0F0F0F0Fu;
    uint32_t o[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t mw = p < 2 ? m.x : m.y;
        const uint32_t u = __byte_perm(mw, 0, (p & 1) == 0 ? 0x4140 : 0x4342);
        const uint32_t e = __byte_perm(qe, qo, 0x4040 | p | ((4 + p) << 8)) & 0x000F000Fu;
        o[p] = ((u & 0x00800080u) << 8) | (u & 0x007F007Fu) | ((e + base2) << 7);
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}

struct PkW {   // packed out_proj weight; m8 == nullptr selects the plain bf16 path
    const uint8_t * m8; const uint8_t * e4; const uint8_t * base; const int * ex_ptr; const int * ex_k; const float * ex_v;
    const int8_t * q8; const uint16_t * s8;   // w8g: int8 [2560, 6144] + bf16 group scales [2560, 48] (W8 path)
};

// 8 int8 (two words) * scale -> 8 bf16 (bf16(q * s), round to nearest even): the exact w8g dequantized weights
__device__ __forceinline__ uint4 i8x8_to_bf16(uint32_t lo, uint32_t hi, float sc) {
    float f[8];
    lo ^= 0x80808080u; hi ^= 0x80808080u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[i] = __int_as_float(__byte_perm(lo, 0x4B000000u, 0x7440 + i)) - 8388736.f;
        f[4 + i] = __int_as_float(__byte_perm(hi, 0x4B000000u, 0x7440 + i)) - 8388736.f;
    }
    uint4 o;
    __nv_bfloat162 * op = reinterpret_cast<__nv_bfloat162 *>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) op[i] = __floats2bfloat162_rn(f[2 * i] * sc, f[2 * i + 1] * sc);
    return o;
}

template <int T, bool PK, bool W8>
__global__ void __launch_bounds__(THREADS, 1) gdn_norm_oproj_kernel(
        const __nv_bfloat16 * __restrict__ core, const __nv_bfloat16 * __restrict__ z,
        const __nv_bfloat16 * __restrict__ w_norm, const __nv_bfloat16 * __restrict__ w, PkW pk,
        __nv_bfloat16 * __restrict__ y, float eps, int swish, int prefetch) {
    extern __shared__ __align__(16) uint4 smem[];
    __nv_bfloat16 * xs = reinterpret_cast<__nv_bfloat16 *>(smem);              // [T][STRIDE]
    float (*red)[16][8] = reinterpret_cast<float (*)[16][8]>(smem);           // reuses xs after the multiply
    float (*corr)[8] = reinterpret_cast<float (*)[8]>(reinterpret_cast<char *>(smem) + T * STRIDE * 2);  // pk exceptions
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, grp = lane >> 2, tq = lane & 3;
    const int G = gridDim.x, n0 = (int) ((long) blockIdx.x * N / G), n1 = (int) ((long) (blockIdx.x + 1) * N / G);
    const int nr = n1 - n0;

    // ---- weights first: this CTA's rows are one contiguous range; pull it into L2 ----
    if (prefetch && tid == 0) {
        if constexpr (W8) {
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(pk.q8 + (size_t) n0 * K), "r"((unsigned) (nr * K)) : "memory");
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(pk.s8 + (size_t) n0 * (K / 128)), "r"((unsigned) (nr * K / 64)) : "memory");
        } else if constexpr (PK) {
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(pk.m8 + (size_t) n0 * K), "r"((unsigned) (nr * K)) : "memory");
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(pk.e4 + (size_t) n0 * (K / 2)), "r"((unsigned) (nr * K / 2)) : "memory");
        } else {
            const __nv_bfloat16 * src = w + (size_t) n0 * K;
            const unsigned bytes = (unsigned) nr * K * 2;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src), "r"(bytes) : "memory");
        }
    }
    if constexpr (PK) {
        if (tid < 16 * 8) corr[tid >> 3][tid & 7] = 0.f;
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
    const int ra = n0 + (v0 ? r0 : 0), rb = n0 + (v1 ? r1 : 0);
    const uint4 * a0p = reinterpret_cast<const uint4 *>(w + (size_t) ra * K + kb);
    const uint4 * a1p = reinterpret_cast<const uint4 *>(w + (size_t) rb * K + kb);
    uint32_t ba = 0, bb = 0;
    if constexpr (PK) {
        ba = pk.base[ra]; ba |= ba << 16;
        bb = pk.base[rb]; bb |= bb << 16;
    }
    constexpr int KC = KW / 32, BATCH = 12;
#pragma unroll
    for (int kc0 = 0; kc0 < KC; kc0 += BATCH) {
        uint4 a0[BATCH], a1[BATCH];
        if constexpr (W8) {
            uint2 w0[BATCH], w1[BATCH];
            uint32_t s0[BATCH], s1[BATCH];
#pragma unroll
            for (int i = 0; i < BATCH; ++i) {
                const int k = kb + (kc0 + i) * 32;
                const int gk = (warp * KW + (kc0 + i) * 32) / 128;
                w0[i] = v0 ? __ldcs(reinterpret_cast<const uint2 *>(pk.q8 + (size_t) ra * K + k)) : make_uint2(0, 0);
                w1[i] = v1 ? __ldcs(reinterpret_cast<const uint2 *>(pk.q8 + (size_t) rb * K + k)) : make_uint2(0, 0);
                s0[i] = v0 ? __ldg(pk.s8 + (size_t) ra * (K / 128) + gk) : 0u;
                s1[i] = v1 ? __ldg(pk.s8 + (size_t) rb * (K / 128) + gk) : 0u;
            }
#pragma unroll
            for (int i = 0; i < BATCH; ++i) {
                a0[i] = v0 ? i8x8_to_bf16(w0[i].x, w0[i].y, __uint_as_float(s0[i] << 16)) : zero;
                a1[i] = v1 ? i8x8_to_bf16(w1[i].x, w1[i].y, __uint_as_float(s1[i] << 16)) : zero;
            }
        } else if constexpr (PK) {
            uint2 m0[BATCH], m1[BATCH];
            uint32_t q0[BATCH], q1[BATCH];
#pragma unroll
            for (int i = 0; i < BATCH; ++i) {
                const int k = kb + (kc0 + i) * 32;
                m0[i] = v0 ? __ldcs(reinterpret_cast<const uint2 *>(pk.m8 + (size_t) ra * K + k)) : make_uint2(0, 0);
                m1[i] = v1 ? __ldcs(reinterpret_cast<const uint2 *>(pk.m8 + (size_t) rb * K + k)) : make_uint2(0, 0);
                q0[i] = v0 ? __ldcs(reinterpret_cast<const uint32_t *>(pk.e4 + (size_t) ra * (K / 2) + k / 2)) : 0u;
                q1[i] = v1 ? __ldcs(reinterpret_cast<const uint32_t *>(pk.e4 + (size_t) rb * (K / 2) + k / 2)) : 0u;
            }
#pragma unroll
            for (int i = 0; i < BATCH; ++i) {
                a0[i] = v0 ? pk_decode8_bf16(m0[i], q0[i], ba) : zero;
                a1[i] = v1 ? pk_decode8_bf16(m1[i], q1[i], bb) : zero;
            }
        } else {
#pragma unroll
            for (int i = 0; i < BATCH; ++i) {
                a0[i] = v0 ? __ldcs(a0p + (kc0 + i) * 4) : zero;
                a1[i] = v1 ? __ldcs(a1p + (kc0 + i) * 4) : zero;
            }
        }
#pragma unroll
        for (int i = 0; i < BATCH; ++i) {
            const uint4 b = grp < T ? *reinterpret_cast<const uint4 *>(xs + (size_t) grp * STRIDE + kb + (kc0 + i) * 32) : zero;
            mma16816(d, a0[i].x, a1[i].x, a0[i].y, a1[i].y, b.x, b.y);
            mma16816(d, a0[i].z, a1[i].z, a0[i].w, a1[i].w, b.z, b.w);
        }
    }
    if constexpr (PK) {
        // exact exceptions (|w| < 2^-14 of the row max, decoded as d = 2^(base-127)): add (w - d) * xn
        for (int rr = warp; rr < nr; rr += WARPS) {
            const int n = n0 + rr;
            const float dd = __uint_as_float(((uint32_t) pk.base[n]) << 23);
            for (int i = pk.ex_ptr[n] + lane; i < pk.ex_ptr[n + 1]; i += 32) {
                const int k = pk.ex_k[i];
                const float v = pk.ex_v[i] - dd;
#pragma unroll
                for (int t = 0; t < T; ++t) atomicAdd(&corr[rr][t], v * __bfloat162float(xs[(size_t) t * STRIDE + k]));
            }
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
            if constexpr (PK) s += corr[r][t];
            y[(size_t) t * N + n0 + r] = __float2bfloat16(s);
        }
    }
}

template <int T, bool PK, bool W8>
void launch(const at::Tensor & core, const at::Tensor & z, const at::Tensor & w_norm, const __nv_bfloat16 * w, PkW pk,
            at::Tensor & y, float eps, bool swish, int grid, cudaStream_t s, bool pdl, bool prefetch) {
    const int smem = std::max(T * STRIDE * 2 + (PK ? 16 * 8 * 4 : 0), WARPS * 16 * 8 * 4);
    static bool attr = false;
    if (!attr) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_norm_oproj_kernel<T, PK, W8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
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
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, gdn_norm_oproj_kernel<T, PK, W8>, (const __nv_bfloat16 *) core.data_ptr(),
                                      (const __nv_bfloat16 *) z.data_ptr(), (const __nv_bfloat16 *) w_norm.data_ptr(),
                                      w, pk, (__nv_bfloat16 *) y.data_ptr(), eps, swish ? 1 : 0, prefetch ? 1 : 0));
}

}  // namespace

namespace {
template <bool PK, bool W8 = false>
at::Tensor run(const at::Tensor & core, const at::Tensor & z, const at::Tensor & w_norm, double eps, bool swish,
               const __nv_bfloat16 * w, PkW pk, bool pdl, bool prefetch) {
    TORCH_CHECK(core.is_cuda() && core.scalar_type() == at::kBFloat16 && core.dim() == 2 && core.size(1) == HD && core.is_contiguous());
    TORCH_CHECK(z.scalar_type() == at::kBFloat16 && z.sizes() == core.sizes() && z.is_contiguous());
    TORCH_CHECK(w_norm.scalar_type() == at::kBFloat16 && w_norm.numel() == HD && w_norm.is_contiguous());
    TORCH_CHECK(core.size(0) % HEADS == 0);
    const int T = core.size(0) / HEADS;
    TORCH_CHECK(T >= 1 && T <= MAXT);
    const c10::cuda::CUDAGuard guard(core.device());
    auto y = at::empty({T, N}, core.options());
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int grid = std::min(sms, N / 8);
    TORCH_CHECK((N + grid - 1) / grid <= 16, "too few SMs for gdn_norm_oproj");
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    const float e = (float) eps;
    switch (T) {
        case 1: launch<1, PK, W8>(core, z, w_norm, w, pk, y, e, swish, grid, s, pdl, prefetch); break;
        case 2: launch<2, PK, W8>(core, z, w_norm, w, pk, y, e, swish, grid, s, pdl, prefetch); break;
        case 3: launch<3, PK, W8>(core, z, w_norm, w, pk, y, e, swish, grid, s, pdl, prefetch); break;
        case 4: launch<4, PK, W8>(core, z, w_norm, w, pk, y, e, swish, grid, s, pdl, prefetch); break;
        case 5: launch<5, PK, W8>(core, z, w_norm, w, pk, y, e, swish, grid, s, pdl, prefetch); break;
        case 6: launch<6, PK, W8>(core, z, w_norm, w, pk, y, e, swish, grid, s, pdl, prefetch); break;
        case 7: launch<7, PK, W8>(core, z, w_norm, w, pk, y, e, swish, grid, s, pdl, prefetch); break;
        default: launch<8, PK, W8>(core, z, w_norm, w, pk, y, e, swish, grid, s, pdl, prefetch); break;
    }
    return y;
}
}  // namespace

// core, z: [T*48, 128] bf16 contiguous (T 1..8); w_norm [128] bf16; w [2560, 6144] bf16. Returns y [T, 2560] bf16.
at::Tensor gdn_norm_oproj(const at::Tensor & core, const at::Tensor & z, const at::Tensor & w_norm, double eps,
                          bool swish, const at::Tensor & w, bool pdl, bool prefetch) {
    TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.dim() == 2 && w.size(0) == N && w.size(1) == K && w.is_contiguous());
    TORCH_CHECK(w.data_ptr() != nullptr && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0);
    PkW pk{};
    return run<false>(core, z, w_norm, eps, swish, (const __nv_bfloat16 *) w.data_ptr(), pk, pdl, prefetch);
}

// same with the out_proj weight stored pk12 (m8 [2560, 6144], e4 [2560, 3072], base [2560], CSR exceptions)
at::Tensor gdn_norm_oproj_pk(const at::Tensor & core, const at::Tensor & z, const at::Tensor & w_norm, double eps,
                             bool swish, const at::Tensor & m8, const at::Tensor & e4, const at::Tensor & base,
                             const at::Tensor & ex_ptr, const at::Tensor & ex_k, const at::Tensor & ex_v, bool pdl, bool prefetch) {
    TORCH_CHECK(m8.scalar_type() == at::kByte && m8.size(0) == N && m8.size(1) == K && m8.is_contiguous());
    TORCH_CHECK(e4.scalar_type() == at::kByte && e4.size(0) == N && e4.size(1) == K / 2 && e4.is_contiguous());
    TORCH_CHECK(base.numel() == N && ex_ptr.numel() == N + 1 && ex_k.scalar_type() == at::kInt && ex_v.scalar_type() == at::kFloat);
    PkW pk{m8.data_ptr<uint8_t>(), e4.data_ptr<uint8_t>(), base.data_ptr<uint8_t>(), ex_ptr.data_ptr<int>(),
           ex_k.data_ptr<int>(), ex_v.data_ptr<float>()};
    return run<true>(core, z, w_norm, eps, swish, nullptr, pk, pdl, prefetch);
}

// same with the out_proj weight stored w8g (int8 q [2560, 6144], bf16 group scales s [2560, 48])
at::Tensor gdn_norm_oproj_w8(const at::Tensor & core, const at::Tensor & z, const at::Tensor & w_norm, double eps,
                             bool swish, const at::Tensor & q, const at::Tensor & sc, bool pdl, bool prefetch) {
    TORCH_CHECK(q.scalar_type() == at::kChar && q.size(0) == N && q.size(1) == K && q.is_contiguous());
    TORCH_CHECK(sc.scalar_type() == at::kBFloat16 && sc.size(0) == N && sc.size(1) == K / 128 && sc.is_contiguous());
    PkW pk{};
    pk.q8 = reinterpret_cast<const int8_t *>(q.data_ptr());
    pk.s8 = reinterpret_cast<const uint16_t *>(sc.data_ptr());
    return run<false, true>(core, z, w_norm, eps, swish, nullptr, pk, pdl, prefetch);
}
