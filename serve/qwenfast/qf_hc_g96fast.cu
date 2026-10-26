// hc_mix_fused with group-scaled INT8 mix weights (w8g, see qf_w8g.cu): the same persistent kernel and math as
// qf_hc2.cu's hc_mix_fused, with
//   Wd [320, 10240] int8 + bf16 scales [320, 80] (groups of 128),   W_inject [4, 10240] kept BF16,
//   Wu [10240, 320] int8 + bf16 scales [10240, 5] (groups of 64).
// Each weight is expanded to bf16(q * s), the exact dequantized value, right before the tensor-core MMA, so the
// result equals hc_mix_fused run on those dequantized BF16 weights. Half the weight bytes of the BF16 kernel.
//
// hc_combine_mix_w8_g96fast: the previous block's combine (hc_combine_apply) fused in front: x = bf16(r + a * y) with
// a = 2 / (1 + exp(-inj_prev / hc)) is formed while normalizing and written out (the new hyper_input), so the
// mix chains (PDL) straight after the block's last kernel and its weight loads overlap that kernel's tail.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

#ifndef HC_EARLY_DQ
#define HC_EARLY_DQ 1
#endif
#ifndef HC_TRACE
#define HC_TRACE 0
#endif
__device__ __forceinline__ void hc_stamp(unsigned *bar, int phase) {
#if HC_TRACE
 if(threadIdx.x==0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory"); reinterpret_cast<unsigned long long*>(bar+512)[blockIdx.x*12+phase]=t; }
#endif
}


constexpr int HC = 4, HS = 2560, K = HC * HS, R = 320, NINJ = 4, NROWS = R + NINJ;
constexpr int THREADS = 256, WARPS = THREADS / 32, MAXM = 8;
constexpr int KQ_VEC = HS / 8;               // 320 uint4 (bf16) per branch row
constexpr int GD = 128, GU = 64;             // group sizes of Wd and Wu
constexpr int WD_NG = K / GD, WU_NG = R / GU; // 80, 5
constexpr int WU8_VEC = R / 16;              // 20 uint4 (int8) per Wu row
constexpr int WU8_VEC_PER_J = HC * WU8_VEC;  // 80 uint4 (4 rows) per output column
constexpr int MAXJ = 32;                     // output columns per CTA (needs >= 160 CTAs)

__device__ __forceinline__ void bf16x8_to_f32(const uint4 & v, float (&f)[8]) {
    const __nv_bfloat162 * h = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x; f[2 * i + 1] = t.y;
    }
}

// 8 int8 (two words) * scale -> 8 bf16 (bf16(q * s), round to nearest even)
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

__device__ __forceinline__ float bf16_bits(uint32_t b) { return __uint_as_float(b << 16); }

__device__ __forceinline__ unsigned ld_acquire(const unsigned * p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

constexpr int XS_STRIDE = HS + 32;        // bf16 elements per token row in shared memory
constexpr int WU8_STRIDE = WU8_VEC + 1;   // uint4 per int8 Wu row in shared memory (336 B)
constexpr int TS_STRIDE = R + 32;         // bf16 elements per token row of t

// rows of x kept in shared memory (MMA rows >= M read zeros from registers); t reuses the region with 8 rows
template <int M>
constexpr int xs_elems() { return (M * XS_STRIDE > 8 * TS_STRIDE) ? M * XS_STRIDE : 8 * TS_STRIDE; }

template <int M, bool COMB>
__global__ void __launch_bounds__(THREADS, 1) hc_mix_fused_w8_g96fast_kernel(
        const __nv_bfloat16 * __restrict__ x, const __nv_bfloat16 * __restrict__ w_norm,
        const int8_t * __restrict__ wd, const uint16_t * __restrict__ wd_s, const __nv_bfloat16 * __restrict__ winj,
        const int8_t * __restrict__ wu, const uint16_t * __restrict__ wu_s,
        __nv_bfloat16 * __restrict__ out, float * __restrict__ inj_out,
        __nv_bfloat16 * __restrict__ normed, float * __restrict__ partial, unsigned * __restrict__ bar,
        int rows, float eps, const __nv_bfloat16 * __restrict__ yb, const float * __restrict__ inj_prev,
        __nv_bfloat16 * __restrict__ xout) {
    hc_stamp(bar,0);
    extern __shared__ __align__(16) uint4 smem[];
    uint4 * wus = smem;                                                               // [MAXJ*4][WU8_STRIDE]
    uint16_t * wuss = reinterpret_cast<uint16_t *>(smem + MAXJ * HC * WU8_STRIDE);  // [MAXJ*4][WU_NG] (+pad)
    __nv_bfloat16 * xs = reinterpret_cast<__nv_bfloat16 *>(smem + MAXJ * HC * WU8_STRIDE + 80);   // [M][XS_STRIDE]
    __shared__ float s_red[WARPS][M];
    __shared__ float s_rstd[M];
    __shared__ float s_c[WARPS][16][8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, grp = lane >> 2, tq = lane & 3;
    const int G = gridDim.x, cta = blockIdx.x;
    const int nrows = winj != nullptr ? NROWS : R;

    // ---- weights first (independent of the previous kernel) ----
    const int per_q = G / HC, q = cta / per_q, qi = cta % per_q;
    uint4 wa[2][10];
#pragma unroll
    for(int rr=0;rr<2;++rr) {
        const int n_row=qi+(grp+rr*8)*per_q;
        const bool has_row=n_row<nrows, row8=n_row<R;
        const int kofs=q*HS+warp*320+tq*8;
        if(has_row && row8) {
            const int8_t *wrow=wd+(size_t)n_row*K+kofs;
#pragma unroll
            for(int kc=0;kc<10;++kc) {
                const uint2 v=__ldcs(reinterpret_cast<const uint2*>(wrow+kc*32));
                const uint32_t sb=__ldg(wd_s+(size_t)n_row*WD_NG+(q*HS+warp*320+kc*32)/GD);
                wa[rr][kc]=make_uint4(v.x,v.y,sb,0);
            }
        } else if(has_row) {
            const __nv_bfloat16 *wrow=winj+(size_t)(n_row-R)*K+kofs;
#pragma unroll
            for(int kc=0;kc<10;++kc) wa[rr][kc]=__ldcs(reinterpret_cast<const uint4*>(wrow+kc*32));
        } else {
#pragma unroll
            for(int kc=0;kc<10;++kc) wa[rr][kc]=make_uint4(0,0,0,0);
        }
    }
    const int j0 = (int) ((long) cta * HS / G), j1 = (int) ((long) (cta + 1) * HS / G), nj = j1 - j0;
    for (int c = tid; c < nj * WU8_VEC_PER_J; c += THREADS) {
        const int jl = c / WU8_VEC_PER_J, v = c % WU8_VEC_PER_J, g = v / WU8_VEC, k = v % WU8_VEC;
        const void * src = wu + ((size_t) (g * HS + j0 + jl)) * R + k * 16;
        const unsigned dst = (unsigned) __cvta_generic_to_shared(wus + (jl * HC + g) * WU8_STRIDE + k);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    for (int c = tid; c < nj * HC * WU_NG; c += THREADS) {
        const int r = c / WU_NG, gg = c % WU_NG, jl = r / HC, g = r % HC;
        wuss[r * WU_NG + gg] = __ldg(wu_s + ((size_t) (g * HS + j0 + jl)) * WU_NG + gg);
    }
    constexpr int NW=WARPS/M, NE=(2+NW-1)/NW, NV=M+NE;
    const int nm=warp/NW, ww=warp%NW;
    uint4 wn[NV];
#pragma unroll
    for(int v=0;v<NV;++v) {
        const int ow=ww+(v<M ? v : v-M)*NW;
        const int k=(v<M ? 0 : 256)+ow*32+lane;
        wn[v]=k<KQ_VEC ? __ldg(reinterpret_cast<const uint4*>(w_norm+q*HS)+k) : make_uint4(0,0,0,0);
    }
#if HC_EARLY_DQ
#pragma unroll
    for(int rr=0;rr<2;++rr) if(qi+(grp+rr*8)*per_q<R) {
#pragma unroll
        for(int kc=0;kc<10;++kc) wa[rr][kc]=i8x8_to_bf16(wa[rr][kc].x,wa[rr][kc].y,bf16_bits(wa[rr][kc].z));
    }
#endif
    hc_stamp(bar,1);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    hc_stamp(bar,2);

    // Each physical warp group owns one token, simulating the original eight
    // logical warps in exactly the same reduction order. Padding work disappears.
    {
        uint4 xv[NV];
#pragma unroll
        for(int v=0;v<NV;++v) {
            const int ow=ww+(v<M ? v : v-M)*NW;
            const int k=(v<M ? 0 : 256)+ow*32+lane;
            xv[v]=nm<rows && k<KQ_VEC ? __ldcg(reinterpret_cast<const uint4*>(x+(size_t)nm*K+q*HS)+k) : make_uint4(0,0,0,0);
        }
        if constexpr(COMB) {
            uint4 yv[NV];
            const float ip=nm<rows ? __ldcg(inj_prev+nm*NINJ+q) : 0.f;
#pragma unroll
            for(int v=0;v<NV;++v) {
                const int ow=ww+(v<M ? v : v-M)*NW;
                const int k=(v<M ? 0 : 256)+ow*32+lane;
                yv[v]=nm<rows && k<KQ_VEC ? __ldcg(reinterpret_cast<const uint4*>(yb+(size_t)nm*HS)+k) : make_uint4(0,0,0,0);
            }
            const float a=2.f/(1.f+expf(-ip/HC));
#pragma unroll
            for(int v=0;v<NV;++v) {
                const int ow=ww+(v<M ? v : v-M)*NW;
                const int k=(v<M ? 0 : 256)+ow*32+lane;
                if(k>=KQ_VEC) continue;
                float rf[8],yf[8];bf16x8_to_f32(xv[v],rf);bf16x8_to_f32(yv[v],yf);
                uint4 o;auto op=reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
                for(int e=0;e<4;++e) op[e]=__floats2bfloat162_rn(rf[2*e]+a*yf[2*e],rf[2*e+1]+a*yf[2*e+1]);
                xv[v]=o;
                if(nm<rows && k%per_q==qi) reinterpret_cast<uint4*>(xout+(size_t)nm*K+q*HS)[k]=o;
            }
        }
#pragma unroll
        for(int h=0;h<M;++h) {
            const int ow=ww+h*NW;
            float ss=0.f,f[8];
            bf16x8_to_f32(xv[h],f);
#pragma unroll
            for(int e=0;e<8;++e) ss+=f[e]*f[e];
            if(ow<2) {
                bf16x8_to_f32(xv[M+h],f);
#pragma unroll
                for(int e=0;e<8;++e) ss+=f[e]*f[e];
            }
#pragma unroll
            for(int o=16;o>0;o>>=1) ss+=__shfl_xor_sync(0xffffffff,ss,o);
            if(lane==0) s_red[ow][nm]=ss;
        }
        __syncthreads();
        if(tid<M) {
            float ss=0.f;
#pragma unroll
            for(int w=0;w<WARPS;++w) ss+=s_red[w][tid];
            s_rstd[tid]=rsqrtf(ss/HS+eps);
        }
        __syncthreads();
#pragma unroll
        for(int v=0;v<NV;++v) {
            const int ow=ww+(v<M ? v : v-M)*NW;
            const int k=(v<M ? 0 : 256)+ow*32+lane;
            if(k>=KQ_VEC) continue;
            float wf[8],f[8];bf16x8_to_f32(wn[v],wf);bf16x8_to_f32(xv[v],f);
            const float r=s_rstd[nm];
            uint4 o;auto op=reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
            for(int e=0;e<4;++e) op[e]=__floats2bfloat162_rn(f[2*e]*r*(1.f+wf[2*e]),f[2*e+1]*r*(1.f+wf[2*e+1]));
            reinterpret_cast<uint4*>(xs+nm*XS_STRIDE)[k]=o;
            if(k%per_q==qi && nm<rows) reinterpret_cast<uint4*>(normed+(size_t)nm*K+q*HS)[k]=o;
        }
        __syncthreads();
    }
    hc_stamp(bar,3);
    // ---- down projection (+ inject rows) on tensor cores ----
    {
#if !HC_EARLY_DQ
#pragma unroll
    for(int rr=0;rr<2;++rr) if(qi+(grp+rr*8)*per_q<R) {
#pragma unroll
        for(int kc=0;kc<10;++kc) wa[rr][kc]=i8x8_to_bf16(wa[rr][kc].x,wa[rr][kc].y,bf16_bits(wa[rr][kc].z));
    }
#endif
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        const bool brow = grp < M;
        const __nv_bfloat16 * xb = xs + (brow ? grp : 0) * XS_STRIDE + warp * 320 + tq * 8;
#pragma unroll
        for (int kc = 0; kc < 10; ++kc) {
            const uint4 bv = brow ? *reinterpret_cast<const uint4 *>(xb + kc * 32) : make_uint4(0, 0, 0, 0);
            mma16816(d, wa[0][kc].x, wa[1][kc].x, wa[0][kc].y, wa[1][kc].y, bv.x, bv.y);
            mma16816(d, wa[0][kc].z, wa[1][kc].z, wa[0][kc].w, wa[1][kc].w, bv.z, bv.w);
        }
        s_c[warp][grp][2 * tq] = d[0];
        s_c[warp][grp][2 * tq + 1] = d[1];
        s_c[warp][grp+8][2*tq]=d[2];
        s_c[warp][grp+8][2*tq+1]=d[3];
        __syncthreads();
        if (tid < 16 * M) {
            const int r = tid / M, m = tid % M, n = qi + r * per_q;
            if (n < nrows) {
                float v = 0.f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) v += s_c[w][r][m];
                partial[((size_t) q * MAXM + m) * NROWS + n] = v;
            }
        }
    }

    hc_stamp(bar,4);
    // ---- grid barrier ----
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        atomicAdd(bar, 1u);
        while (ld_acquire(bar) < (unsigned) G) { }
    }
    __syncthreads();
    hc_stamp(bar,5);
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // the up projection's gate multiplies normed[m, g*HS + j] (written before the barrier by other CTAs): load this
    // thread's four values now, alongside the partial sums, instead of after the MMA
    float nv[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
    if (warp < 8 && warp * 4 < nj) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rr = warp * 16 + grp + 8 * h, jl = rr >> 2, g = rr & 3;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int m = 2 * tq + c;
                if (rr < nj * HC && m < rows) nv[h][c] = __bfloat162float(__ldcg(normed + (size_t) m * K + g * HS + j0 + jl));
            }
        }
    }
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

    hc_stamp(bar,6);
    // ---- up projection on tensor cores: rows (jl, g) of Wu x [8 tokens]; warp w < 4 owns rows 16w..16w+15 ----
    if (warp < 8 && warp * 4 < nj) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        const int r0 = warp * 16 + grp, r1 = r0 + 8;
        const bool v0 = r0 < nj * HC, v1 = r1 < nj * HC;
        const uint8_t * a0p = reinterpret_cast<const uint8_t *>(wus + r0 * WU8_STRIDE) + tq * 8;
        const uint8_t * a1p = reinterpret_cast<const uint8_t *>(wus + r1 * WU8_STRIDE) + tq * 8;
        const __nv_bfloat16 * bp = ts + grp * TS_STRIDE + tq * 8;
#pragma unroll
        for (int kc = 0; kc < R / 32; ++kc) {
            uint4 av0 = make_uint4(0, 0, 0, 0), av1 = make_uint4(0, 0, 0, 0);
            if (v0) {
                const uint2 w = *reinterpret_cast<const uint2 *>(a0p + kc * 32);
                av0 = i8x8_to_bf16(w.x, w.y, bf16_bits(wuss[r0 * WU_NG + kc * 32 / GU]));
            }
            if (v1) {
                const uint2 w = *reinterpret_cast<const uint2 *>(a1p + kc * 32);
                av1 = i8x8_to_bf16(w.x, w.y, bf16_bits(wuss[r1 * WU_NG + kc * 32 / GU]));
            }
            const uint4 bv = *reinterpret_cast<const uint4 *>(bp + kc * 32);
            mma16816(d, av0.x, av1.x, av0.y, av1.y, bv.x, bv.y);
            mma16816(d, av0.z, av1.z, av0.w, av1.w, bv.z, bv.w);
        }
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
                    v = gate * nv[h][c];
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

    hc_stamp(bar,7);
    if (tid == 0) {
        const unsigned ticket = atomicAdd(bar + 1, 1u);
        if (ticket == (unsigned) G - 1) {
            bar[0] = 0;
            bar[1] = 0;
            __threadfence();
        }
    }
}

template <int M, bool COMB>
void launch_mix8(const at::Tensor & x, const at::Tensor & w_norm, const at::Tensor & wd, const at::Tensor & wd_s,
                 const at::Tensor * winj, const at::Tensor & wu, const at::Tensor & wu_s, at::Tensor & out, at::Tensor & inj,
                 at::Tensor & normed, at::Tensor & partial, at::Tensor & bar, int rows, float eps, int grid,
                 cudaStream_t s, bool pdl, const at::Tensor * y, const at::Tensor * inj_prev, at::Tensor * xout) {
    const int smem = (MAXJ * HC * WU8_STRIDE + 80) * 16 + xs_elems<M>() * 2;
    auto k = hc_mix_fused_w8_g96fast_kernel<M, COMB>;
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
                                      (const int8_t *) wd.data_ptr(), (const uint16_t *) wd_s.data_ptr(),
                                      winj ? (const __nv_bfloat16 *) winj->data_ptr() : (const __nv_bfloat16 *) nullptr,
                                      (const int8_t *) wu.data_ptr(), (const uint16_t *) wu_s.data_ptr(),
                                      (__nv_bfloat16 *) out.data_ptr(),
                                      winj ? inj.data_ptr<float>() : (float *) nullptr, (__nv_bfloat16 *) normed.data_ptr(),
                                      partial.data_ptr<float>(), reinterpret_cast<unsigned *>(bar.data_ptr<int>()), rows, eps,
                                      y ? (const __nv_bfloat16 *) y->data_ptr() : (const __nv_bfloat16 *) nullptr,
                                      inj_prev ? inj_prev->data_ptr<float>() : (float *) nullptr,
                                      xout ? (__nv_bfloat16 *) xout->data_ptr() : (__nv_bfloat16 *) nullptr));
}

template <bool COMB>
void dispatch_mix8(const at::Tensor & x, const at::Tensor & w_norm, const at::Tensor & wd, const at::Tensor & wd_s,
                   const at::Tensor * wi, const at::Tensor & wu, const at::Tensor & wu_s, at::Tensor & out, at::Tensor & inj,
                   at::Tensor & normed, at::Tensor & partial, at::Tensor & bar, int rows, float e, int grid,
                   cudaStream_t s, bool pdl, const at::Tensor * y, const at::Tensor * ip, at::Tensor * xo) {
    if (rows == 1) launch_mix8<1, COMB>(x, w_norm, wd, wd_s, wi, wu, wu_s, out, inj, normed, partial, bar, rows, e, grid, s, pdl, y, ip, xo);
    else if (rows == 2) launch_mix8<2, COMB>(x, w_norm, wd, wd_s, wi, wu, wu_s, out, inj, normed, partial, bar, rows, e, grid, s, pdl, y, ip, xo);
    else if (rows <= 4) launch_mix8<4, COMB>(x, w_norm, wd, wd_s, wi, wu, wu_s, out, inj, normed, partial, bar, rows, e, grid, s, pdl, y, ip, xo);
    else launch_mix8<8, COMB>(x, w_norm, wd, wd_s, wi, wu, wu_s, out, inj, normed, partial, bar, rows, e, grid, s, pdl, y, ip, xo);
}

void check_mix8(const at::Tensor & x, const at::Tensor & w_norm, const at::Tensor & wd, const at::Tensor & wd_s,
                const c10::optional<at::Tensor> & winj, const at::Tensor & wu, const at::Tensor & wu_s,
                const at::Tensor & normed, const at::Tensor & partial, const at::Tensor & bar) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == K && x.is_contiguous());
    TORCH_CHECK(w_norm.scalar_type() == at::kBFloat16 && w_norm.numel() == K && w_norm.is_contiguous());
    TORCH_CHECK(wd.scalar_type() == at::kChar && wd.size(0) == R && wd.size(1) == K && wd.is_contiguous());
    TORCH_CHECK(wd_s.scalar_type() == at::kBFloat16 && wd_s.size(0) == R && wd_s.size(1) == WD_NG && wd_s.is_contiguous());
    TORCH_CHECK(wu.scalar_type() == at::kChar && wu.size(0) == K && wu.size(1) == R && wu.is_contiguous());
    TORCH_CHECK(wu_s.scalar_type() == at::kBFloat16 && wu_s.size(0) == K && wu_s.size(1) == WU_NG && wu_s.is_contiguous());
    if (winj.has_value())
        TORCH_CHECK(winj->scalar_type() == at::kBFloat16 && winj->size(0) == NINJ && winj->size(1) == K && winj->is_contiguous());
    TORCH_CHECK(normed.scalar_type() == at::kBFloat16 && normed.numel() >= MAXM * K);
    TORCH_CHECK(partial.scalar_type() == at::kFloat && partial.numel() >= HC * MAXM * NROWS);
    TORCH_CHECK(bar.scalar_type() == at::kInt && bar.numel() >= 2);
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= MAXM);
}

}  // namespace

// As hc_mix_fused, with wd int8 [320, 10240] + wd_s bf16 [320, 80], wu int8 [10240, 320] + wu_s bf16 [10240, 5].
std::vector<at::Tensor> hc_mix_fused_w8_g96fast(const at::Tensor & x, const at::Tensor & w_norm, const at::Tensor & wd,
                                        const at::Tensor & wd_s, const c10::optional<at::Tensor> & winj,
                                        const at::Tensor & wu, const at::Tensor & wu_s, double eps,
                                        at::Tensor & normed, at::Tensor & partial, at::Tensor & bar, bool pdl) {
    check_mix8(x, w_norm, wd, wd_s, winj, wu, wu_s, normed, partial, bar);
    const int rows = x.size(0);
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty({rows, HS}, x.options());
    auto inj = at::empty({rows, NINJ}, x.options().dtype(at::kFloat));
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int grid = rows<=4 ? 96 : (sms/HC)*HC;
    TORCH_CHECK(grid * MAXJ >= HS && (grid / HC) * 16 >= NROWS, "too few SMs for hc_mix_fused_w8_g96fast");
    dispatch_mix8<false>(x, w_norm, wd, wd_s, winj.has_value() ? &winj.value() : nullptr, wu, wu_s, out, inj, normed,
                         partial, bar, rows, (float) eps, grid, at::cuda::getCurrentCUDAStream(), pdl, nullptr, nullptr, nullptr);
    return {out, inj};
}

// hc_combine_apply(y, r, inj_prev) followed by hc_mix_fused_w8_g96fast on its result, in one kernel. Returns
// {mix out [rows, 2560], inject dots [rows, 4], hyper_input [rows, 10240] (= the combine's output)}.
std::vector<at::Tensor> hc_combine_mix_w8_g96fast(const at::Tensor & y, const at::Tensor & r, const at::Tensor & inj_prev,
                                          const at::Tensor & w_norm, const at::Tensor & wd,
                                          const at::Tensor & wd_s, const c10::optional<at::Tensor> & winj,
                                          const at::Tensor & wu, const at::Tensor & wu_s, double eps,
                                          at::Tensor & normed, at::Tensor & partial, at::Tensor & bar, bool pdl) {
    check_mix8(r, w_norm, wd, wd_s, winj, wu, wu_s, normed, partial, bar);
    const int rows = r.size(0);
    TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.dim() == 2 && y.size(0) == rows && y.size(1) == HS && y.is_contiguous());
    TORCH_CHECK(inj_prev.scalar_type() == at::kFloat && inj_prev.numel() == rows * NINJ && inj_prev.is_contiguous());
    const c10::cuda::CUDAGuard guard(r.device());
    auto out = at::empty({rows, HS}, r.options());
    auto inj = at::empty({rows, NINJ}, r.options().dtype(at::kFloat));
    auto xo = at::empty({rows, K}, r.options());
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int grid = rows<=4 ? 96 : (sms/HC)*HC;
    TORCH_CHECK(grid * MAXJ >= HS && (grid / HC) * 16 >= NROWS, "too few SMs for hc_combine_mix_w8_g96fast");
    dispatch_mix8<true>(r, w_norm, wd, wd_s, winj.has_value() ? &winj.value() : nullptr, wu, wu_s, out, inj, normed,
                        partial, bar, rows, (float) eps, grid, at::cuda::getCurrentCUDAStream(), pdl, &y, &inj_prev, &xo);
    return {out, inj, xo};
}
