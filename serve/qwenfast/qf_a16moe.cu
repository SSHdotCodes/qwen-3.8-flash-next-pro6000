// Decode-size (1..8 tokens) W4A16 routed experts for Qwen3.8-Flash-Next on the Marlin-packed NVFP4 weights the
// production loader already holds (no second copy). Same rounding points as SGLang's Marlin MoE path:
//   gemm1 = bf16(bf16(acc) * gs13)  ->  silu_and_mul  ->  gemm2 = bf16(bf16(acc) * bf16(gs2 * bf16(w_topk)))
//   -> moe_sum_reduce (fp32 sum over top-k positions, bf16)  ->  fused_gate_sigmoid_mul_add (shared expert)
// in two kernels instead of align + 2 GEMMs + activation + sum + gate:
//   gemm1: one CTA per (top-k slot, 64 gate columns + the matching 64 up columns). A slot that repeats an earlier
//          slot's expert exits, so each selected expert is read once. SwiGLU in the epilogue -> h[slot][token][640].
//   gemm2: one CTA per (top-k slot, 128 output columns). Weighted per-token outputs go to y[token][topk pos];
//          the last CTA of each column tile sums them in top-k order and adds sigmoid(x . w_gate) * shared.
// Weight fragments are read the way Marlin's m_block_size_8 kernel reads them: a 16(k) x 64(n) tile is 32 int4,
// one per lane; word j of lane (grp, tq) holds columns 16j+grp (nibbles 0,16,4,20 of q) and 16j+8+grp (nibbles
// 8,24,12,28) at k = 2tq, 2tq+1, 2tq+8, 2tq+9; the fp8 group scales of a lane group are 8 bytes at
// [group][64*ng + 8*grp], byte order (j0h0, j1h0, j0h1, j1h1, j2h0, j3h0, j2h1, j3h1). Weights are the A operand
// of m16n8k16 (16 columns), tokens are N = 8.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int H = 2560, I = 640, TOPK = 10, MAXT = 8, MAXSLOT = MAXT * TOPK;
constexpr int THREADS = 256, WARPS = 8;
constexpr int KT1 = H / 16, KT2 = I / 16;            // 160, 40 k-tiles
constexpr int ROW1 = (2 * I) / 2, ROW2 = H / 2;      // int4 per k-tile row: 640 (w13), 1280 (w2)
constexpr int NG1 = I / 64;                          // 10 gate (and up) column groups
constexpr int NT2 = H / 128;                         // 20 output tiles of gemm2
constexpr int XS1 = H + 8, XS2 = I + 8;              // bf16 row strides in shared memory (conflict-free B reads)
constexpr int D1 = 4, D2 = 5;                        // k-tiles prefetched per warp

__device__ __forceinline__ uint4 ldg_nc(const uint4 * p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 ldg_nc2(const uint2 * p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

// Marlin dequant<nv_bfloat162, kFE2M1f, skip_flop=true>: e2m1 nibbles -> bf16 bit patterns (scaled by 2^-126;
// the processed group/global scales carry the compensation). f[0] <- nibbles 8,24 ; f[1] <- nibbles 12,28.
__device__ __forceinline__ void dq_fp4(uint32_t q, uint32_t (&f)[2]) {
    const uint32_t o1 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
    q <<= 4;
    const uint32_t o2 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
    f[0] = o2; f[1] = o1;
}
// Marlin dequant_fp8_scales<nv_bfloat162, kFE4M3fn>: s[0] <- bytes 0,2 ; s[1] <- bytes 1,3.
__device__ __forceinline__ void dq_s8(uint32_t q, uint32_t & s0, uint32_t & s1) {
    const uint32_t o1 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
    q <<= 8;
    const uint32_t o2 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
    s0 = o2; s1 = o1;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
    __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162 *>(&a), *reinterpret_cast<__nv_bfloat162 *>(&b));
    return *reinterpret_cast<uint32_t *>(&r);
}
__device__ __forceinline__ uint32_t bdup_lo(uint32_t s) { return (s & 0xFFFFu) | (s << 16); }
__device__ __forceinline__ uint32_t bdup_hi(uint32_t s) { return (s >> 16) | (s & 0xFFFF0000u); }

__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One 16 x 64 weight tile (this lane's int4 w + its lane group's 8 scale bytes sc) against the token fragment
// (b0, b1); acc[j] = {C[16j+grp][2tq], C[16j+grp][2tq+1], C[16j+8+grp][2tq], C[16j+8+grp][2tq+1]}.
__device__ __forceinline__ void tile_mma(float (&acc)[4][4], const uint4 & w, const uint2 & sc, uint32_t b0, uint32_t b1) {
    uint32_t s[4];
    dq_s8(sc.x, s[0], s[1]);
    dq_s8(sc.y, s[2], s[3]);
    const uint32_t q[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t f0[2], f1[2];
        dq_fp4(q[j] << 8, f0);      // columns 16j + grp
        dq_fp4(q[j], f1);           // columns 16j + 8 + grp
        const uint32_t s0 = bdup_lo(s[j]), s1 = bdup_hi(s[j]);
        f0[0] = bmul2(f0[0], s0); f0[1] = bmul2(f0[1], s0);
        f1[0] = bmul2(f1[0], s1); f1[1] = bmul2(f1[1], s1);
        mma(acc[j], f0[0], f1[0], f0[1], f1[1], b0, b1);
    }
}

__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ void i8x4_to_f32_(uint32_t u, float * f) {    // as qf_w8g.cu (exact)
    u ^= 0x80808080u;
    f[0] = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7440)) - 8388736.f;
    f[1] = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7441)) - 8388736.f;
    f[2] = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7442)) - 8388736.f;
    f[3] = __int_as_float(__byte_perm(u, 0x4B000000u, 0x7443)) - 8388736.f;
}
__device__ __forceinline__ float bf16_bits(uint16_t b) { return __uint_as_float(((uint32_t) b) << 16); }

// Shared slot bookkeeping: ids in shared memory; first[s] = 1 when slot s is the first with its expert.
struct SlotInfo {
    int ids[MAXSLOT];
    unsigned char first[MAXSLOT];
    int n_unique;
};

__device__ __forceinline__ void load_slots(SlotInfo & si, const int * __restrict__ topk_ids, int nslots) {
    const int tid = threadIdx.x;
    if (tid < nslots) si.ids[tid] = topk_ids[tid];
    __syncthreads();
    if (tid < nslots) {
        const int e = si.ids[tid];
        bool f = true;
        for (int s = 0; s < tid; ++s) f &= si.ids[s] != e;
        si.first[tid] = f;
    }
    __syncthreads();
    if (tid < 32) {
        int n = 0;
        for (int s = tid; s < nslots; s += 32) n += si.first[s];
#pragma unroll
        for (int o = 16; o; o >>= 1) n += __shfl_xor_sync(0xffffffffu, n, o);
        if (tid == 0) si.n_unique = n;
    }
    __syncthreads();
}

// ------------------------------------------------------------------------------------------------------------
// gemm1: grid (nslots, NG1). Block (slot, g) computes gate columns [64g, 64g+64) and up columns [640+64g, ...)
// of expert ids[slot] for all tokens, SwiGLU -> h[slot][t][64g ...]. Block (0, 0) also computes the shared
// expert gate logits x . w_gate -> gl[t].
__global__ void __launch_bounds__(THREADS, 2) a16_gemm1_kernel(
        const __nv_bfloat16 * __restrict__ x, const int * __restrict__ topk_ids,
        const uint4 * __restrict__ w13, const uint8_t * __restrict__ s13, const uint16_t * __restrict__ gs13,
        const __nv_bfloat16 * __restrict__ w_gate, __nv_bfloat16 * __restrict__ hbuf, float * __restrict__ gl, int M) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16 * xs = reinterpret_cast<__nv_bfloat16 *>(smem_raw);                       // [8][XS1]
    float * red = reinterpret_cast<float *>(smem_raw);       // [WARPS][32][33], reuses xs after the k loop
    __shared__ SlotInfo si;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, grp = lane >> 2, tq = lane & 3;
    const int slot = blockIdx.x, g = blockIdx.y, nslots = M * TOPK;

    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (slot == nslots) {               // extra CTA: shared-expert gate logits (fp32 dot, as fused_gate_sigmoid_mul_add)
        if (g != 0) return;
        for (int t = warp; t < M; t += WARPS) {
            float a = 0.f;
            for (int k = lane; k < H; k += 32) a += __bfloat162float(x[(size_t) t * H + k]) * __bfloat162float(w_gate[k]);
#pragma unroll
            for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) gl[t] = a;
        }
        return;
    }
    load_slots(si, topk_ids, nslots);
    if (!si.first[slot]) return;
    const int e = si.ids[slot];

    // weights: warp w owns k-tiles w, w+8, ...; gate group g and up group NG1+g of each.
    const uint4 * wb = w13 + (size_t) e * KT1 * ROW1 + lane;
    const uint8_t * sb = s13 + (size_t) e * KT1 * (2 * I) + grp * 8;
    uint4 wg[D1], wu[D1];
    uint2 sg[D1], su[D1];
#pragma unroll
    for (int d = 0; d < D1; ++d) {
        const int kt = warp + d * WARPS;
        wg[d] = ldg_nc(wb + kt * ROW1 + g * 32);
        wu[d] = ldg_nc(wb + kt * ROW1 + (NG1 + g) * 32);
        sg[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * (2 * I) + g * 64));
        su[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * (2 * I) + (NG1 + g) * 64));
    }
    // tokens -> shared memory (rows >= M zero)
    for (int c = tid; c < MAXT * (H / 8); c += THREADS) {
        const int t = c / (H / 8), v = c % (H / 8);
        uint4 val = make_uint4(0, 0, 0, 0);
        if (t < M) val = *reinterpret_cast<const uint4 *>(x + (size_t) t * H + v * 8);
        *reinterpret_cast<uint4 *>(xs + t * XS1 + v * 8) = val;
    }
    __syncthreads();

    float ag[4][4], au[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) ag[j][i] = au[j][i] = 0.f;

    const uint32_t * xr = reinterpret_cast<const uint32_t *>(xs + grp * XS1) + tq;
#pragma unroll 4
    for (int it = 0; it < KT1 / WARPS; ++it) {
        const int d = it % D1;
        const int kt = warp + it * WARPS;
        const uint32_t b0 = xr[kt * 8], b1 = xr[kt * 8 + 4];
        const uint4 cwg = wg[d], cwu = wu[d];
        const uint2 csg = sg[d], csu = su[d];
        const int nk = kt + D1 * WARPS;
        if (nk < KT1) {
            wg[d] = ldg_nc(wb + nk * ROW1 + g * 32);
            wu[d] = ldg_nc(wb + nk * ROW1 + (NG1 + g) * 32);
            sg[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + nk * (2 * I) + g * 64));
            su[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + nk * (2 * I) + (NG1 + g) * 64));
        }
        tile_mma(ag, cwg, csg, b0, b1);
        tile_mma(au, cwu, csu, b0, b1);
    }

    // cross-warp reduction: red[warp][lane][0..15] gate, [16..31] up
    __syncthreads();
    float * r = red + (warp * 32 + lane) * 33;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) { r[j * 4 + i] = ag[j][i]; r[16 + j * 4 + i] = au[j][i]; }
    __syncthreads();
    const float gs = bf16_bits(gs13[e]);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int idx = tid * 2 + p;            // 0..511: (lane, j, i)
        const int ln = idx >> 4, ji = idx & 15, j = ji >> 2, i = ji & 3;
        float sg_ = 0.f, su_ = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) { sg_ += red[(w * 32 + ln) * 33 + ji]; su_ += red[(w * 32 + ln) * 33 + 16 + ji]; }
        const int lg = ln >> 2, lq = ln & 3;
        const int n = 16 * j + lg + ((i & 2) ? 8 : 0);
        const int t = 2 * lq + (i & 1);
        if (t < M) {
            const float gv = bf16r(bf16r(sg_) * gs), uv = bf16r(bf16r(su_) * gs);
            const float hv = gv / (1.f + __expf(-gv)) * uv;
            hbuf[((size_t) slot * MAXT + t) * I + g * 64 + n] = __float2bfloat16(hv);
        }
    }
}

// ------------------------------------------------------------------------------------------------------------
// gemm2: grid (nslots, NT2). Block (slot, c) computes output columns [128c, 128c+128) of expert ids[slot] for its
// routed tokens, y[t][kpos] = bf16(bf16(acc) * bf16(gs2 * bf16(w))); the last block of column tile c writes
// out[t] = bf16(sigmoid(gl[t]) * shared[t] + bf16(sum_kpos y[t][kpos])).
__global__ void __launch_bounds__(THREADS, 2) a16_gemm2_kernel(
        const int * __restrict__ topk_ids, const float * __restrict__ topk_w,
        const uint4 * __restrict__ w2, const uint8_t * __restrict__ s2, const uint16_t * __restrict__ gs2,
        const __nv_bfloat16 * __restrict__ hbuf, const float * __restrict__ gl,
        const __nv_bfloat16 * __restrict__ shared, __nv_bfloat16 * __restrict__ ybuf,
        unsigned * __restrict__ counters, __nv_bfloat16 * __restrict__ out, int M) {
    __shared__ __align__(16) __nv_bfloat16 hs[MAXT * XS2];
    __shared__ float red[WARPS][32][33];
    __shared__ SlotInfo si;
    __shared__ unsigned s_last;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, grp = lane >> 2, tq = lane & 3;
    const int slot = blockIdx.x, c = blockIdx.y, nslots = M * TOPK;

    // topk ids were final before gemm1 started, so the weights can be fetched before waiting on gemm1.
    load_slots(si, topk_ids, nslots);
    if (!si.first[slot]) return;
    const int e = si.ids[slot];
    const uint4 * wb = w2 + (size_t) e * KT2 * ROW2 + lane;
    const uint8_t * sb = s2 + (size_t) e * KT2 * H + grp * 8;
    uint4 wa[D2], wc[D2];
    uint2 sa[D2], sc[D2];
#pragma unroll
    for (int d = 0; d < D2; ++d) {
        const int kt = warp + d * WARPS;          // 5 k-tiles per warp = all of them
        wa[d] = ldg_nc(wb + kt * ROW2 + (2 * c) * 32);
        wc[d] = ldg_nc(wb + kt * ROW2 + (2 * c + 1) * 32);
        sa[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * H + (2 * c) * 64));
        sc[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * H + (2 * c + 1) * 64));
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int q = tid; q < MAXT * (I / 8); q += THREADS) {
        const int t = q / (I / 8), v = q % (I / 8);
        uint4 val = make_uint4(0, 0, 0, 0);
        if (t < M) val = __ldcg(reinterpret_cast<const uint4 *>(hbuf + ((size_t) slot * MAXT + t) * I + v * 8));
        *reinterpret_cast<uint4 *>(hs + t * XS2 + v * 8) = val;
    }
    __syncthreads();

    float aa[4][4], ac[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) aa[j][i] = ac[j][i] = 0.f;
    const uint32_t * hr = reinterpret_cast<const uint32_t *>(hs + grp * XS2) + tq;
#pragma unroll
    for (int d = 0; d < D2; ++d) {
        const int kt = warp + d * WARPS;
        const uint32_t b0 = hr[kt * 8], b1 = hr[kt * 8 + 4];
        tile_mma(aa, wa[d], sa[d], b0, b1);
        tile_mma(ac, wc[d], sc[d], b0, b1);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) { red[warp][lane][j * 4 + i] = aa[j][i]; red[warp][lane][16 + j * 4 + i] = ac[j][i]; }
    __syncthreads();

    // epilogue: (lane, half, j, i) -> column, token; per routed token write y[t][kpos]
    const float gsv = bf16_bits(gs2[e]);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int idx = tid * 4 + p;            // 0..1023: (lane, half, j, i)
        const int ln = idx >> 5, hj = idx & 31, half = hj >> 4, ji = hj & 15, j = ji >> 2, i = ji & 3;
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) s += red[w][ln][hj];
        const int lg = ln >> 2, lq = ln & 3;
        const int n = 128 * c + 64 * half + 16 * j + lg + ((i & 2) ? 8 : 0);
        const int t = 2 * lq + (i & 1);
        if (t < M) {
#pragma unroll
            for (int kp = 0; kp < TOPK; ++kp) {
                if (si.ids[t * TOPK + kp] == e) {
                    const float tw = bf16r(gsv * bf16r(topk_w[t * TOPK + kp]));
                    ybuf[((size_t) t * TOPK + kp) * H + n] = __float2bfloat16(bf16r(s) * tw);
                }
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(counters + c, 1u) == (unsigned) (si.n_unique - 1);
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    for (int q = tid; q < M * 128; q += THREADS) {
        const int t = q / 128, n = 128 * c + q % 128;
        float acc = 0.f;
#pragma unroll
        for (int kp = 0; kp < TOPK; ++kp) acc += __bfloat162float(__ldcg(ybuf + ((size_t) t * TOPK + kp) * H + n));
        const float routed = bf16r(acc);
        float v = routed;
        if (shared != nullptr) {
            const float gate = 1.f / (1.f + __expf(-__ldcg(gl + t)));
            v = gate * __bfloat162float(shared[(size_t) t * H + n]) + routed;
        }
        out[(size_t) t * H + n] = __float2bfloat16(v);
    }
    if (tid == 0) counters[c] = 0u;
}


// ------------------------------------------------------------------------------------------------------------
// Persistent single-kernel version: CTAs pull work items in a fixed order (first item = blockIdx.x, then a queue)
//   item 0                    : shared-expert gate logits gl[t] (then ctrl[GLF] = 1)
//   [shared expert, when its w8g weights are given: 80 items gate/up rows (8 gate + matching 8 up each,
//    w8g_gemm arithmetic, SwiGLU -> hs; ctrl[SDONE] += 1)]
//   1 .. U*NG1*2              : gemm1 half (unique expert u, column group g, k half kh); expert-major order.
//                               The second half to finish for (u, g) adds the other's partial (pbuf), applies
//                               SwiGLU -> h[u], then done1[u] += 1.
//   [shared expert: 40 items, 64 down rows each (half of tile c; wait ctrl[SDONE] == 80) -> sbuf; count toward cnt[c]]
//   .. + U*NT2                : gemm2 (u, column tile c): weights fetched first, then wait done1[u] == NG1;
//                               the last item of tile c (cnt[c] == U (+1 with the shared expert)) sums y and adds
//                               sigmoid(gl) * shared (from sbuf, or the precomputed shared tensor).
// Items are handed out in order, so an item only ever waits on items already held by running CTAs.
// ctrl (int32): [0] queue, [1] exited CTAs, [2] gl flag, [8, 28) tile counters, [32, 112) done1, [128, 928) half
// counters per (u, g).
constexpr int QC = 0, XC = 1, GLF = 2, SDONE = 3, CNT = 8, DONE = 32, HALF = 128;
constexpr int SH1 = 80, SH2 = 2 * NT2;           // shared-expert items: 8 gate + 8 up rows; 64 down rows (half tile)
constexpr int KH = KT1 / 2, DH = 5;                  // 80 k-tiles per half; 10 per warp; prefetch depth 5

__device__ __forceinline__ unsigned ld_acq(const unsigned * p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__global__ void __launch_bounds__(THREADS, 2) a16_moe_persist_kernel(
        const __nv_bfloat16 * __restrict__ x, const int * __restrict__ topk_ids, const float * __restrict__ topk_w,
        const uint4 * __restrict__ w13, const uint8_t * __restrict__ s13, const uint16_t * __restrict__ gs13,
        const uint4 * __restrict__ w2, const uint8_t * __restrict__ s2, const uint16_t * __restrict__ gs2,
        const __nv_bfloat16 * __restrict__ w_gate, const __nv_bfloat16 * __restrict__ shared,
        __nv_bfloat16 * __restrict__ hbuf, __nv_bfloat16 * __restrict__ ybuf, float * __restrict__ gl,
        float * __restrict__ pbuf, unsigned * __restrict__ ctrl, __nv_bfloat16 * __restrict__ out, int M,
        long long * __restrict__ dbg,
        const int8_t * __restrict__ shq1, const uint16_t * __restrict__ shs1,     // [1280, 2560], [1280, 20]
        const int8_t * __restrict__ shq2, const uint16_t * __restrict__ shs2,     // [2560, 640], [2560, 5]
        __nv_bfloat16 * __restrict__ shbuf) {                                       // hs [8][640] + sbuf [8][2560]
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16 * xs = reinterpret_cast<__nv_bfloat16 *>(smem_raw);     // gemm1: [8][XS1/2]; gemm2: [8][XS2]
    float * red = reinterpret_cast<float *>(smem_raw);                    // after the k loop: [WARPS][32][33]
    int dn = 0;
    auto stamp = [&](long long tag) {
        if (dbg != nullptr && threadIdx.x == 0 && dn < 30) {
            long long tm; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tm));
            dbg[blockIdx.x * 32 + dn++] = (tm << 12) | tag;
        }
    };
    __shared__ SlotInfo si;
    __shared__ int ulist[MAXSLOT];
    __shared__ int s_item;
    __shared__ unsigned s_last;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, grp = lane >> 2, tq = lane & 3;
    const int nslots = M * TOPK;
    constexpr int XSH = KH * 16 + 8;                                       // half-K token row stride

    stamp(4095);
    const bool has_sh = shq1 != nullptr;
    const int ns1 = has_sh ? SH1 : 0, ns2 = has_sh ? SH2 : 0;
    // item ranges: [0] gl | [b1, +ns1) shared gate/up | [b3, +ns2) shared down | [b2, +n1) gemm1 | [b4, +n2) gemm2
    // (the shared expert goes first: its down items then sit in the first round with their weights in flight)
    const int b1 = 1, b3 = b1 + ns1, b2 = b3 + ns2;
    int U = 0, b4 = 0, bend = 0;
    unsigned tile_target = 0u;
    // The previous kernel (router) is waited for only when an item needs its output or x: shared-expert items
    // request their weights first, so those loads overlap the router's tail. Every path waits before it touches
    // ctrl (counters are re-armed by the previous launch's last CTA), x, topk or any scratch. Routing (U, ulist,
    // later ranges) is loaded once per CTA, by the first item that needs it.
    bool waited = false, routed = false;
    auto ensure_wait = [&]() {
        if (waited) return;
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        waited = true;
        stamp(4094);
    };
    auto ensure_routing = [&]() {
        if (routed) return;
        ensure_wait();
        load_slots(si, topk_ids, nslots);
        if (warp == 0) {                       // ulist = experts of first slots, in slot order (warp prefix sum)
            int base = 0;
            for (int s0 = 0; s0 < nslots; s0 += 32) {
                const int s = s0 + lane;
                const int f = s < nslots ? si.first[s] : 0;
                int incl = f;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += v; }
                if (f) ulist[base + incl - 1] = si.ids[s];
                base += __shfl_sync(0xffffffffu, incl, 31);
            }
        }
        __syncthreads();
        U = si.n_unique;
        b4 = b2 + U * NG1 * 2;
        bend = b4 + U * NT2;
        tile_target = (unsigned) (U - 1 + (has_sh ? 2 : 0));
        routed = true;
    };
    __nv_bfloat16 * hsh = shbuf;                 // [8][640]
    __nv_bfloat16 * sbuf = shbuf + MAXT * I;     // [8][2560]
    // last arrival at column tile c: out = bf16(sigmoid(gl) * shared + bf16(sum over top-k positions of y))
    auto final_tile = [&](int c) {
        if (tid == 0) {
            while (ld_acq(ctrl + GLF) == 0u) __nanosleep(32);
        }
        __threadfence();
        __syncthreads();
        for (int q = tid; q < M * 128; q += THREADS) {
            const int t = q / 128, n = 128 * c + q % 128;
            float acc = 0.f;
#pragma unroll
            for (int kp = 0; kp < TOPK; ++kp) acc += __bfloat162float(__ldcg(ybuf + ((size_t) t * TOPK + kp) * H + n));
            const float routed = bf16r(acc);
            float v = routed;
            if (has_sh) {
                const float gate = 1.f / (1.f + __expf(-__ldcg(gl + t)));
                v = gate * __bfloat162float(__ldcg(sbuf + t * H + n)) + routed;
            } else if (shared != nullptr) {
                const float gate = 1.f / (1.f + __expf(-__ldcg(gl + t)));
                v = gate * __bfloat162float(shared[(size_t) t * H + n]) + routed;
            }
            out[(size_t) t * H + n] = __float2bfloat16(v);
        }
        if (tid == 0) ctrl[CNT + c] = 0u;
        stamp(4090);
    };

    int item = blockIdx.x;
    while (true) {
        stamp(item < 4000 ? item : 3999);
        if (item == 0) {
            ensure_wait();
            for (int t = warp; t < M; t += WARPS) {
                float a = 0.f;
                for (int k = lane; k < H; k += 32) a += __bfloat162float(x[(size_t) t * H + k]) * __bfloat162float(w_gate[k]);
#pragma unroll
                for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                if (lane == 0) gl[t] = a;
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) atomicExch(ctrl + GLF, 1u);
        } else if (item >= b1 && item < b3) {      // shared gate/up
            // ---------------- shared expert gate/up rows (w8g_gemm arithmetic) ----------------
            // item s: warp w computes gate row 8s + w and up row 640 + 8s + w
            const int s = item - b1;
            const int rows[2] = {8 * s + warp, I + 8 * s + warp};
            uint4 wv[2][5];
            uint16_t sv[2][5];
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int u2 = 0; u2 < 5; ++u2) {
                    const int k0 = u2 * 512 + lane * 16;
                    wv[r][u2] = __ldcs(reinterpret_cast<const uint4 *>(shq1 + (size_t) rows[r] * H + k0));
                    sv[r][u2] = __ldg(shs1 + rows[r] * (H / 128) + k0 / 128);
                }
            ensure_wait();
            for (int c = tid; c < MAXT * (H / 8); c += THREADS) {
                const int tt = c / (H / 8), v = c % (H / 8);
                uint4 val = make_uint4(0, 0, 0, 0);
                if (tt < M) val = *reinterpret_cast<const uint4 *>(x + (size_t) tt * H + v * 8);
                *reinterpret_cast<uint4 *>(xs + tt * H + v * 8) = val;
            }
            __syncthreads();
            float acc[2][MAXT];
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int m = 0; m < MAXT; ++m) acc[r][m] = 0.f;
#pragma unroll
            for (int u2 = 0; u2 < 5; ++u2) {
                const int k0 = u2 * 512 + lane * 16;
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    float wf[16];
                    const uint32_t * ww = reinterpret_cast<const uint32_t *>(&wv[r][u2]);
#pragma unroll
                    for (int i = 0; i < 4; ++i) i8x4_to_f32_(ww[i], wf + 4 * i);
                    const float sc = bf16_bits(sv[r][u2]);
#pragma unroll
                    for (int m = 0; m < MAXT; ++m) {
                        if (m < M) {
                            float pp = 0.f;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const uint2 xv = *reinterpret_cast<const uint2 *>(xs + m * H + k0 + 4 * i);
                                pp = fmaf(wf[4 * i], __uint_as_float(xv.x << 16), pp);
                                pp = fmaf(wf[4 * i + 1], __uint_as_float(xv.x & 0xFFFF0000u), pp);
                                pp = fmaf(wf[4 * i + 2], __uint_as_float(xv.y << 16), pp);
                                pp = fmaf(wf[4 * i + 3], __uint_as_float(xv.y & 0xFFFF0000u), pp);
                            }
                            acc[r][m] = fmaf(pp, sc, acc[r][m]);
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int m = 0; m < MAXT; ++m) {
                    float v = acc[r][m];
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                    acc[r][m] = v;
                }
            if (lane == 0) {
#pragma unroll
                for (int m = 0; m < MAXT; ++m)
                    if (m < M) {
                        const float gv = bf16r(acc[0][m]), uv = bf16r(acc[1][m]);
                        hsh[m * I + 8 * s + warp] = __float2bfloat16(gv / (1.f + __expf(-gv)) * uv);
                    }
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) atomicAdd(ctrl + SDONE, 1u);
            stamp(4088);
        } else if (item >= b3 && item < b2) {      // shared down
            // ---------------- shared expert down rows of tile c (w8g_gemm arithmetic, K = 640) ----------------
            // item (c, half): warp w rows 128c + 64 half + 8w + [0, 8) in 2 blocks of 4; block 1 loads while block 0 computes
            const int c = (item - b3) >> 1, hf = (item - b3) & 1;
            const int nw = 128 * c + 64 * hf + 8 * warp;
            uint32_t wv[2][4][5];
            uint16_t sv[2][4][5];
            auto load_blk = [&](int bi, int slot) {
#pragma unroll
                for (int r = 0; r < 4; ++r)
#pragma unroll
                    for (int u2 = 0; u2 < 5; ++u2) {
                        const int k0 = u2 * 128 + lane * 4;
                        const int n = nw + 4 * bi + r;
                        wv[slot][r][u2] = __ldcs(reinterpret_cast<const uint32_t *>(shq2 + (size_t) n * I + k0));
                        sv[slot][r][u2] = __ldg(shs2 + n * (I / 128) + k0 / 128);
                    }
            };
            load_blk(0, 0);
            ensure_wait();          // ctrl belongs to this launch only once the previous kernel has completed
            if (tid == 0) {
                while (ld_acq(ctrl + SDONE) < (unsigned) SH1) __nanosleep(32);
            }
            __syncthreads();
            stamp(4087);
            for (int q = tid; q < MAXT * (I / 8); q += THREADS) {
                const int tt = q / (I / 8), v = q % (I / 8);
                uint4 val = make_uint4(0, 0, 0, 0);
                if (tt < M) val = __ldcg(reinterpret_cast<const uint4 *>(hsh + tt * I + v * 8));
                *reinterpret_cast<uint4 *>(xs + tt * I + v * 8) = val;
            }
            __syncthreads();
#pragma unroll
            for (int bi = 0; bi < 2; ++bi) {
                const int cur = bi & 1;
                if (bi + 1 < 2) load_blk(bi + 1, cur ^ 1);
                float acc[4][MAXT];
#pragma unroll
                for (int r = 0; r < 4; ++r)
#pragma unroll
                    for (int m = 0; m < MAXT; ++m) acc[r][m] = 0.f;
#pragma unroll
                for (int u2 = 0; u2 < 5; ++u2) {
                    const int k0 = u2 * 128 + lane * 4;
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        float wf[4];
                        i8x4_to_f32_(wv[cur][r][u2], wf);
                        const float sc = bf16_bits(sv[cur][r][u2]);
#pragma unroll
                        for (int m = 0; m < MAXT; ++m) {
                            if (m < M) {
                                const uint2 xv = *reinterpret_cast<const uint2 *>(xs + m * I + k0);
                                float pp = 0.f;
                                pp = fmaf(wf[0], __uint_as_float(xv.x << 16), pp);
                                pp = fmaf(wf[1], __uint_as_float(xv.x & 0xFFFF0000u), pp);
                                pp = fmaf(wf[2], __uint_as_float(xv.y << 16), pp);
                                pp = fmaf(wf[3], __uint_as_float(xv.y & 0xFFFF0000u), pp);
                                acc[r][m] = fmaf(pp, sc, acc[r][m]);
                            }
                        }
                    }
                }
#pragma unroll
                for (int r = 0; r < 4; ++r)
#pragma unroll
                    for (int m = 0; m < MAXT; ++m) {
                        float v = acc[r][m];
#pragma unroll
                        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                        acc[r][m] = v;
                    }
                if (lane == 0) {
#pragma unroll
                    for (int r = 0; r < 4; ++r)
#pragma unroll
                        for (int m = 0; m < MAXT; ++m)
                            if (m < M) sbuf[m * H + nw + 4 * bi + r] = __float2bfloat16(acc[r][m]);
                }
            }
            ensure_routing();
            __threadfence();
            __syncthreads();
            if (tid == 0) s_last = atomicAdd(ctrl + CNT + c, 1u) == tile_target;
            __syncthreads();
            if (s_last) final_tile(c);
        } else if ((ensure_routing(), item >= b2 && item < b4)) {      // gemm1
            // ---------------- gemm1 half item ----------------
            const int it = item - b2, u = it / (2 * NG1), g = (it >> 1) % NG1, kh = it & 1, e = ulist[u];
            const int k0 = kh * KH;
            const uint4 * wb = w13 + (size_t) e * KT1 * ROW1 + lane;
            const uint8_t * sb = s13 + (size_t) e * KT1 * (2 * I) + grp * 8;
            uint4 wg[DH], wu[DH];
            uint2 sg[DH], su[DH];
#pragma unroll
            for (int d = 0; d < DH; ++d) {
                const int kt = k0 + warp + d * WARPS;
                wg[d] = ldg_nc(wb + kt * ROW1 + g * 32);
                wu[d] = ldg_nc(wb + kt * ROW1 + (NG1 + g) * 32);
                sg[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * (2 * I) + g * 64));
                su[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * (2 * I) + (NG1 + g) * 64));
            }
            for (int c = tid; c < MAXT * (KH * 2); c += THREADS) {            // half of each token row: KH*16 bf16
                const int t = c / (KH * 2), v = c % (KH * 2);
                uint4 val = make_uint4(0, 0, 0, 0);
                if (t < M) val = *reinterpret_cast<const uint4 *>(x + (size_t) t * H + k0 * 16 + v * 8);
                *reinterpret_cast<uint4 *>(xs + t * XSH + v * 8) = val;
            }
            __syncthreads();
            float ag[4][4], au[4][4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) ag[j][i] = au[j][i] = 0.f;
            const uint32_t * xr = reinterpret_cast<const uint32_t *>(xs + grp * XSH) + tq;
#pragma unroll
            for (int i2 = 0; i2 < KH / WARPS; ++i2) {
                const int d = i2 % DH;
                const int ktl = warp + i2 * WARPS;                 // k-tile within the half
                const uint32_t b0 = xr[ktl * 8], b1 = xr[ktl * 8 + 4];
                const uint4 cwg = wg[d], cwu = wu[d];
                const uint2 csg = sg[d], csu = su[d];
                const int nk = ktl + DH * WARPS;
                if (nk < KH) {
                    const int kt = k0 + nk;
                    wg[d] = ldg_nc(wb + kt * ROW1 + g * 32);
                    wu[d] = ldg_nc(wb + kt * ROW1 + (NG1 + g) * 32);
                    sg[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * (2 * I) + g * 64));
                    su[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * (2 * I) + (NG1 + g) * 64));
                }
                tile_mma(ag, cwg, csg, b0, b1);
                tile_mma(au, cwu, csu, b0, b1);
            }
            __syncthreads();
            float * r = red + (warp * 32 + lane) * 33;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) { r[j * 4 + i] = ag[j][i]; r[16 + j * 4 + i] = au[j][i]; }
            __syncthreads();
            // reduce over warps: thread -> 4 of the 1024 (lane, 32) values; this half's partial
            float part[4];
            float * pb = pbuf + ((size_t) (u * NG1 + g) * 2) * 1024;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int idx = tid + p * THREADS;          // 0..1023 = ln * 32 + v
                const int ln = idx >> 5, v = idx & 31;
                float s = 0.f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) s += red[(w * 32 + ln) * 33 + v];
                part[p] = s;
                __stcg(pb + kh * 1024 + idx, s);
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) s_last = atomicAdd(ctrl + HALF + u * NG1 + g, 1u) == 1u;
            __syncthreads();
            if (s_last) {
                __threadfence();
                const float gs = bf16_bits(gs13[e]);
                // both halves summed in fixed order (half 0 + half 1) so the result does not depend on timing
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const int idx = tid + p * THREADS;
                    const float other = __ldcg(pb + (1 - kh) * 1024 + idx);
                    red[idx] = kh == 0 ? part[p] + other : other + part[p];
                }
                __syncthreads();
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int idx = tid * 2 + p;            // (ln, ji) gate at ln*32+ji, up at ln*32+16+ji
                    const int ln = idx >> 4, ji = idx & 15, j = ji >> 2, i = ji & 3;
                    const float sg_ = red[ln * 32 + ji], su_ = red[ln * 32 + 16 + ji];
                    const int n = 16 * j + (ln >> 2) + ((i & 2) ? 8 : 0);
                    const int t = 2 * (ln & 3) + (i & 1);
                    if (t < M) {
                        const float gv = bf16r(bf16r(sg_) * gs), uv = bf16r(bf16r(su_) * gs);
                        hbuf[((size_t) u * MAXT + t) * I + g * 64 + n] = __float2bfloat16(gv / (1.f + __expf(-gv)) * uv);
                    }
                }
                __threadfence();
                __syncthreads();
                if (tid == 0) { ctrl[HALF + u * NG1 + g] = 0u; atomicAdd(ctrl + DONE + u, 1u); }
            }
            stamp(4093);
        } else if (item >= b4 && item < bend) {
            // ---------------- gemm2 item ----------------
            const int j2 = item - b4, u = j2 / NT2, c = j2 % NT2, e = ulist[u];
            const uint4 * wb = w2 + (size_t) e * KT2 * ROW2 + lane;
            const uint8_t * sb = s2 + (size_t) e * KT2 * H + grp * 8;
            uint4 wa[D2], wc[D2];
            uint2 sa[D2], sc[D2];
#pragma unroll
            for (int d = 0; d < D2; ++d) {
                const int kt = warp + d * WARPS;
                wa[d] = ldg_nc(wb + kt * ROW2 + (2 * c) * 32);
                wc[d] = ldg_nc(wb + kt * ROW2 + (2 * c + 1) * 32);
                sa[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * H + (2 * c) * 64));
                sc[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * H + (2 * c + 1) * 64));
            }
            if (tid == 0) {
                while (ld_acq(ctrl + DONE + u) < (unsigned) NG1) __nanosleep(32);
            }
            __syncthreads();
            stamp(4092);
            uint32_t hb[D2][2];
            {
                const unsigned * hrow = reinterpret_cast<const unsigned *>(hbuf + ((size_t) u * MAXT + grp) * I) + tq;
#pragma unroll
                for (int d = 0; d < D2; ++d) {
                    const int kt = warp + d * WARPS;
                    hb[d][0] = grp < M ? __ldcg(hrow + kt * 8) : 0u;
                    hb[d][1] = grp < M ? __ldcg(hrow + kt * 8 + 4) : 0u;
                }
            }
            float aa[4][4], ac[4][4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) aa[j][i] = ac[j][i] = 0.f;
#pragma unroll
            for (int d = 0; d < D2; ++d) {
                tile_mma(aa, wa[d], sa[d], hb[d][0], hb[d][1]);
                tile_mma(ac, wc[d], sc[d], hb[d][0], hb[d][1]);
            }
            __syncthreads();
            float * r = red + (warp * 32 + lane) * 33;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) { r[j * 4 + i] = aa[j][i]; r[16 + j * 4 + i] = ac[j][i]; }
            __syncthreads();
            const float gsv = bf16_bits(gs2[e]);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int idx = tid * 4 + p;
                const int ln = idx >> 5, hj = idx & 31, half = hj >> 4, ji = hj & 15, j = ji >> 2, i = ji & 3;
                float s = 0.f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) s += red[(w * 32 + ln) * 33 + hj];
                const int n = 128 * c + 64 * half + 16 * j + (ln >> 2) + ((i & 2) ? 8 : 0);
                const int t = 2 * (ln & 3) + (i & 1);
                if (t < M) {
#pragma unroll
                    for (int kp = 0; kp < TOPK; ++kp) {
                        if (si.ids[t * TOPK + kp] == e) {
                            const float tw = bf16r(gsv * bf16r(topk_w[t * TOPK + kp]));
                            ybuf[((size_t) t * TOPK + kp) * H + n] = __float2bfloat16(bf16r(s) * tw);
                        }
                    }
                }
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) s_last = atomicAdd(ctrl + CNT + c, 1u) == tile_target;
            __syncthreads();
            stamp(4091);
            if (s_last) final_tile(c);
        } else {
            break;
        }
        if (tid == 0) s_item = (int) (gridDim.x + atomicAdd(ctrl + QC, 1u));
        __syncthreads();
        item = s_item;
        __syncthreads();
    }
    __syncthreads();
    stamp(4089);
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(ctrl + XC, 1u) == gridDim.x - 1) {
            for (int u = 0; u < U; ++u) ctrl[DONE + u] = 0u;
            ctrl[GLF] = 0u;
            ctrl[SDONE] = 0u;
            ctrl[QC] = 0u;
            __threadfence();
            ctrl[XC] = 0u;
        }
    }
}

template <typename K, typename... Args>
void launch_pdl(K kernel, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}

}  // namespace

// x [M, 2560] bf16; topk_ids [M, 10] int32; topk_w [M, 10] fp32; w13 int32 [E, 160, 2560] (Marlin); s13 fp8
// [E, 160, 1280]; gs13 bf16 [E]; w2 int32 [E, 40, 5120]; s2 fp8 [E, 40, 2560]; gs2 bf16 [E]; w_gate bf16 [2560];
// shared bf16 [M, 2560] or empty (no shared expert term). Scratch: hbuf bf16 [80*8*640], ybuf bf16 [8*10*2560],
// gl fp32 [8], counters int32 [20] (zero-initialised, left zero).
torch::Tensor a16_moe_decode(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_w,
                             torch::Tensor w13, torch::Tensor s13, torch::Tensor gs13,
                             torch::Tensor w2, torch::Tensor s2, torch::Tensor gs2,
                             torch::Tensor w_gate, torch::Tensor shared,
                             torch::Tensor hbuf, torch::Tensor ybuf, torch::Tensor gl, torch::Tensor counters,
                             torch::Tensor out, int64_t phase, torch::Tensor dbg, torch::Tensor pbuf,
                             torch::Tensor shq1, torch::Tensor shs1, torch::Tensor shq2, torch::Tensor shs2,
                             torch::Tensor shbuf) {
    const int M = (int) x.size(0);
    TORCH_CHECK(M >= 1 && M <= MAXT && x.size(1) == H && x.scalar_type() == at::kBFloat16 && x.is_contiguous());
    TORCH_CHECK(topk_ids.scalar_type() == at::kInt && topk_ids.size(1) == TOPK && topk_ids.is_contiguous());
    TORCH_CHECK(topk_w.scalar_type() == at::kFloat && topk_w.is_contiguous());
    TORCH_CHECK(w13.size(1) == KT1 && w13.size(2) == 4 * ROW1 && w2.size(1) == KT2 && w2.size(2) == 4 * ROW2);
    TORCH_CHECK(s13.size(1) == KT1 && s13.size(2) == 2 * I && s2.size(1) == KT2 && s2.size(2) == H);
    TORCH_CHECK(gs13.scalar_type() == at::kBFloat16 && gs2.scalar_type() == at::kBFloat16);
    const bool has_shared = shared.numel() > 0;
    if (shq1.numel()) {
        TORCH_CHECK(phase == 4, "shared-expert items need the persistent kernel");
        TORCH_CHECK(shq1.scalar_type() == at::kChar && shq1.size(0) == 2 * I && shq1.size(1) == H && shq1.is_contiguous());
        TORCH_CHECK(shs1.scalar_type() == at::kBFloat16 && shs1.size(0) == 2 * I && shs1.size(1) == H / 128 && shs1.is_contiguous());
        TORCH_CHECK(shq2.scalar_type() == at::kChar && shq2.size(0) == H && shq2.size(1) == I && shq2.is_contiguous());
        TORCH_CHECK(shs2.scalar_type() == at::kBFloat16 && shs2.size(0) == H && shs2.size(1) == I / 128 && shs2.is_contiguous());
        TORCH_CHECK(shbuf.scalar_type() == at::kBFloat16 && shbuf.numel() >= MAXT * (I + H));
    }
    if (has_shared) TORCH_CHECK(shared.scalar_type() == at::kBFloat16 && shared.is_contiguous() && shared.size(0) == M);
    const at::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const size_t smem1 = MAXT * XS1 * 2;     // >= WARPS * 32 * 33 * 4 (the reduction reuses it)
    static bool attr_set = false;
    if (!attr_set) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(a16_gemm1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem1));
        attr_set = true;
    }
    if (phase != 4 && (phase & 1)) launch_pdl(a16_gemm1_kernel, dim3(M * TOPK + 1, NG1), dim3(THREADS), smem1, stream,
               (const __nv_bfloat16 *) x.data_ptr(), (const int *) topk_ids.data_ptr(),
               (const uint4 *) w13.data_ptr(), (const uint8_t *) s13.data_ptr(), (const uint16_t *) gs13.data_ptr(),
               (const __nv_bfloat16 *) w_gate.data_ptr(), (__nv_bfloat16 *) hbuf.data_ptr(), (float *) gl.data_ptr(), M);
    if (phase == 4) {
        static int grid4 = 0;
        if (!grid4) {
            int per_sm = 0, sms = 0;
            C10_CUDA_CHECK(cudaFuncSetAttribute(a16_moe_persist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem1));
            C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, a16_moe_persist_kernel, THREADS, smem1));
            C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.device().index()));
            grid4 = per_sm * sms;
        }
        launch_pdl(a16_moe_persist_kernel, dim3(grid4), dim3(THREADS), smem1, stream,
                   (const __nv_bfloat16 *) x.data_ptr(), (const int *) topk_ids.data_ptr(), (const float *) topk_w.data_ptr(),
                   (const uint4 *) w13.data_ptr(), (const uint8_t *) s13.data_ptr(), (const uint16_t *) gs13.data_ptr(),
                   (const uint4 *) w2.data_ptr(), (const uint8_t *) s2.data_ptr(), (const uint16_t *) gs2.data_ptr(),
                   (const __nv_bfloat16 *) w_gate.data_ptr(), has_shared ? (const __nv_bfloat16 *) shared.data_ptr() : nullptr,
                   (__nv_bfloat16 *) hbuf.data_ptr(), (__nv_bfloat16 *) ybuf.data_ptr(), (float *) gl.data_ptr(),
                   (float *) pbuf.data_ptr(), (unsigned *) counters.data_ptr(), (__nv_bfloat16 *) out.data_ptr(), M,
                   dbg.numel() ? (long long *) dbg.data_ptr() : nullptr,
                   shq1.numel() ? (const int8_t *) shq1.data_ptr() : nullptr, shq1.numel() ? (const uint16_t *) shs1.data_ptr() : nullptr,
                   shq1.numel() ? (const int8_t *) shq2.data_ptr() : nullptr, shq1.numel() ? (const uint16_t *) shs2.data_ptr() : nullptr,
                   shq1.numel() ? (__nv_bfloat16 *) shbuf.data_ptr() : nullptr);
        return out;
    }
    if (phase & 2) launch_pdl(a16_gemm2_kernel, dim3(M * TOPK, NT2), dim3(THREADS), 0, stream,
               (const int *) topk_ids.data_ptr(), (const float *) topk_w.data_ptr(),
               (const uint4 *) w2.data_ptr(), (const uint8_t *) s2.data_ptr(), (const uint16_t *) gs2.data_ptr(),
               (const __nv_bfloat16 *) hbuf.data_ptr(), (const float *) gl.data_ptr(),
               has_shared ? (const __nv_bfloat16 *) shared.data_ptr() : nullptr,
               (__nv_bfloat16 *) ybuf.data_ptr(), (unsigned *) counters.data_ptr(), (__nv_bfloat16 *) out.data_ptr(), M);
    return out;
}
