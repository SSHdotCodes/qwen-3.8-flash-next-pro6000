// MoE router for Qwen3.8-Flash-Next decode batches (1..8 tokens): router GEMM + softmax top-k in one launch.
//
//   logits[t, e] = bf16(x[t] . w[e])                    (what the BF16 gate linear returns)
//   p[t, e]      = softmax_e(logits[t])                  (fp32)
//   ids[t]       = the k experts with the largest logits (lowest expert id wins ties)
//   weights[t]   = p[t, ids] / sum(p[t, ids])  if renormalize, else p[t, ids]
// Same selection and weights as sglang's _router_triton_kernel (softmax scoring, no bias, no groups).
// Each warp owns one expert row and requests it before waiting on the previous kernel (programmatic
// dependent launch); the last CTA to finish (ticket counter, re-armed) does the per-token top-k.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int H = 2560, E = 512, MAXT = 8, MAXK = 16;
constexpr int WARPS = 8, THREADS = WARPS * 32, CHUNKS = H / 8 / 32;   // 10 uint4 per lane per row
constexpr int XV = H / 8;                                             // 320 uint4 per token row
constexpr int PER_LANE = E / 32;                                      // 16 logits per lane in the top-k

__device__ __forceinline__ float dot8(const uint4 & a, const uint4 & b, float acc) {
    const __nv_bfloat162 * x = reinterpret_cast<const __nv_bfloat162 *>(&a);
    const __nv_bfloat162 * y = reinterpret_cast<const __nv_bfloat162 *>(&b);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 u = __bfloat1622float2(x[i]), v = __bfloat1622float2(y[i]);
        acc = fmaf(u.x, v.x, acc);
        acc = fmaf(u.y, v.y, acc);
    }
    return acc;
}

__global__ void __launch_bounds__(THREADS) router_topk_kernel(
        const __nv_bfloat16 * __restrict__ x, const __nv_bfloat16 * __restrict__ w, __nv_bfloat16 * __restrict__ logits,
        float * __restrict__ weights, int * __restrict__ ids, float * __restrict__ scratch, unsigned * __restrict__ counter,
        int rows, int topk, int renormalize) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int e = blockIdx.x * WARPS + warp;
    uint4 wv[CHUNKS];
    const uint4 * wr = reinterpret_cast<const uint4 *>(w + (size_t) e * H);
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) wv[c] = __ldg(wr + c * 32 + lane);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // stage the tokens in shared memory (every warp reads all of them)
    __shared__ uint4 xs[MAXT * XV];
    {
        constexpr int PER = (MAXT * XV + THREADS - 1) / THREADS;
        uint4 tmp[PER];
        const uint4 * xg = reinterpret_cast<const uint4 *>(x);
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int c = threadIdx.x + i * THREADS;
            tmp[i] = c < rows * XV ? __ldg(xg + c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int c = threadIdx.x + i * THREADS;
            if (c < rows * XV) xs[c] = tmp[i];
        }
    }
    __syncthreads();
    float acc[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) acc[t] = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
#pragma unroll
        for (int t = 0; t < MAXT; ++t)
            if (t < rows) acc[t] = dot8(xs[t * XV + c * 32 + lane], wv[c], acc[t]);
    }
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t < rows) {
            float a = acc[t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
            if (lane == t) {
                const __nv_bfloat16 bv = __float2bfloat16(a);
                logits[t * E + e] = bv;
                scratch[t * E + e] = __bfloat162float(bv);
            }
        }
    }

    // ---- last CTA: softmax + top-k per token ----
    __shared__ bool s_last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        s_last = atomicAdd(counter, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (threadIdx.x == 0) *counter = 0;

    for (int t = warp; t < rows; t += WARPS) {
        float v[PER_LANE], r[PER_LANE];
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < PER_LANE; ++i) {
            v[i] = __ldcg(scratch + t * E + i * 32 + lane);
            mx = fmaxf(mx, v[i]);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < PER_LANE; ++i) {
            r[i] = v[i] == v[i] ? v[i] : -1e30f;          // NaN ranks last (as the Triton router)
            v[i] = __expf(v[i] - mx);
            sum += v[i];
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);

        float my_p = 0.f, routed = 0.f;
        int my_id = 0;
        for (int k = 0; k < topk; ++k) {
            // lane-local best (strict > keeps the lowest expert id: ids grow with i)
            float bv = -INFINITY;
            int bi = E;
#pragma unroll
            for (int i = 0; i < PER_LANE; ++i)
                if (r[i] > bv) { bv = r[i]; bi = i * 32 + lane; }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffff, bv, o);
                const int oi = __shfl_xor_sync(0xffffffff, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            const int owner = bi & 31, slot = bi >> 5;
            float p = 0.f;
#pragma unroll
            for (int i = 0; i < PER_LANE; ++i)
                if (lane == owner && i == slot) { p = v[i] / sum; r[i] = -INFINITY; }
            p = __shfl_sync(0xffffffff, p, owner);
            routed += p;
            if (lane == k) { my_p = p; my_id = bi; }
        }
        if (lane < topk) {
            const float norm = routed > 0.f ? routed : 1.f;
            weights[t * topk + lane] = renormalize ? my_p / norm : my_p;
            ids[t * topk + lane] = my_id;
        }
    }
}

}  // namespace

// x [rows, 2560] bf16 (rows 1..8), w [512, 2560] bf16. Scratch: fp32 >= 8*512, counter int32 [1] zeroed.
// Returns (topk_weights fp32 [rows, k], topk_ids int32 [rows, k], router_logits bf16 [rows, 512]).
std::vector<at::Tensor> router_topk(const at::Tensor & x, const at::Tensor & w, int64_t topk, bool renormalize,
                                    at::Tensor & scratch, at::Tensor & counter, bool pdl) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == H && x.is_contiguous());
    TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.dim() == 2 && w.size(0) == E && w.size(1) == H && w.is_contiguous());
    TORCH_CHECK(topk >= 1 && topk <= MAXK);
    TORCH_CHECK(scratch.scalar_type() == at::kFloat && scratch.numel() >= MAXT * E);
    TORCH_CHECK(counter.scalar_type() == at::kInt && counter.numel() >= 1);
    const int rows = x.size(0);
    TORCH_CHECK(rows >= 1 && rows <= MAXT);
    const c10::cuda::CUDAGuard guard(x.device());
    auto weights = at::empty({rows, topk}, x.options().dtype(at::kFloat));
    auto ids = at::empty({rows, topk}, x.options().dtype(at::kInt));
    auto logits = at::empty({rows, E}, x.options());
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(E / WARPS);
    cfg.blockDim = dim3(THREADS);
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute lattr[1];
    lattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    lattr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = lattr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, router_topk_kernel, (const __nv_bfloat16 *) x.data_ptr(),
                                      (const __nv_bfloat16 *) w.data_ptr(), (__nv_bfloat16 *) logits.data_ptr(),
                                      weights.data_ptr<float>(), ids.data_ptr<int>(), scratch.data_ptr<float>(),
                                      reinterpret_cast<unsigned *>(counter.data_ptr<int>()), rows, (int) topk,
                                      renormalize ? 1 : 0));
    return {weights, ids, logits};
}
