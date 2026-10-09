// Row-wise top-k (k <= 64) of long rows (the 248,320-entry vocabulary) for the speculative-decoding samplers.
//
// torch.topk runs one block per row (~50 us for a 248K row). Here, two kernels:
//   1. grid (NB, rows): each block takes a contiguous segment, keeps its keys in registers and finds the
//      segment's top-k with a 4 x 8-bit radix select over a shared histogram; writes k candidates.
//   2. grid (rows): radix select over the NB * k candidates, then each of the k winners computes its rank
//      (key descending, index ascending) and writes itself to vals/idx in sorted order.
// Keys are the order-preserving uint32 image of the float values; ties at the threshold are taken in no
// particular order inside a block, but the final order is (value desc, index asc), and the selected set is
// a correct top-k set (any element > threshold is in it).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int THREADS = 256, NB = 64, KMAX = 64;

__device__ __forceinline__ uint32_t f2key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float key2f(uint32_t k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k);
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

// Block-wide radix select over keys held in registers (EPT per thread, invalid = 0). Returns the k-th largest
// key (thresh) and how many elements equal to it belong to the top-k (need_eq).
template <int EPT>
__device__ void radix_select(const uint32_t (&key)[EPT], int k, uint32_t & thresh, int & need_eq,
                             int * hist, int * s_bucket) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    uint32_t prefix = 0, mask = 0;
#pragma unroll 1
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[tid] = 0;
        __syncthreads();
#pragma unroll
        for (int e = 0; e < EPT; ++e)
            if ((key[e] & mask) == prefix && key[e] != 0u) atomicAdd(&hist[(key[e] >> shift) & 255], 1);
        __syncthreads();
        if (warp == 0) {
            // lane l owns bins 255-8l .. 248-8l (descending); suffix counts from the top
            int c[8], tot = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) { c[j] = hist[255 - 8 * lane - j]; tot += c[j]; }
            int incl = tot;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffff, incl, o);
                if (lane >= o) incl += v;
            }
            int above = incl - tot;   // elements in higher bins than this lane's first bin
            if (above < k && k <= incl) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (above < k && k <= above + c[j]) { s_bucket[0] = 255 - 8 * lane - j; s_bucket[1] = k - above; }
                    above += c[j];
                }
            }
        }
        __syncthreads();
        const uint32_t b = (uint32_t) s_bucket[0];
        k = s_bucket[1];
        prefix |= b << shift;
        mask |= 0xFFu << shift;
        __syncthreads();
    }
    thresh = prefix;
    need_eq = k;
}

template <typename T, int EPT>
__global__ void __launch_bounds__(THREADS) topk_stage1(const T * __restrict__ x, int V, long ldx, int seg, int k,
                                                       uint32_t * __restrict__ ckey, int * __restrict__ cidx) {
    __shared__ int hist[256];
    __shared__ int s_bucket[2];
    __shared__ int s_cnt[2];
    const int b = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
    const int lo = b * seg, hi = min(V, lo + seg);
    const T * row = x + (size_t) r * ldx;
    uint32_t key[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = lo + e * THREADS + tid;
        key[e] = i < hi ? f2key(to_f<T>(row[i])) : 0u;
    }
    uint32_t th; int need;
    radix_select<EPT>(key, k, th, need, hist, s_bucket);
    if (tid < 2) s_cnt[tid] = 0;
    __syncthreads();
    uint32_t * ok = ckey + ((size_t) r * NB + b) * k;
    int * oi = cidx + ((size_t) r * NB + b) * k;
    const int ngt = k - need;
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = lo + e * THREADS + tid;
        if (key[e] > th) {
            const int s = atomicAdd(&s_cnt[0], 1);
            ok[s] = key[e]; oi[s] = i;
        } else if (key[e] == th && key[e] != 0u) {
            const int s = atomicAdd(&s_cnt[1], 1);
            if (s < need) { ok[ngt + s] = key[e]; oi[ngt + s] = i; }
        }
    }
    __syncthreads();
    // fewer than k valid elements in the segment: pad
    const int filled = s_cnt[0] + min(s_cnt[1], need);
    for (int s = filled + tid; s < k; s += THREADS) { ok[s] = 0u; oi[s] = -1; }
}

template <typename TO>
__global__ void __launch_bounds__(THREADS) topk_stage2(const uint32_t * __restrict__ ckey, const int * __restrict__ cidx,
                                                       int k, TO * __restrict__ vals, int64_t * __restrict__ idx) {
    constexpr int EPT = NB * KMAX / THREADS;   // 16
    __shared__ int hist[256];
    __shared__ int s_bucket[2];
    __shared__ int s_cnt[2];
    __shared__ uint32_t s_key[KMAX];
    __shared__ int s_idx[KMAX];
    const int r = blockIdx.x, tid = threadIdx.x, n = NB * k;
    const uint32_t * ck = ckey + (size_t) r * n;
    const int * ci = cidx + (size_t) r * n;
    uint32_t key[EPT];
    int ix[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = e * THREADS + tid;
        key[e] = i < n ? ck[i] : 0u;
        ix[e] = i < n ? ci[i] : -1;
    }
    uint32_t th; int need;
    radix_select<EPT>(key, k, th, need, hist, s_bucket);
    if (tid < 2) s_cnt[tid] = 0;
    __syncthreads();
    const int ngt = k - need;
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        if (key[e] > th) {
            const int s = atomicAdd(&s_cnt[0], 1);
            s_key[s] = key[e]; s_idx[s] = ix[e];
        } else if (key[e] == th && key[e] != 0u) {
            const int s = atomicAdd(&s_cnt[1], 1);
            if (s < need) { s_key[ngt + s] = key[e]; s_idx[ngt + s] = ix[e]; }
        }
    }
    __syncthreads();
    if (tid < k) {
        const uint32_t mk = s_key[tid];
        const int mi = s_idx[tid];
        int rank = 0;
        for (int j = 0; j < k; ++j) {
            const uint32_t oj = s_key[j];
            rank += (oj > mk) || (oj == mk && s_idx[j] < mi);
        }
        vals[(size_t) r * k + rank] = (TO) key2f(mk);
        idx[(size_t) r * k + rank] = mi;
    }
}

}  // namespace

// x [rows, V] fp32 or bf16 (row stride may exceed V); returns (vals [rows, k] fp32 sorted desc, idx [rows, k] int64)
std::vector<at::Tensor> topk_rows(const at::Tensor & x, int64_t k) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.stride(1) == 1);
    TORCH_CHECK(k >= 1 && k <= KMAX);
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16);
    const int rows = x.size(0), V = x.size(1);
    const int seg = (V + NB - 1) / NB;
    TORCH_CHECK(seg >= k && seg <= 16 * THREADS, "topk_rows: unsupported row length ", V);
    const c10::cuda::CUDAGuard guard(x.device());
    auto vals = at::empty({rows, k}, x.options().dtype(at::kFloat));
    auto idx = at::empty({rows, k}, x.options().dtype(at::kLong));
    if (rows == 0) return {vals, idx};
    auto ckey = at::empty({rows, NB, k}, x.options().dtype(at::kInt));
    auto cidx = at::empty({rows, NB, k}, x.options().dtype(at::kInt));
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const dim3 g1(NB, rows);
    const int ept = (seg + THREADS - 1) / THREADS;
    auto ck = reinterpret_cast<uint32_t *>(ckey.data_ptr<int>());
#define LAUNCH1(T, E) topk_stage1<T, E><<<g1, THREADS, 0, st>>>(reinterpret_cast<const T *>(x.data_ptr()), V, x.stride(0), seg, (int) k, ck, cidx.data_ptr<int>())
    if (x.scalar_type() == at::kFloat) {
        if (ept <= 4) LAUNCH1(float, 4); else if (ept <= 8) LAUNCH1(float, 8); else LAUNCH1(float, 16);
    } else {
        if (ept <= 4) LAUNCH1(__nv_bfloat16, 4); else if (ept <= 8) LAUNCH1(__nv_bfloat16, 8); else LAUNCH1(__nv_bfloat16, 16);
    }
#undef LAUNCH1
    topk_stage2<float><<<rows, THREADS, 0, st>>>(ck, cidx.data_ptr<int>(), (int) k, vals.data_ptr<float>(), idx.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {vals, idx};
}

// ---------------------------------------------------------------------------------------------------------------
// trunc_sample: the truncated draft proposal from each row's sorted top-K logits in one launch (one block of K
// threads per row): z = logit / T, keep the first top_k, softmax, keep the top-p prefix (incl. the crossing
// token), renormalize, X = argmax(p / Exp(1)) (Gumbel race, i.e. X ~ p). Writes p into q (pre-zeroed dense
// [rows, V]) at the candidates' token ids, and X / p(X). Rejection sampling stays exact: X is drawn from the q
// written here.
namespace {
template <typename TK>
__global__ void trunc_sample_kernel(const float * __restrict__ vals, const int64_t * __restrict__ idx, int K,
                                    const float * __restrict__ temps, const TK * __restrict__ top_ks,
                                    const float * __restrict__ top_ps, const float * __restrict__ expo,
                                    float * __restrict__ q, long ldq, float * __restrict__ px, int64_t * __restrict__ x) {
    __shared__ float s_red[2];
    __shared__ float s_scan[2];
    __shared__ float s_best[2];
    __shared__ int s_bi[2];
    const int r = blockIdx.x, i = threadIdx.x, lane = i & 31, w = i >> 5;
    const int kk = (int) min(max((long) top_ks[r], 1L), (long) K);
    const float T = temps[r];
    const float z0 = vals[(size_t) r * K] / T;
    const float z = vals[(size_t) r * K + i] / T;
    float e = (i < kk) ? __expf(z - z0) : 0.f;
    // block sum (K <= 64: two warps)
    float s = e;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) s_red[w] = s;
    __syncthreads();
    const float tot = s_red[0] + (K > 32 ? s_red[1] : 0.f);
    float p = e / tot;
    // inclusive prefix sum of p in candidate order
    float c = p;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float v = __shfl_up_sync(0xffffffff, c, o);
        if (lane >= o) c += v;
    }
    if (lane == 31) s_scan[w] = c;
    __syncthreads();
    if (w == 1) c += s_scan[0];
    const bool keep = p > 0.f && (c - p) < top_ps[r];
    p = keep ? p : 0.f;
    float s2 = p;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, o);
    __syncthreads();
    if (lane == 0) s_red[w] = s2;
    __syncthreads();
    const float tot2 = s_red[0] + (K > 32 ? s_red[1] : 0.f);
    p = p / tot2;
    // Gumbel race over kept candidates
    float score = p > 0.f ? p / fmaxf(expo[(size_t) r * K + i], 1.17549435e-38f) : -1.f;
    int bi = i;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float os = __shfl_xor_sync(0xffffffff, score, o);
        const int ob = __shfl_xor_sync(0xffffffff, bi, o);
        if (os > score || (os == score && ob < bi)) { score = os; bi = ob; }
    }
    if (lane == 0) { s_best[w] = score; s_bi[w] = bi; }
    const int64_t tok = idx[(size_t) r * K + i];
    if (p > 0.f) q[(size_t) r * ldq + tok] = p;
    __syncthreads();
    int b = s_bi[0];
    if (K > 32 && (s_best[1] > s_best[0] || (s_best[1] == s_best[0] && s_bi[1] < b))) b = s_bi[1];
    if (i == b) { x[r] = tok; px[r] = p; }
}
}  // namespace

// vals/idx [rows, K] (sorted desc, from topk_rows), temps [rows] or [rows, 1] fp32, top_ks [rows] int32/int64,
// top_ps [rows] fp32, expo [rows, K] fp32 ~ Exp(1); q [rows, V] fp32 zeroed. Returns (p(X) [rows, 1], X [rows, 1]).
std::vector<at::Tensor> trunc_sample(const at::Tensor & vals, const at::Tensor & idx, const at::Tensor & temps,
                                     const at::Tensor & top_ks, const at::Tensor & top_ps, const at::Tensor & expo,
                                     at::Tensor & q) {
    TORCH_CHECK(vals.scalar_type() == at::kFloat && vals.is_contiguous() && vals.dim() == 2);
    const int rows = vals.size(0), K = vals.size(1);
    TORCH_CHECK(K >= 1 && K <= 64 && idx.scalar_type() == at::kLong && idx.is_contiguous() && idx.sizes() == vals.sizes());
    TORCH_CHECK(temps.scalar_type() == at::kFloat && temps.numel() == rows && temps.is_contiguous());
    TORCH_CHECK(top_ps.scalar_type() == at::kFloat && top_ps.numel() == rows && top_ps.is_contiguous());
    TORCH_CHECK(top_ks.numel() == rows && top_ks.is_contiguous());
    TORCH_CHECK(expo.scalar_type() == at::kFloat && expo.is_contiguous() && expo.sizes() == vals.sizes());
    TORCH_CHECK(q.scalar_type() == at::kFloat && q.dim() == 2 && q.size(0) == rows && q.stride(1) == 1);
    const c10::cuda::CUDAGuard guard(vals.device());
    auto px = at::empty({rows, 1}, vals.options());
    auto x = at::empty({rows, 1}, idx.options());
    if (rows == 0) return {px, x};
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const int threads = K > 32 ? 64 : 32;
    TORCH_CHECK(K == threads || K == 64 || K == 32, "trunc_sample: K must be 32 or 64");
    if (top_ks.scalar_type() == at::kInt)
        trunc_sample_kernel<int><<<rows, threads, 0, st>>>(vals.data_ptr<float>(), idx.data_ptr<int64_t>(), K,
            temps.data_ptr<float>(), top_ks.data_ptr<int>(), top_ps.data_ptr<float>(), expo.data_ptr<float>(),
            q.data_ptr<float>(), q.stride(0), px.data_ptr<float>(), x.data_ptr<int64_t>());
    else
        trunc_sample_kernel<int64_t><<<rows, threads, 0, st>>>(vals.data_ptr<float>(), idx.data_ptr<int64_t>(), K,
            temps.data_ptr<float>(), top_ks.data_ptr<int64_t>(), top_ps.data_ptr<float>(), expo.data_ptr<float>(),
            q.data_ptr<float>(), q.stride(0), px.data_ptr<float>(), x.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {px, x};
}
