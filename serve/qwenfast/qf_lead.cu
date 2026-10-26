// cand_trunc_sample: the truncated draft proposal (trunc_sample in qf_topk.cu) built straight from the draft hot-head
// screening's candidates instead of a top-K over the full-vocabulary logits they were scattered into: exact logits
// [rows, K] fp32 (rounded to BF16 exactly as the BF16 logits tensor held them) at hot-head rows cidx [rows, K],
// tokens hot_ids[cidx]. Every logit outside the candidates is -inf there, so the full-vocabulary top-K returns these
// same K values; each row is ranked here (value desc, ties by candidate order) and sampled with trunc_sample's body.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

template <typename TK>
__global__ void cand_trunc_sample_kernel(const float * __restrict__ exact, const int64_t * __restrict__ cidx,
                                         const int64_t * __restrict__ hot_ids, int K,
                                         const float * __restrict__ temps, const TK * __restrict__ top_ks,
                                         const float * __restrict__ top_ps, const float * __restrict__ expo,
                                         float * __restrict__ q, long ldq, float * __restrict__ px, int64_t * __restrict__ x) {
    __shared__ float sv[64];
    __shared__ float sv2[64];
    __shared__ int64_t stok[64];
    __shared__ float s_red[2];
    __shared__ float s_scan[2];
    __shared__ float s_best[2];
    __shared__ int s_bi[2];
    const int r = blockIdx.x, i = threadIdx.x, lane = i & 31, w = i >> 5;
    float v = __bfloat162float(__float2bfloat16(exact[(size_t) r * K + i]));
    if (v != v) v = -INFINITY;
    const int64_t tok_i = hot_ids[cidx[(size_t) r * K + i]];
    sv[i] = v;
    __syncthreads();
    int rank = 0;
    for (int j = 0; j < K; ++j) {
        const float u = sv[j];
        rank += (u > v) || (u == v && j < i);
    }
    sv2[rank] = v;
    stok[rank] = tok_i;
    __syncthreads();
    // ---- trunc_sample body on the sorted candidates ----
    const int kk = (int) min(max((long) top_ks[r], 1L), (long) K);
    const float T = temps[r];
    const float z0 = sv2[0] / T;
    const float z = sv2[i] / T;
    float e = (i < kk) ? __expf(z - z0) : 0.f;
    float s = e;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) s_red[w] = s;
    __syncthreads();
    const float tot = s_red[0] + (K > 32 ? s_red[1] : 0.f);
    float p = e / tot;
    float c = p;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float t = __shfl_up_sync(0xffffffff, c, o);
        if (lane >= o) c += t;
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
    float score = p > 0.f ? p / fmaxf(expo[(size_t) r * K + i], 1.17549435e-38f) : -1.f;
    int bi = i;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float os = __shfl_xor_sync(0xffffffff, score, o);
        const int ob = __shfl_xor_sync(0xffffffff, bi, o);
        if (os > score || (os == score && ob < bi)) { score = os; bi = ob; }
    }
    if (lane == 0) { s_best[w] = score; s_bi[w] = bi; }
    const int64_t tok = stok[i];
    if (p > 0.f) q[(size_t) r * ldq + tok] = p;
    __syncthreads();
    int b = s_bi[0];
    if (K > 32 && (s_best[1] > s_best[0] || (s_best[1] == s_best[0] && s_bi[1] < b))) b = s_bi[1];
    if (i == b) { x[r] = tok; px[r] = p; }
}

}  // namespace

// exact [rows, K] fp32, cidx [rows, K] int64 (rows of the hot head), hot_ids [H] int64 (token id per hot row),
// temps/top_ks/top_ps/expo/q as trunc_sample. K = 32 or 64. Returns (p(X) [rows, 1], X [rows, 1]).
std::vector<at::Tensor> cand_trunc_sample(const at::Tensor & exact, const at::Tensor & cidx, const at::Tensor & hot_ids,
                                          const at::Tensor & temps, const at::Tensor & top_ks, const at::Tensor & top_ps,
                                          const at::Tensor & expo, at::Tensor & q) {
    TORCH_CHECK(exact.scalar_type() == at::kFloat && exact.is_contiguous() && exact.dim() == 2);
    const int rows = exact.size(0), K = exact.size(1);
    TORCH_CHECK(K == 32 || K == 64, "cand_trunc_sample: K must be 32 or 64");
    TORCH_CHECK(cidx.scalar_type() == at::kLong && cidx.is_contiguous() && cidx.sizes() == exact.sizes());
    TORCH_CHECK(hot_ids.scalar_type() == at::kLong && hot_ids.is_contiguous());
    TORCH_CHECK(temps.scalar_type() == at::kFloat && temps.numel() == rows && temps.is_contiguous());
    TORCH_CHECK(top_ps.scalar_type() == at::kFloat && top_ps.numel() == rows && top_ps.is_contiguous());
    TORCH_CHECK(top_ks.numel() == rows && top_ks.is_contiguous());
    TORCH_CHECK(expo.scalar_type() == at::kFloat && expo.is_contiguous() && expo.sizes() == exact.sizes());
    TORCH_CHECK(q.scalar_type() == at::kFloat && q.dim() == 2 && q.size(0) == rows && q.stride(1) == 1);
    const c10::cuda::CUDAGuard guard(exact.device());
    auto px = at::empty({rows, 1}, exact.options());
    auto x = at::empty({rows, 1}, cidx.options());
    if (rows == 0) return {px, x};
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    if (top_ks.scalar_type() == at::kInt)
        cand_trunc_sample_kernel<int><<<rows, K, 0, st>>>(exact.data_ptr<float>(), cidx.data_ptr<int64_t>(),
            hot_ids.data_ptr<int64_t>(), K, temps.data_ptr<float>(), top_ks.data_ptr<int>(), top_ps.data_ptr<float>(),
            expo.data_ptr<float>(), q.data_ptr<float>(), q.stride(0), px.data_ptr<float>(), x.data_ptr<int64_t>());
    else
        cand_trunc_sample_kernel<int64_t><<<rows, K, 0, st>>>(exact.data_ptr<float>(), cidx.data_ptr<int64_t>(),
            hot_ids.data_ptr<int64_t>(), K, temps.data_ptr<float>(), top_ks.data_ptr<int64_t>(), top_ps.data_ptr<float>(),
            expo.data_ptr<float>(), q.data_ptr<float>(), q.stride(0), px.data_ptr<float>(), x.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {px, x};
}
