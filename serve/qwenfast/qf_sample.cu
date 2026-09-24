// Verify-side sampling for speculative decoding with top-k/top-p (chain drafts, rejection sampling).
//
// Replaces softmax(logits / T) -> top_k_renorm_prob -> top_p_renorm_prob -> chain speculative sampling
// over the full vocabulary. With a top-k active the full softmax denominator cancels in the top-k
// renormalization, so everything is computed on the (sorted) top candidates of each row:
//   p_i = exp(z_i - z_0) / sum_kept exp(z_j - z_0),  z = logit / T,  kept = {z >= z_(k-1)}   (ties kept)
//   top-p: keep {p >= p_n}, n = first index where the running sum reaches top_p; renormalize
// then the same accept / residual-resample rules as sglang's speculative_sampling_classic_kernel
// (accept draft token if coin * q < p; residual max(p - q, 0) sampled in token-id order).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int KMAX = 64, MAXD = 8;

template <typename TI, typename TK>
__global__ void __launch_bounds__(32 * MAXD) spec_topk_chain_kernel(
        const float * __restrict__ vals, const int64_t * __restrict__ idx, int km,
        const float * __restrict__ temps, const TK * __restrict__ top_ks, const float * __restrict__ top_ps, int use_top_p,
        const float * __restrict__ draft_probs, long dp_sb, long dp_ss,
        const TI * __restrict__ candidates, long cand_sb, const TI * __restrict__ retrive_index, long ri_sb,
        const float * __restrict__ coins, long coin_sb, const float * __restrict__ coins_final,
        int * __restrict__ predicts, int * __restrict__ accept_index, long ai_sb, int * __restrict__ accept_num,
        int D, int V) {
    __shared__ float s_p[MAXD][KMAX];
    __shared__ int s_tok[MAXD][KMAX];
    __shared__ int s_n[MAXD];
    const int b = blockIdx.x, lane = threadIdx.x & 31, row = threadIdx.x >> 5;

    if (row < D) {
        const size_t r = (size_t) b * D + row;
        const float T = temps[b];
        const int k = (int) min((long) top_ks[b], (long) km);
        float z[2];
        int tk[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int i = lane + 32 * h;
            z[h] = i < km ? vals[r * km + i] / T : -INFINITY;
            tk[h] = i < km ? (int) idx[r * km + i] : 0;
        }
        const float zmax = __shfl_sync(0xffffffff, z[0], 0);
        const float zk = __shfl_sync(0xffffffff, (k - 1) < 32 ? z[0] : z[1], (k - 1) & 31);
        float e[2];
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int i = lane + 32 * h;
            e[h] = (i < km && z[h] >= zk) ? __expf(z[h] - zmax) : 0.f;
        }
        // ordered sum (descending probability) so every lane sees the same value
        for (int i = 0; i < 64; ++i) {
            const float v = __shfl_sync(0xffffffff, i < 32 ? e[0] : e[1], i & 31);
            s += v;
        }
        float p[2] = {e[0] / s, e[1] / s};
        if (use_top_p) {
            const float tp = top_ps[b];
            float cum = 0.f, pivot = 0.f;
            bool found = false;
            for (int i = 0; i < 64 && !found; ++i) {
                const float v = __shfl_sync(0xffffffff, i < 32 ? p[0] : p[1], i & 31);
                if (v <= 0.f) break;
                cum += v;
                if (cum >= tp) { pivot = v; found = true; }
            }
            if (found) {
                float s2 = 0.f;
#pragma unroll
                for (int h = 0; h < 2; ++h) if (p[h] < pivot) p[h] = 0.f;
                for (int i = 0; i < 64; ++i) s2 += __shfl_sync(0xffffffff, i < 32 ? p[0] : p[1], i & 31);
#pragma unroll
                for (int h = 0; h < 2; ++h) p[h] = p[h] / s2;
            }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            s_p[row][lane + 32 * h] = p[h];
            s_tok[row][lane + 32 * h] = tk[h];
        }
        if (lane == 0) s_n[row] = km;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    auto target_p = [&](int rr, int tok) -> float {
        for (int i = 0; i < s_n[rr]; ++i)
            if (s_tok[rr][i] == tok) return s_p[rr][i];
        return 0.f;
    };
    const TI * cand = candidates + (size_t) b * cand_sb;
    const TI * ri = retrive_index + (size_t) b * ri_sb;
    int last = (int) ri[0];
    accept_index[(size_t) b * ai_sb] = last;
    int cur = 0, num = 0, cont = 1;
    for (int step = 1; step < D && cont; ++step) {
        const int tok = (int) cand[step];
        const float pt = target_p(cur, tok);
        const float q = draft_probs[(size_t) b * dp_sb + (size_t) cur * dp_ss + tok];
        const float coin = coins[(size_t) b * coin_sb + step - 1];
        if (coin * q < pt) {
            ++num;
            cur = step;
            predicts[last] = tok;
            const int ci = (int) ri[step];
            accept_index[(size_t) b * ai_sb + num] = ci;
            last = ci;
        } else {
            cont = 0;
        }
    }
    accept_num[b] = num;

    // final token: target row `cur` (all accepted) or residual max(p - q, 0) there, sampled in token-id order
    float val[KMAX];
    int ord[KMAX];
    int n = 0;
    for (int i = 0; i < s_n[cur]; ++i) {
        const float pv = s_p[cur][i];
        if (pv <= 0.f) continue;
        float v = pv;
        if (!cont) {
            float q = draft_probs[(size_t) b * dp_sb + (size_t) cur * dp_ss + s_tok[cur][i]];
            q = q == q ? q : 0.f;
            v = pv - q > 0.f ? pv - q : 0.f;
        }
        // insertion by token id
        int j = n++;
        while (j > 0 && s_tok[cur][ord[j - 1]] > s_tok[cur][i]) { ord[j] = ord[j - 1]; val[j] = val[j - 1]; --j; }
        ord[j] = i;
        val[j] = v;
    }
    float norm = 0.f;
    for (int i = 0; i < n; ++i) norm += val[i];
    const float u = coins_final[b] * norm;
    float cum = 0.f;
    int final_tok = V - 1;
    for (int i = 0; i < n; ++i) {
        cum += val[i];
        if (cum > u) { final_tok = s_tok[cur][ord[i]]; break; }
    }
    predicts[last] = final_tok;
}

}  // namespace

// vals/idx: torch.topk(logits, km) of the [bs*D, V] target logits (sorted, km <= 64, km >= every top_k + ties);
// temps [bs] (or [bs,1]) fp32, top_ks [bs] int32/int64, top_ps [bs] fp32; draft_probs [bs, >=D-1, V] fp32;
// candidates, retrive_index [bs, D] int32/int64; coins [bs, >=D-1] fp32; coins_final [bs] fp32.
// Writes predicts (flat int32), accept_index [bs, depth] int32, accept_num [bs] int32 like the Triton kernel.
void spec_topk_chain(const at::Tensor & vals, const at::Tensor & idx, const at::Tensor & temps, const at::Tensor & top_ks,
                     const at::Tensor & top_ps, bool use_top_p, const at::Tensor & draft_probs,
                     const at::Tensor & candidates, const at::Tensor & retrive_index, const at::Tensor & coins,
                     const at::Tensor & coins_final, at::Tensor & predicts, at::Tensor & accept_index,
                     at::Tensor & accept_num) {
    const int bs = candidates.size(0), D = candidates.size(1);
    const int km = vals.size(1), V = draft_probs.size(-1);
    TORCH_CHECK(D >= 2 && D <= MAXD && km >= 1 && km <= KMAX);
    TORCH_CHECK(vals.scalar_type() == at::kFloat && vals.is_contiguous() && vals.size(0) == (long) bs * D);
    TORCH_CHECK(idx.scalar_type() == at::kLong && idx.is_contiguous() && idx.sizes() == vals.sizes());
    TORCH_CHECK(temps.scalar_type() == at::kFloat && temps.is_contiguous() && temps.numel() == bs);
    TORCH_CHECK(top_ps.scalar_type() == at::kFloat && top_ps.is_contiguous() && top_ps.numel() == bs);
    TORCH_CHECK(top_ks.is_contiguous() && top_ks.numel() == bs);
    TORCH_CHECK(draft_probs.scalar_type() == at::kFloat && draft_probs.dim() == 3 && draft_probs.stride(2) == 1);
    TORCH_CHECK(draft_probs.size(0) == bs && draft_probs.size(1) >= D - 1);
    TORCH_CHECK(candidates.scalar_type() == retrive_index.scalar_type() && candidates.stride(1) == 1 && retrive_index.stride(1) == 1);
    TORCH_CHECK(coins.scalar_type() == at::kFloat && coins.stride(1) == 1 && coins.size(1) >= D - 1);
    TORCH_CHECK(coins_final.scalar_type() == at::kFloat && coins_final.is_contiguous());
    TORCH_CHECK(predicts.scalar_type() == at::kInt && accept_index.scalar_type() == at::kInt && accept_num.scalar_type() == at::kInt);
    TORCH_CHECK(accept_index.stride(1) == 1 && accept_index.size(1) >= D);
    const c10::cuda::CUDAGuard guard(vals.device());
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    const bool i64 = candidates.scalar_type() == at::kLong, k64 = top_ks.scalar_type() == at::kLong;
    TORCH_CHECK(i64 || candidates.scalar_type() == at::kInt);
    TORCH_CHECK(k64 || top_ks.scalar_type() == at::kInt);
#define QF_LAUNCH(TI, TK)                                                                                            \
    spec_topk_chain_kernel<TI, TK><<<bs, 32 * MAXD, 0, s>>>(                                                         \
        vals.data_ptr<float>(), idx.data_ptr<int64_t>(), km, temps.data_ptr<float>(), top_ks.data_ptr<TK>(),        \
        top_ps.data_ptr<float>(), use_top_p ? 1 : 0, draft_probs.data_ptr<float>(), draft_probs.stride(0),          \
        draft_probs.stride(1), candidates.data_ptr<TI>(), candidates.stride(0), retrive_index.data_ptr<TI>(),       \
        retrive_index.stride(0), coins.data_ptr<float>(), coins.stride(0), coins_final.data_ptr<float>(),           \
        predicts.data_ptr<int>(), accept_index.data_ptr<int>(), accept_index.stride(0), accept_num.data_ptr<int>(), \
        D, V)
    if (i64 && k64) QF_LAUNCH(int64_t, int64_t);
    else if (i64) QF_LAUNCH(int64_t, int);
    else if (k64) QF_LAUNCH(int, int64_t);
    else QF_LAUNCH(int, int);
#undef QF_LAUNCH
    C10_CUDA_CHECK(cudaGetLastError());
}
