// Weight prefetcher: a single small CTA on a side stream that streams upcoming weight ranges into L2 with
// bulk prefetches while the main stream is busy with latency-bound kernels. It stays at most `lookahead`
// bytes ahead of the main stream's position, which the main stream publishes as a monotonically increasing
// counter (progress); pos_bytes[k] maps a counter value to the byte position reached. Only moves bytes earlier:
// no results depend on it. Gives up (exits) if the main stream stops advancing for `timeout_ns`.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

__device__ __forceinline__ void prefetch_l2(const void * p, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
}

__device__ __forceinline__ int ld_acquire(const int * p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint64_t now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// chunk i: [ptr, ptr+bytes), its end offset in the forward's weight stream = end_pos[i]
__global__ void pf_kernel(const uint64_t * __restrict__ ptrs, const uint32_t * __restrict__ bytes,
                          const int64_t * __restrict__ end_pos, int n, const int64_t * __restrict__ pos_of,
                          int n_pos, const int * progress, int64_t lookahead, int64_t timeout_ns, int piece) {
    const int lane = threadIdx.x;
    int k = ld_acquire(progress);
    int64_t pos = pos_of[min(max(k, 0), n_pos - 1)];
    uint64_t last = now_ns();
    for (int i = 0; i < n; ++i) {
        const int64_t need = end_pos[i] - lookahead;
        while (pos < need) {
            const int k2 = ld_acquire(progress);
            if (k2 != k) {
                k = k2;
                pos = pos_of[min(max(k, 0), n_pos - 1)];
                last = now_ns();
            } else {
                if (now_ns() - last > (uint64_t) timeout_ns) return;
                __nanosleep(200);
            }
        }
        const char * p = reinterpret_cast<const char *>(ptrs[i]);
        const uint32_t b = bytes[i];
        const uint32_t stride = gridDim.x * 32u * (uint32_t) piece;
        for (uint32_t off = (blockIdx.x * 32u + lane) * (uint32_t) piece; off < b; off += stride)
            prefetch_l2(p + off, min((uint32_t) piece, b - off));
        __syncwarp();
    }
}

__global__ void progress_kernel(int * progress, int value) {
    asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(progress), "r"(value) : "memory");
}

}  // namespace

void l2_prefetch_stream(const at::Tensor & ptrs, const at::Tensor & bytes, const at::Tensor & end_pos,
                        const at::Tensor & pos_of, const at::Tensor & progress, int64_t lookahead,
                        int64_t timeout_ns, int64_t piece, int64_t ctas) {
    TORCH_CHECK(ptrs.scalar_type() == at::kLong && bytes.scalar_type() == at::kInt && end_pos.scalar_type() == at::kLong);
    TORCH_CHECK(pos_of.scalar_type() == at::kLong && progress.scalar_type() == at::kInt);
    TORCH_CHECK(piece % 16 == 0 && piece > 0);
    const c10::cuda::CUDAGuard guard(ptrs.device());
    pf_kernel<<<(int) ctas, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint64_t *>(ptrs.data_ptr()), reinterpret_cast<const uint32_t *>(bytes.data_ptr()),
        end_pos.data_ptr<int64_t>(), (int) ptrs.numel(), pos_of.data_ptr<int64_t>(), (int) pos_of.numel(),
        progress.data_ptr<int>(), lookahead, timeout_ns, (int) piece);
}

void l2_progress(const at::Tensor & progress, int64_t value) {
    const c10::cuda::CUDAGuard guard(progress.device());
    progress_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(progress.data_ptr<int>(), (int) value);
}
