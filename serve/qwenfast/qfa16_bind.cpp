#include <torch/extension.h>
torch::Tensor a16_moe_decode(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_w,
                             torch::Tensor w13, torch::Tensor s13, torch::Tensor gs13,
                             torch::Tensor w2, torch::Tensor s2, torch::Tensor gs2,
                             torch::Tensor w_gate, torch::Tensor shared,
                             torch::Tensor hbuf, torch::Tensor ybuf, torch::Tensor gl, torch::Tensor counters,
                             torch::Tensor out, int64_t phase, torch::Tensor dbg, torch::Tensor pbuf,
                             torch::Tensor shq1, torch::Tensor shs1, torch::Tensor shq2, torch::Tensor shs2, torch::Tensor shbuf);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("a16_moe_decode", &a16_moe_decode); }
