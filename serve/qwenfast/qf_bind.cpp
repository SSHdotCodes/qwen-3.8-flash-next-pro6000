#include <torch/extension.h>

at::Tensor smallm_gemm(const at::Tensor & x, const at::Tensor & w, int64_t rpw, int64_t unroll);
at::Tensor hc_mix(const at::Tensor & x, const at::Tensor & wd, const at::Tensor & wu, at::Tensor & partial, at::Tensor & bar);
at::Tensor nvfp4_moe_decode(const at::Tensor & x, const at::Tensor & topk_ids, const at::Tensor & topk_w,
                            const at::Tensor & w13, const at::Tensor & w13_sf, const at::Tensor & g1_alpha,
                            const at::Tensor & a1_gs, const at::Tensor & w2, const at::Tensor & w2_sf,
                            const at::Tensor & g2_alpha, const at::Tensor & a2_gs, bool pdl);
std::vector<at::Tensor> hc_mix_fused(const at::Tensor & x, const at::Tensor & w_norm, const at::Tensor & wd,
                                     const c10::optional<at::Tensor> & winj, const at::Tensor & wu, double eps,
                                     at::Tensor & normed, at::Tensor & partial, at::Tensor & bar, bool pdl);
at::Tensor hc_combine_apply(const at::Tensor & y, const at::Tensor & r, const at::Tensor & inj, bool pdl);
std::vector<at::Tensor> router_topk(const at::Tensor & x, const at::Tensor & w, int64_t topk, bool renormalize,
                                    at::Tensor & scratch, at::Tensor & counter, bool pdl);

at::Tensor gdn_norm_oproj(const at::Tensor & core, const at::Tensor & z, const at::Tensor & w_norm, double eps,
                          bool swish, const at::Tensor & w, bool pdl, bool prefetch);

void spec_topk_chain(const at::Tensor & vals, const at::Tensor & idx, const at::Tensor & temps, const at::Tensor & top_ks,
                     const at::Tensor & top_ps, bool use_top_p, const at::Tensor & draft_probs,
                     const at::Tensor & candidates, const at::Tensor & retrive_index, const at::Tensor & coins,
                     const at::Tensor & coins_final, at::Tensor & predicts, at::Tensor & accept_index,
                     at::Tensor & accept_num);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smallm_gemm", &smallm_gemm, "small-M bf16 gemm y = x @ w^T", py::arg("x"), py::arg("w"), py::arg("rpw") = 0, py::arg("unroll") = 0);
    m.def("hc_mix", &hc_mix, "hyper-connection low-rank mix (1..8 rows)");
    m.def("nvfp4_moe_decode", &nvfp4_moe_decode, "decode-size NVFP4 W4A4 MoE (1..8 tokens, top-10)",
          py::arg("x"), py::arg("topk_ids"), py::arg("topk_w"), py::arg("w13"), py::arg("w13_sf"), py::arg("g1_alpha"),
          py::arg("a1_gs"), py::arg("w2"), py::arg("w2_sf"), py::arg("g2_alpha"), py::arg("a2_gs"), py::arg("pdl") = true);
    m.def("hc_mix_fused", &hc_mix_fused, "grouped RMSNorm + HC low-rank mix + combine-gate dots (1..8 rows)",
          py::arg("x"), py::arg("w_norm"), py::arg("wd"), py::arg("winj"), py::arg("wu"), py::arg("eps"),
          py::arg("normed"), py::arg("partial"), py::arg("bar"), py::arg("pdl") = true);
    m.def("hc_combine_apply", &hc_combine_apply, "HC combine with precomputed gate dots",
          py::arg("y"), py::arg("r"), py::arg("inj"), py::arg("pdl") = true);
    m.def("router_topk", &router_topk, "MoE router gemm + softmax top-k (1..8 tokens)",
          py::arg("x"), py::arg("w"), py::arg("topk"), py::arg("renormalize"), py::arg("scratch"), py::arg("counter"),
          py::arg("pdl") = true);
    m.def("gdn_norm_oproj", &gdn_norm_oproj, "GDN gated RMSNorm + out_proj (1..8 tokens)",
          py::arg("core"), py::arg("z"), py::arg("w_norm"), py::arg("eps"), py::arg("swish"), py::arg("w"),
          py::arg("pdl") = true, py::arg("prefetch") = true);
    m.def("spec_topk_chain", &spec_topk_chain, "top-k/top-p renorm on top candidates + chain rejection sampling");
}
