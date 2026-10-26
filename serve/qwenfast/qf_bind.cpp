#include <torch/extension.h>
#include <optional>

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

at::Tensor w8_gemm(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, int64_t rpw);

void l2_prefetch_stream(const at::Tensor & ptrs, const at::Tensor & bytes, const at::Tensor & end_pos,
                        const at::Tensor & pos_of, const at::Tensor & progress, int64_t lookahead,
                        int64_t timeout_ns, int64_t piece, int64_t ctas);
void l2_progress(const at::Tensor & progress, int64_t value);

at::Tensor pk_gemm(const at::Tensor & x, const at::Tensor & m8, const at::Tensor & e4, const at::Tensor & base,
                   const at::Tensor & ex_ptr, const at::Tensor & ex_k, const at::Tensor & ex_v, int64_t rpw);
void pk_unpack(const at::Tensor & m8, const at::Tensor & e4, const at::Tensor & base, const at::Tensor & ex_ptr,
               const at::Tensor & ex_k, const at::Tensor & ex_v, at::Tensor & w);

at::Tensor gdn_norm_oproj_pk(const at::Tensor & core, const at::Tensor & z, const at::Tensor & w_norm, double eps,
                             bool swish, const at::Tensor & m8, const at::Tensor & e4, const at::Tensor & base,
                             const at::Tensor & ex_ptr, const at::Tensor & ex_k, const at::Tensor & ex_v, bool pdl, bool prefetch);

at::Tensor w8g_gemm(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, int64_t rpw);
void w8g_unpack(const at::Tensor & q, const at::Tensor & s, at::Tensor & w);
std::vector<at::Tensor> hc_mix_fused_w8(const at::Tensor & x, const at::Tensor & w_norm, const at::Tensor & wd,
                                        const at::Tensor & wd_s, const c10::optional<at::Tensor> & winj,
                                        const at::Tensor & wu, const at::Tensor & wu_s, double eps,
                                        at::Tensor & normed, at::Tensor & partial, at::Tensor & bar, bool pdl);
std::vector<at::Tensor> hc_combine_mix_w8(const at::Tensor & y, const at::Tensor & r, const at::Tensor & inj_prev,
                                          const at::Tensor & w_norm, const at::Tensor & wd,
                                          const at::Tensor & wd_s, const c10::optional<at::Tensor> & winj,
                                          const at::Tensor & wu, const at::Tensor & wu_s, double eps,
                                          at::Tensor & normed, at::Tensor & partial, at::Tensor & bar, bool pdl);
std::vector<at::Tensor> topk_rows(const at::Tensor & x, int64_t k);
at::Tensor w4g_gemm(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s);
at::Tensor rows_dot_i8(const at::Tensor & x, const at::Tensor & q, const at::Tensor & s, const at::Tensor & idx);
std::vector<at::Tensor> hc_mix_fused_w8_fast(const at::Tensor & x, const at::Tensor & w_norm, const at::Tensor & wd,
                                        const at::Tensor & wd_s, const c10::optional<at::Tensor> & winj,
                                        const at::Tensor & wu, const at::Tensor & wu_s, double eps,
                                        at::Tensor & normed, at::Tensor & partial, at::Tensor & bar, bool pdl);
std::vector<at::Tensor> hc_combine_mix_w8_fast(const at::Tensor & y, const at::Tensor & r, const at::Tensor & inj_prev,
                                          const at::Tensor & w_norm, const at::Tensor & wd,
                                          const at::Tensor & wd_s, const c10::optional<at::Tensor> & winj,
                                          const at::Tensor & wu, const at::Tensor & wu_s, double eps,
                                          at::Tensor & normed, at::Tensor & partial, at::Tensor & bar, bool pdl);
std::vector<at::Tensor> hc_mix_fused_w8_g96fast(const at::Tensor & x, const at::Tensor & w_norm, const at::Tensor & wd,
                                        const at::Tensor & wd_s, const c10::optional<at::Tensor> & winj,
                                        const at::Tensor & wu, const at::Tensor & wu_s, double eps,
                                        at::Tensor & normed, at::Tensor & partial, at::Tensor & bar, bool pdl);
std::vector<at::Tensor> hc_combine_mix_w8_g96fast(const at::Tensor & y, const at::Tensor & r, const at::Tensor & inj_prev,
                                          const at::Tensor & w_norm, const at::Tensor & wd,
                                          const at::Tensor & wd_s, const c10::optional<at::Tensor> & winj,
                                          const at::Tensor & wu, const at::Tensor & wu_s, double eps,
                                          at::Tensor & normed, at::Tensor & partial, at::Tensor & bar, bool pdl);
void wpgdn_replay_commit(at::Tensor, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                         const at::Tensor &, const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &,
                         const std::optional<at::Tensor> &);
at::Tensor head_pack_hi(const at::Tensor &);
void head_set_carveout(int64_t);
std::vector<at::Tensor> head_run(const at::Tensor & h, const at::Tensor & q, const at::Tensor & s, const at::Tensor & hi,
                                 const at::Tensor & karg, const at::Tensor & dense, double z, int64_t cap, int64_t rpw);
std::vector<at::Tensor> cand_trunc_sample(const at::Tensor & exact, const at::Tensor & cidx, const at::Tensor & hot_ids,
                                          const at::Tensor & temps, const at::Tensor & top_ks, const at::Tensor & top_ps,
                                          const at::Tensor & expo, at::Tensor & q);
std::vector<at::Tensor> trunc_sample(const at::Tensor & vals, const at::Tensor & idx, const at::Tensor & temps,
                                     const at::Tensor & top_ks, const at::Tensor & top_ps, const at::Tensor & expo,
                                     at::Tensor & q);
at::Tensor gdn_norm_oproj_w8(const at::Tensor & core, const at::Tensor & z, const at::Tensor & w_norm, double eps,
                             bool swish, const at::Tensor & q, const at::Tensor & sc, bool pdl, bool prefetch);

torch::Tensor a16_moe_decode(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_w,
                             torch::Tensor w13, torch::Tensor s13, torch::Tensor gs13,
                             torch::Tensor w2, torch::Tensor s2, torch::Tensor gs2,
                             torch::Tensor w_gate, torch::Tensor shared,
                             torch::Tensor hbuf, torch::Tensor ybuf, torch::Tensor gl, torch::Tensor counters,
                             torch::Tensor out, int64_t phase, torch::Tensor dbg, torch::Tensor pbuf,
                             torch::Tensor shq1, torch::Tensor shs1, torch::Tensor shq2, torch::Tensor shs2,
                             torch::Tensor shbuf);

// WP-DraftHead: INT2 screen -> threshold superset -> INT4 rescore -> top-64 (M=1 draft hot head)
at::Tensor drafthead_q2(const at::Tensor&, const at::Tensor&, const at::Tensor&);
at::Tensor drafthead_threshold(const at::Tensor&, int64_t);
std::vector<at::Tensor> drafthead_rescore(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                          const at::Tensor&);
std::vector<at::Tensor> drafthead_screen(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                         const at::Tensor&, int64_t);
// WP-GDNout: gated norm once + INT8 out_proj (bitwise equal to gdn_norm_oproj_w8); variant 0 stream, 1 replay
at::Tensor wpgdnout_w8(const at::Tensor&, const at::Tensor&, const at::Tensor&, double, bool, const at::Tensor&,
                       const at::Tensor&, int64_t);

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
    m.def("w8_gemm", &w8_gemm, "small-M int8 weight-only gemm (draft)", py::arg("x"), py::arg("q"), py::arg("s"), py::arg("rpw") = 0);
    m.def("l2_prefetch_stream", &l2_prefetch_stream, "progress-gated L2 weight prefetcher (side stream)");
    m.def("l2_progress", &l2_progress, "publish main-stream progress for the prefetcher");
    m.def("pk_gemm", &pk_gemm, "small-M gemm on exact 12-bit packed bf16 weights");
    m.def("pk_unpack", &pk_unpack, "unpack 12-bit packed weights to bf16 (exact)");
    m.def("gdn_norm_oproj_pk", &gdn_norm_oproj_pk, "GDN gated RMSNorm + pk12 out_proj (1..8 tokens)");
    m.def("w8g_gemm", &w8g_gemm, "small-M gemm on group-scaled int8 weights", py::arg("x"), py::arg("q"), py::arg("s"), py::arg("rpw") = 0);
    m.def("w8g_unpack", &w8g_unpack, "unpack group-scaled int8 weights to bf16");
    m.def("hc_mix_fused_w8", &hc_mix_fused_w8, "hc_mix_fused with int8 group-scaled mix weights");
    m.def("hc_combine_mix_w8", &hc_combine_mix_w8, "hc_combine_apply + hc_mix_fused_w8 in one kernel (1..8 rows)");
    m.def("gdn_norm_oproj_w8", &gdn_norm_oproj_w8, "GDN gated RMSNorm + int8 group-scaled out_proj");
    m.def("topk_rows", &topk_rows, "row-wise top-k (k <= 64) of long rows, sorted descending");
    m.def("trunc_sample", &trunc_sample, "truncated draft proposal sample from sorted top-K logits");
    m.def("w4g_gemm", &w4g_gemm, "small-M gemm on 4-bit group-scaled weights (draft hot-head screening)");
    m.def("rows_dot_i8", &rows_dot_i8, "exact int8 logits of selected rows (draft hot-head screening)");
    m.def("a16_moe_decode", &a16_moe_decode, "decode W4A16 routed experts on Marlin-packed NVFP4 + shared gate (1..8 tokens)");
    m.def("hc_mix_fused_w8_fast", &hc_mix_fused_w8_fast, "WP-HC: hc_mix_fused_w8, token-parallel exact-order norm");
    m.def("hc_combine_mix_w8_fast", &hc_combine_mix_w8_fast, "WP-HC: hc_combine_mix_w8, token-parallel exact-order norm");
    m.def("hc_mix_fused_w8_g96fast", &hc_mix_fused_w8_g96fast, "WP-HC: 96-CTA hc_mix_fused_w8 (rows <= 4)");
    m.def("hc_combine_mix_w8_g96fast", &hc_combine_mix_w8_g96fast, "WP-HC: 96-CTA hc_combine_mix_w8 (rows <= 4)");
    m.def("wpgdn_replay_commit", &wpgdn_replay_commit, "WP-GDN: exact replay-on-commit of accepted GDN verify tokens");
    m.def("head_pack_hi", &head_pack_hi, "WP-Head: high-nibble plane of the INT8 target head");
    m.def("head_set_carveout", &head_set_carveout, "WP-Head: shared-memory carveout of the fused screen kernel");
    m.def("head_run", &head_run, "WP-Head: screened exact target lm_head (device k and dense flag)");
    m.def("cand_trunc_sample", &cand_trunc_sample, "truncated draft proposal straight from hot-head screening candidates");
    m.def("drafthead_q2", &drafthead_q2, "WP-DraftHead: packed INT2 g64 hot-head screen logits");
    m.def("drafthead_threshold", &drafthead_threshold, "WP-DraftHead: histogram threshold keeping a superset of the top-C");
    m.def("drafthead_rescore", &drafthead_rescore, "WP-DraftHead: INT4 rescore of the threshold survivors, top-64");
    m.def("drafthead_screen", &drafthead_screen, "WP-DraftHead: INT2 screen + threshold + INT4 rescore -> (vals, ids) top-64");
    m.def("wpgdnout_w8", &wpgdnout_w8, "WP-GDNout: gated RMSNorm once + INT8 out_proj, early release");
    m.def("spec_topk_chain", &spec_topk_chain, "top-k/top-p renorm on top candidates + chain rejection sampling");
}
