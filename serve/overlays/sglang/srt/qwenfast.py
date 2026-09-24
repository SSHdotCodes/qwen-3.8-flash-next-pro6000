"""Decode-size kernel overrides for Qwen3.8-Flash-Next on the RTX PRO 6000 (SM120).

Imported once by the patched qwen4_exp model module. Everything here is shape-gated to the
decode/verify batch sizes where the replacement measured faster; every other call keeps the
stock SGLang path. Weights, dtypes and quantization are unchanged.

  QWENFAST=0          disable all overrides
  QWENFAST_MOE=0      keep FlashInfer CUTLASS for the NVFP4 experts
  QWENFAST_HC=0       keep the stock hyper-connection norm + mix + combine kernels
  QWENFAST_ROUTER=1   fused router on the main stream (measured no gain: the MoE block is bandwidth-bound)
  QWENFAST_GDN=0      keep the stock GDN output norm + out_proj (default: fused, weights prefetched into L2)
  QWENFAST_GDN_MAXT=4 largest token count for the fused GDN output
  QWENFAST_SAMPLE=0   keep the dense verify sampling (softmax -> top-k/top-p renorm -> chain sampling)
  QWENFAST_QSAMETA=0  keep the single-warp, 128-pages-per-iteration QSA graph page-table build
  QWENFAST_SO=path    prebuilt extension (smallm_gemm, nvfp4_moe_decode, hc_mix_fused, hc_combine_apply)
"""

import importlib.machinery
import importlib.util
import logging
import os

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("QWENFAST", "1") == "1"
MOE_ENABLED = ENABLED and os.environ.get("QWENFAST_MOE", "1") == "1"
HC_ENABLED = ENABLED and os.environ.get("QWENFAST_HC", "1") == "1"
ROUTER_ENABLED = ENABLED and os.environ.get("QWENFAST_ROUTER", "0") == "1"
GDN_ENABLED = ENABLED and os.environ.get("QWENFAST_GDN", "1") == "1"
GDN_MAXT = int(os.environ.get("QWENFAST_GDN_MAXT", "4"))
SAMPLE_ENABLED = ENABLED and os.environ.get("QWENFAST_SAMPLE", "1") == "1"
QSAMETA_ENABLED = ENABLED and os.environ.get("QWENFAST_QSAMETA", "1") == "1"
_ext = None

# (N, K) -> {M: (rows per warp, unroll)}; measured against cuBLAS on this GPU (bench_smallm.json)
_BIG = {1: (2, 2), 2: (1, 2), 3: (4, 4), 4: (4, 4)}
_SMALL = {1: (1, 2), 2: (1, 4), 3: (1, 4), 4: (1, 4)}
SMALLM_TABLE = {
    (16480, 2560): _BIG,                    # GDN fused in_proj (qkvz + ba)
    (13312, 2560): _BIG,                    # attention qkv (+ gate)
    (512, 2560): _SMALL,                    # MoE router
    (640, 2560): _SMALL,                    # QSA index qk
    (1280, 2560): _SMALL,                   # shared expert gate_up
    (2560, 640): {1: (2, 4), 2: (2, 2), 3: (2, 2), 4: (2, 2)},  # shared expert down
    (248320, 2560): {1: (1, 4), 2: (1, 4), 3: (2, 4), 4: (2, 4)},  # lm_head
}
HOT_HEAD_CFG = {1: (1, 4), 2: (1, 4), 3: (4, 2), 4: (4, 2)}
# Shapes that run concurrently with another GEMM on a second stream lose in the model even
# when faster alone (few large blocks starve the neighbour); leave them to cuBLAS by default.
for _key in os.environ.get("QWENFAST_SKIP", "13312x2560,640x2560,1280x2560,2560x640").split(","):
    if _key:
        SMALLM_TABLE.pop(tuple(int(v) for v in _key.split("x")), None)


def ext():
    global _ext
    if _ext is None:
        path = os.environ.get("QWENFAST_SO", "/opt/qwenfast/qwenfast.so")
        loader = importlib.machinery.ExtensionFileLoader("qwenfast", path)
        spec = importlib.util.spec_from_loader("qwenfast", loader, origin=path)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        _ext = mod
    return _ext


def _smallm_ok(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and w.is_contiguous()
        and w.data_ptr() % 16 == 0
        and not torch.compiler.is_compiling()
    )


def try_smallm(x: torch.Tensor, w: torch.Tensor, bias=None):
    """y = x @ w.T via the small-M kernel when (N, K, M) is in the measured table; else None."""
    if bias is not None or w.dim() != 2:
        return None
    cfg = SMALLM_TABLE.get((w.shape[0], w.shape[1]))
    if cfg is None:
        return None
    m = x.shape[0] if x.dim() == 2 else -1
    rc = cfg.get(m)
    if rc is None or not _smallm_ok(x, w):
        return None
    return ext().smallm_gemm(x, w, rc[0], rc[1])


def hot_head(h: torch.Tensor, w: torch.Tensor):
    rc = HOT_HEAD_CFG.get(h.shape[0]) if h.dim() == 2 else None
    if rc is None or not _smallm_ok(h, w):
        return None
    return ext().smallm_gemm(h, w, rc[0], rc[1])


def fast_softmax(x: torch.Tensor) -> torch.Tensor:
    """softmax over the last dim of a few long rows using multi-block reductions
    (torch's softmax runs one block per row: ~50-70 us for a 248K-vocab row)."""
    if not ENABLED or x.dim() != 2 or x.shape[0] > 16 or x.shape[1] < 16384 or not x.is_cuda:
        return torch.softmax(x, dim=-1)
    x = x.float()
    e = torch.exp(x - x.amax(dim=-1, keepdim=True))
    return e.div_(e.sum(dim=-1, keepdim=True))


def _install():
    from sglang.srt.layers.quantization import unquant
    from sglang.srt.layers import logits_processor as lp

    orig_apply = unquant.UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):
        if _gdn_state["gdn"] is not None and _gdn_state["pending"]:
            y = _gdn_out_proj(layer, x, bias)
            if y is not None:
                return y
        y = try_smallm(x, layer.weight, bias) if x.is_cuda else None
        return y if y is not None else orig_apply(self, layer, x, bias)

    unquant.UnquantizedLinearMethod.apply = apply

    orig_impl = unquant._bf16_gemm_dispatch_impl

    def dispatch_impl(x, weight, bias, addend=None):
        if addend is None:
            y = try_smallm(x, weight, bias)
            if y is not None:
                return y
        return orig_impl(x, weight, bias, addend)

    unquant._bf16_gemm_dispatch_impl = dispatch_impl

    orig_lm = lp.LogitsProcessor._compute_lm_head

    def compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        # Same branch the stock code takes for a plain BF16 head (torch.matmul), nothing else.
        if (
            embedding_bias is None
            and not self.use_fp32_lm_head
            and self.rl_on_policy_target is None
            and hasattr(lm_head, "weight")
            and not (hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"))
            and not lp.should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None))
            and not lp.use_intel_amx_backend(lm_head)
        ):
            y = try_smallm(hidden_states, lm_head.weight)
            if y is not None:
                return y
        return orig_lm(self, hidden_states, lm_head, embedding_bias)

    lp.LogitsProcessor._compute_lm_head = compute_lm_head
    logger.info("qwenfast: small-M GEMM overrides installed for %d shapes", len(SMALLM_TABLE))
    if MOE_ENABLED:
        _install_moe()
    if HC_ENABLED:
        _install_hc()
    if ROUTER_ENABLED:
        _install_router()
    if GDN_ENABLED:
        _install_gdn()
    if QSAMETA_ENABLED:
        _install_qsa_meta()


def _install_moe():
    """Decode-size NVFP4 experts (1..8 tokens): same W4A4 recipe as FlashInfer CUTLASS, fewer launches,
    each selected expert read once. Anything else (prefill, EP/TP, other activations) keeps FlashInfer."""
    from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as fic

    orig = fic._run_flashinfer_cutlass

    def run(*, dispatch_output, quant_info, runner_config, output=None, enable_alltoall=False):
        x = dispatch_output.hidden_states
        if (
            output is None
            and not enable_alltoall
            and quant_info.quant_type == "fp4"
            and x.dim() == 2
            and 1 <= x.shape[0] <= 8
            and x.shape[1] == 2560
            and x.dtype == torch.bfloat16
            and getattr(dispatch_output, "hidden_states_scale", None) is None
            and quant_info.moe_ep_size == 1
            and quant_info.moe_tp_size == 1
            and runner_config.is_gated
            and runner_config.activation == "silu"
            and not runner_config.apply_router_weight_on_input
            and getattr(runner_config, "gemm1_alpha", None) is None
            and getattr(runner_config, "gemm1_clamp_limit", None) is None
            and getattr(runner_config, "swiglu_limit", None) is None
            and dispatch_output.topk_output.topk_ids.shape[-1] == 10
            and quant_info.w13_weight.dtype == torch.uint8
            and not torch.compiler.is_compiling()
        ):
            topk = dispatch_output.topk_output
            qs = quant_info.quant_scales
            return ext().nvfp4_moe_decode(
                x.contiguous(),
                topk.topk_ids.to(torch.int32).contiguous(),
                topk.topk_weights.to(torch.float32).contiguous(),
                quant_info.w13_weight, qs[1], qs[2], qs[0],
                quant_info.w2_weight, qs[4], qs[5], qs[3],
            )
        return orig(dispatch_output=dispatch_output, quant_info=quant_info, runner_config=runner_config,
                    output=output, enable_alltoall=enable_alltoall)

    fic._run_flashinfer_cutlass = run
    logger.info("qwenfast: decode NVFP4 MoE override installed")


def sparse_verify_topk(batch, sampling_info, verify_input, logits):
    """Candidates per row for the sparse verify sampler (spec_topk_chain), or None to keep the dense path.
    Needs a top-k on every request (the renormalization then only involves the top candidates)."""
    if not SAMPLE_ENABLED:
        return None
    try:
        reqs = getattr(batch, "reqs", None)
        bs = len(batch.seq_lens)
        if not reqs or len(reqs) != bs or not sampling_info.need_top_k_sampling:
            return None
        top_ks = [int(r.sampling_params.top_k) for r in reqs]
        if min(top_ks) < 1 or max(top_ks) > 56:
            return None
        d = int(verify_input.draft_token_num)
        dp = verify_input.draft_probs
        cand, ri = verify_input.draft_token, verify_input.retrieve_index
        ok = (
            2 <= d <= 8
            and verify_input.tree_topk == 1
            and dp is not None
            and dp.dtype == torch.float32
            and dp.dim() == 3
            and dp.shape[0] == bs
            and dp.shape[1] >= d - 1
            and dp.shape[2] == logits.shape[-1]
            and dp.stride(2) == 1
            and cand.dtype == ri.dtype
            and cand.dtype in (torch.int32, torch.int64)
            and ri.dim() == 2
            and ri.stride(1) == 1
            and cand.is_contiguous()
            and logits.dim() == 2
            and logits.shape[0] == bs * d
            and logits.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and sampling_info.temperatures.dtype == torch.float32
            and sampling_info.temperatures.numel() == bs
            and sampling_info.top_ps.dtype == torch.float32
            and sampling_info.top_ks.dtype in (torch.int32, torch.int64)
        )
        return min(64, max(top_ks) + 8) if ok else None
    except Exception:  # anything unexpected: dense path
        return None


# Scratch for hc_mix_fused (normed rows, down-projection partials, grid-barrier counters), one set per
# (device, stream): calls on one stream are ordered, so every layer can share it.
_hc_scratch = {}


def _hc_bufs(device):
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    b = _hc_scratch.get(key)
    if b is None:
        b = (
            torch.empty(8 * 10240, dtype=torch.bfloat16, device=device),
            torch.empty(4 * 8 * 324, dtype=torch.float32, device=device),
            torch.zeros(2, dtype=torch.int32, device=device),
        )
        _hc_scratch[key] = b
    return b


def _hc_mix_ok(mod, x: torch.Tensor) -> bool:
    cfg = mod.config
    w = mod.hc_norm.weight
    return (
        x.dim() == 2
        and 1 <= x.shape[0] <= 8
        and x.shape[1] == 10240
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and mod.hc_count == 4
        and mod.hidden_size == 2560
        and cfg.hc_lowrank == 320
        and cfg.hc_per_branch_norm
        and mod.hc_norm.group_size == 2560
        and w.dtype == torch.bfloat16
        and w.is_contiguous()
        and hasattr(mod, "input_mix_weight_down")
        and mod.input_mix_weight_down.weight.dtype == torch.bfloat16
        and mod.input_mix_weight_up.weight.dtype == torch.bfloat16
        and (not hasattr(mod, "block_inject_weight") or mod.block_inject_weight.weight.dtype == torch.bfloat16)
        and not torch.compiler.is_compiling()
    )


def _install_hc():
    """Decode-size hyper-connections (1..8 rows): per-branch Gemma RMSNorm + low-rank gated mix in one
    kernel that also produces the combine's inject dot products, and a one-pass combine. The residual
    tuple then carries (hyper_input, inject fp32 [rows, 4]) instead of (hyper_input, normed bf16)."""
    from sglang.srt.layers import hyperconnection as hcm

    orig_mix = hcm.GatedResidual.mix
    orig_combine = hcm.GatedResidual.combine

    logged = set()

    def mix(self, hyper_input):
        if not _hc_mix_ok(self, hyper_input):
            if hyper_input.dim() == 2 and 1 <= hyper_input.shape[0] <= 8 and "off" not in logged:
                logged.add("off")
                logger.info("qwenfast: hc fast path declined (hc_count=%s norm_w=%s x=%s contiguous=%s)", self.hc_count,
                            self.hc_norm.weight.dtype, hyper_input.dtype, hyper_input.is_contiguous())
            return orig_mix(self, hyper_input)
        if "on" not in logged:
            logged.add("on")
            logger.info("qwenfast: hc fast path active")
        winj = self.block_inject_weight.weight if hasattr(self, "block_inject_weight") else None
        normed, partial, bar = _hc_bufs(hyper_input.device)
        out, inj = ext().hc_mix_fused(
            hyper_input, self.hc_norm.weight, self.input_mix_weight_down.weight, winj,
            self.input_mix_weight_up.weight, self.hc_norm.variance_epsilon, normed, partial, bar, True,
        )
        return out, (hyper_input, inj)

    def combine(self, block_output, residuals):
        hyper_input, second = residuals
        if second.dtype != torch.float32:
            return orig_combine(self, block_output, residuals)
        if (
            block_output.dtype == torch.bfloat16
            and block_output.dim() == 2
            and block_output.shape[1] == 2560
            and not torch.compiler.is_compiling()
        ):
            return ext().hc_combine_apply(block_output.contiguous(), hyper_input.contiguous(), second, True)
        rows = hyper_input.shape[0]
        a = 2.0 * torch.sigmoid(second / self.hc_count)
        out = hyper_input.view(rows, self.hc_count, -1).float() + a.unsqueeze(-1) * block_output.float().unsqueeze(1)
        return out.to(self.params_dtype).view(rows, -1)

    hcm.GatedResidual.mix = mix
    hcm.GatedResidual.combine = combine
    logger.info("qwenfast: decode hyper-connection override installed")


_router_scratch = {}


def _router_bufs(device):
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    b = _router_scratch.get(key)
    if b is None:
        b = (torch.empty(8 * 512, dtype=torch.float32, device=device), torch.zeros(1, dtype=torch.int32, device=device))
        _router_scratch[key] = b
    return b


def _install_router():
    """Decode-size MoE blocks (1..8 tokens, CUDA-graph dual-stream path): router GEMM + softmax top-k in one
    kernel on the main stream, chained (PDL) between the hyper-connection mix and the routed experts; the
    shared expert moves to the side stream. Stock order runs the router next to the shared expert's GEMMs,
    which halves its bandwidth while every routed-expert kernel waits on it."""
    from sglang.srt.models import qwen2_moe as qm
    from sglang.srt.layers.moe import topk as tk
    from sglang.srt.layers.moe.utils import get_moe_runner_backend
    from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder

    orig = qm.Qwen2MoeSparseMoeBlock.forward_normal_dual_stream

    def why_not(self, x):
        """None when the fused router applies, else the first failing condition."""
        cfg = self.topk.topk_config
        gw = getattr(self.gate, "weight", None)
        checks = (
            ("shape", x.dim() == 2 and 1 <= x.shape[0] <= 8 and x.shape[1] == 2560),
            ("dtype", x.dtype == torch.bfloat16 and x.is_contiguous()),
            ("shared", self.shared_expert is not None and not self.enable_shared_expert_fusion),
            ("lora", not getattr(qm, "_SGLANG_EXPERIMENTAL_LORA_OPTI", False)),
            ("inplace", getattr(getattr(self.experts, "moe_runner_config", None), "inplace", True) is False),
            ("gate", gw is not None and gw.dtype == torch.bfloat16 and tuple(gw.shape) == (512, 2560)
             and gw.is_contiguous() and getattr(self.gate, "bias", None) is None
             and type(getattr(self.gate, "quant_method", None)).__name__ == "UnquantizedLinearMethod"),
            ("topk", cfg.top_k <= 16 and not cfg.use_grouped_topk and cfg.num_fused_shared_experts == 0
             and cfg.custom_routing_function is None and cfg.correction_bias is None
             and not cfg.apply_routed_scaling_factor_on_output and cfg.scoring_func == "softmax"
             and cfg.output_format is None),
            ("waterfill", not getattr(self.topk, "enable_waterfill", False)
             and getattr(self.topk, "waterfill_balancer", None) is None),
            ("backend", get_moe_runner_backend().is_flashinfer_cutlass()),
            ("simulate", not tk.envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
             and not tk.envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()),
            ("compile", not torch.compiler.is_compiling()),
        )
        return next((name for name, good in checks if not good), None)

    logged = set()

    def dual(self, hidden_states, use_fused_gate=False, defer_finalize=False):
        reason = "defer_finalize" if defer_finalize else why_not(self, hidden_states)
        if reason is not None:
            if reason not in logged and reason != "shape":
                logged.add(reason)
                logger.info("qwenfast: router fast path declined (%s, layer %s)", reason, getattr(self, "layer_id", "?"))
            return orig(self, hidden_states, use_fused_gate=use_fused_gate, defer_finalize=defer_finalize)
        if "on" not in logged:
            logged.add("on")
            logger.info("qwenfast: router fast path active")
        # Shared expert forks first (it needs SM slots before the routed experts' CTAs fill the GPU); the
        # router follows the mix directly on the main stream so it can prefetch its weight under PDL.
        cur = torch.cuda.current_stream()
        self.alt_stream.wait_stream(cur)
        with torch.cuda.stream(self.alt_stream):
            shared_output = self._forward_shared_experts(hidden_states.clone(), apply_gate=not use_fused_gate)
        cfg = self.topk.topk_config
        scratch, counter = _router_bufs(hidden_states.device)
        topk_w, topk_ids, logits = ext().router_topk(
            hidden_states, self.gate.weight, cfg.top_k, bool(cfg.renormalize), scratch, counter, True)
        topk_ids, topk_w, rec_ids = tk._post_process_topk_ids(
            topk_ids=topk_ids, topk_weights=topk_w, topk_config=cfg, router_logits=logits,
            layer_id=self.topk.layer_id, num_token_non_padded=None, expert_location_dispatch_info=None)
        get_global_expert_distribution_recorder().on_select_experts(topk_ids=rec_ids)
        topk_output = tk.StandardTopKOutput(topk_w, topk_ids, logits)
        router_output = self.experts(hidden_states, topk_output)
        cur.wait_stream(self.alt_stream)
        return router_output, shared_output

    qm.Qwen2MoeSparseMoeBlock.forward_normal_dual_stream = dual
    logger.info("qwenfast: MoE router-first override installed")


# GDN output: the gated norm returns a placeholder and records its inputs; the out_proj linear that consumes
# the placeholder runs gated-norm + GEMM in one kernel. Any other consumer path materializes the real norm.
_gdn_state = {"gdn": None, "pending": {}, "norm_forward": None}


def _gdn_out_proj(layer, x, bias):
    g = _gdn_state["gdn"]
    if layer is not g.out_proj:
        return None
    item = _gdn_state["pending"].pop(x.data_ptr(), None)
    if item is None:
        return None
    core, z, norm, placeholder = item
    w = layer.weight
    if (
        bias is None
        and x.dim() == 2
        and x.shape == (core.shape[0] // 48, 6144)
        and w.dtype == torch.bfloat16
        and tuple(w.shape) == (2560, 6144)
        and w.is_contiguous()
    ):
        return ext().gdn_norm_oproj(core, z, norm.weight, norm.eps, norm.activation != "sigmoid", w, True, True)
    placeholder.copy_(_gdn_state["norm_forward"](norm, core, z))
    return None


def _install_gdn():
    from sglang.kernels.ops.attention.fla import layernorm_gated as lg
    from sglang.srt.models import qwen3_5 as q35

    orig_norm = lg.RMSNorm.forward
    orig_gdn = q35.Qwen3_5GatedDeltaNet.forward
    _gdn_state["norm_forward"] = orig_norm
    logged = set()

    def gdn_forward(self, hidden_states, forward_batch):
        prev = _gdn_state["gdn"]
        _gdn_state["gdn"] = self
        try:
            return orig_gdn(self, hidden_states, forward_batch)
        finally:
            _gdn_state["gdn"] = prev
            for core, z, norm, placeholder in _gdn_state["pending"].values():   # never consumed: materialize
                placeholder.copy_(orig_norm(norm, core, z))
            _gdn_state["pending"].clear()

    def norm_forward(self, x, z=None):
        g = _gdn_state["gdn"]
        if (
            g is not None
            and g.norm is self
            and z is not None
            and x.dim() == 2
            and x.shape[1] == 128
            and x.shape[0] % 48 == 0
            and 1 <= x.shape[0] // 48 <= GDN_MAXT
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and z.dtype == torch.bfloat16
            and z.shape == x.shape
            and z.is_contiguous()
            and self.group_size is None
            and self.norm_before_gate
            and self.bias is None
            and self.activation in ("sigmoid", "swish", "silu")
            and self.weight.dtype == torch.bfloat16
            and self.weight.numel() == 128
            and x.is_cuda
            and not torch.compiler.is_compiling()
        ):
            if "on" not in logged:
                logged.add("on")
                logger.info("qwenfast: GDN fused output path active (activation=%s)", self.activation)
            placeholder = torch.empty_like(x)
            _gdn_state["pending"][placeholder.data_ptr()] = (x, z, self, placeholder)
            return placeholder
        return orig_norm(self, x, z)

    lg.RMSNorm.forward = norm_forward
    q35.Qwen3_5GatedDeltaNet.forward = gdn_forward
    logger.info("qwenfast: GDN gated-norm + out_proj override installed")


def _install_qsa_meta():
    """The QSA graph row-metadata kernel rebuilds each row's full-width page table with one warp, 128 pages
    per loop iteration, each iteration a dependent memory round trip (~13-18 us per call, 4 calls per step).
    Same Triton kernel, launched with one block covering the whole table and 4 warps."""
    import triton
    from sglang.srt.layers.attention.qsa import graph_metadata as gm

    kernel = gm._qsa_graph_row_metadata_kernel

    class RowMetadataLauncher:
        def __getitem__(self, grid):
            launch = kernel[grid]

            def run(*args, **kwargs):
                max_pages = int(args[10]) if len(args) > 10 else int(kwargs["max_pages"])
                block = min(4096, triton.next_power_of_2(max(max_pages, 128)))
                kwargs["PAGE_BLOCK"] = max(block, int(kwargs.get("PAGE_BLOCK", 128)))
                kwargs["num_warps"] = 4
                return launch(*args, **kwargs)

            return run

    gm._qsa_graph_row_metadata_kernel = RowMetadataLauncher()
    logger.info("qwenfast: QSA graph row-metadata launch override installed")


if ENABLED:
    ext()
    _install()
