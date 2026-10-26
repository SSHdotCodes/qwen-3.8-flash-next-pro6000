"""Inference-only Qwen4-Exp MTP speculative decoding."""

import copy
import json
import logging
import os
from contextlib import ExitStack
from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP, _mtp_quant_config
from sglang.srt.models.qwen4_exp import Qwen4ExpModel
from sglang.srt.runtime_context import get_model, get_parallel, get_spec
from sglang.srt.utils import add_prefix, is_npu

logger = logging.getLogger(__name__)
_HOT_INT8 = os.environ.get("QWEN_MTP_HOT_INT8", "0") == "1"
_FC_INT8 = os.environ.get("QWEN_MTP_FC_INT8", "0") == "1"
# Draft-only hot-head screening (needs QWEN_MTP_HOT_INT8): an INT4 copy (groups of 64) of the hot head picks 64
# candidates per row, their exact INT8 logits are recomputed and every other logit stays -inf. With the truncated
# draft proposal (QWEN_DRAFT_TRUNC, top-k <= 64) q is then built from the same INT8 logits as without screening
# whenever the INT8 top-k lies inside the candidates.
_HOT_SCREEN = os.environ.get("QWEN_MTP_HOT_SCREEN", "0") == "1"
_LIN_INT8 = os.environ.get("QWEN_MTP_LIN_INT8", "0") == "1"


class ShortlistDraftLogitsProcessor(LogitsProcessor):
    """Draft-only reduced head reads with a full target-vocabulary distribution.

    Exact rejection sampling still verifies against the unchanged target. IDs
    outside these two ranges have zero proposal probability, not zero target
    probability. Weight slices are views; no weights are copied or quantized.
    """

    def __init__(self, config, cutoff):
        super().__init__(config)
        assert get_parallel().tp_size == 1
        assert get_spec().speculative_use_rejection_sampling
        assert get_spec().speculative_eagle_topk == 1
        assert config.vocab_size == 248320
        assert 0 < cutoff < 248044
        assert not self.final_logit_softcapping
        self.cutoff = cutoff
        token_map = os.environ.get("QWEN_MTP_HOTMAP")
        self.hot_ids = None
        self.hot_weight = None
        self.hot_weight_key = None
        if token_map:
            with open(token_map) as f:
                ids = json.load(f)
            assert isinstance(ids, list) and all(type(i) is int and 0 <= i < 248320 for i in ids)
            ids = sorted(set(ids) | set(range(248044, 248320)))
            self.hot_ids = torch.tensor(ids, dtype=torch.long, device="cuda")
            logger.info("Experimental BF16 draft-only hot map: %d IDs", len(ids))
        logger.info("Draft-only shortlist: %d regular IDs plus all 276 special IDs; full target vocabulary retained", cutoff)

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
        # Unsupported heads keep the original path, including initial model
        # profiling before the unquantized target head is shared with the draft.
        pk = getattr(lm_head, "_qf_pk", None)
        from sglang.srt.qwenfast import draft_cand_clear
        draft_cand_clear()
        if (not hasattr(lm_head, "weight")
            or lm_head.weight.dtype not in (torch.bfloat16, torch.float16)
            or (pk is None and lm_head.weight.shape[0] != 248320)
            or should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None))
            or embedding_bias is not None
            or hasattr(lm_head, "set_lora")
            or self.rl_on_policy_target is not None):
            return super()._compute_lm_head(hidden_states, lm_head, embedding_bias)
        h = hidden_states.to(lm_head.weight.dtype)
        dtype = torch.float32 if self.use_fp32_lm_head else lm_head.weight.dtype
        logits = torch.full((h.shape[0], 248320), -float("inf"), dtype=dtype, device=h.device)
        if self.hot_ids is not None:
            # Cache a gathered BF16/FP16 copy of ONLY the draft head. Full target
            # lm_head and target sampling are unchanged. Exact rejection uses
            # the full-vocabulary q below, including zero mass outside the map.
            try:
                version = lm_head.weight._version
            except RuntimeError:
                version = None
            key = (lm_head.weight.data_ptr(), version)
            if key != self.hot_weight_key:
                if pk is not None:   # target head stored 12-bit packed (exact); unpack only the hot rows
                    from sglang.srt.qwenfast import pk_gather_rows
                    hot = pk_gather_rows(pk, self.hot_ids)
                else:
                    hot = lm_head.weight.index_select(0, self.hot_ids)
                if _HOT_INT8 and not self.use_fp32_lm_head and hot.dtype == torch.bfloat16:
                    # Draft-only INT8 copy (symmetric per row): half the bytes per draft step. The target head
                    # is untouched; exact rejection sampling sees the q actually sampled from.
                    amax = hot.float().abs().amax(dim=1).clamp_min_(1e-30)
                    self.hot_scale = (amax / 127.0).contiguous()
                    self.hot_q = torch.round(hot.float() / self.hot_scale[:, None]).clamp_(-127, 127).to(torch.int8).contiguous()
                    self.hot_weight = None
                    self.hot_q4 = self.hot_s4 = None
                    self.hot_q2 = self.hot_s2 = None
                    if _HOT_SCREEN and hot.shape[1] == 2560:
                        xg = hot.float().view(hot.shape[0], 40, 64)
                        s4 = (xg.abs().amax(2) / 7.0).clamp_min_(1e-30).to(torch.bfloat16)
                        q = (xg / s4.float().unsqueeze(-1)).round_().clamp_(-7, 7).add_(8).to(torch.uint8).view(hot.shape[0], 2560)
                        self.hot_q4 = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
                        self.hot_s4 = s4.contiguous()
                        del xg, q, s4
                        from sglang.srt.qwenfast import drafthead_make_q2
                        self.hot_q2, self.hot_s2 = drafthead_make_q2(hot)
                        logger.info("Draft hot head screening: INT4 copy %s", tuple(self.hot_q4.shape))
                    del hot, amax
                    logger.info("Draft hot head stored as INT8 per-row: %s", tuple(self.hot_q.shape))
                else:
                    self.hot_weight = hot.contiguous()
                self.hot_weight_key = key
            part = None
            if (self.hot_weight is None and getattr(self, "hot_q4", None) is not None and 1 <= h.shape[0] <= 8
                    and h.dtype == torch.bfloat16 and logits.dtype == torch.bfloat16):
                from sglang.srt.qwenfast import ext
                hc = h.contiguous()
                from sglang.srt.qwenfast import drafthead_candidates
                _, cidx = drafthead_candidates(hc, self.hot_q4, self.hot_s4,
                                               getattr(self, "hot_q2", None), getattr(self, "hot_s2", None))
                exact = ext().rows_dot_i8(hc, self.hot_q, self.hot_scale, cidx)
                logits.scatter_(1, self.hot_ids[cidx], exact.to(logits.dtype))
                from sglang.srt.qwenfast import draft_cand_register
                draft_cand_register(logits, exact, cidx, self.hot_ids)
                return logits
            if self.hot_weight is None:
                from sglang.srt.qwenfast import ext
                if 1 <= h.shape[0] <= 8:
                    part = ext().w8_gemm(h.contiguous(), self.hot_q, self.hot_scale, 0)
                elif h.shape[0] == 0:
                    part = h.new_empty((0, self.hot_q.shape[0]))
                else:
                    part = torch.cat([ext().w8_gemm(h[i:i + 8].contiguous(), self.hot_q, self.hot_scale, 0)
                                      for i in range(0, h.shape[0], 8)])
            elif not self.use_fp32_lm_head:
                from sglang.srt.qwenfast import hot_head
                part = hot_head(h, self.hot_weight)
            if part is None:
                part = (torch.mm(h, self.hot_weight.T, out_dtype=torch.float32)
                        if self.use_fp32_lm_head else torch.matmul(h, self.hot_weight.T))
            logits[:, self.hot_ids] = part
            return logits
        for lo, hi in ((0, self.cutoff), (248044, 248320)):
            if self.use_fp32_lm_head:
                part = torch.mm(h, lm_head.weight[lo:hi].T, out_dtype=torch.float32)
            else:
                part = torch.matmul(h, lm_head.weight[lo:hi].T)
            logits[:, lo:hi] = part
        return logits


class Qwen4ExpForCausalLMMTP(Qwen3_5ForCausalLMMTP):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.is_multimodal = hasattr(config, "text_config")
        if self.is_multimodal:
            config = config.text_config

        # Deepcopy so MTP-only mutations below don't leak into the main model.
        config = copy.deepcopy(config)
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]
        config.full_attention_interval = 1
        config.ple_layer_ids = []

        quant_config = _mtp_quant_config(quant_config)

        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self._mtp_input_fusion = self._init_mtp_input_fusion(config)

        self.model = Qwen4ExpModel(
            config,
            quant_config,
            prefix=add_prefix("mtp", prefix),
            is_nextn=True,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        cutoff = int(os.environ.get("QWEN_MTP_SHORTLIST", "0"))
        self.logits_processor = (
            ShortlistDraftLogitsProcessor(config, cutoff) if cutoff else LogitsProcessor(config)
        )

    def _init_pre_fc_norms(self, config: PretrainedConfig) -> None:
        self.pre_fc_norm_embedding = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        hidden_norm_size = (
            self.hc_count * config.hidden_size
            if self.hc_count > 1
            else config.hidden_size
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            hidden_norm_size, eps=config.rms_norm_eps
        )

    def _init_linear_projections(self, config: PretrainedConfig) -> None:
        self.fc_embedding = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.fc_hidden = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _init_standard_fusion(self, config: PretrainedConfig):
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self._init_pre_fc_norms(config)
        return self._fuse_standard

    def _init_mtp_input_fusion(self, config: PretrainedConfig):
        if self.hc_count <= 1:
            return self._init_standard_fusion(config)

        self._init_linear_projections(config)
        self._init_pre_fc_norms(config)
        return self._fuse_residual_linear_shared

    def _fc_int8(self, lin: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        """Draft-only INT8 (per-row) copy of a 2560x2560 MTP input projection for <= 8-row calls."""
        if not _FC_INT8 or lin.bias is not None or lin.weight.dtype != torch.bfloat16:
            return lin(x)
        cache = getattr(lin, "_qf_i8", None)
        if cache is None:
            if torch.cuda.is_current_stream_capturing():
                return lin(x)
            w = lin.weight.float()
            s = (w.abs().amax(dim=1).clamp_min(1e-30) / 127.0).contiguous()
            q = torch.round(w / s[:, None]).clamp_(-127, 127).to(torch.int8).contiguous()
            cache = lin._qf_i8 = (q, s)
        x2 = x.reshape(-1, x.shape[-1])
        if x2.shape[0] > 16:          # prefill / long extends: BF16 GEMM
            return lin(x)
        from sglang.srt.qwenfast import ext, w8_rows_ok
        outs = []
        for i in range(0, x2.shape[0], 8):
            part = x2[i:i + 8]
            if not w8_rows_ok(part):
                return lin(x)
            outs.append(ext().w8_gemm(part, cache[0], cache[1], 0))
        y = outs[0] if len(outs) == 1 else torch.cat(outs)
        return y.view(*x.shape[:-1], cache[0].shape[0])

    def _fuse_residual_linear_shared(
        self, input_embeds: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        input_embeds = self._fc_int8(self.fc_embedding, self.pre_fc_norm_embedding(input_embeds))
        orig_shape = hidden_states.shape
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        decoder_view = hidden_states.view(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        encoder_inputs = self._fc_int8(self.fc_hidden, decoder_view)
        return (input_embeds.unsqueeze(-2) + encoder_inputs).view(orig_shape)

    def _fuse_standard(
        self, input_embeds: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        input_embeds = self.pre_fc_norm_embedding(input_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        return self.fc(torch.cat((input_embeds, hidden_states), dim=-1))

    def _npu_quant_context(self):
        exit_stack = ExitStack()
        if (
            is_npu()
            and self.quant_config is None
            and get_model().quantization is not None
        ):
            exit_stack.enter_context(envs.SGLANG_DEEPEP_BF16_DISPATCH.override(True))
            exit_stack.enter_context(
                envs.DEEP_NORMAL_MODE_USE_INT8_QUANT.override(False)
            )
        return exit_stack

    def _prepare_input_embeds(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert input_embeds is None
        input_embeds = forward_batch.mm_input_embeds
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.contains_mm_inputs()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            assert input_embeds is not None
            last_indices = (
                forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
            ).long()
            input_embeds[last_indices] = self.model.embed_tokens(
                input_ids[last_indices]
            )
        if input_embeds is None:
            input_embeds = self.model.embed_tokens(input_ids)
        return input_embeds

    def _set_hc_logits_hidden_states(
        self,
        logits_output,
        hc_hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> None:
        if hc_hidden_states is None:
            return

        # The EAGLE v2 future map holds one hidden state per request;
        # reduce a token-shaped draft-extend HC tensor to its last token per request.
        if forward_batch.forward_mode.is_draft_extend_v2():
            # Mirror LogitsProcessor: the graph path selects rows via
            # spec_info.select_index and the worker no longer re-indexes them.
            select_index = forward_batch.spec_info.select_index
            if select_index is not None:
                hc_hidden_states = hc_hidden_states[select_index]
        elif (
            forward_batch.extend_seq_lens is not None
            and hc_hidden_states.shape[0] != forward_batch.extend_seq_lens.shape[0]
        ):
            last_index = (
                torch.cumsum(forward_batch.extend_seq_lens.to(torch.int64), dim=0) - 1
            )
            hc_hidden_states = hc_hidden_states[last_index]

        assert hc_hidden_states.shape[-1] == self.hc_count * self.hidden_size
        logits_output.hidden_states = hc_hidden_states

    def _qf_int8_linears(self):
        """Draft-only INT8 (per-row) copies of the MTP layer's remaining BF16 linears (attention qkv, indexer
        qk). Used for <= 8-row calls through qwenfast; the BF16 weights stay for larger (prefill) calls."""
        if getattr(self, "_qf_i8_done", False) or not _LIN_INT8 or torch.cuda.is_current_stream_capturing():
            return
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        n = 0
        for name, mod in self.model.named_modules():
            w = getattr(mod, "weight", None)
            if (isinstance(getattr(mod, "quant_method", None), UnquantizedLinearMethod) and isinstance(w, torch.Tensor)
                    and w.dim() == 2 and w.dtype == torch.bfloat16 and w.is_cuda and w.shape[1] in (640, 2560, 6144)
                    and getattr(mod, "bias", None) is None and w.shape[0] >= 640 and not name.endswith("gate")):
                wf = w.float()
                sc = (wf.abs().amax(dim=1).clamp_min(1e-30) / 127.0).contiguous()
                mod._qf_i8 = (torch.round(wf / sc[:, None]).clamp_(-127, 127).to(torch.int8).contiguous(), sc)
                n += 1
                del wf
        self._qf_i8_done = True
        logger.info("Draft-only INT8 copies for %d BF16 linears", n)
        for name, mod in self.model.named_modules():
            w = getattr(mod, "weight", None)
            if isinstance(w, torch.Tensor) and w.dim() == 2 and w.shape[0] >= 512:
                logger.info("draft module %s %s qm=%s w=%s %s", name, type(mod).__name__,
                            type(getattr(mod, "quant_method", None)).__name__, tuple(w.shape), w.dtype)

    def _qf_draft_fast(self):
        """Draft-only decode copies, made once outside graph capture (precision changes acceptance only):
        QWEN_MTP_A16=path  routed experts as Marlin NVFP4 + shared expert as w8g INT8 (draft/convert.py output) for
                           qwenfast's persistent a16 MoE kernel at <= 8 rows; the FP8 experts stay for prefill calls
        QWEN_MTP_HC8=1     hyper-connection mix weights as w8g INT8 (BF16 parameters become the dequantized values)"""
        if getattr(self, "_qf_fast_done", False) or torch.cuda.is_current_stream_capturing():
            return
        self._qf_fast_done = True
        from sglang.srt import qwenfast as qfm
        path = os.environ.get("QWEN_MTP_A16")
        if path and not any(hasattr(getattr(m, "experts", None), "_qf_a16") for m in self.model.modules()):
            d = torch.load(path, map_location="cuda")
            blocks = [m for m in self.model.modules() if hasattr(m, "experts") and hasattr(m, "shared_expert_gate")]
            for b in blocks:
                b.experts._qf_a16 = (d["w13"], d["s13"], d["g13"], d["w2"], d["s2"], d["g2"])
                b._qf_a16sh = (d["sq1"], d["ss1"], d["sq2"], d["ss2"])
            logger.info("Draft a16 MoE: NVFP4 experts + INT8 shared expert from %s for %d blocks", path, len(blocks))
        if os.environ.get("QWEN_MTP_HC8", "0") == "1":
            from sglang.srt.layers import hyperconnection as hcm
            n = 0
            for mod in self.model.modules():
                if not (isinstance(mod, hcm.GatedResidual) and hasattr(mod, "input_mix_weight_down")):
                    continue
                wd, wu = mod.input_mix_weight_down.weight, mod.input_mix_weight_up.weight
                if wd.dtype == torch.bfloat16 and tuple(wd.shape) == (320, 10240) and tuple(wu.shape) == (10240, 320):
                    pd, pu = qfm.w8_quant(wd.data), qfm.w8_quant(wu.data)
                    qfm.ext().w8g_unpack(pd["q"], pd["s"], wd.data)
                    qfm.ext().w8g_unpack(pu["q"], pu["s"], wu.data)
                    mod._qf_w8hc = (pd["q"], pd["s"], pu["q"], pu["s"])
                    n += 1
            logger.info("Draft hyper-connection mix weights as w8g INT8: %d modules", n)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self._qf_int8_linears()
        self._qf_draft_fast()
        with self._npu_quant_context():
            input_embeds = self._prepare_input_embeds(
                input_ids, forward_batch, input_embeds
            )
            hidden_states = forward_batch.spec_info.hidden_states
            if not forward_batch.forward_mode.is_idle():
                hidden_states = self._mtp_input_fusion(input_embeds, hidden_states)

            with get_global_expert_distribution_recorder().disable_this_region():
                model_output = self.model(
                    input_ids,
                    positions,
                    forward_batch,
                    hidden_states,
                )

            hc_hidden_states = None
            if isinstance(model_output, tuple):
                hidden_states, hc_hidden_states = model_output
            else:
                hidden_states = model_output

        logits_output = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        self._set_hc_logits_hidden_states(
            logits_output, hc_hidden_states, forward_batch
        )
        return logits_output


EntryClass = [Qwen4ExpForCausalLMMTP]
