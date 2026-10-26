"""Decode-size kernel overrides for Qwen3.8-Flash-Next on the RTX PRO 6000 (SM120).

Imported once by the patched qwen4_exp model module. Everything here is shape-gated to the
decode/verify batch sizes where the replacement measured faster; every other call keeps the
stock SGLang path. Weights, dtypes and quantization are unchanged.

  QWENFAST=0          disable all overrides
  QWENFAST_MOE=0      keep FlashInfer CUTLASS for the NVFP4 experts
  QWENFAST_HC=0       keep the stock hyper-connection norm + mix + combine kernels
  QWENFAST_ROUTER=1   fused router GEMM + top-10 on the main stream (RedHat checkpoint: MoE block 91 -> 83 us)
  QWENFAST_GDN=0      keep the stock GDN output norm + out_proj (default: fused, weights prefetched into L2)
  QWENFAST_GDN_MAXT=4 largest token count for the fused GDN output
  QWENFAST_SAMPLE=0   keep the dense verify sampling (softmax -> top-k/top-p renorm -> chain sampling)
  QWENFAST_QSAMETA=0  keep the single-warp, 128-pages-per-iteration QSA graph page-table build
  QWENFAST_PK=1       exact 12-bit packed BF16 weights for GDN in_proj / attention qkv + o_proj / lm_head
                      (QWENFAST_PK_SHAPES, QWENFAST_PK_LMHEAD=0 keeps the head in BF16)
  QWENFAST_DRAFT_W8=0 keep SGLang's FP8 GEMM for the online-FP8 draft linears (default: weight-only FP8 at <= 8 rows)
  QWENFAST_SO=path    prebuilt extension (smallm_gemm, nvfp4_moe_decode, hc_mix_fused, hc_combine_apply)
"""

import importlib.machinery
import importlib.util
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("QWENFAST", "1") == "1"
MOE_ENABLED = ENABLED and os.environ.get("QWENFAST_MOE", "1") == "1"
HC_ENABLED = ENABLED and os.environ.get("QWENFAST_HC", "1") == "1"
ROUTER_ENABLED = ENABLED and os.environ.get("QWENFAST_ROUTER", "0") == "1"
GDN_ENABLED = ENABLED and os.environ.get("QWENFAST_GDN", "1") == "1"
GDN_MAXT = int(os.environ.get("QWENFAST_GDN_MAXT", "4"))
SAMPLE_ENABLED = ENABLED and os.environ.get("QWENFAST_SAMPLE", "1") == "1"
CANDSAMPLE_ENABLED = ENABLED and os.environ.get("QWENFAST_CANDSAMPLE", "1") == "1"
EXTCAND_ENABLED = CANDSAMPLE_ENABLED and os.environ.get("QWENFAST_EXTCAND", "1") == "1"   # also for the draft-extend graph
# WP-Head: screened exact target lm_head in the verify graph (4-bit high-nibble screen, per-row sigma bound, exact
# W8M rescoring of the candidates, -inf elsewhere); a device flag set per verify replay selects the full dense head
HEAD_SCREEN_ENABLED = ENABLED and os.environ.get("QWENFAST_HEAD_SCREEN", "0") == "1"
HEAD_SCREEN_Z = float(os.environ.get("QWENFAST_HEAD_Z", "8"))
HEAD_SCREEN_CAP = int(os.environ.get("QWENFAST_HEAD_CAP", "4096"))
_HEAD = {}
A16MOE_ENABLED = ENABLED and os.environ.get("QWENFAST_A16MOE", "0") == "1"
# Decode: a block's hyper-connection combine runs inside the next mix kernel (hc_combine_mix_w8), within a layer
# (attention combine -> MLP mix) and across layers (MLP combine -> next attention mix, when no PLE sits between)
HCCOMB_ENABLED = ENABLED and os.environ.get("QWENFAST_HCCOMB", "0") == "1"
QSAMETA_ENABLED = ENABLED and os.environ.get("QWENFAST_QSAMETA", "1") == "1"
# WP-HC kernels (same signatures and math, bitwise identical): QWENFAST_HC_WP=1; variant QWENFAST_HC_WP_VARIANT
# (g96fast: 96-CTA schedule for <= 4 rows, the token-parallel "fast" kernel above that)
HC_WP_VARIANT = os.environ.get("QWENFAST_HC_WP_VARIANT", "g96fast") if os.environ.get("QWENFAST_HC_WP", "0") == "1" else None
# Exact 12-bit packing of selected BF16 target weights (bit-identical weights, 75% of the bytes): GDN in_proj,
# attention qkv + o_proj and the target lm_head. Decode/verify rows (<= 8) use pk_gemm; larger calls unpack to BF16.
PK_ENABLED = ENABLED and os.environ.get("QWENFAST_PK", "0") == "1"
PK_SHAPES = {tuple(int(v) for v in k.split("x")) for k in os.environ.get("QWENFAST_PK_SHAPES", "16480x2560,13312x2560,2560x6144").split(",") if k}
# 2560x6144 is both the attention o_proj and the GDN out_proj; only the former is packed (the latter has a fused
# norm + out_proj kernel whose packed variant measured slower)
_PK_NAME_SUFFIX = {(2560, 6144): ".o_proj"}
PK_LM_HEAD = PK_ENABLED and os.environ.get("QWENFAST_PK_LMHEAD", "1") == "1"
# Target 8-bit weights ("w8g", qf_w8g.cu): symmetric INT8 with one BF16 scale per 128 weights of a row, for the
# listed categories of dense BF16 target linears (inproj = GDN in_proj incl. a/b rows, qkv, oproj = attention o_proj,
# head = lm_head, shared = shared expert, outproj = GDN out_proj, hc = hyper-connection mix). Lossy (~0.8% relative
# weight error); categories not listed keep pk12 / BF16. Decode rows (<= 8) read the INT8 form directly; larger
# batches unpack to BF16 and run the stock GEMM.
W8_CATS = {c for c in os.environ.get("QWENFAST_W8", "").split(",") if c} if ENABLED else set()
# Draft-only: SGLang's online FP8 draft linears run as weight-only FP8 GEMMs (BF16 activations) at <= 8 rows.
DRAFT_W8_ENABLED = ENABLED and os.environ.get("QWENFAST_DRAFT_W8", "1") == "1"
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



# ------------------------------------------------------------------------------------------------
# pk12: exact 12-bit packing (see qf_pk.cu). Packed modules keep a 1-element placeholder parameter; the
# registry maps its data pointer to (packed dict, first row, last row) so every call site can find it.
_PK_BY_PTR = {}
_pk_scratch = {}
_pk_logged = set()


def pk_pack(w: torch.Tensor, rows_per_chunk: int = 8192):
    assert w.dtype == torch.bfloat16 and w.dim() == 2 and w.is_cuda and w.shape[1] % 16 == 0
    N, K = w.shape
    m8 = torch.empty((N, K), dtype=torch.uint8, device=w.device)
    e4 = torch.empty((N, K // 2), dtype=torch.uint8, device=w.device)
    base = torch.empty((N,), dtype=torch.uint8, device=w.device)
    ex_rows, ex_k, ex_v = [], [], []
    for r0 in range(0, N, rows_per_chunk):
        r1 = min(N, r0 + rows_per_chunk)
        u = w[r0:r1].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        exp = (u >> 7) & 0xFF
        b = (exp.amax(dim=1) - 15).clamp_(min=1)
        nib = exp - b[:, None]
        esc = nib <= 0
        nib = torch.where(esc, 0, nib)
        m = torch.where(esc, 0, ((u >> 8) & 0x80) | (u & 0x7F))
        m8[r0:r1] = m.to(torch.uint8)
        e4[r0:r1] = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
        base[r0:r1] = b.to(torch.uint8)
        idx = esc.nonzero()
        if idx.numel():
            ex_rows.append(idx[:, 0] + r0)
            ex_k.append(idx[:, 1])
            ex_v.append(w[r0:r1][idx[:, 0], idx[:, 1]].float())
        del u, exp, nib, esc, m, idx
    if ex_rows:
        rows = torch.cat(ex_rows); ks = torch.cat(ex_k); vs = torch.cat(ex_v)
        order = torch.argsort(rows * K + ks)
        rows, ks, vs = rows[order], ks[order].int(), vs[order]
    else:
        rows = torch.zeros(0, dtype=torch.long, device=w.device)
        ks = torch.zeros(0, dtype=torch.int32, device=w.device)
        vs = torch.zeros(0, dtype=torch.float32, device=w.device)
    ptr = torch.zeros(N + 1, dtype=torch.int32, device=w.device)
    ptr[1:] = torch.bincount(rows, minlength=N).cumsum(0).int()
    p = dict(m8=m8, e4=e4, base=base, ex_ptr=ptr, ex_k=ks.contiguous(), ex_v=vs.contiguous(), shape=(N, K))
    # exactness check of this packing (cheap: one unpack into a temporary of the same size, row blocks)
    for r0 in range(0, N, rows_per_chunk):
        r1 = min(N, r0 + rows_per_chunk)
        out = torch.empty((r1 - r0, K), dtype=torch.bfloat16, device=w.device)
        ext().pk_unpack(m8[r0:r1], e4[r0:r1], base[r0:r1], ptr[r0:r1 + 1], p["ex_k"], p["ex_v"], out)
        if not torch.equal(out.view(torch.int16), w[r0:r1].view(torch.int16)):
            raise RuntimeError("qwenfast pk12: packed weights do not reproduce the BF16 original")
        del out
    return p


def _pk_nbytes(p):
    return sum(p[k].numel() * p[k].element_size() for k in ("m8", "e4", "base", "ex_ptr", "ex_k", "ex_v"))


def _pk_placeholder(param: torch.Tensor, entry):
    ph = torch.zeros(1, dtype=param.dtype, device=param.device)
    param.data = ph
    _PK_BY_PTR[param.data_ptr()] = entry
    return ph


def pk_lookup(w):
    if not _PK_BY_PTR or not isinstance(w, torch.Tensor) or w.numel() != 1:
        return None
    return _PK_BY_PTR.get(w.data_ptr())


def _pk_unpack_rows(p, r0, r1):
    """BF16 rows [r0, r1) of a packed matrix in the shared scratch buffer (valid until the next unpack)."""
    N, K = p["shape"]
    dev = p["m8"].device
    buf = _pk_scratch.get(dev)
    need = (r1 - r0) * K
    if buf is None or buf.numel() < need:
        buf = torch.empty(max(need, 16480 * 2560), dtype=torch.bfloat16, device=dev)
        _pk_scratch[dev] = buf
    w = buf[:need].view(r1 - r0, K)
    ext().pk_unpack(p["m8"][r0:r1], p["e4"][r0:r1], p["base"][r0:r1], p["ex_ptr"][r0:r1 + 1], p["ex_k"], p["ex_v"], w)
    return w


def _pk_rpw(M, K):
    # measured on this GPU (ext/test_pk.py, test_pk6144.py): K=2560 favours 4 rows per warp from M=2, K=6144 1-2
    if K == 6144:
        return 1 if M <= 4 else 2
    return 1 if M == 1 else 4


def pk_linear(entry, x: torch.Tensor, bias=None):
    if entry[0].get("kind") == "w8":
        return w8_linear(entry, x, bias)
    p, r0, r1 = entry
    N, K = p["shape"]
    shp = x.shape
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    n_out = r1 - r0
    if M == 0:
        return x.new_empty((*shp[:-1], n_out))
    if (
        bias is None and M <= 8 and M * K * 2 <= 99 * 1024 and x2.dtype == torch.bfloat16 and x2.stride(1) == 1
        and x2.stride(0) % 8 == 0 and x2.data_ptr() % 16 == 0 and not torch.compiler.is_compiling()
    ):
        if r0 == 0 and r1 == N:
            y = ext().pk_gemm(x2, p["m8"], p["e4"], p["base"], p["ex_ptr"], p["ex_k"], p["ex_v"], _pk_rpw(M, K))
        else:
            y = ext().pk_gemm(x2, p["m8"][r0:r1], p["e4"][r0:r1], p["base"][r0:r1], p["ex_ptr"][r0:r1 + 1],
                              p["ex_k"], p["ex_v"], _pk_rpw(M, K))
        return y.view(*shp[:-1], n_out)
    key = ("unpack", N, K, M if M <= 8 else ">8")
    if key not in _pk_logged:
        _pk_logged.add(key)
        logger.info("qwenfast pk12: unpack path for %sx%s rows=%s M=%s", N, K, (r0, r1), M)
    w = _pk_unpack_rows(p, r0, r1)
    return torch.nn.functional.linear(x2, w, bias).view(*shp[:-1], n_out)


def pk_lm_head(p, h: torch.Tensor):
    if p.get("kind") == "w8":
        return _w8_lm_head(p, h)
    N, K = p["shape"]
    M = h.shape[0]
    if 1 <= M <= 8 and h.dtype == torch.bfloat16 and h.is_contiguous() and h.data_ptr() % 16 == 0:
        return ext().pk_gemm(h, p["m8"], p["e4"], p["base"], p["ex_ptr"], p["ex_k"], p["ex_v"], _pk_rpw(M, K))
    out = torch.empty((M, N), dtype=h.dtype, device=h.device)
    for r0 in range(0, N, 16384):
        r1 = min(N, r0 + 16384)
        out[:, r0:r1] = torch.matmul(h, _pk_unpack_rows(p, r0, r1).T)
    return out


def pk_gather_rows(p, ids: torch.Tensor):
    """BF16 copies of the given rows of a packed matrix (draft hot-head construction)."""
    N, K = p["shape"]
    unpack = _w8_unpack_rows if p.get("kind") == "w8" else _pk_unpack_rows
    out = torch.empty((ids.numel(), K), dtype=torch.bfloat16, device=ids.device)
    for r0 in range(0, N, 16384):
        r1 = min(N, r0 + 16384)
        sel = ((ids >= r0) & (ids < r1)).nonzero().flatten()
        if sel.numel():
            out[sel] = unpack(p, r0, r1).index_select(0, ids[sel] - r0)
    return out


@torch.no_grad()
def w8_quant(w: torch.Tensor, rows_per_chunk: int = 8192):
    """BF16 [N, K] -> INT8 q [N, K] + BF16 group scales s [N, K / G] (G = 128, or 64 when K % 128 != 0).
    Scales are rounded to BF16 before quantizing so q * s is the exact dequantized weight."""
    N, K = w.shape
    G = 128 if K % 128 == 0 else 64
    q = torch.empty((N, K), dtype=torch.int8, device=w.device)
    sc = torch.empty((N, K // G), dtype=torch.bfloat16, device=w.device)
    for r0 in range(0, N, rows_per_chunk):
        r1 = min(N, r0 + rows_per_chunk)
        xg = w[r0:r1].float().view(r1 - r0, K // G, G)
        s16 = (xg.abs().amax(2) / 127).clamp_(min=1e-30).to(torch.bfloat16)
        q[r0:r1] = (xg / s16.float().unsqueeze(-1)).round_().clamp_(-127, 127).to(torch.int8).view(r1 - r0, K)
        sc[r0:r1] = s16
        del xg, s16
    return dict(kind="w8", q=q, s=sc, shape=(N, K), G=G)


def _w8_unpack_rows(p, r0, r1):
    N, K = p["shape"]
    dev = p["q"].device
    buf = _pk_scratch.get(dev)
    need = (r1 - r0) * K
    if buf is None or buf.numel() < need:
        buf = torch.empty(max(need, 16480 * 2560), dtype=torch.bfloat16, device=dev)
        _pk_scratch[dev] = buf
    w = buf[:need].view(r1 - r0, K)
    ext().w8g_unpack(p["q"][r0:r1], p["s"][r0:r1], w)
    return w


def _w8_rpw(M, K):
    # measured (ext/test_w8k.py bench, weights rotated past L2): 2 rows per warp up to 4 tokens; 4 for 5..8 tokens at
    # K = 2560 (in_proj, qkv, lm_head); K = 6144 / 640 keep 2
    return 4 if M > 4 and K == 2560 else 2


def w8_linear(entry, x: torch.Tensor, bias=None):
    p, r0, r1 = entry
    N, K = p["shape"]
    shp = x.shape
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    n_out = r1 - r0
    if M == 0:
        return x.new_empty((*shp[:-1], n_out))
    if (
        bias is None and M <= 8 and p["G"] == 128 and K in (640, 2560, 6144) and x2.dtype == torch.bfloat16
        and x2.stride(1) == 1 and x2.stride(0) % 8 == 0 and x2.data_ptr() % 16 == 0 and not torch.compiler.is_compiling()
    ):
        if r0 == 0 and r1 == N:
            y = ext().w8g_gemm(x2, p["q"], p["s"], _w8_rpw(M, K))
        else:
            y = ext().w8g_gemm(x2, p["q"][r0:r1], p["s"][r0:r1], _w8_rpw(M, K))
        return y.view(*shp[:-1], n_out)
    key = ("w8unpack", N, K, M if M <= 8 else ">8")
    if key not in _pk_logged:
        _pk_logged.add(key)
        logger.info("qwenfast w8g: unpack path for %sx%s rows=%s M=%s", N, K, (r0, r1), M)
    w = _w8_unpack_rows(p, r0, r1)
    return torch.nn.functional.linear(x2, w, bias).view(*shp[:-1], n_out)


def _w8_lm_head(p, h: torch.Tensor):
    N, K = p["shape"]
    M = h.shape[0]
    if ("head_hi" in p and _HEAD and 1 <= M <= 8 and h.dtype == torch.bfloat16 and h.is_contiguous()
            and h.data_ptr() % 16 == 0 and not torch.compiler.is_compiling()):
        if torch.cuda.is_current_stream_capturing():
            _HEAD.setdefault("h_graph", {})[M] = h     # static input buffer of the captured verify graph for M rows
        else:
            _HEAD["h"] = h
        out = ext().head_run(h, p["q"], p["s"], p["head_hi"], _HEAD["k"], _HEAD["dense"], HEAD_SCREEN_Z,
                             HEAD_SCREEN_CAP, 4 if M > 4 else 2)
        return out[0]
    if 1 <= M <= 8 and h.dtype == torch.bfloat16 and h.is_contiguous() and h.data_ptr() % 16 == 0:
        return ext().w8g_gemm(h, p["q"], p["s"], _w8_rpw(M, K))
    out = torch.empty((M, N), dtype=h.dtype, device=h.device)
    for r0 in range(0, N, 16384):
        r1 = min(N, r0 + 16384)
        out[:, r0:r1] = torch.matmul(h, _w8_unpack_rows(p, r0, r1).T)
    return out


def _w8_category(name: str):
    if "visual" in name:
        return None
    leaf = name.split(".")[-1]
    if leaf == "lm_head":
        return "head"
    if name.endswith("linear_attn.out_proj"):
        return "outproj"
    if leaf == "qkv_proj":
        return "qkv"
    if leaf == "o_proj":
        return "oproj"
    if ".shared_expert." in name and leaf in ("gate_up_proj", "down_proj"):
        return "shared"
    return None


def w8_model(model):
    """Convert the selected target BF16 linears to w8g (after weights are loaded, before pk12 and graphs)."""
    from sglang.srt.layers.quantization import unquant
    t0 = time.time()
    n = before = after = 0
    seen = {}
    for name, mod in model.named_modules():
        fused = getattr(mod, "_fused_in_proj_weight", None)
        if "inproj" in W8_CATS and isinstance(fused, torch.Tensor) and fused.dtype == torch.bfloat16 and fused.dim() == 2:
            p = w8_quant(fused)
            N = fused.shape[0]
            wq = mod._fused_in_proj_qkvz_width
            mod._fused_in_proj_weight = _pk_placeholder(torch.empty(0, dtype=fused.dtype, device=fused.device), (p, 0, N))
            _pk_placeholder(mod.in_proj_qkvz.weight, (p, 0, wq))
            _pk_placeholder(mod.in_proj_ba.weight, (p, wq, N))
            before += fused.numel() * 2; after += p["q"].numel() + p["s"].numel() * 2; n += 1
            seen["inproj"] = seen.get("inproj", 0) + 1
            del fused
            continue
        if "hc" in W8_CATS and "visual" not in name and hasattr(mod, "input_mix_weight_down") and hasattr(mod, "input_mix_weight_up"):
            wd, wu = mod.input_mix_weight_down.weight, mod.input_mix_weight_up.weight
            if (wd.dtype == torch.bfloat16 and tuple(wd.shape) == (320, 10240) and tuple(wu.shape) == (10240, 320)):
                pd, pu = w8_quant(wd.data), w8_quant(wu.data)
                # decode reads the INT8 copies; the BF16 parameters become the exact dequantized weights so every
                # other path (prefill mix) computes with the same weights
                ext().w8g_unpack(pd["q"], pd["s"], wd.data)
                ext().w8g_unpack(pu["q"], pu["s"], wu.data)
                mod._qf_w8hc = (pd["q"], pd["s"], pu["q"], pu["s"])
                mod._qf_hc_mlp = name.endswith("mlp_hyper_connection")   # mix feeding the MoE router (HC kernel choice)
                before += (wd.numel() + wu.numel()) * 2
                after += pd["q"].numel() + pu["q"].numel() + 2 * (pd["s"].numel() + pu["s"].numel())
                n += 1
                seen["hc"] = seen.get("hc", 0) + 1
            continue
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.dtype != torch.bfloat16 or not w.is_cuda:
            continue
        cat = _w8_category(name)
        if cat is None or cat not in W8_CATS:
            continue
        is_head = cat == "head"
        if not is_head and not (isinstance(getattr(mod, "quant_method", None), unquant.UnquantizedLinearMethod)
                                and getattr(mod, "bias", None) is None):
            continue
        if cat == "outproj":
            mod._qf_w8_outproj = True
        p = w8_quant(w.data)
        before += w.numel() * 2; after += p["q"].numel() + p["s"].numel() * 2; n += 1
        seen[cat] = seen.get(cat, 0) + 1
        if is_head:
            mod._qf_pk = p
            if HEAD_SCREEN_ENABLED and tuple(p["shape"]) == (248320, 2560) and hasattr(ext(), "head_run"):
                p["head_hi"] = ext().head_pack_hi(p["q"])
                if hasattr(ext(), "head_set_carveout"):
                    ext().head_set_carveout(int(os.environ.get("QWENFAST_HEAD_CARVEOUT", "25")))
                _HEAD_P[:] = [p]
                _HEAD.update(dense=torch.ones(1, dtype=torch.int32, device=w.device),
                             k=torch.full((1,), 64, dtype=torch.int32, device=w.device), k_host=64, active=False, h=None)
                logger.info("qwenfast: screened target head prepared (z=%s, cap=%d)", HEAD_SCREEN_Z, HEAD_SCREEN_CAP)
        _pk_placeholder(w, (p, 0, w.shape[0]))
    torch.cuda.empty_cache()
    logger.info("qwenfast w8g: %d matrices %s, %.2f GB -> %.2f GB, %.1f s", n, seen, before / 1e9, after / 1e9,
                time.time() - t0)


def pk_model(model):
    """Pack the target's BF16 GDN in_proj / attention qkv / lm_head (after weights are loaded, before graphs)."""
    import time
    from sglang.srt.layers.quantization import unquant
    t0 = time.time()
    n = before = after = nex = 0
    for name, mod in model.named_modules():
        fused = getattr(mod, "_fused_in_proj_weight", None)
        if isinstance(fused, torch.Tensor) and fused.dtype == torch.bfloat16 and tuple(fused.shape) in PK_SHAPES:
            p = pk_pack(fused)
            N = fused.shape[0]
            wq = mod._fused_in_proj_qkvz_width
            entry = (p, 0, N)
            mod._fused_in_proj_weight = _pk_placeholder(torch.empty(0, dtype=fused.dtype, device=fused.device), entry)
            _pk_placeholder(mod.in_proj_qkvz.weight, (p, 0, wq))
            _pk_placeholder(mod.in_proj_ba.weight, (p, wq, N))
            before += fused.numel() * 2; after += _pk_nbytes(p); nex += p["ex_k"].numel(); n += 1
            del fused
            continue
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.dtype != torch.bfloat16 or not w.is_cuda:
            continue
        is_lin = (isinstance(getattr(mod, "quant_method", None), unquant.UnquantizedLinearMethod)
                  and tuple(w.shape) in PK_SHAPES and getattr(mod, "bias", None) is None
                  and name.endswith(_PK_NAME_SUFFIX.get(tuple(w.shape), "")))
        is_head = PK_LM_HEAD and name.split(".")[-1] == "lm_head" and tuple(w.shape) == (248320, 2560)
        if not (is_lin or is_head):
            continue
        p = pk_pack(w.data)
        before += w.numel() * 2; after += _pk_nbytes(p); nex += p["ex_k"].numel(); n += 1
        N = w.shape[0]
        if is_head:
            mod._qf_pk = p
        _pk_placeholder(w, (p, 0, N))
    torch.cuda.empty_cache()
    logger.info("qwenfast pk12: packed %d matrices, %.2f GB -> %.2f GB (%d exception values), exact, %.1f s",
                n, before / 1e9, after / 1e9, nex, time.time() - t0)


# Evaluation only: fake-quantize selected target BF16 weights in place (quantize -> dequantize to BF16) so the
# KLD harness measures what an 8-bit weight format would cost before any kernel exists.
#   QWENFAST_FQ=inproj,inproj_ba,outproj,qkv,oproj,head,hc,shared,router,ple,index   QWENFAST_FQ_FMT=i8g128|i8row|f8row
FQ_CATS = {c for c in os.environ.get("QWENFAST_FQ", "").split(",") if c}
FQ_FMT = os.environ.get("QWENFAST_FQ_FMT", "i8g128")


@torch.no_grad()
def _fq_rows(w: torch.Tensor, fmt: str):
    N, K = w.shape
    for r0 in range(0, N, 4096):
        r1 = min(N, r0 + 4096)
        x = w[r0:r1].float()
        if fmt.startswith("i8g"):
            g = int(fmt[3:])
            g = g if K % g == 0 else 64
            xg = x.view(r1 - r0, K // g, g)
            s = (xg.abs().amax(2, keepdim=True) / 127).clamp_(min=1e-30).to(torch.bfloat16).float()
            x = ((xg / s).round_().clamp_(-127, 127) * s).view(r1 - r0, K)
        elif fmt == "i8row":
            s = (x.abs().amax(1, keepdim=True) / 127).clamp_(min=1e-30)
            x = (x / s).round_().clamp_(-127, 127) * s
        elif fmt == "f8row":
            s = (x.abs().amax(1, keepdim=True) / 448).clamp_(min=1e-30)
            x = (x / s).to(torch.float8_e4m3fn).float() * s
        else:
            raise ValueError(fmt)
        w[r0:r1].copy_(x.to(torch.bfloat16))


def _fq_category(name: str, w: torch.Tensor):
    if "visual" in name or "mtp" in name.split(".")[0]:
        return None
    leaf = name.split(".")[-1]
    if leaf == "lm_head":
        return "head"
    if name.endswith("linear_attn.out_proj"):
        return "outproj"
    if leaf == "qkv_proj":          # full-attention layers (".self_attn" is stripped from module names)
        return "qkv"
    if leaf == "o_proj":
        return "oproj"
    if leaf in ("input_mix_weight_down", "input_mix_weight_up"):
        return "hc"
    if ".shared_expert." in name and leaf in ("gate_up_proj", "down_proj"):
        return "shared"
    if name.endswith("mlp.gate"):
        return "router"
    if leaf in ("key_proj", "value_proj") and ".ple" in name:
        return "ple"
    if "index" in leaf and leaf.endswith("proj"):
        return "index"
    return None


def fq_model(model):
    import collections
    t0 = time.time()
    seen = collections.Counter()
    for name, mod in model.named_modules():
        fused = getattr(mod, "_fused_in_proj_weight", None)
        if isinstance(fused, torch.Tensor) and fused.dtype == torch.bfloat16 and fused.dim() == 2:
            wq = mod._fused_in_proj_qkvz_width
            if "inproj" in FQ_CATS:
                _fq_rows(fused[:wq], FQ_FMT); seen["inproj"] += fused[:wq].numel()
            if "inproj_ba" in FQ_CATS:
                _fq_rows(fused[wq:], FQ_FMT); seen["inproj_ba"] += fused[wq:].numel()
            continue
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.dtype != torch.bfloat16 or not w.is_cuda:
            continue
        cat = _fq_category(name, w)
        if cat is None or cat not in FQ_CATS:
            continue
        if cat == "outproj" and "in_proj" in name:
            continue
        _fq_rows(w.data, FQ_FMT)
        seen[cat] += w.numel()
    logger.info("qwenfast FQ (%s): fake-quantized %s in %.1f s", FQ_FMT,
                {k: f"{v / 1e9:.3f}G params" for k, v in sorted(seen.items())}, time.time() - t0)


def _install_pk():
    from sglang.srt.model_loader import loader as ld
    orig = ld.DefaultModelLoader.postprocess_weights

    def postprocess_weights(model, target_device):
        orig(model, target_device)
        if type(model).__name__ == "Qwen4ExpForConditionalGeneration":
            if FQ_CATS:
                fq_model(model)
            if W8_CATS:
                w8_model(model)
            if HCCOMB_ENABLED and HC_ENABLED and "hc" in W8_CATS:
                _install_hc_comb(model)
            if PK_ENABLED:
                pk_model(model)
        elif type(model).__name__ == "Qwen4ExpForCausalLMMTP" and os.environ.get("QWEN_MTP_A16"):
            draft_a16_install(model, os.environ["QWEN_MTP_A16"])

    ld.DefaultModelLoader.postprocess_weights = staticmethod(postprocess_weights)
    logger.info("qwenfast pk12: target weight packing hook installed (shapes %s, lm_head %s)", sorted(PK_SHAPES), PK_LM_HEAD)

def w8_rows_ok(x2: torch.Tensor) -> bool:
    return (x2.dim() == 2 and 1 <= x2.shape[0] <= 8 and x2.dtype == torch.bfloat16 and x2.stride(1) == 1
            and x2.stride(0) % 8 == 0 and x2.data_ptr() % 16 == 0 and not torch.compiler.is_compiling())


def _install_draft_w8():
    """Weight-only FP8 GEMM for the online-FP8 MTP draft (the only Fp8LinearMethod user in this deployment):
    same FP8 weights and per-channel scales, BF16 activations instead of per-token FP8. Changes draft
    acceptance only (exact rejection sampling); CUTLASS FP8 tiles launch 5-20 CTAs at these sizes."""
    from sglang.srt.layers.quantization import fp8 as f8
    orig = f8.Fp8LinearMethod.apply
    logged = set()

    def apply(self, layer, x, bias=None):
        w = getattr(layer, "weight", None)
        if (
            bias is None and isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor)
            and w.dtype == torch.float8_e4m3fn and w.dim() == 2
            and not getattr(self, "block_quant", False) and not getattr(self, "use_marlin", False)
            and getattr(layer, "input_scale", None) is None
        ):
            K, N = w.shape
            q = w.t()
            ws = getattr(layer, "weight_scale", None)
            if (q.is_contiguous() and K in (640, 2560, 6144) and isinstance(ws, torch.Tensor)
                    and ws.dtype == torch.float32 and ws.numel() in (1, N) and ws.is_contiguous()):
                x2 = x.reshape(-1, K)
                if w8_rows_ok(x2):
                    if (N, K) not in logged:
                        logged.add((N, K))
                        logger.info("qwenfast: draft weight-only FP8 GEMM active for %sx%s", N, K)
                    return ext().w8_gemm(x2, q, ws.view(-1), 0).view(*x.shape[:-1], N)
        if isinstance(w, torch.Tensor) and w.dim() == 2 and ("decl", tuple(w.shape)) not in logged:
            logged.add(("decl", tuple(w.shape)))
            ws = getattr(layer, "weight_scale", None)
            logger.info("qwenfast: draft FP8 GEMM declined: %s w=%s %s stride=%s ws=%s %s input_scale=%s bias=%s x=%s",
                        type(layer).__name__, tuple(w.shape), w.dtype, w.stride(), None if ws is None else tuple(ws.shape),
                        None if ws is None else ws.dtype, getattr(layer, "input_scale", None) is not None, bias is not None,
                        None if not isinstance(x, torch.Tensor) else (tuple(x.shape), x.dtype, x.stride()))
        return orig(self, layer, x, bias)

    f8.Fp8LinearMethod.apply = apply
    logger.info("qwenfast: draft weight-only FP8 linear override installed")


def _install():
    from sglang.srt.layers.quantization import unquant
    from sglang.srt.layers import logits_processor as lp

    orig_apply = unquant.UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):
        # GDN out_proj first: its input may be the deferred gated-norm placeholder (materialized or fused there)
        if _gdn_state["gdn"] is not None and _gdn_state["pending"]:
            y = _gdn_out_proj(layer, x, bias)
            if y is not None:
                return y
        entry = pk_lookup(getattr(layer, "weight", None))
        if entry is not None:
            return pk_linear(entry, x, bias)
        i8 = getattr(layer, "_qf_i8", None)   # draft-only INT8 copy (set by the MTP model)
        if i8 is not None and bias is None and x.is_cuda:
            x2 = x.reshape(-1, x.shape[-1])
            if w8_rows_ok(x2):
                return ext().w8_gemm(x2, i8[0], i8[1], 0).view(*x.shape[:-1], i8[0].shape[0])
        y = try_smallm(x, layer.weight, bias) if x.is_cuda else None
        return y if y is not None else orig_apply(self, layer, x, bias)

    unquant.UnquantizedLinearMethod.apply = apply

    orig_apply_into = getattr(unquant.UnquantizedLinearMethod, "apply_into", None)
    if orig_apply_into is not None:
        def apply_into(self, layer, x, output, bias=None):
            entry = pk_lookup(getattr(layer, "weight", None))
            if entry is not None:
                output.copy_(pk_linear(entry, x, bias).view_as(output))
                return output
            return orig_apply_into(self, layer, x, output, bias)

        unquant.UnquantizedLinearMethod.apply_into = apply_into

    orig_impl = unquant._bf16_gemm_dispatch_impl

    def dispatch_impl(x, weight, bias, addend=None):
        entry = pk_lookup(weight)
        if entry is not None:
            y = pk_linear(entry, x, bias)
            return y if addend is None else y + addend
        if addend is None:
            y = try_smallm(x, weight, bias)
            if y is not None:
                return y
        return orig_impl(x, weight, bias, addend)

    unquant._bf16_gemm_dispatch_impl = dispatch_impl

    orig_lm = lp.LogitsProcessor._compute_lm_head

    def compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        p = getattr(lm_head, "_qf_pk", None)
        if p is not None:
            assert embedding_bias is None and not self.use_fp32_lm_head
            return pk_lm_head(p, hidden_states.to(torch.bfloat16))
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
    if PK_ENABLED or W8_CATS or FQ_CATS:
        _install_pk()
    if DRAFT_W8_ENABLED:
        _install_draft_w8()


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


def _hc_fn(name, rows, mod=None):
    """WP-HC kernel for one mix call. g96fast (96 CTAs) wins in front of the INT8 GEMMs (attention-side mixes, final
    mixer) but its shared-memory footprint hurts the router's PDL overlap, so MLP-side mixes and > 4 rows use fast."""
    e = ext()
    if HC_WP_VARIANT:
        v = HC_WP_VARIANT
        if v == "g96fast" and (rows > 4 or getattr(mod, "_qf_hc_mlp", False)):
            v = "fast"
        f = getattr(e, name + "_" + v, None) or getattr(e, name + "_fast", None)
        if f is not None:
            return f
    return getattr(e, name)


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
        w8 = getattr(self, "_qf_w8hc", None)
        if w8 is not None:
            out, inj = _hc_fn("hc_mix_fused_w8", hyper_input.shape[0], self)(
                hyper_input, self.hc_norm.weight, w8[0], w8[1], winj, w8[2], w8[3],
                self.hc_norm.variance_epsilon, normed, partial, bar, True,
            )
            return out, (hyper_input, inj)
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


class _HCDeferred:
    """Residual slot of a layer whose MLP combine was left for the next layer's attention mix to run."""
    __slots__ = ("mod", "res")

    def __init__(self, mod, res):
        self.mod, self.res = mod, res


def _hc_comb_ok(mix_mod, block_output, residuals):
    if not isinstance(residuals, tuple) or len(residuals) != 2:
        return False
    hyper_input, second = residuals
    return (
        isinstance(second, torch.Tensor) and second.dtype == torch.float32 and second.is_contiguous()
        and getattr(mix_mod, "_qf_w8hc", None) is not None
        and isinstance(block_output, torch.Tensor) and block_output.dtype == torch.bfloat16 and block_output.dim() == 2
        and block_output.shape[0] <= int(os.environ.get("QWENFAST_HCCOMB_MAXROWS", "4"))
        and block_output.shape[1] == 2560 and block_output.is_contiguous()
        and hyper_input.dim() == 2 and block_output.shape[0] == hyper_input.shape[0]
        and tuple(second.shape) == (hyper_input.shape[0], 4)
        and _hc_mix_ok(mix_mod, hyper_input)
    )


def _hc_comb_mix(mix_mod, block_output, residuals):
    """combine(block_output, residuals) then mix_mod.mix(...) in one kernel; same results as the two steps."""
    hyper_input, second = residuals
    winj = mix_mod.block_inject_weight.weight if hasattr(mix_mod, "block_inject_weight") else None
    normed, partial, bar = _hc_bufs(block_output.device)
    w8 = mix_mod._qf_w8hc
    out, inj, xo = _hc_fn("hc_combine_mix_w8", block_output.shape[0], mix_mod)(
        block_output, hyper_input, second, mix_mod.hc_norm.weight, w8[0], w8[1], winj, w8[2], w8[3],
        mix_mod.hc_norm.variance_epsilon, normed, partial, bar, True)
    return out, (xo, inj)


def _install_hc_comb(model):
    """After the target loads (qwen4_exp is only importable then): link each decoder layer to the next one whose
    attention mix can absorb its MLP combine, and route the layer prepare/postprocess steps through the fused kernel."""
    from sglang.srt.models import qwen4_exp as q4
    mixin = q4.Qwen4ExpLayerExtensionMixin
    layers = [m for m in model.modules() if isinstance(m, mixin)]
    linked = 0
    for a, b in zip(layers, layers[1:]):
        if getattr(b, "layer_id", None) == getattr(a, "layer_id", -2) + 1 and b.ple is None:
            a._qf_hc_next = b
            linked += 1
    if getattr(mixin, "_qf_hc_comb", False):
        logger.info("qwenfast: hc combine->mix re-linked %d of %d layers", linked, len(layers))
        return
    mixin._qf_hc_comb = True
    orig_prep_attn = mixin._prepare_qwen4_exp_attn
    orig_prep_mlp = mixin._prepare_qwen4_exp_mlp
    orig_post = mixin._postprocess_qwen4_exp_layer

    def prep_attn(self, hidden_states, residual, forward_batch, *, ple_batch):
        if isinstance(residual, _HCDeferred):
            if self.ple is None and _hc_comb_ok(self.attn_hyper_connection, hidden_states, residual.res):
                return _hc_comb_mix(self.attn_hyper_connection, hidden_states, residual.res)
            hidden_states = residual.mod.combine(hidden_states, residual.res)
            residual = None
        return orig_prep_attn(self, hidden_states, residual, forward_batch, ple_batch=ple_batch)

    def prep_mlp(self, hidden_states, residual, forward_batch):
        if q4.get_parallel().attn_tp_size == 1 and _hc_comb_ok(self.mlp_hyper_connection, hidden_states, residual):
            return _hc_comb_mix(self.mlp_hyper_connection, hidden_states, residual)
        return orig_prep_mlp(self, hidden_states, residual, forward_batch)

    def post(self, hidden_states, residual, forward_batch):
        nxt = getattr(self, "_qf_hc_next", None)
        if (nxt is not None and not getattr(self, "_is_layer_to_capture", False)
                and _hc_comb_ok(nxt.attn_hyper_connection, hidden_states, residual)):
            return hidden_states, _HCDeferred(self.mlp_hyper_connection, residual)
        return orig_post(self, hidden_states, residual, forward_batch)

    mixin._prepare_qwen4_exp_attn = prep_attn
    mixin._prepare_qwen4_exp_mlp = prep_mlp
    mixin._postprocess_qwen4_exp_layer = post
    logger.info("qwenfast: hc combine->mix fusion installed (%d of %d layers hand their MLP combine on)", linked, len(layers))


_router_scratch = {}


def _router_bufs(device):
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    b = _router_scratch.get(key)
    if b is None:
        b = (torch.empty(8 * 512, dtype=torch.float32, device=device), torch.zeros(1, dtype=torch.int32, device=device))
        _router_scratch[key] = b
    return b


_a16_scratch = {}


def _a16_bufs(device):
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    b = _a16_scratch.get(key)
    if b is None:
        b = dict(
            hbuf=torch.empty(80 * 8 * 640, dtype=torch.bfloat16, device=device),
            ybuf=torch.empty(8 * 10 * 2560, dtype=torch.bfloat16, device=device),
            gl=torch.empty(8, dtype=torch.float32, device=device),
            ctrl=torch.zeros(1024, dtype=torch.int32, device=device),
            pbuf=torch.empty(80 * 10 * 2 * 1024, dtype=torch.float32, device=device),
            dbg=torch.empty(0, dtype=torch.int64, device=device),
            shbuf=torch.empty(8 * (640 + 2560), dtype=torch.bfloat16, device=device),
            none=torch.empty(0, dtype=torch.bfloat16, device=device),
        )
        _a16_scratch[key] = b
    return b


def _a16_weights(block):
    """(w13, s13, g13, w2, s2, g2) of a block's routed experts in the Marlin NVFP4 layout: the experts' own parameters
    (target, QWEN_NVFP4_MARLIN loader), or a decode-only copy attached as experts._qf_a16 (the MTP draft, whose own
    FP8 experts stay for prefill-size calls)."""
    ex = block.experts
    o = getattr(ex, "_qf_a16", None)
    if o is not None:
        return o
    return tuple(getattr(ex, n, None) for n in ("w13_weight", "w13_weight_scale", "w13_weight_scale_2",
                                                "w2_weight", "w2_weight_scale", "w2_weight_scale_2"))


def _a16_ok(block, x):
    """Decode routed experts on the Marlin-packed NVFP4 weights (QWEN_NVFP4_MARLIN loader) + shared-expert gate."""
    w13, s13, g13, w2, s2, g2 = _a16_weights(block)
    sg = getattr(block, "shared_expert_gate", None)
    return (
        x.dim() == 2 and 1 <= x.shape[0] <= 8 and x.shape[1] == 2560 and x.dtype == torch.bfloat16 and x.is_contiguous()
        and w13 is not None and w13.dtype == torch.int32 and tuple(w13.shape) == (512, 160, 2560)
        and w2 is not None and w2.dtype == torch.int32 and tuple(w2.shape) == (512, 40, 5120)
        and s13 is not None and s13.dtype == torch.float8_e4m3fn and tuple(s13.shape) == (512, 160, 1280)
        and s2 is not None and s2.dtype == torch.float8_e4m3fn and tuple(s2.shape) == (512, 40, 2560)
        and g13 is not None and g13.dtype == torch.bfloat16 and g13.numel() == 512
        and g2 is not None and g2.dtype == torch.bfloat16 and g2.numel() == 512
        and sg is not None and getattr(sg, "weight", None) is not None and sg.weight.numel() == 2560
        and sg.weight.dtype == torch.bfloat16 and getattr(sg, "bias", None) is None
    )


def _a16_shared(block):
    """w8g INT8 weights of the shared expert (gate_up [1280, 2560], down [2560, 640]) when the persistent a16 kernel
    can compute it in-kernel (same arithmetic as w8g_gemm + SiluAndMul), else None (side-stream shared expert)."""
    hit = getattr(block, "_qf_a16sh", 0)
    if hit != 0:
        return hit
    hit = None
    se = getattr(block, "shared_expert", None)
    act = type(getattr(se, "act_fn", None)).__name__
    if os.environ.get("QWENFAST_A16SHARED", "1") == "1" and se is not None:
        gu = pk_lookup(getattr(getattr(se, "gate_up_proj", None), "weight", None))
        dn = pk_lookup(getattr(getattr(se, "down_proj", None), "weight", None))
        if (gu is not None and dn is not None and act == "SiluAndMul"
                and gu[0].get("kind") == "w8" and dn[0].get("kind") == "w8"
                and gu[0]["shape"] == (1280, 2560) and dn[0]["shape"] == (2560, 640)
                and gu[1:] == (0, 1280) and dn[1:] == (0, 2560)
                and getattr(se.gate_up_proj, "bias", None) is None and getattr(se.down_proj, "bias", None) is None):
            hit = (gu[0]["q"], gu[0]["s"], dn[0]["q"], dn[0]["s"])
    block._qf_a16sh = hit
    logger.info("qwenfast: a16 shared expert %s (layer %s)", "in-kernel" if hit else "side stream (%s)" % act if se is not None else "none",
                getattr(block, "layer_id", "?"))
    return hit


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
            ("backend", True),
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
            logger.info("qwenfast: router fast path active (experts quant method %s)", type(getattr(self.experts, "quant_method", None)).__name__)
        # Shared expert forks first (it needs SM slots before the routed experts' CTAs fill the GPU); the
        # router follows the mix directly on the main stream so it can prefetch its weight under PDL.
        use_a16 = A16MOE_ENABLED and self.topk.topk_config.top_k == 10 and _a16_ok(self, hidden_states)
        if A16MOE_ENABLED and ("a16" if use_a16 else "a16off") not in logged:
            logged.add("a16" if use_a16 else "a16off")
            logger.info("qwenfast: a16 MoE decode %s (layer %s)", "active" if use_a16 else "declined", getattr(self, "layer_id", "?"))
        cur = torch.cuda.current_stream()
        sh8 = _a16_shared(self) if use_a16 else None
        if sh8 is None:
            self.alt_stream.wait_stream(cur)
            with torch.cuda.stream(self.alt_stream):
                shared_output = self._forward_shared_experts(hidden_states.clone(), apply_gate=not use_fused_gate and not use_a16)
        cfg = self.topk.topk_config
        scratch, counter = _router_bufs(hidden_states.device)
        topk_w, topk_ids, logits = ext().router_topk(
            hidden_states, self.gate.weight, cfg.top_k, bool(cfg.renormalize), scratch, counter, True)
        topk_ids, topk_w, rec_ids = tk._post_process_topk_ids(
            topk_ids=topk_ids, topk_weights=topk_w, topk_config=cfg, router_logits=logits,
            layer_id=self.topk.layer_id, num_token_non_padded=None, expert_location_dispatch_info=None)
        get_global_expert_distribution_recorder().on_select_experts(topk_ids=rec_ids)
        if use_a16:
            # routed experts + sigmoid(x . w_gate) * shared in one persistent kernel; the shared expert (side
            # stream) must be complete before it starts, and the caller then skips its gate-add (shared None).
            w13, s13, g13, w2, s2, g2 = _a16_weights(self)
            b = _a16_bufs(hidden_states.device)
            if sh8 is None:
                cur.wait_stream(self.alt_stream)
                shared_arg, sh8 = shared_output.contiguous(), (b["none"],) * 4
            else:
                shared_arg = b["none"]
            out = torch.empty_like(hidden_states)
            ext().a16_moe_decode(
                hidden_states, topk_ids.to(torch.int32).contiguous(), topk_w.to(torch.float32).contiguous(),
                w13, s13, g13.view(-1), w2, s2, g2.view(-1),
                self.shared_expert_gate.weight.view(-1), shared_arg,
                b["hbuf"], b["ybuf"], b["gl"], b["ctrl"], out, 4, b["dbg"], b["pbuf"], *sh8, b["shbuf"])
            return out, None
        topk_output = tk.StandardTopKOutput(topk_w, topk_ids, logits)
        router_output = self.experts(hidden_states, topk_output)
        cur.wait_stream(self.alt_stream)
        return router_output, shared_output

    qm.Qwen2MoeSparseMoeBlock.forward_normal_dual_stream = dual
    logger.info("qwenfast: MoE router-first override installed")


# GDN output: the gated norm returns a placeholder and records its inputs; the out_proj linear that consumes
# the placeholder runs gated-norm + GEMM in one kernel. Any other consumer path materializes the real norm.
_gdn_state = {"gdn": None, "pending": {}, "norm_forward": None}

# WP-GDNout: same fused gated-norm + INT8 out_proj, with the norm computed once (not once per CTA), bounded projection
# batches (64 regs, 4 KB smem at M=4) and an early dependent release. Bitwise equal to gdn_norm_oproj_w8 (same w8g
# weights, BF16 rounding points, eight K partials, MMA/reduction order). VARIANT tunes it for the recurrence in front:
# stream = the stock fused recurrence (M=1 keeps the original kernel), replay = the WP-GDN replay-on-commit recurrence.
GDNOUT_WP_ENABLED = ENABLED and os.environ.get("QWENFAST_GDNOUT_WP", "0") == "1"
GDNOUT_VARIANT = {"stream": 0, "replay": 1}.get(os.environ.get("QWENFAST_GDNOUT_VARIANT", "stream"))


def _gdnout_w8(core, z, wn, eps, swish, q, sc):
    x = ext()
    if (GDNOUT_WP_ENABLED and GDNOUT_VARIANT is not None and hasattr(x, "wpgdnout_w8")
            and core.dtype == torch.bfloat16 and core.ndim == 2 and core.shape[1] == 128 and core.is_contiguous()
            and core.shape[0] % 48 == 0 and 1 <= core.shape[0] // 48 <= 8
            and not (GDNOUT_VARIANT == 0 and core.shape[0] == 48)
            and z.dtype == core.dtype and z.shape == core.shape and z.is_contiguous()
            and wn.dtype == core.dtype and wn.numel() == 128 and wn.is_contiguous()
            and q.dtype == torch.int8 and q.shape == (2560, 6144) and q.is_contiguous()
            and sc.dtype == torch.bfloat16 and sc.shape == (2560, 48) and sc.is_contiguous()):
        return x.wpgdnout_w8(core, z, wn, eps, swish, q, sc, GDNOUT_VARIANT)
    return x.gdn_norm_oproj_w8(core, z, wn, eps, swish, q, sc, True, True)


def _gdn_out_proj(layer, x, bias):
    g = _gdn_state["gdn"]
    if layer is not g.out_proj:
        return None
    item = _gdn_state["pending"].pop(x.data_ptr(), None)
    if item is None:
        return None
    core, z, norm, placeholder = item
    w = layer.weight
    entry = pk_lookup(w)
    if entry is not None:
        p = entry[0]
        if (p.get("kind") == "w8" and bias is None and x.dim() == 2 and x.shape == (core.shape[0] // 48, 6144)
                and p["shape"] == (2560, 6144) and entry[1] == 0 and entry[2] == 2560):
            return _gdnout_w8(core, z, norm.weight, norm.eps, norm.activation != "sigmoid", p["q"], p["s"])
        placeholder.copy_(_gdn_state["norm_forward"](norm, core, z))
        return None
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




def draft_a16_install(model, path):
    """MTP draft MoE on NVFP4 copies of its experts (draft/convert.py output), made at load time so the KV pool is
    sized after the draft's FP8 experts are freed: <= 8-row calls run the persistent a16 kernel (dual()), larger
    calls (draft prefill) SGLang's Marlin MoE on the same NVFP4 weights. Draft precision only changes acceptance."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    d = torch.load(path, map_location="cuda")
    blocks = [m for m in model.modules() if hasattr(m, "experts") and hasattr(m, "shared_expert_gate")]
    freed = 0
    for b in blocks:
        ex = b.experts
        w = (d["w13"], d["s13"], d["g13"], d["w2"], d["s2"], d["g2"])
        ex._qf_a16 = w
        b._qf_a16sh = (d["sq1"], d["ss1"], d["sq2"], d["ss2"])
        ws = marlin_make_workspace(d["w13"].device, 4)
        n_exp = d["w13"].shape[0]

        def forward(hidden_states, topk_output, *args, _w=w, _ws=ws, _n=n_exp, **kwargs):
            return fused_marlin_moe(hidden_states=hidden_states, w1=_w[0], w2=_w[3], w1_scale=_w[1], w2_scale=_w[4],
                                    gating_output=topk_output.router_logits, topk_weights=topk_output.topk_weights,
                                    topk_ids=topk_output.topk_ids, global_num_experts=_n, w1_global_scale=_w[2],
                                    w2_global_scale=_w[5], workspace=_ws, num_bits=4, is_k_full=True, inplace=False)

        ex.forward = forward
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale", "w13_weight_scale_inv",
                     "w2_weight_scale_inv", "w13_input_scale", "w2_input_scale"):
            t = getattr(ex, name, None)
            if isinstance(t, torch.Tensor) and t.numel() > 1:
                freed += t.numel() * t.element_size()
                setattr(ex, name, torch.nn.Parameter(torch.empty(0, dtype=t.dtype, device=t.device), requires_grad=False))
    del d
    torch.cuda.empty_cache()
    logger.info("Draft a16 MoE at load: NVFP4 experts + INT8 shared expert for %d blocks, Marlin for prefill-size calls, "
                "%.2f GB of draft FP8 experts freed", len(blocks), freed / 1e9)

# Draft hot-head screening candidates (qwen4_exp_mtp ShortlistDraftLogitsProcessor) handed to the truncated draft
# proposal (eagle_worker_v2._truncated_from_logits), so it skips the top-K over the full-vocabulary logits tensor the
# candidates were scattered into. Single slot: every draft head call clears it, a screened call fills it, the next
# proposal consumes it if the shape matches (the logits processor copies logits into its own buffer, so storage differs).
_draft_cand = {}
# The draft-extend graph samples outside its graph, after replay, where no Python of the captured forward runs.
# A registration made while capturing points at the graph's static buffers; the runner keeps it per graph and restores
# it after each replay (draft_cand_captured / draft_cand_restore), when those buffers hold this replay's candidates.
_draft_cand_cap = {}


def draft_cand_clear():
    _draft_cand.clear()


def draft_cand_register(logits, exact, cidx, hot_ids):
    _draft_cand.clear()
    if CANDSAMPLE_ENABLED and hasattr(ext(), "cand_trunc_sample"):
        _draft_cand["e"] = (tuple(logits.shape), exact, cidx, hot_ids)
        if torch.cuda.is_current_stream_capturing():
            _draft_cand_cap["e"] = _draft_cand["e"]


def draft_cand_take(logits):
    e = _draft_cand.pop("e", None)
    if e is None or e[0] != tuple(logits.shape):
        return None
    return e[1:]


# WP-DraftHead: M=1 hot-head screen in two stages. A packed INT2 g64 screen over every hot row, a histogram threshold
# keeping a superset of its top-C, then the production INT4 arithmetic on the survivors only -> top-64 ids. The INT8
# row-dot / scatter / sampler after it are unchanged; a missed INT4 top-64 member could only change which proposal q the
# draft samples from (rejection sampling stays exact), never the target distribution. M=2..8 keep the INT4 screen.
DRAFTHEAD2_ENABLED = ENABLED and os.environ.get("QWEN_MTP_HOT_SCREEN2", "0") == "1"
DRAFTHEAD2_C = int(os.environ.get("QWEN_MTP_HOT_SCREEN2_C", "4608"))


@torch.no_grad()
def drafthead_make_q2(hot):
    """Packed INT2 copy of the BF16 hot rows: code b is (2*b-3)*s, s = BF16(RMS(group of 64)/2), four codes per byte
    little endian. Chunked to bound load-time memory. (None, None) when disabled or unsupported."""
    if not (DRAFTHEAD2_ENABLED and hot.is_cuda and hot.dtype == torch.bfloat16 and hot.ndim == 2
            and hot.shape[1] == 2560 and 64 <= hot.shape[0] <= 131072):
        return None, None
    n = hot.shape[0]
    q2 = torch.empty((n, 640), dtype=torch.uint8, device=hot.device)
    s2 = torch.empty((n, 40), dtype=torch.bfloat16, device=hot.device)
    for lo in range(0, n, 4096):
        hi = min(n, lo + 4096)
        xg = hot[lo:hi].float().view(-1, 40, 64)
        s = (xg.square().mean(2).sqrt().clamp_min_(1e-30) / 2).to(torch.bfloat16)
        codes = (xg / (2 * s.float().unsqueeze(-1))).floor_().clamp_(-2, 1).add_(2).to(torch.uint8).view(-1, 2560)
        q2[lo:hi] = codes[:, 0::4] | (codes[:, 1::4] << 2) | (codes[:, 2::4] << 4) | (codes[:, 3::4] << 6)
        s2[lo:hi] = s
    logger.info("Draft hot head two-stage screen: INT2 copy %s, C=%d", tuple(q2.shape), DRAFTHEAD2_C)
    return q2, s2


def drafthead_candidates(h, q4, s4, q2=None, s2=None):
    """(INT4 values, ids) of the top-64 hot rows: two-stage at M=1, the production INT4 screen otherwise."""
    x = ext()
    if (DRAFTHEAD2_ENABLED and q2 is not None and s2 is not None and h.dtype == torch.bfloat16 and h.ndim == 2
            and h.shape[0] == 1 and h.shape[1] == 2560 and h.stride(1) == 1
            and 64 <= DRAFTHEAD2_C <= q4.shape[0] and hasattr(x, "drafthead_screen")):
        return x.drafthead_screen(h, q2, s2, q4, s4, DRAFTHEAD2_C)
    m = h.shape[0]
    # the w4g kernel dispatches M=5..7 to its M=8 instance: pad so it reads only real rows
    hp = torch.cat((h, h[:1].expand(8 - m, -1))) if 5 <= m <= 7 else h
    return x.topk_rows(x.w4g_gemm(hp, q4, s4)[:m], 64)


def draft_cand_captured():
    """The registration made by the graph capture that just ran (static graph buffers), or None. Clears it."""
    return _draft_cand_cap.pop("e", None)


def draft_cand_restore(entry):
    """After replaying a graph whose capture registered `entry`; the next draft_cand_take still checks the shape."""
    _draft_cand.clear()
    if entry is not None and EXTCAND_ENABLED:
        _draft_cand["e"] = entry


def head_screen_begin(batch):
    """Before a target-verify forward: let the head run screened (device flag 0) when every request is sampled by the
    sparse verify sampler (top_k 1..56, no logprobs/grammar/penalties/logit bias/min_p); k = its candidate count."""
    if not _HEAD:
        return
    ok, k = False, 64
    try:
        reqs = getattr(batch, "reqs", None)
        si = getattr(batch, "sampling_info", None)
        if reqs and si is not None and not getattr(batch, "has_grammar", False) and si.need_top_k_sampling \
                and not si.need_min_p_sampling and si.logit_bias is None \
                and not (si.penalizer_orchestrator is not None and si.penalizer_orchestrator.is_required) \
                and not getattr(batch, "return_logprob", False) and not any(getattr(r, "return_logprob", False) for r in reqs):
            tks = [int(r.sampling_params.top_k) for r in reqs]
            if min(tks) >= 1 and max(tks) <= 56:
                ok, k = True, min(64, max(tks) + 8)
    except Exception:
        ok = False
    if ok:
        if _HEAD["k_host"] != k:
            _HEAD["k"].fill_(k)
            _HEAD["k_host"] = k
        _HEAD["dense"].fill_(0)
        _HEAD["active"] = True


def head_screen_end():
    if _HEAD and _HEAD["active"]:
        _HEAD["dense"].fill_(1)
        _HEAD["active"] = False


def head_dense_fixup(logits):
    """The sampler is about to read every logit (dense path) of a screened verify: recompute them exactly."""
    if not (_HEAD and _HEAD["active"]):
        return
    h = _HEAD.get("h_graph", {}).get(logits.shape[0] if logits.dim() == 2 else -1, _HEAD.get("h"))
    p = _HEAD_P[0] if _HEAD_P else None
    if p is None or logits.dim() != 2 or logits.shape[0] != h.shape[0]:
        raise RuntimeError("qwenfast: screened head without dense fixup state")
    logger.warning("qwenfast: screened verify head declined by the sampler; recomputing dense logits")
    logits.copy_(ext().w8g_gemm(h, p["q"], p["s"], _w8_rpw(h.shape[0], 2560)).to(logits.dtype))


_HEAD_P = []

if ENABLED:
    ext()
    _install()
