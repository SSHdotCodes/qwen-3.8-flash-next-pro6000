"""QK norm + RoPE + FP8 K/V conversion and scatter, preserving BF16 rounding.
Derived from the production Triton QK prepare kernel.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _attn_norm_rope_store_kernel(
    v_ptr, k_cache_ptr, v_cache_ptr, loc_ptr, stride_v_t,
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    mrope_axis_map_ptr,
    stride_qg_t,
    stride_k_t,
    stride_qo_t,
    stride_ko_t,
    stride_gate_t,
    stride_cos_t,
    stride_pos_axis,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    FP16: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    MROPE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= NUM_Q_HEADS
    local_head = tl.where(is_k, head - NUM_Q_HEADS, head)
    out_dtype = tl.float16 if FP16 else tl.bfloat16

    if is_k:
        in_base = k_ptr + token * stride_k_t + local_head * HEAD_DIM
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * stride_ko_t + local_head * HEAD_DIM
    else:
        if HAS_GATE:
            in_base = q_gate_ptr + token * stride_qg_t + local_head * 2 * HEAD_DIM
        else:
            in_base = q_gate_ptr + token * stride_qg_t + local_head * HEAD_DIM
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + token * stride_qo_t + local_head * HEAD_DIM

    loc = tl.load(loc_ptr + token)
    cache_base = k_cache_ptr + loc.to(tl.int64) * NUM_KV_HEADS * HEAD_DIM + local_head * HEAD_DIM

    # Full load -> RMSNorm variance
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < HEAD_DIM
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / HEAD_DIM
    inv_rms = tl.rsqrt(var + EPS)
    x_norm = (x * inv_rms * (w + 1.0)).to(out_dtype).to(tl.float32)

    # Pass-through tail [rotary_dim, head_dim)
    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= ROTARY_DIM)
        tl.store(out_base + head_offs, x_norm, mask=pass_mask)
        if is_k:
            tl.store(cache_base + head_offs, x_norm.to(out_dtype), mask=pass_mask & (loc != 0))

    # Reload rotary portion from L1 -> re-norm -> RoPE
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < HALF_ROTARY
    xr1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    xr2 = tl.load(in_base + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    wr1 = tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    wr2 = tl.load(w_ptr + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    xr1 = (xr1 * inv_rms * (wr1 + 1.0)).to(out_dtype).to(tl.float32)
    xr2 = (xr2 * inv_rms * (wr2 + 1.0)).to(out_dtype).to(tl.float32)

    if MROPE:
        axis = tl.load(mrope_axis_map_ptr + rot_offs, mask=rot_mask, other=0)
        pos = tl.load(
            positions_ptr + axis * stride_pos_axis + token, mask=rot_mask, other=0
        )
    else:
        pos = tl.load(positions_ptr + token)
    cache_off = pos.to(tl.int64) * stride_cos_t
    cos = tl.load(
        cos_sin_cache_ptr + cache_off + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_off + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    r1 = (xr1 * cos - xr2 * sin).to(out_dtype)
    r2 = (xr2 * cos + xr1 * sin).to(out_dtype)
    tl.store(out_base + rot_offs, r1, mask=rot_mask)
    tl.store(out_base + HALF_ROTARY + rot_offs, r2, mask=rot_mask)
    if is_k:
        tl.store(cache_base + rot_offs, r1, mask=rot_mask & (loc != 0))
        tl.store(cache_base + HALF_ROTARY + rot_offs, r2, mask=rot_mask & (loc != 0))
        vv = tl.load(v_ptr + token * stride_v_t + local_head * HEAD_DIM + head_offs, mask=head_mask, other=0.0)
        vdst = v_cache_ptr + loc.to(tl.int64) * NUM_KV_HEADS * HEAD_DIM + local_head * HEAD_DIM
        tl.store(vdst + head_offs, vv, mask=head_mask & (loc != 0))

    # Gate copy (Q heads only)
    if False:
        gate_in = in_base + HEAD_DIM
        gate_out = gate_out_ptr + token * stride_gate_t + local_head * HEAD_DIM
        g = tl.load(gate_in + head_offs, mask=head_mask, other=0.0)
        tl.store(gate_out + head_offs, g, mask=head_mask)

    # PDL: signal dependent kernels (attention/allreduce) can start early.
    # Only available on NVIDIA Hopper+ (sm_90+); guarded for AMD/other backends.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def prepare_store(qkv,q_weight,k_weight,cos_sin_cache,positions,loc,k_cache,v_cache,
                  eps=1.e-6,num_q_heads=24,num_kv_heads=2,head_dim=256,rotary_dim=64,
                  mrope_axis_map=None):
    m=qkv.shape[0]; qsize=num_q_heads*head_dim;ksize=num_kv_heads*head_dim
    qg,k,v=qkv.split([qsize*2,ksize,ksize],dim=-1)
    qo=torch.empty((m,qsize),device=qkv.device,dtype=qkv.dtype)
    ko=torch.empty((m,ksize),device=qkv.device,dtype=qkv.dtype)
    gate=qg.view(m,num_q_heads,2*head_dim)[:,:,head_dim:]
    _attn_norm_rope_store_kernel[(m,num_q_heads+num_kv_heads)](
        v,k_cache,v_cache,loc,v.stride(0),qg,k,qo,ko,gate,q_weight,k_weight,
        cos_sin_cache,positions,mrope_axis_map,qg.stride(0),k.stride(0),qo.stride(0),
        ko.stride(0),gate.stride(0),cos_sin_cache.stride(0),positions.stride(0),
        NUM_Q_HEADS=num_q_heads,NUM_KV_HEADS=num_kv_heads,HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,HALF_ROTARY=rotary_dim//2,
        HEAD_BLOCK=triton.next_power_of_2(head_dim),ROT_HALF_BLOCK=triton.next_power_of_2(rotary_dim//2),
        EPS=eps,FP16=qkv.dtype==torch.float16,HAS_PASS=rotary_dim<head_dim,HAS_GATE=True,
        MROPE=positions.ndim==2,ENABLE_PDL=False, num_warps=4)
    return qo,ko,v,gate
