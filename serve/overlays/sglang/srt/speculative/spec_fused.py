"""WP-Spec: single-launch replacements for chains of tiny torch ops in the EAGLE/MTP decode step.

Each helper returns exactly the tensors (values, dtypes, shapes) the replaced torch expressions produce, or None when
its inputs are outside the shapes/dtypes it handles (callers then run the original code). QWEN_SPEC_FUSED=0 disables.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_ENABLED = os.environ.get("QWEN_SPEC_FUSED", "1") == "1"


def _ok(*tensors) -> bool:
    return _ENABLED and all(t.is_cuda and t.is_contiguous() for t in tensors)


@triton.jit
def _post_verify_kernel(
    seq_lens_ptr,
    accept_lens_ptr,
    predict_ptr,
    accept_index_ptr,
    new_seq_lens_ptr,
    bonus_ptr,
    bs,
    predict_numel,
    DEPTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    m = i < bs
    acc = tl.load(accept_lens_ptr + i, mask=m, other=1)
    seq = tl.load(seq_lens_ptr + i, mask=m, other=0)
    tl.store(new_seq_lens_ptr + i, seq + acc.to(tl.int64), mask=m)
    # bonus = predict[accept_index][i, acc - 1] (torch advanced indexing: negative indices wrap)
    node = tl.load(accept_index_ptr + i * DEPTH + acc - 1, mask=m, other=0).to(tl.int64)
    node = tl.where(node < 0, node + predict_numel, node)
    tok = tl.load(predict_ptr + node, mask=m, other=0)
    tl.store(bonus_ptr + i, tok, mask=m)


def post_verify(
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(seq_lens + accept_lens, bonus) where bonus is fill_bonus_tokens over predict[accept_index]."""
    bs = seq_lens.shape[0]
    if not (
        _ok(seq_lens, accept_lens, predict, accept_index)
        and 0 < bs <= 64
        and seq_lens.dtype == torch.int64
        and accept_lens.dtype == torch.int32
        and accept_lens.shape == (bs,)
        and predict.dim() == 1
        and predict.dtype == torch.int32
        and accept_index.dim() == 2
        and accept_index.shape[0] == bs
        and accept_index.dtype == torch.int32
    ):
        return None
    new_seq_lens = torch.empty_like(seq_lens)
    bonus = torch.empty((bs,), dtype=torch.int32, device=seq_lens.device)
    _post_verify_kernel[(1,)](
        seq_lens,
        accept_lens,
        predict,
        accept_index,
        new_seq_lens,
        bonus,
        bs,
        predict.numel(),
        DEPTH=accept_index.shape[1],
        BLOCK=triton.next_power_of_2(bs),
        num_warps=1,
    )
    return new_seq_lens, bonus


@triton.jit
def _draft_extend_prep_kernel(
    accept_lens_ptr,
    predict_ptr,
    num_correct_ptr,
    select_index_ptr,
    next_ids_ptr,
    bs,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    m = i < bs
    acc = tl.load(accept_lens_ptr + i, mask=m, other=0)
    tl.store(num_correct_ptr + i, acc - 1, mask=m)
    tl.store(select_index_ptr + i, i.to(tl.int64) * N + acc.to(tl.int64) - 1, mask=m)
    t = tl.arange(0, BLOCK_T)
    mt = t < bs * N
    tl.store(next_ids_ptr + t, tl.load(predict_ptr + t, mask=mt, other=0).to(tl.int64), mask=mt)


def draft_extend_prep(
    accept_lens: torch.Tensor, predict: torch.Tensor, num_draft_tokens: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """(accept_lens - 1, arange(0, bs*N, N) + accept_lens - 1, predict.to(int64))."""
    bs = accept_lens.shape[0]
    n = int(num_draft_tokens)
    if not (
        _ok(accept_lens, predict)
        and 0 < bs <= 64
        and n > 0
        and accept_lens.dim() == 1
        and accept_lens.dtype == torch.int32
        and predict.dtype == torch.int32
        and predict.shape == (bs * n,)
    ):
        return None
    dev = accept_lens.device
    num_correct = torch.empty((bs,), dtype=torch.int32, device=dev)
    select_index = torch.empty((bs,), dtype=torch.int64, device=dev)
    next_ids = torch.empty((bs * n,), dtype=torch.int64, device=dev)
    _draft_extend_prep_kernel[(1,)](
        accept_lens,
        predict,
        num_correct,
        select_index,
        next_ids,
        bs,
        N=n,
        BLOCK=triton.next_power_of_2(bs),
        BLOCK_T=triton.next_power_of_2(bs * n),
        num_warps=1,
    )
    return num_correct, select_index, next_ids


@triton.jit
def _extend_lens_kernel(
    seq_lens_ptr,
    prefix_ptr,
    extend_ptr,
    seq_plus_ptr,
    bs,
    front_offset,
    window,
    seq_add,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    m = i < bs
    seq = tl.load(seq_lens_ptr + i, mask=m, other=0)
    tl.store(prefix_ptr + i, tl.maximum(seq - front_offset, 0).to(tl.int32), mask=m)
    tl.store(extend_ptr + i, tl.full([BLOCK], 0, tl.int32) + window, mask=m)
    tl.store(seq_plus_ptr + i, seq + seq_add, mask=m)


def extend_lens(
    seq_lens: torch.Tensor, front_offset: int, num_window_tokens: int, seq_add: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """((seq_lens - front_offset).clamp(min=0).to(int32), full((bs,), window, int32), seq_lens + seq_add)."""
    bs = seq_lens.shape[0]
    if not (
        _ok(seq_lens)
        and 0 < bs <= 64
        and seq_lens.dim() == 1
        and seq_lens.dtype == torch.int64
        and 0 <= int(num_window_tokens) < 2**31
    ):
        return None
    dev = seq_lens.device
    prefix = torch.empty((bs,), dtype=torch.int32, device=dev)
    ext = torch.empty((bs,), dtype=torch.int32, device=dev)
    seq_plus = torch.empty_like(seq_lens)
    _extend_lens_kernel[(1,)](
        seq_lens,
        prefix,
        ext,
        seq_plus,
        bs,
        int(front_offset),
        int(num_window_tokens),
        int(seq_add),
        BLOCK=triton.next_power_of_2(bs),
        num_warps=1,
    )
    return prefix, ext, seq_plus


# ---- validation mode (QWEN_SPEC_CHECK=1): every WP-Spec fast path also runs the original code and compares (host
# syncs; for correctness runs only, never for timing).
CHECK = os.environ.get("QWEN_SPEC_CHECK", "0") == "1"
_check_counts = {}


def check(name: str, ok: bool) -> None:
    good, bad = _check_counts.get(name, (0, 0))
    good, bad = (good + 1, bad) if ok else (good, bad + 1)
    _check_counts[name] = (good, bad)
    n = good + bad
    if not ok or n & (n - 1) == 0 or n % 500 == 0:
        import logging

        logging.getLogger(__name__).warning("WP-Spec check %s: ok=%d bad=%d", name, good, bad)


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))
