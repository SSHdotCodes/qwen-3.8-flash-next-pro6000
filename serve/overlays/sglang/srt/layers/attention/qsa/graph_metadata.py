"""GPU builders for QSA CUDA-graph replay metadata.

Compressed addressing is pure arithmetic over the page-aligned full-KV cache:
a group's compressed slot is any of its raw slots floor-divided by the compress ratio,
so per-row graph buffers are rebuilt from request lengths and ``req_to_token`` alone;
accept-dependent speculative lengths never need the host.

Both kernels run once eagerly at capture warmup,
then are recorded into the main CUDA graph through ``init_forward_metadata_in_graph``;
all inputs are stable-address runner buffers.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_graph_layout_kernel(
    # Request-level inputs.
    seq_lens_ptr,  # [bs] base lengths
    req_pool_ptr,  # [bs] request pool slots
    extend_lens_ptr,  # [bs] per-request extend lengths (draft extend)
    # Row layout buffers (graph persistent state).
    row_seq_lens_ptr,
    row_prefix_lens_ptr,
    row_req_pool_ptr,
    bs,
    num_tokens,
    num_padding,
    extend_len,  # uniform extend length (target verify); 0 -> extend_lens_ptr
    MODE: tl.constexpr,  # 0 = decode, 1 = target verify, 2 = draft extend
):
    pid = tl.program_id(0)

    if MODE == 0:
        if pid < bs:
            real = pid < bs - num_padding
            seq_len = tl.load(seq_lens_ptr + pid).to(tl.int32)
            req = tl.load(req_pool_ptr + pid).to(tl.int64)
            # Padding rows alias request slot 0: it is never allocated, so
            # its pending-ring rows are the inert dump for their state
            # stores, and its req_to_token row reads stay in-bounds.
            req = tl.where(real, req, 0)
            seq_len = tl.where(real, seq_len, 1)
            prefix = tl.maximum(seq_len - 1, 0)
            tl.store(row_seq_lens_ptr + pid, seq_len)
            tl.store(row_req_pool_ptr + pid, req.to(tl.int32))
            tl.store(row_prefix_lens_ptr + pid, prefix)
        return

    real_reqs = bs - num_padding
    if pid == bs:
        # Tail program: dummy rows for the static capacity past the real layout.
        if MODE == 1:
            row_start = real_reqs * extend_len
        else:
            row_start = 0
            for j in range(real_reqs):
                row_start += tl.load(extend_lens_ptr + j)
        for row in range(row_start, num_tokens):
            tl.store(row_seq_lens_ptr + row, 1)
            tl.store(row_prefix_lens_ptr + row, 0)
            # Request slot 0 is never allocated: inert for ring stores and
            # in-bounds for every row read.
            tl.store(row_req_pool_ptr + row, 0)
        return

    if MODE == 1:
        eff = tl.where(pid < real_reqs, extend_len, 0)
        offset = tl.minimum(pid, real_reqs) * extend_len
    else:
        eff = 0
        offset = 0
        for j in range(bs):
            e_j = tl.where(j < real_reqs, tl.load(extend_lens_ptr + j), 0)
            offset += tl.where(j < pid, e_j, 0)
            eff = tl.where(j == pid, e_j, eff)
    base = tl.load(seq_lens_ptr + pid).to(tl.int32)
    req = tl.load(req_pool_ptr + pid).to(tl.int64)
    if MODE == 1:
        prefix = base
        limit = base + eff
    else:
        prefix = tl.maximum(base - eff, 0)
        limit = base
    for j in range(eff):
        row = offset + j
        seq_len = tl.minimum(prefix + 1 + j, limit)
        tl.store(row_seq_lens_ptr + row, seq_len)
        tl.store(row_prefix_lens_ptr + row, prefix)
        tl.store(row_req_pool_ptr + row, req.to(tl.int32))


@triton.jit
def _qsa_graph_row_metadata_kernel(
    # Row layout buffers (filled by the layout kernel).
    row_seq_lens_ptr,
    row_req_pool_ptr,
    # Graph output buffers.
    compressed_lens_ptr,
    write_locs_ptr,
    page_table_ptr,
    logical_positions_ptr,
    state_slots_ptr,
    ring_locs_ptr,
    # Pool state.
    req_to_token_ptr,
    req_to_token_row_stride,
    max_pages,
    RATIO: tl.constexpr,
    RING_SIZE: tl.constexpr,
    FULL_PAGE: tl.constexpr,  # full-KV tokens per page
    PAGE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    seq_len = tl.load(row_seq_lens_ptr + row).to(tl.int32)
    req = tl.load(row_req_pool_ptr + row).to(tl.int64)
    token_row = req * req_to_token_row_stride
    current = tl.maximum(seq_len - 1, 0)
    last_loc = tl.load(req_to_token_ptr + token_row + current).to(tl.int32)

    compressed = seq_len // RATIO
    tl.store(compressed_lens_ptr + row, compressed)

    # The page-aligned allocator keeps each compression group inside one page,
    # so last_loc // RATIO is the group's compressed slot; slot 0 is the padding slot.
    boundary = (seq_len > 0) & (seq_len % RATIO == 0)
    write_loc = tl.where(boundary, last_loc // RATIO, 0)
    tl.store(write_locs_ptr + row, write_loc)

    tl.store(logical_positions_ptr + row, current)
    tl.store(state_slots_ptr + row, req * RING_SIZE + (current % RING_SIZE).to(tl.int64))
    ring_base = row.to(tl.int64) * RATIO
    for k in tl.static_range(RATIO):
        member = tl.maximum(current - (RATIO - 1 - k), 0)
        slot = req * RING_SIZE + (member % RING_SIZE).to(tl.int64)
        tl.store(ring_locs_ptr + ring_base + k, slot.to(tl.int32))

    # Page-table entries are the request's FULL-KV page ids, read from the
    # page-aligned req_to_token row; the scoring kernels turn them into
    # compressed slots as page_id * (FULL_PAGE // RATIO) + block_in_page.
    table_row = page_table_ptr + row.to(tl.int64) * max_pages
    offs = tl.arange(0, PAGE_BLOCK)
    row_width_pages = req_to_token_row_stride // FULL_PAGE
    for p0 in range(0, max_pages, PAGE_BLOCK):
        idx = p0 + offs
        valid = idx < tl.minimum(max_pages, row_width_pages)
        loc = tl.load(
            req_to_token_ptr + token_row + idx * FULL_PAGE, mask=valid, other=0
        )
        tl.store(table_row + idx, tl.maximum(loc // FULL_PAGE, 0), mask=valid)


# WP-Spec: one launch per replay instead of layout + row-metadata launches (and, for the MTP draft, one launch for
# every draft step). Each program owns one (metadata set, row, page chunk): it recomputes the row layout in registers
# with the layout kernel's arithmetic, chunk 0 stores the layout and per-row outputs, and every chunk copies its slice
# of the page table, so the 4096-page rows are spread over many CTAs. Every stored value and every untouched byte is
# the same as the two-kernel path (test_spec_qsa.py checks all buffers bitwise).
_QSA_FUSED = os.environ.get("QWEN_SPEC_QSA_FUSED", "1") == "1"
_PAGE_CHUNK = 512


@triton.jit
def _qsa_meta_row(
    row,
    chunk,
    seq_lens_ptr,
    req_pool_ptr,
    extend_lens_ptr,
    row_seq_lens_ptr,
    row_prefix_lens_ptr,
    row_req_pool_ptr,
    compressed_lens_ptr,
    write_locs_ptr,
    page_table_ptr,
    logical_positions_ptr,
    state_slots_ptr,
    ring_locs_ptr,
    req_to_token_ptr,
    req_to_token_row_stride,
    max_pages,
    bs,
    num_padding,
    extend_len,
    seq_add,
    MODE: tl.constexpr,
    RATIO: tl.constexpr,
    RING_SIZE: tl.constexpr,
    FULL_PAGE: tl.constexpr,
    PAGE_CHUNK: tl.constexpr,
):
    # ---- layout (same arithmetic as _qsa_graph_layout_kernel) ----
    real_reqs = bs - num_padding
    if MODE == 0:
        real = row < real_reqs
        seq_len = (tl.load(seq_lens_ptr + row) + seq_add).to(tl.int32)
        req = tl.load(req_pool_ptr + row).to(tl.int64)
        req = tl.where(real, req, 0)
        seq_len = tl.where(real, seq_len, 1)
        prefix = tl.maximum(seq_len - 1, 0)
    elif MODE == 1:
        real = row < real_reqs * extend_len
        pid = tl.where(real, row // extend_len, 0)
        j = row - pid * extend_len
        base = tl.load(seq_lens_ptr + pid).to(tl.int32)
        req = tl.load(req_pool_ptr + pid).to(tl.int64)
        prefix = base
        seq_len = tl.minimum(prefix + 1 + j, base + extend_len)
        seq_len = tl.where(real, seq_len, 1)
        prefix = tl.where(real, prefix, 0)
        req = tl.where(real, req, 0)
    else:
        offset = row * 0
        pid = row * 0
        jj = row * 0
        eff = row * 0
        for r in range(bs):
            e_r = tl.where(r < real_reqs, tl.load(extend_lens_ptr + r), 0)
            hit = (row >= offset) & (row < offset + e_r)
            pid = tl.where(hit, r, pid)
            jj = tl.where(hit, row - offset, jj)
            eff = tl.where(hit, e_r, eff)
            offset += e_r
        real = row < offset
        base = tl.load(seq_lens_ptr + pid).to(tl.int32)
        req = tl.load(req_pool_ptr + pid).to(tl.int64)
        prefix = tl.maximum(base - eff, 0)
        seq_len = tl.minimum(prefix + 1 + jj, base)
        seq_len = tl.where(real, seq_len, 1)
        prefix = tl.where(real, prefix, 0)
        req = tl.where(real, req, 0)
    req32 = req.to(tl.int32)
    # ---- row metadata (same arithmetic as _qsa_graph_row_metadata_kernel) ----
    req = req32.to(tl.int64)
    token_row = req * req_to_token_row_stride
    current = tl.maximum(seq_len - 1, 0)
    if chunk == 0:
        tl.store(row_seq_lens_ptr + row, seq_len)
        tl.store(row_prefix_lens_ptr + row, prefix)
        tl.store(row_req_pool_ptr + row, req32)
        last_loc = tl.load(req_to_token_ptr + token_row + current).to(tl.int32)
        compressed = seq_len // RATIO
        tl.store(compressed_lens_ptr + row, compressed)
        boundary = (seq_len > 0) & (seq_len % RATIO == 0)
        write_loc = tl.where(boundary, last_loc // RATIO, 0)
        tl.store(write_locs_ptr + row, write_loc)
        tl.store(logical_positions_ptr + row, current)
        tl.store(
            state_slots_ptr + row, req * RING_SIZE + (current % RING_SIZE).to(tl.int64)
        )
        ring_base = row.to(tl.int64) * RATIO
        for k in tl.static_range(RATIO):
            member = tl.maximum(current - (RATIO - 1 - k), 0)
            slot = req * RING_SIZE + (member % RING_SIZE).to(tl.int64)
            tl.store(ring_locs_ptr + ring_base + k, slot.to(tl.int32))
    table_row = page_table_ptr + row.to(tl.int64) * max_pages
    row_width_pages = req_to_token_row_stride // FULL_PAGE
    idx = chunk * PAGE_CHUNK + tl.arange(0, PAGE_CHUNK)
    valid = idx < tl.minimum(max_pages, row_width_pages)
    loc = tl.load(req_to_token_ptr + token_row + idx * FULL_PAGE, mask=valid, other=0)
    tl.store(table_row + idx, tl.maximum(loc // FULL_PAGE, 0), mask=valid)


@triton.jit
def _qsa_graph_meta_fused_kernel(
    seq_lens_ptr,
    req_pool_ptr,
    extend_lens_ptr,
    a_row_seq_lens,
    a_row_prefix_lens,
    a_row_req_pool,
    a_compressed_lens,
    a_write_locs,
    a_page_table,
    a_logical_positions,
    a_state_slots,
    a_ring_locs,
    b_row_seq_lens,
    b_row_prefix_lens,
    b_row_req_pool,
    b_compressed_lens,
    b_write_locs,
    b_page_table,
    b_logical_positions,
    b_state_slots,
    b_ring_locs,
    req_to_token_ptr,
    req_to_token_row_stride,
    max_pages,
    bs,
    num_rows,
    num_padding,
    extend_len,
    seq_add_a,
    seq_add_b,
    MODE: tl.constexpr,
    NSETS: tl.constexpr,
    RATIO: tl.constexpr,
    RING_SIZE: tl.constexpr,
    FULL_PAGE: tl.constexpr,
    PAGE_CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk = tl.program_id(1)
    if NSETS == 1 or pid < num_rows:
        _qsa_meta_row(
            pid, chunk, seq_lens_ptr, req_pool_ptr, extend_lens_ptr,
            a_row_seq_lens, a_row_prefix_lens, a_row_req_pool, a_compressed_lens, a_write_locs,
            a_page_table, a_logical_positions, a_state_slots, a_ring_locs,
            req_to_token_ptr, req_to_token_row_stride, max_pages, bs, num_padding, extend_len, seq_add_a,
            MODE, RATIO, RING_SIZE, FULL_PAGE, PAGE_CHUNK,
        )
    else:
        _qsa_meta_row(
            pid - num_rows, chunk, seq_lens_ptr, req_pool_ptr, extend_lens_ptr,
            b_row_seq_lens, b_row_prefix_lens, b_row_req_pool, b_compressed_lens, b_write_locs,
            b_page_table, b_logical_positions, b_state_slots, b_ring_locs,
            req_to_token_ptr, req_to_token_row_stride, max_pages, bs, num_padding, extend_len, seq_add_b,
            MODE, RATIO, RING_SIZE, FULL_PAGE, PAGE_CHUNK,
        )


def _meta_buffers(metadata):
    indexer = metadata.indexer_metadata
    return (
        metadata.sequence_lengths,
        indexer.graph_prefix_lengths,
        metadata.row_req_pool_indices,
        indexer.graph_compressed_lengths,
        indexer.graph_write_locs,
        indexer.graph_compressed_page_table,
        indexer.decode_logical_positions,
        indexer.pending_ring_slots,
        indexer.graph_ring_group_locs,
    )


def _fused_ok(metadatas, req_to_token) -> bool:
    if not _QSA_FUSED or not metadatas or len(metadatas) > 2:
        return False
    first = metadatas[0].indexer_metadata
    for md in metadatas:
        idx = md.indexer_metadata
        bufs = _meta_buffers(md)
        if not all(b.is_contiguous() for b in bufs):
            return False
        if (
            idx.graph_compressed_page_table.shape[1] != first.graph_compressed_page_table.shape[1]
            or idx.compress_ratio != first.compress_ratio
            or idx.graph_ring_group_locs.shape[1] != idx.compress_ratio
            or idx.token_to_kv_pool is not first.token_to_kv_pool
        ):
            return False
    return req_to_token.dim() == 2 and req_to_token.stride(1) == 1


def launch_graph_metadata_fused(
    *,
    mode,
    bs,
    num_rows,
    seq_lens,
    req_pool_indices,
    extend_lens,
    extend_len,
    num_padding,
    metadatas,
    seq_adds,
    req_to_token,
    pool,
) -> bool:
    """One-launch replacement for launch_graph_metadata over 1 or 2 metadata sets that share the row layout inputs
    (seq_adds[i] is added to seq_lens for set i, mode 0 only). Returns False (nothing launched) when unsupported."""
    if not _fused_ok(metadatas, req_to_token) or (mode == 1 and extend_len <= 0):
        return False
    if mode != 0 and any(s != 0 for s in seq_adds):
        return False
    for md in metadatas:
        if md.sequence_lengths.numel() < num_rows:
            return False
    indexer = metadatas[0].indexer_metadata
    max_pages = indexer.graph_compressed_page_table.shape[1]
    a = _meta_buffers(metadatas[0])
    b = _meta_buffers(metadatas[1]) if len(metadatas) == 2 else a
    nsets = len(metadatas)
    chunks = triton.cdiv(max(max_pages, 1), _PAGE_CHUNK)
    _qsa_graph_meta_fused_kernel[(num_rows * nsets, chunks)](
        seq_lens,
        req_pool_indices,
        extend_lens if extend_lens is not None else seq_lens,
        *a,
        *b,
        req_to_token,
        req_to_token.stride(0),
        max_pages,
        bs,
        num_rows,
        num_padding,
        extend_len,
        int(seq_adds[0]),
        int(seq_adds[-1]),
        MODE=mode,
        NSETS=nsets,
        RATIO=indexer.compress_ratio,
        RING_SIZE=pool.qsa_ring_size,
        FULL_PAGE=pool.qsa_compressed_page_size * indexer.compress_ratio,
        PAGE_CHUNK=_PAGE_CHUNK,
        num_warps=4,
    )
    return True


def supports_graph_metadata_kernels(pool, device) -> bool:
    """Whether the CUDA fast path can serve this pool/device pair."""

    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    return torch.device(device).type == "cuda" and isinstance(pool, QSATokenToKVPool)


def launch_graph_metadata(
    *,
    mode,
    bs,
    num_rows,
    seq_lens,
    req_pool_indices,
    extend_lens,
    extend_len,
    num_padding,
    metadata,
    req_to_token,
    pool,
) -> None:

    if launch_graph_metadata_fused(
        mode=mode,
        bs=bs,
        num_rows=num_rows,
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        extend_lens=extend_lens,
        extend_len=extend_len,
        num_padding=num_padding,
        metadatas=[metadata],
        seq_adds=[0],
        req_to_token=req_to_token,
        pool=pool,
    ):
        if _check_enabled():
            fused = [b.clone() for b in _meta_buffers(metadata)]
            _launch_two_kernel(mode, bs, num_rows, seq_lens, req_pool_indices, extend_lens, extend_len,
                               num_padding, metadata, req_to_token, pool)
            _check_report("qsa_meta_mode%d" % mode, fused, metadata)
        return
    _launch_two_kernel(mode, bs, num_rows, seq_lens, req_pool_indices, extend_lens, extend_len, num_padding,
                       metadata, req_to_token, pool)


def _check_enabled() -> bool:
    try:
        from sglang.srt.speculative import spec_fused

        return spec_fused.CHECK
    except ImportError:
        return False


def _check_report(name, fused, metadata) -> None:
    from sglang.srt.speculative import spec_fused

    spec_fused.check(name, all(torch.equal(a, b) for a, b in zip(fused, _meta_buffers(metadata))))


def _launch_two_kernel(mode, bs, num_rows, seq_lens, req_pool_indices, extend_lens, extend_len, num_padding,
                       metadata, req_to_token, pool) -> None:

    indexer = metadata.indexer_metadata
    max_pages = indexer.graph_compressed_page_table.shape[1]
    row_seq_lens = metadata.sequence_lengths
    row_req_pool = metadata.row_req_pool_indices
    row_prefix_lens = indexer.graph_prefix_lengths

    _qsa_graph_layout_kernel[(bs + 1,)](
        seq_lens,
        req_pool_indices,
        (
            extend_lens
            if extend_lens is not None
            else row_seq_lens  # unused dummy pointer
        ),
        row_seq_lens,
        row_prefix_lens,
        row_req_pool,
        bs,
        num_rows,
        num_padding,
        extend_len,
        MODE=mode,
        num_warps=1,
    )
    _qsa_graph_row_metadata_kernel[(num_rows,)](
        row_seq_lens,
        row_req_pool,
        indexer.graph_compressed_lengths,
        indexer.graph_write_locs,
        indexer.graph_compressed_page_table,
        indexer.decode_logical_positions,
        indexer.pending_ring_slots,
        indexer.graph_ring_group_locs,
        req_to_token,
        req_to_token.stride(0),
        max_pages,
        RATIO=indexer.compress_ratio,
        RING_SIZE=pool.qsa_ring_size,
        FULL_PAGE=pool.qsa_compressed_page_size * indexer.compress_ratio,
        PAGE_BLOCK=128,
        num_warps=1,
    )
