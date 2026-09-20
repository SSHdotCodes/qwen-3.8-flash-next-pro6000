"""Adversarial QSA ring tests against an unbounded token-history oracle.

Runs inside the pinned serving image with the five wide-ring overlays.
No target weights are loaded. Includes CUDA graph metadata replay checks.
"""

import json
import torch
from sglang.srt.layers.attention.qsa.metadata import (
    build_pending_ring_slots,
    build_group_ring_slots,
)
from sglang.srt.layers.attention.qsa.graph_metadata import (
    _qsa_graph_row_metadata_kernel,
)


def slots(positions, request, ring, extend=False):
    return build_pending_ring_slots(
        token_to_batch_idx=torch.zeros(len(positions), dtype=torch.long),
        req_pool_indices=torch.tensor([request]),
        sequence_lengths=torch.tensor([len(positions)]),
        logical_positions=positions,
        compress_ratio=4,
        is_extend=extend,
        ring_size=ring,
    )


def rejection_cases(ring, widths):
    passed = failed = 0
    for request in (1, 2):
        for start in range(1, 34):
            for width in widths:
                for accept in range(1, width + 1):
                    history = [request * 100000 + p for p in range(start)]
                    state = torch.full((3 * ring,), -1, dtype=torch.long)
                    pos = torch.arange(start)
                    # Only the incomplete prefill group must persist. Other
                    # prefill groups compress directly from current tensors.
                    loc = slots(pos, request, ring, extend=True)
                    valid = loc >= request * ring
                    state[loc[valid]] = torch.tensor(history)[valid]
                    for turn in range(4):
                        n = len(history)
                        pos = torch.arange(n, n + width)
                        proposed = [
                            request * 100000 + (turn + 1) * 1000 + p
                            for p in range(n, n + width)
                        ]
                        state[slots(pos, request, ring)] = torch.tensor(proposed)
                        full = history + proposed
                        ends = pos[(pos + 1) % 4 == 0]
                        groups = build_group_ring_slots(
                            req_pool_indices=torch.tensor([request]),
                            group_end_positions=ends,
                            sequence_ids=torch.zeros(len(ends), dtype=torch.long),
                            compress_ratio=4,
                            ring_size=ring,
                        )
                        expected = torch.tensor(
                            [full[int(e) - 3 : int(e) + 1] for e in ends],
                            dtype=torch.long,
                        ).reshape(-1, 4)
                        if not torch.equal(state[groups], expected):
                            failed += 1
                            break
                        history += proposed[:accept]
                    else:
                        passed += 1
    return dict(passed=passed, failed=failed)


def graph_cases():
    device = "cuda"
    rows = 32
    ring = 16
    req = torch.tensor([1] * 16 + [2] * 16, device=device, dtype=torch.int32)
    seq = torch.arange(1, rows + 1, device=device, dtype=torch.int32)
    table = torch.arange(3 * 128, device=device, dtype=torch.int32).reshape(3, 128) + 16
    compressed = torch.empty_like(seq)
    write = torch.empty_like(seq)
    pages = torch.empty((rows, 8), device=device, dtype=torch.int32)
    logical = torch.empty_like(seq)
    state = torch.empty(rows, device=device, dtype=torch.int64)
    groups = torch.empty((rows, 4), device=device, dtype=torch.int32)

    def launch():
        _qsa_graph_row_metadata_kernel[(rows,)](
            seq,
            req,
            compressed,
            write,
            pages,
            logical,
            state,
            groups,
            table,
            128,
            8,
            RATIO=4,
            RING_SIZE=ring,
            FULL_PAGE=16,
            PAGE_BLOCK=128,
            num_warps=1,
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for shift in range(48):
        seq.copy_(
            torch.arange(1 + shift, rows + 1 + shift, device=device, dtype=torch.int32)
        )
        graph.replay()
        positions = seq.long() - 1
        expected_slots = build_pending_ring_slots(
            token_to_batch_idx=torch.arange(rows, device=device),
            req_pool_indices=req,
            sequence_lengths=seq,
            logical_positions=positions,
            compress_ratio=4,
            is_extend=False,
            ring_size=ring,
        )
        expected_groups = build_group_ring_slots(
            req_pool_indices=req,
            group_end_positions=positions,
            sequence_ids=torch.arange(rows, device=device),
            compress_ratio=4,
            ring_size=ring,
        )
        assert torch.equal(state, expected_slots)
        assert torch.equal(groups.long(), expected_groups)
        assert torch.equal(compressed, seq // 4)
        expected_write = torch.where(seq % 4 == 0, table[req.long(), positions] // 4, 0)
        assert torch.equal(write, expected_write)
    return {"replays": 48, "rows_per_replay": rows, "passed": True}


def main():
    original = rejection_cases(4, [4])
    assert original["failed"] > 0, "Test must expose the old partial-group overwrite"
    wide = rejection_cases(16, [1, 2, 3, 4, 5, 6, 7, 8, 12, 13])
    assert wide["failed"] == 0, wide
    result = {
        "original_four_slot_ring": original,
        "wide_ring": wide,
        "cuda_graph": graph_cases(),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
