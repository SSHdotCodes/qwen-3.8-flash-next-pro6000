"""WP-Spec: fused QSA graph metadata vs the production two-kernel path (rt graph_metadata + qwenfast row launcher).

Checks every output buffer bitwise (buffers pre-filled with identical garbage, so untouched bytes are compared too)
for decode (incl. the 2-step MTP draft with per-step seq offsets), target verify and draft extend, bs 1..2 (+3, 4),
with padding, and times old vs new as CUDA-graph replays (us per replay call).
Run: python3 /opt/qwenfast/src/test_spec_qsa.py
"""
import importlib.util, json, os, sys, types
import torch
import triton

from sglang.srt.layers.attention.qsa import graph_metadata as shipped
W = shipped.__file__


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


old = load("gm_old", W)
new = load("gm_new", W)

old._QSA_FUSED = False
new._QSA_FUSED = True

# production row launcher override (qwenfast._install_qsa_meta): same kernel, one block per row, 4 warps
_k = old._qsa_graph_row_metadata_kernel


class RowMetadataLauncher:
    def __getitem__(self, grid):
        launch = _k[grid]

        def run(*args, **kwargs):
            max_pages = int(args[10]) if len(args) > 10 else int(kwargs["max_pages"])
            block = min(4096, triton.next_power_of_2(max(max_pages, 128)))
            kwargs["PAGE_BLOCK"] = max(block, int(kwargs.get("PAGE_BLOCK", 128)))
            kwargs["num_warps"] = 4
            return launch(*args, **kwargs)

        return run


old._qsa_graph_row_metadata_kernel = RowMetadataLauncher()

dev = "cuda"
RATIO, RING, CPAGE = 4, 16, 16
FULL_PAGE = RATIO * CPAGE
CTX = 262144
MAX_PAGES = -(-(-(-CTX // RATIO)) // CPAGE)  # 4096
MAX_TOKENS = 8
pool = types.SimpleNamespace(qsa_ring_size=RING, qsa_compressed_page_size=CPAGE)
g = torch.Generator(device="cpu").manual_seed(1234)
REQS = 16
ROW = CTX + 64  # req_to_token row width (> max_pages * FULL_PAGE)
req_to_token = torch.randint(0, 2_000_000, (REQS, ROW), dtype=torch.int32, generator=g).to(dev)
req_to_token[:, ::7] = -3  # some negative slots (clamped by max(.,0) in the page table)


def make_meta(seed):
    gg = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s, dt=torch.int32: torch.randint(-50, 50, s, dtype=dt, generator=gg).to(dev)
    idx = types.SimpleNamespace(
        graph_prefix_lengths=r(MAX_TOKENS),
        graph_compressed_page_table=r(MAX_TOKENS, MAX_PAGES),
        graph_compressed_lengths=r(MAX_TOKENS),
        graph_write_locs=r(MAX_TOKENS),
        decode_logical_positions=r(MAX_TOKENS),
        pending_ring_slots=r(MAX_TOKENS, dt=torch.int64),
        graph_ring_group_locs=r(MAX_TOKENS, RATIO),
        compress_ratio=RATIO,
        token_to_kv_pool=pool,
    )
    return types.SimpleNamespace(sequence_lengths=r(MAX_TOKENS), row_req_pool_indices=r(MAX_TOKENS), indexer_metadata=idx)


def sliced(m, rows):
    i = m.indexer_metadata
    idx = types.SimpleNamespace(**{k: (v[:rows] if isinstance(v, torch.Tensor) else v) for k, v in vars(i).items()})
    return types.SimpleNamespace(sequence_lengths=m.sequence_lengths[:rows], row_req_pool_indices=m.row_req_pool_indices[:rows], indexer_metadata=idx)


def bufs(m):
    i = m.indexer_metadata
    return [m.sequence_lengths, m.row_req_pool_indices] + [v for v in vars(i).values() if isinstance(v, torch.Tensor)]


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(bufs(a), bufs(b)))


def run_case(mode, bs, num_padding, seq, reqs, ext_lens=None, steps=None):
    rows = bs if mode == 0 else bs * 4
    seq_lens = torch.tensor(seq, dtype=torch.int64, device=dev)
    rpi = torch.tensor(reqs, dtype=torch.int64, device=dev)
    el = torch.tensor(ext_lens, dtype=torch.int32, device=dev) if ext_lens is not None else None
    nsets = len(steps) if steps else 1
    A = [sliced(make_meta(100 + s), rows) for s in range(nsets)]
    B = [sliced(make_meta(100 + s), rows) for s in range(nsets)]
    if steps:
        for s, add in enumerate(steps):  # production: per-step seq_lens (+step+1, int32, padded rows = 1) + two kernels
            sl = (seq_lens + add).to(torch.int32)
            if num_padding:
                sl = sl.clone(); sl[-num_padding:] = 1
            old.launch_graph_metadata(mode=0, bs=bs, num_rows=rows, seq_lens=sl, req_pool_indices=rpi, extend_lens=None,
                                      extend_len=0, num_padding=num_padding, metadata=A[s], req_to_token=req_to_token, pool=pool)
        ok = new.launch_graph_metadata_fused(mode=0, bs=bs, num_rows=rows, seq_lens=seq_lens, req_pool_indices=rpi,
                                             extend_lens=None, extend_len=0, num_padding=num_padding, metadatas=B,
                                             seq_adds=list(steps), req_to_token=req_to_token, pool=pool)
        assert ok
    else:
        kw = dict(mode=mode, bs=bs, num_rows=rows, seq_lens=seq_lens, req_pool_indices=rpi, extend_lens=el,
                  extend_len=4 if mode == 1 else 0, num_padding=num_padding, req_to_token=req_to_token, pool=pool)
        old.launch_graph_metadata(metadata=A[0], **kw)
        new.launch_graph_metadata(metadata=B[0], **kw)
    torch.cuda.synchronize()
    return all(same(a, b) for a, b in zip(A, B))


cases = []
seqs = [[1], [2], [3], [4], [5], [63], [64], [65], [4095], [100000], [CTX - 8]]
for s in seqs:
    for mode in (0, 1, 2):
        cases.append((mode, 1, 0, s, [3], [4] if mode == 2 else None, None))
    cases.append((0, 1, 0, s, [3], None, [1, 2]))
for mode in (0, 1, 2):
    for (seq, reqs) in [([17, 900], [5, 9]), ([4, 8], [1, 2]), ([130000, 7], [15, 0])]:
        for npad in (0, 1, 2):
            cases.append((mode, 2, npad, seq, reqs, [4, 4] if mode == 2 else None, None))
for (seq, reqs) in [([17, 900], [5, 9]), ([3, 4], [1, 2]), ([62, 63], [7, 8])]:
    for npad in (0, 1, 2):
        cases.append((0, 2, npad, seq, reqs, None, [1, 2]))
        cases.append((0, 2, npad, seq, reqs, None, [1]))
# non-uniform / partial draft-extend lengths and bs 1/2 with fewer accepted rows
cases += [(2, 2, 0, [40, 41], [2, 3], [3, 4], None), (2, 2, 0, [40, 41], [2, 3], [1, 2], None),
          (2, 2, 1, [40, 41], [2, 3], [4, 4], None), (2, 1, 0, [9], [4], [2], None), (2, 1, 0, [2], [4], [4], None)]
bad = [c for c in cases if not run_case(*c)]
assert not bad, f"QSA metadata mismatch: {bad[:5]}"
print(json.dumps({"cases": len(cases), "mismatch": len(bad), "bad": [str(c) for c in bad[:5]]}), flush=True)


def gtime(fn, reps=50, iters=20):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        for _ in range(reps):
            fn()
    graph.replay(); torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = 1e9
    for _ in range(iters):
        st.record(); graph.replay(); en.record(); torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en) * 1000 / reps)
    return best


timing = {}
for name, mode, bs, steps in [("verify bs1", 1, 1, None), ("verify bs2", 1, 2, None), ("extend bs1", 2, 1, None),
                              ("extend bs2", 2, 2, None), ("draft2 bs1", 0, 1, [1, 2]), ("draft2 bs2", 0, 2, [1, 2])]:
    rows = bs if mode == 0 else bs * 4
    seq_lens = torch.tensor([5000, 7001][:bs], dtype=torch.int64, device=dev)
    rpi = torch.tensor([3, 5][:bs], dtype=torch.int64, device=dev)
    el = torch.tensor([4] * bs, dtype=torch.int32, device=dev)
    nsets = len(steps) if steps else 1
    A = [sliced(make_meta(7 + s), rows) for s in range(nsets)]
    kw = dict(bs=bs, num_rows=rows, req_pool_indices=rpi, num_padding=0, req_to_token=req_to_token, pool=pool)
    if steps:
        def f_old():
            for s, add in enumerate(steps):
                sl = (seq_lens + add).to(torch.int32)
                old.launch_graph_metadata(mode=0, seq_lens=sl, extend_lens=None, extend_len=0, metadata=A[s], **kw)

        def f_new():
            new.launch_graph_metadata_fused(mode=0, seq_lens=seq_lens, extend_lens=None, extend_len=0, metadatas=A,
                                            seq_adds=list(steps), **kw)
    else:
        def f_old():
            old.launch_graph_metadata(mode=mode, seq_lens=seq_lens, extend_lens=el if mode == 2 else None,
                                      extend_len=4 if mode == 1 else 0, metadata=A[0], **kw)

        def f_new():
            new.launch_graph_metadata(mode=mode, seq_lens=seq_lens, extend_lens=el if mode == 2 else None,
                                      extend_len=4 if mode == 1 else 0, metadata=A[0], **kw)
    timing[name] = (round(gtime(f_old), 2), round(gtime(f_new), 2))
print(json.dumps({"us_per_call_old_new": timing}), flush=True)
