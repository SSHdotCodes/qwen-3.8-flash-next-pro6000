"""WP-Spec: FutureMap relay elision vs stock overlap_utils, driven through stash / publish / resolve_forward_inputs
sequences like the overlap scheduler's (steady decode, filter, merge with a prefill stash, mixed chunked prefill,
stash_bonus_tokens, idle, payload buffers overwritten after their stash the way CUDA-graph outputs are).
Every resolved draft-input field (values, dtype, shape) and, after a final flush, every relay buffer must be equal.
Run: python3 /opt/qwenfast/src/test_spec_relay.py
"""
import importlib.util, json, sys, types
import torch

import sglang.srt.speculative.spec_utils as su

su.spec_need_hidden_states = lambda: True
import sglang.srt.managers.overlap_utils as shipped


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


stock = load("ou_reference", shipped.__file__)
stock._RELAY_ELIDE = False
new = load("ou_new", shipped.__file__)
new._RELAY_ELIDE = True
assert new._RELAY_ELIDE
dev = torch.device("cuda")
V, H = 248320, 2560 * 4


class Algo:
    def is_none(self): return False
    def is_some(self): return True
    def is_ngram(self): return False
    def need_topk(self): return True


pool = types.SimpleNamespace(req_to_token=torch.zeros((8, 64), dtype=torch.int32, device=dev))


def make(mod):
    fm = mod.FutureMap(dev, Algo(), pool, needs_cpu_seq_lens=False)
    return fm


g = torch.Generator(device="cuda").manual_seed(5)


def payload(m, step, graph_hidden=None):
    """Worker-like next_draft_input tensors (graph_hidden: a 'static graph output' buffer reused every step)."""
    topk_p = torch.rand((m, 1), device=dev, generator=g)
    topk_index = torch.randint(0, V, (m, 1), device=dev, generator=g)
    bonus = torch.randint(0, V, (m,), device=dev, generator=g).to(torch.int32)
    if graph_hidden is not None:
        graph_hidden[:m].copy_(torch.randn((m, H), device=dev, generator=g).to(torch.bfloat16))
        hidden = graph_hidden[:m]
    else:
        hidden = torch.randn((m, H), device=dev, generator=g).to(torch.bfloat16)
    q = torch.zeros((m, V), device=dev)
    q[:, torch.randint(0, V, (64,), device=dev, generator=g)] = torch.rand((m, 64), device=dev, generator=g)
    return dict(topk_p=topk_p, topk_index=topk_index, bonus_tokens=bonus, hidden_states=hidden, draft_probs=q)


def draft_input(p, fi):
    return types.SimpleNamespace(future_indices=fi, future_dsa_topk_indices_available=False, dsa_topk_indices=None, **p)


def batch(di, prefill_cpu=None, mix=None):
    return types.SimpleNamespace(prefill_input_ids_cpu=prefill_cpu, mix_running_indices=mix, enable_overlap=True,
                                 spec_algorithm=Algo(), input_ids=None, spec_info=di, device=dev, req_pool_indices=None)


FIELDS = ["topk_p", "topk_index", "bonus_tokens", "hidden_states", "draft_probs", "dsa_topk_indices"]


def snap(di):
    out = {}
    for f in FIELDS:
        t = getattr(di, f)
        out[f] = None if t is None else (t.dtype, tuple(t.shape), t.clone())
    return out


def eq(a, b):
    for f in FIELDS:
        x, y = a[f], b[f]
        if (x is None) != (y is None):
            return False
        if x is not None and (x[0] != y[0] or x[1] != y[1] or not torch.equal(x[2], y[2])):
            return False
    return True


def bufs_equal(fa, fb):
    fb._flush_deferred()
    names = ["output_tokens_buf", "topk_p_buf", "topk_index_buf", "hidden_states_buf", "draft_probs_buf"]
    res = {}
    for n in names:
        a, b = getattr(fa, n), getattr(fb, n)
        # rows never written are uninitialized (torch.empty); compare the rows the scenario wrote
        rows = torch.tensor(sorted(WRITTEN), device=dev)
        res[n] = torch.equal(a[rows], b[rows]) if n != "output_tokens_buf" else torch.equal(a[rows], b[rows])
    return all(res.values()), res


ROWS = None


WRITTEN = set()


def scenario(kind):
    WRITTEN.clear()
    fa, fb = make(stock), make(new)
    cnt = [0]
    orig_try = fb._try_elided_resolve

    def counting(di, idx):
        r = orig_try(di, idx)
        cnt[0] += int(r)
        return r

    fb._try_elided_resolve = counting
    st0 = fa.stash

    def tracking(idx, pl):
        WRITTEN.update(idx.tolist())
        return st0(idx, pl)

    fa.stash = tracking
    gh_a = torch.zeros((8, H), dtype=torch.bfloat16, device=dev)
    gh_b = torch.zeros((8, H), dtype=torch.bfloat16, device=dev)
    elided = 0
    ok = True
    X = torch.tensor([3, 5], device=dev, dtype=torch.int64)
    m = 2 if kind != "bs1" else 1
    if kind == "bs1":
        X = X[:1].clone()
    # prefill relay for the requests (non-deferred payload shapes: same spec payload from draft-extend-for-prefill)
    seed_state = g.get_state()
    pa = payload(m, 0)
    g.set_state(seed_state)
    pb = payload(m, 0)
    fa.stash(X, stock.RelayPayload(**pa)); fb.stash(X, new.RelayPayload(**pb))
    da, db = draft_input(pa, X), draft_input(pb, X)
    for step in range(1, 7):
        # next iteration: resolve
        if kind == "filter" and step == 3:
            keep = torch.tensor([1], device=dev)
            da.future_indices = da.future_indices[keep]; db.future_indices = db.future_indices[keep]
        if kind == "merge" and step == 4:
            Y = torch.tensor([6], device=dev)
            s = g.get_state(); qa = payload(1, 99); g.set_state(s); qb = payload(1, 99)
            fa.stash(Y, stock.RelayPayload(**qa)); fb.stash(Y, new.RelayPayload(**qb))
            da.future_indices = torch.cat([da.future_indices, Y]); db.future_indices = torch.cat([db.future_indices, Y])
        if kind == "bonus_only" and step == 3:
            Z = torch.tensor([1, 2], device=dev)
            toks = torch.tensor([11, 12], device=dev)
            fa.stash_bonus_tokens(Z, toks); fb.stash_bonus_tokens(Z, toks)
        if kind == "prefill_between" and step == 3:
            # a prefill forward of a new request runs between two decode steps, overwrites the graph output
            # buffer, stashes its own relay, and is then merged into the running batch
            pba, pbb = batch(None, prefill_cpu=torch.tensor([1, 2, 3])), batch(None, prefill_cpu=torch.tensor([1, 2, 3]))
            stock.resolve_forward_inputs(pba, fa); new.resolve_forward_inputs(pbb, fb)
            ok &= not fb._deferred
            gh_a.normal_(); gh_b.copy_(gh_a)
            Y = torch.tensor([6], device=dev)
            s = g.get_state(); qa = payload(1, 98, gh_a); g.set_state(s); qb = payload(1, 98, gh_b)
            fa.stash(Y, stock.RelayPayload(**qa)); fb.stash(Y, new.RelayPayload(**qb))
            da.future_indices = torch.cat([da.future_indices, Y]); db.future_indices = torch.cat([db.future_indices, Y])
        ba, bb = batch(da), batch(db)
        stock.resolve_forward_inputs(ba, fa)
        new.resolve_forward_inputs(bb, fb)
        ok &= eq(snap(da), snap(db))
        if kind == "clobber":
            # forward reuses the graph output buffer before the next stash (as the draft-extend replay does)
            gh_a.normal_(); gh_b.copy_(gh_a)
        # forward produces the next draft input; scheduler stashes and installs it
        fi = da.future_indices
        m = fi.shape[0]
        s = g.get_state(); pa = payload(m, step, gh_a); g.set_state(s); pb = payload(m, step, gh_b)
        fa.stash(fi, stock.RelayPayload(**pa)); fb.stash(fi, new.RelayPayload(**pb))
        da, db = draft_input(pa, fi), draft_input(pb, fi)
    # end of the scenario: every row any stash wrote must hold the same bytes once the deferred stash is written
    bok, detail = bufs_equal(fa, fb)
    torch.cuda.synchronize()
    return dict(kind=kind, resolved_equal=bool(ok), bufs_equal=bok, elided_resolves=cnt[0], detail=detail)


res = [scenario(k) for k in ["steady", "bs1", "filter", "merge", "bonus_only", "prefill_between", "clobber"]]
assert all(r["resolved_equal"] and r["bufs_equal"] for r in res), res
for r in res:
    print(json.dumps(r))
print(json.dumps({"all_ok": all(r["resolved_equal"] and r["bufs_equal"] for r in res)}))


# timing: stash + resolve of one steady step, stock vs elided (graph-free, as in production: eager)
def t(mod, fm):
    X = torch.tensor([3], device=dev)
    p = payload(1, 0)
    di = draft_input(p, X)
    fm.stash(X, mod.RelayPayload(**p))
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(30):
        st.record()
        for _ in range(20):
            mod.resolve_forward_inputs(batch(di), fm)
            p = dict(topk_p=di.topk_p, topk_index=di.topk_index, bonus_tokens=p["bonus_tokens"], hidden_states=di.hidden_states, draft_probs=di.draft_probs)
            fm.stash(X, mod.RelayPayload(**p))
            di = draft_input(p, X)
        en.record(); torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en) * 1000 / 20)
    return best


print(json.dumps({"stash+resolve us/step (GPU span, eager) stock/new": [round(t(stock, make(stock)), 1), round(t(new, make(new)), 1)]}))
