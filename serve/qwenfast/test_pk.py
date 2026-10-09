import sys, time, json, struct, torch
sys.path.insert(0, '/experiment/ext')
from build_qf import build
import pk
qf = build()
dev = 'cuda'
def real(name, rows=None):
    D = '/opt/redhat-model/target/'
    idx = json.load(open(D + 'model.safetensors.index.json'))['weight_map']
    f = D + idx[name]
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]; h = json.loads(fh.read(n)); meta = h[name]; a, b = meta['data_offsets']
        fh.seek(8 + n + a); raw = fh.read(b - a)
    t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(*meta['shape'])
    return (t[:rows] if rows else t).to(dev).contiguous()
qkv = real('model.language_model.layers.0.linear_attn.in_proj_qkv.weight')
z = real('model.language_model.layers.0.linear_attn.in_proj_z.weight')
a = real('model.language_model.layers.0.linear_attn.in_proj_a.weight'); b = real('model.language_model.layers.0.linear_attn.in_proj_b.weight')
W = torch.cat([qkv, z, b, a]).contiguous()   # 16480 x 2560 like the fused in_proj
O = real('model.language_model.layers.0.linear_attn.out_proj.weight')      # 2560 x 6144
LM = real('lm_head.weight', 65536)
for name, w in (('in_proj', W), ('out_proj', O), ('lm_head[:64K]', LM)):
    p = pk.pack(w)
    nex = p['ex_k'].numel()
    out = torch.empty_like(w)
    qf.pk_unpack(p['m8'], p['e4'], p['base'], p['ex_ptr'], p['ex_k'], p['ex_v'], out)
    exact = torch.equal(out.view(torch.int16), w.view(torch.int16))
    print(f'{name} {tuple(w.shape)}: packed {pk.nbytes(p)/w.numel()/2*100:.2f}% of bf16, exceptions {nex} ({nex/w.numel()*1e4:.2f}/1e4), unpack exact {exact}')
    assert exact
    for M in (1, 2, 4):
        x = torch.randn(M, w.shape[1], device=dev).to(torch.bfloat16)
        ref = (x.float() @ w.float().T)
        for rpw in (1, 2, 4):
            y = qf.pk_gemm(x, p['m8'], p['e4'], p['base'], p['ex_ptr'], p['ex_k'], p['ex_v'], rpw).float()
            err = ((y - ref).abs().max() / ref.abs().max()).item()
            assert err < 4e-3, (name, M, rpw, err)
    # timing vs smallm (bf16) with DRAM rotation
    NC = max(2, 700_000_000 // (w.numel() * 2))
    ws = [w.clone() for _ in range(NC)]
    ps = [pk.pack(w) for _ in range(NC)]
    def gt(fn, reps=24):
        g = torch.cuda.CUDAGraph(); fn(0); torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for i in range(reps): fn(i)
        g.replay(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(10): g.replay()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / 10 / reps * 1e6
    for M in (1, 4):
        x = torch.randn(M, w.shape[1], device=dev).to(torch.bfloat16)
        tb = gt(lambda i: qf.smallm_gemm(x, ws[i % NC], 0, 0))
        res = {r: gt(lambda i, r=r: qf.pk_gemm(x, ps[i % NC]['m8'], ps[i % NC]['e4'], ps[i % NC]['base'], ps[i % NC]['ex_ptr'], ps[i % NC]['ex_k'], ps[i % NC]['ex_v'], r)) for r in (1, 2, 4)}
        best = min(res.values())
        print(f'   M={M}: bf16 smallm {tb:.1f} us ({w.numel()*2/tb/1e6:.2f} TB/s) | pk12 ' + ' '.join(f'rpw{r} {t:.1f}' for r, t in res.items()) + f' us -> {best/tb:.2f}x')
    del ws, ps
