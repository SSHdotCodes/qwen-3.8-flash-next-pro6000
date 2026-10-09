import sys, time, json, struct, torch
sys.path.insert(0, '/experiment/ext')
from build_qf import build
import pk
qf = build()
dev = 'cuda'
def real(name):
    D = '/opt/redhat-model/target/'
    idx = json.load(open(D + 'model.safetensors.index.json'))['weight_map']
    f = D + idx[name]
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]; h = json.loads(fh.read(n)); meta = h[name]; a, b = meta['data_offsets']
        fh.seek(8 + n + a); raw = fh.read(b - a)
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(*meta['shape']).to(dev).contiguous()
W = real('model.language_model.layers.4.linear_attn.out_proj.weight')
wn = real('model.language_model.layers.4.linear_attn.norm.weight')
p = pk.pack(W)
print('exceptions', p['ex_k'].numel())
for T in (1, 2, 4, 8):
    core = (torch.randn(T * 48, 128, device=dev) * 0.3).to(torch.bfloat16)
    z = torch.randn(T * 48, 128, device=dev).to(torch.bfloat16)
    ref = qf.gdn_norm_oproj(core, z, wn, 1e-6, True, W, False, True).float()
    out = qf.gdn_norm_oproj_pk(core, z, wn, 1e-6, True, p['m8'], p['e4'], p['base'], p['ex_ptr'], p['ex_k'], p['ex_v'], False, True).float()
    err = ((out - ref).abs().max() / ref.abs().max()).item()
    x = torch.randn(T, 6144, device=dev).to(torch.bfloat16)
    y1 = qf.pk_gemm(x, p['m8'], p['e4'], p['base'], p['ex_ptr'], p['ex_k'], p['ex_v'], 0).float(); y0 = x.float() @ W.float().T
    err2 = ((y1 - y0).abs().max() / y0.abs().max()).item()
    print(f'T={T}: gdn pk vs bf16 rel max {err:.2e} | pk_gemm K=6144 vs fp32 ref {err2:.2e}')
    assert err < 1e-2 and err2 < 1e-2
NC = 12
Ws = [W.clone() for _ in range(NC)]; Ps = [pk.pack(W) for _ in range(NC)]
def gt(fn, reps=24):
    g = torch.cuda.CUDAGraph(); fn(0); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for i in range(reps): fn(i)
    g.replay(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / 10 / reps * 1e6
for T in (1, 4, 8):
    core = (torch.randn(T * 48, 128, device=dev) * 0.3).to(torch.bfloat16); z = torch.randn(T * 48, 128, device=dev).to(torch.bfloat16)
    t0 = gt(lambda i: qf.gdn_norm_oproj(core, z, wn, 1e-6, True, Ws[i % NC], False, True))
    t1 = gt(lambda i: qf.gdn_norm_oproj_pk(core, z, wn, 1e-6, True, Ps[i % NC]['m8'], Ps[i % NC]['e4'], Ps[i % NC]['base'], Ps[i % NC]['ex_ptr'], Ps[i % NC]['ex_k'], Ps[i % NC]['ex_v'], False, True))
    x = torch.randn(T, 6144, device=dev).to(torch.bfloat16)
    t2 = gt(lambda i: qf.smallm_gemm(x, Ws[i % NC], 0, 0))
    t3 = gt(lambda i: qf.pk_gemm(x, Ps[i % NC]['m8'], Ps[i % NC]['e4'], Ps[i % NC]['base'], Ps[i % NC]['ex_ptr'], Ps[i % NC]['ex_k'], Ps[i % NC]['ex_v'], 0))
    print(f'T={T}: gdn_norm_oproj bf16 {t0:.1f} us  pk {t1:.1f} us | o_proj smallm {t2:.1f} us  pk_gemm {t3:.1f} us')
