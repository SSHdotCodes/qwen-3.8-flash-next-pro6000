import os, sys, time
from torch.utils.cpp_extension import load
SRC = ['qf_bind.cpp', 'qf_smallm.cu', 'qf_hcmix.cu', 'qf_moe.cu', 'qf_hc2.cu', 'qf_router.cu', 'qf_gdnout.cu', 'qf_sample.cu', 'qf_w8.cu', 'qf_pf.cu', 'qf_pk.cu', 'qf_w8g.cu', 'qf_hc8.cu', 'qf_topk.cu', 'qf_w4s.cu']
def build(verbose=False):
    d = os.path.dirname(os.path.abspath(__file__))
    bd = os.environ.get("QF_BUILD", d + "/build_qf"); os.makedirs(bd, exist_ok=True)
    return load(name='qwenfast', sources=[d + '/' + f for f in SRC],
                build_directory=bd, verbose=verbose,
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3', '-gencode=arch=compute_120a,code=sm_120a', '--use_fast_math', '-lineinfo'])
if __name__ == '__main__':
    t = time.time(); build(verbose='-v' in sys.argv); print('built in', round(time.time() - t, 1), 's')
