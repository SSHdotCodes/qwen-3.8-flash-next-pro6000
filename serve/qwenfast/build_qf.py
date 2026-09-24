import os, sys, time
from torch.utils.cpp_extension import load
def build(verbose=False):
    d = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(d + '/build_qf', exist_ok=True)
    return load(name='qwenfast', sources=[d + '/qf_bind.cpp', d + '/qf_smallm.cu', d + '/qf_hcmix.cu', d + '/qf_moe.cu', d + '/qf_hc2.cu', d + '/qf_router.cu', d + '/qf_gdnout.cu', d + '/qf_sample.cu'],
                build_directory=d + '/build_qf', verbose=verbose,
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3', '-gencode=arch=compute_120a,code=sm_120a', '--use_fast_math', '-Xptxas=-v', '-lineinfo'])
if __name__ == '__main__':
    t = time.time(); build(verbose='-v' in sys.argv); print('built in', round(time.time() - t, 1), 's')
