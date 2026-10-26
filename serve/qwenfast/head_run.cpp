
#include <torch/extension.h>
#include <cmath>
#include <c10/cuda/CUDAGuard.h>
at::Tensor head_pack_hi(const at::Tensor&);
void head_set_carveout(int64_t);
void head_energy(const at::Tensor&,at::Tensor&,const at::Tensor&);
void head_screen(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,const at::Tensor&,double,int64_t);
void head_screen_simt(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,const at::Tensor&,double,int64_t);
void head_screen_mma(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,const at::Tensor&,double);
void head_threshold1(const at::Tensor&,at::Tensor&,const at::Tensor&,const at::Tensor&);
void head_threshold2(const at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,const at::Tensor&,const at::Tensor&);
void head_compact(const at::Tensor&,const at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,const at::Tensor&);
void head_rescore(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor&);
void head_dense_fallback(const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor&,const at::Tensor&,const at::Tensor&);
std::vector<at::Tensor> head_run(const at::Tensor& h,const at::Tensor& q,const at::Tensor& s,const at::Tensor& hi,
 const at::Tensor& karg,const at::Tensor& dense,double z,int64_t cap,int64_t rpw){
 TORCH_CHECK(h.is_cuda()&&h.scalar_type()==at::kBFloat16&&h.dim()==2&&h.is_contiguous()&&h.size(1)==2560&&h.size(0)>=1&&h.size(0)<=8&&reinterpret_cast<uintptr_t>(h.data_ptr())%16==0);
 TORCH_CHECK(q.is_cuda()&&q.scalar_type()==at::kChar&&q.is_contiguous()&&q.dim()==2&&q.size(0)==248320&&q.size(1)==2560);
 TORCH_CHECK(s.is_cuda()&&s.scalar_type()==at::kBFloat16&&s.is_contiguous()&&s.dim()==2&&s.size(0)==q.size(0)&&s.size(1)==20);
 TORCH_CHECK(hi.is_cuda()&&hi.scalar_type()==at::kByte&&hi.is_contiguous()&&hi.dim()==2&&hi.size(0)==q.size(0)&&hi.size(1)==1280);
 TORCH_CHECK(dense.is_cuda()&&dense.scalar_type()==at::kInt&&dense.is_contiguous()&&dense.numel()==1);
 TORCH_CHECK(h.device()==q.device()&&h.device()==s.device()&&h.device()==hi.device()&&h.device()==dense.device());
 TORCH_CHECK(karg.is_cuda()&&karg.scalar_type()==at::kInt&&karg.is_contiguous()&&karg.numel()==1&&karg.device()==h.device());
 TORCH_CHECK(cap>=64&&cap<=65536&&(rpw==1||rpw==2||rpw==4)&&z>=8&&std::isfinite(z));
 c10::cuda::CUDAGuard guard(h.device());
 int64_t M=h.size(0),N=q.size(0);
 auto out=at::empty({M,N},h.options());
 auto f=h.options().dtype(at::kFloat);auto i=h.options().dtype(at::kInt);
 auto en=at::empty({M,20},f),scr=at::empty({M,N},f),sig=at::empty({M,N},f),low=at::empty({M,N},f),up=at::empty({M,N},f);
 auto keys=at::empty({M,64,64},i),tau=at::empty({M},f),counts=at::empty({M},i),overflow=at::empty({1},i),ids=at::empty({M,cap},i);
 head_energy(h,en,dense);
 head_screen(h,hi,s,en,scr,sig,low,up,dense,z,rpw);
 head_threshold1(low,keys,karg,dense);
 head_threshold2(keys,tau,counts,overflow,karg,dense);
 head_compact(up,tau,ids,counts,overflow,out,dense);
 head_rescore(h,q,s,ids,counts,overflow,dense,out);
 head_dense_fallback(h,q,s,out,dense,overflow);
 return {out,counts,overflow,scr,sig,ids,low,up,tau,keys,en};
}
// Integration: merge the declarations and head_run above into qf_bind.cpp and the
// m.def lines below into its existing PYBIND11_MODULE; compile head_*.cu only.
#ifdef WP_HEAD_STANDALONE
at::Tensor w8g_gemm(const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("w8g_gemm",&w8g_gemm);
 m.def("head_pack_hi",&head_pack_hi);
 m.def("head_set_carveout",&head_set_carveout);
 m.def("head_run",&head_run);
 m.def("head_energy",&head_energy);
 m.def("head_screen",&head_screen);
 m.def("head_screen_simt",&head_screen_simt);
 m.def("head_screen_mma",&head_screen_mma);
 m.def("head_threshold1",&head_threshold1);
 m.def("head_threshold2",&head_threshold2);
 m.def("head_compact",&head_compact);
 m.def("head_rescore",&head_rescore);
 m.def("head_dense_fallback",&head_dense_fallback);
}
#endif
