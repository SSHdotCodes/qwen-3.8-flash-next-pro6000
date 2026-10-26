#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <type_traits>
#include <cuda_bf16.h>
#include "gdn_replay_update.cuh"
namespace {
struct Index { const void* p=nullptr; long stride=1; bool wide=false; };
__device__ __forceinline__ long get(Index i,int n){return i.wide?((const int64_t*)i.p)[n*i.stride]:((const int*)i.p)[n*i.stride];}
struct Replay {
 void *state; const void *snap; const float *k,*d,*decay; const uint8_t* active;
 Index dest,end,track,trackstep;
 long state_layer,state_slot; int layers,slots,rows,T,B;
};
// Storage conversion happens only at the initial read and committed stores.
// Verify carries FP32 h between tokens even when snapshots/persistent slots
// are BF16. Rounding h after each replay token would change the stock result.
template<typename T> using Storage4 = std::conditional_t<std::is_same_v<T,float>,float4,uint2>;
template<typename T> __device__ __forceinline__ float4 load_state4(const T* p){
 if constexpr(std::is_same_v<T,float>)return __ldg((const float4*)p);
 else {
  uint2 bits=__ldg((const uint2*)p);
  union Pair {unsigned int bits; __nv_bfloat162 bf;};
  Pair a,b;a.bits=bits.x;b.bits=bits.y;
  float2 x=__bfloat1622float2(a.bf),y=__bfloat1622float2(b.bf);
  return make_float4(x.x,x.y,y.x,y.y);
 }
}
template<typename T> __device__ __forceinline__ void store_state4(T* p,float4 h){
 if constexpr(std::is_same_v<T,float>)*(float4*)p=h;
 else {
  union Pair {unsigned int bits; __nv_bfloat162 bf;};
  Pair a,b;a.bf=__floats2bfloat162_rn(h.x,h.y);b.bf=__floats2bfloat162_rn(h.z,h.w);
  *(uint2*)p=make_uint2(a.bits,b.bits);
 }
}
template<typename T> __global__ void __launch_bounds__(256) wpgdn_replay_commit_kernel(Replay m){
 T* state=static_cast<T*>(m.state);const T* snap=static_cast<const T*>(m.snap);
 asm volatile("griddepcontrol.wait;":::"memory");
 asm volatile("griddepcontrol.launch_dependents;":::"memory");
 int layer=blockIdx.y,req=blockIdx.z;
 int e=(blockIdx.x*256+threadIdx.x)*4;
 if(req>=m.rows || e>=786432)return;
 long dst=get(m.dest,req),end=get(m.end,req);
 long track=m.track.p?get(m.track,req):-1,ts=m.trackstep.p?get(m.trackstep,req):-1;
 bool write_main=dst>=0 && dst<m.slots && end>=0 && end<m.T;
 bool write_track=track>=0 && track<m.slots && ts>=0 && ts<m.T;
 if(!write_main && !write_track)return;
 if(!m.active[layer*m.rows+req]){
  if(write_main){
   Storage4<T> v=__ldg((const Storage4<T>*)(snap+(((long)layer*m.rows+req)*m.T+end)*786432+e));
   *(Storage4<T>*)(state+(long)layer*m.state_layer+dst*m.state_slot+e)=v;
  }
  if(write_track){
   Storage4<T> v=__ldg((const Storage4<T>*)(snap+(((long)layer*m.rows+req)*m.T+ts)*786432+e));
   *(Storage4<T>*)(state+(long)layer*m.state_layer+track*m.state_slot+e)=v;
  }
  return;
 }
 if(dst<0 || dst>=m.slots)return;
 float4 h=load_state4(state+(long)layer*m.state_layer+dst*m.state_slot+e);
 float4 committed=h,tracked=h;
 int hv=e/16384,v=(e%16384)/128,k=e%128,kh=hv/3;
 int stop=max(write_main?(int)end:-1,write_track?(int)ts:-1);
 for(int t=0;t<=stop;t++){
  long row=((long)layer*m.rows+req)*m.T+t;
  float4 nk=__ldg((const float4*)(m.k+row*2048+kh*128+k));
  float dv=__ldg(m.d+row*6144+hv*128+v),decay=__ldg(m.decay+row*48+hv);
  h.x=wpgdn_replay_update(h.x,nk.x,dv,decay);h.y=wpgdn_replay_update(h.y,nk.y,dv,decay);
  h.z=wpgdn_replay_update(h.z,nk.z,dv,decay);h.w=wpgdn_replay_update(h.w,nk.w,dv,decay);
  if(t==end)committed=h;
  if(t==ts)tracked=h;
 }
 // Match the two original scatters even when track aliases the main slot:
 // the track state is the last writer.
 if(write_main)store_state4(state+(long)layer*m.state_layer+dst*m.state_slot+e,committed);
 if(write_track)store_state4(state+(long)layer*m.state_layer+track*m.state_slot+e,tracked);
}
Index idx(const at::Tensor&t){
 TORCH_CHECK(t.is_cuda() && t.dim()==1 && (t.scalar_type()==at::kInt || t.scalar_type()==at::kLong));
 return Index{t.data_ptr(),t.stride(0),t.scalar_type()==at::kLong};
}
}
void wpgdn_replay_commit(at::Tensor state,const at::Tensor&snap,const at::Tensor&k,const at::Tensor&d,const at::Tensor&decay,const at::Tensor&active,const at::Tensor&dest,const at::Tensor&end,const std::optional<at::Tensor>&track,const std::optional<at::Tensor>&trackstep){
 TORCH_CHECK(state.is_cuda() && (state.scalar_type()==at::kFloat || state.scalar_type()==at::kBFloat16) && state.dim()==5 && state.size(2)==48 && state.size(3)==128 && state.size(4)==128);
 TORCH_CHECK(state.stride(2)==16384 && state.stride(3)==128 && state.stride(4)==1);
 TORCH_CHECK(snap.scalar_type()==state.scalar_type() && snap.dim()==6 && snap.is_contiguous() && snap.size(0)==state.size(0) && snap.size(3)==48 && snap.size(4)==128 && snap.size(5)==128);
 int L=state.size(0),R=snap.size(1),T=snap.size(2),B=dest.numel();
 TORCH_CHECK(T>=1 && T<=8 && end.numel()==B);
 TORCH_CHECK(k.scalar_type()==at::kFloat && k.is_contiguous() && k.sizes()==at::IntArrayRef({L,R,T,16,128}));
 TORCH_CHECK(d.scalar_type()==at::kFloat && d.is_contiguous() && d.sizes()==at::IntArrayRef({L,R,T,48,128}));
 TORCH_CHECK(decay.scalar_type()==at::kFloat && decay.is_contiguous() && decay.sizes()==at::IntArrayRef({L,R,T,48}));
 TORCH_CHECK(active.scalar_type()==at::kByte && active.is_contiguous() && active.sizes()==at::IntArrayRef({L,R}));
 TORCH_CHECK(track.has_value()==trackstep.has_value());
 Replay m{state.data_ptr(),snap.data_ptr(),k.data_ptr<float>(),d.data_ptr<float>(),decay.data_ptr<float>(),active.data_ptr<uint8_t>(),idx(dest),idx(end),{}, {},state.stride(0),state.stride(1),L,(int)state.size(1),R,T,B};
 if(track){TORCH_CHECK(track->numel()==B && trackstep->numel()==B);m.track=idx(*track);m.trackstep=idx(*trackstep);}
 if(B==0)return;
 const c10::cuda::CUDAGuard guard(state.device());cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(768,L,B);cfg.blockDim=dim3(256);cfg.stream=at::cuda::getCurrentCUDAStream();
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;cfg.attrs=&attr;cfg.numAttrs=1;
 if(state.scalar_type()==at::kFloat){
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,wpgdn_replay_commit_kernel<float>,m));
 }else{
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,wpgdn_replay_commit_kernel<__nv_bfloat16>,m));
 }
}
