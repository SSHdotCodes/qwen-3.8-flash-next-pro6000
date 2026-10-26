// WP-Head: packed high-nibble screening and device-only candidate selection.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
namespace {
constexpr int HK=2560, HNG=20, HW=8, HT=256, HNB=64;
__device__ __forceinline__ float hbf(uint16_t x){return __uint_as_float(uint32_t(x)<<16);}
__device__ __forceinline__ uint32_t hkey(float f){uint32_t b=__float_as_uint(f);return b&0x80000000u?~b:b|0x80000000u;}
__device__ __forceinline__ float hfloat(uint32_t k){return __uint_as_float(k&0x80000000u?k&0x7fffffffu:~k);}

__global__ void head_pack_kernel(const int8_t* q,uint8_t* hi,int64_t n){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<n)hi[i]=uint8_t(((int(q[2*i])>>4)+8)|(((int(q[2*i+1])>>4)+8)<<4));
}
__global__ void head_energy_kernel(const __nv_bfloat16* h,float* en,int M,int stride,const int* dense){
 if(*dense)return;
 int m=blockIdx.x,g=threadIdx.x>>5,l=threadIdx.x&31;
 if(g>=20)return;
 float v=0;
 for(int j=l;j<128;j+=32){float x=__bfloat162float(h[m*stride+g*128+j]);v=fmaf(x,x,v);}
 for(int o=16;o;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o);
 if(!l)en[m*20+g]=v;
}
__device__ __forceinline__ void head_unpack8(uint32_t u,float* f){
 #pragma unroll
 for(int i=0;i<8;i++){
  uint32_t n=(u>>(4*i))&15;
  f[i]=__uint_as_float(0x47000000u|(n<<12))-32888.5f;
 }
}
template<bool XB> __device__ __forceinline__ float4 head_x4(const char* xs,int i){
 if constexpr(XB){uint2 v=*reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(xs)+i);return make_float4(__uint_as_float(v.x<<16),__uint_as_float(v.x&0xffff0000u),__uint_as_float(v.y<<16),__uint_as_float(v.y&0xffff0000u));}
 else return *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(xs)+i);
}
template<int M,int RPW,bool XB>
__global__ __launch_bounds__(256) void head_screen_kernel(
 const __nv_bfloat16* h,const uint8_t* hi,const uint16_t* s,const float* en,
 float* scr,float* sig,float* lower,float* upper,int N,int stride,float z,const int* dense){
 if(*dense)return;
 extern __shared__ __align__(16) char stage[];
 float* xs=reinterpret_cast<float*>(stage);
 __shared__ float energy[M*20];
 for(int i=threadIdx.x;i<M*20;i+=256)energy[i]=en[i];
 for(int i=threadIdx.x;i<M*(HK/8);i+=256){
  int m=i/(HK/8),k=(i%(HK/8))*8;
  uint4 v=*reinterpret_cast<const uint4*>(h+m*stride+k);
  if constexpr(XB)reinterpret_cast<uint4*>(stage)[i]=v;
  else{
   const __nv_bfloat162* p=reinterpret_cast<const __nv_bfloat162*>(&v);
   float4* out=reinterpret_cast<float4*>(xs+m*HK+k);
   float2 a=__bfloat1622float2(p[0]),b=__bfloat1622float2(p[1]),c=__bfloat1622float2(p[2]),d=__bfloat1622float2(p[3]);
   out[0]=make_float4(a.x,a.y,b.x,b.y);out[1]=make_float4(c.x,c.y,d.x,d.y);
  }
 }
 __syncthreads();
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,row0=(blockIdx.x*8+warp)*RPW;
 if(row0>=N)return;
 uint2 w[RPW][5];uint16_t scale[RPW][5];
 #pragma unroll
 for(int r=0;r<RPW;r++){
  int row=min(row0+r,N-1);
  #pragma unroll
  for(int c=0;c<5;c++){
   int k=c*512+lane*16;
   w[r][c]=__ldcs(reinterpret_cast<const uint2*>(hi+size_t(row)*(HK/2)+k/2));
   scale[r][c]=__ldg(s+row*20+k/128);
  }
 }
 float acc[RPW][M],var[RPW][M];
 #pragma unroll
 for(int r=0;r<RPW;r++)
  #pragma unroll
  for(int m=0;m<M;m++){acc[r][m]=0;var[r][m]=0;}
 #pragma unroll
 for(int c=0;c<5;c++){
  int k=c*512+lane*16;
  #pragma unroll
  for(int r=0;r<RPW;r++){
   float wf[16];head_unpack8(w[r][c].x,wf);head_unpack8(w[r][c].y,wf+8);
   float sc=hbf(scale[r][c]);
   #pragma unroll
   for(int m=0;m<M;m++){
    float p=0;
    #pragma unroll
    for(int j=0;j<4;j++){
     float4 x=head_x4<XB>(stage,m*HK+k+j*4);
     p=fmaf(wf[4*j],x.x,p);p=fmaf(wf[4*j+1],x.y,p);p=fmaf(wf[4*j+2],x.z,p);p=fmaf(wf[4*j+3],x.w,p);
    }
    acc[r][m]=fmaf(p,sc,acc[r][m]);
    var[r][m]=fmaf(sc*sc,energy[m*20+k/128],var[r][m]);
   }
  }
 }
 #pragma unroll
 for(int r=0;r<RPW;r++)
  #pragma unroll
  for(int m=0;m<M;m++){
   float a=acc[r][m],v=var[r][m];
   #pragma unroll
   for(int o=16;o;o>>=1){a+=__shfl_xor_sync(0xffffffff,a,o);v+=__shfl_xor_sync(0xffffffff,v,o);}
   if(!lane&&row0+r<N){
    int64_t ix=int64_t(m)*N+row0+r;
    float sigma=sqrtf(v*(21.25f/8.f));
    scr[ix]=a;sig[ix]=sigma;lower[ix]=a-z*sigma;upper[ix]=a+z*sigma;
   }
  }
}

template<int EPT>
__device__ void head_radix(const uint32_t(&keys)[EPT],int k,uint32_t& th,int& need,int* hist,int* bucket){
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 uint32_t prefix=0,mask=0;
 #pragma unroll 1
 for(int shift=24;shift>=0;shift-=8){
  hist[tid]=0;__syncthreads();
  #pragma unroll
  for(int e=0;e<EPT;e++)if(keys[e]&&(keys[e]&mask)==prefix)atomicAdd(hist+((keys[e]>>shift)&255),1);
  __syncthreads();
  if(!warp){
   int c[8],tot=0;
   #pragma unroll
   for(int j=0;j<8;j++){c[j]=hist[255-8*lane-j];tot+=c[j];}
   int incl=tot;
   #pragma unroll
   for(int o=1;o<32;o<<=1){int v=__shfl_up_sync(0xffffffff,incl,o);if(lane>=o)incl+=v;}
   int above=incl-tot;
   if(above<k&&k<=incl){
    #pragma unroll
    for(int j=0;j<8;j++){if(above<k&&k<=above+c[j]){bucket[0]=255-8*lane-j;bucket[1]=k-above;}above+=c[j];}
   }
  }
  __syncthreads();prefix|=uint32_t(bucket[0])<<shift;mask|=255u<<shift;k=bucket[1];__syncthreads();
 }
 th=prefix;need=k;
}
__global__ void head_threshold1_kernel(const float* lower,uint32_t* ck,int N,const int* karg,const int* dense){
 if(*dense)return;
 const int kc=max(1,min(*karg,64));
 __shared__ int hist[256],bucket[2],count[2];
 int tid=threadIdx.x,r=blockIdx.y,b=blockIdx.x,seg=(N+63)/64,lo=b*seg,high=min(N,lo+seg);
 uint32_t key[16];
 #pragma unroll
 for(int e=0;e<16;e++){int i=lo+e*256+tid;key[e]=i<high?hkey(lower[size_t(r)*N+i]):0;}
 uint32_t th;int need;head_radix(key,kc,th,need,hist,bucket);
 if(tid<2)count[tid]=0;__syncthreads();
 uint32_t* out=ck+(size_t(r)*64+b)*64;
 if(tid>=kc&&tid<64)out[tid]=0;
 #pragma unroll
 for(int e=0;e<16;e++){
  if(key[e]>th)out[atomicAdd(count,1)]=key[e];
  else if(key[e]==th&&key[e]){int j=atomicAdd(count+1,1);if(j<need)out[kc-need+j]=key[e];}
 }
}
__global__ void head_threshold2_kernel(const uint32_t* ck,float* tau,int* counts,int* overflow,const int* karg,const int* dense){
 const int kc=max(1,min(*karg,64));
 int tid=threadIdx.x,r=blockIdx.x;
 if(tid==0){counts[r]=0;if(!r)*overflow=0;}
 if(*dense)return;
 __shared__ int hist[256],bucket[2];
 uint32_t key[16];
 #pragma unroll
 for(int e=0;e<16;e++){int i=e*256+tid;key[e]=ck[size_t(r)*4096+i];}
 uint32_t th;int need;head_radix(key,kc,th,need,hist,bucket);
 if(!tid)tau[r]=hfloat(th);
}
__global__ void head_compact_kernel(const float* upper,const float* tau,int* ids,int* counts,int* overflow,
 __nv_bfloat16* out,int N,int cap,const int* dense){
 int r=blockIdx.y,i=blockIdx.x*256+threadIdx.x;
 if(i>=N)return;
 out[size_t(r)*N+i]=__float2bfloat16(__uint_as_float(0xff800000u));
 if(*dense)return;
 bool take=upper[size_t(r)*N+i]>=tau[r];
 unsigned mask=__ballot_sync(0xffffffff,take);
 int base=0;
 if((threadIdx.x&31)==0&&mask)base=atomicAdd(counts+r,__popc(mask));
 base=__shfl_sync(0xffffffff,base,0);
 if((threadIdx.x&31)==0&&mask&&base+__popc(mask)>cap)atomicExch(overflow,1);
 if(take){int j=base+__popc(mask&((1u<<(threadIdx.x&31))-1));if(j<cap)ids[r*cap+j]=i;}
}

template<int M,int RPW>void screen_launch(const at::Tensor& h,const at::Tensor& hi,const at::Tensor& s,
 const at::Tensor& en,at::Tensor& scr,at::Tensor& sig,at::Tensor& low,at::Tensor& up,const at::Tensor& dense,float z,cudaStream_t st){
 int N=hi.size(0);constexpr bool XB=M>4;
 head_screen_kernel<M,RPW,XB><<<(N+8*RPW-1)/(8*RPW),256,M*HK*(XB?2:4),st>>>(
 reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),hi.data_ptr<uint8_t>(),reinterpret_cast<const uint16_t*>(s.data_ptr()),
 en.data_ptr<float>(),scr.data_ptr<float>(),sig.data_ptr<float>(),low.data_ptr<float>(),up.data_ptr<float>(),N,h.stride(0),z,dense.data_ptr<int>());
}
template<int M>void screen_rpw(int rpw,const at::Tensor& h,const at::Tensor& hi,const at::Tensor& s,
 const at::Tensor& en,at::Tensor& scr,at::Tensor& sig,at::Tensor& low,at::Tensor& up,const at::Tensor& dense,float z,cudaStream_t st){
 if(rpw==1)screen_launch<M,1>(h,hi,s,en,scr,sig,low,up,dense,z,st);
 else if(rpw==4)screen_launch<M,4>(h,hi,s,en,scr,sig,low,up,dense,z,st);
 else screen_launch<M,2>(h,hi,s,en,scr,sig,low,up,dense,z,st);
}
}
at::Tensor head_pack_hi(const at::Tensor& q){
 TORCH_CHECK(q.is_cuda()&&q.scalar_type()==at::kChar&&q.is_contiguous()&&q.dim()==2&&q.size(1)==2560);
 c10::cuda::CUDAGuard guard(q.device());
 auto out=at::empty({q.size(0),q.size(1)/2},q.options().dtype(at::kByte));
 head_pack_kernel<<<(out.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<int8_t>(),out.data_ptr<uint8_t>(),out.numel());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
void head_energy(const at::Tensor& h,at::Tensor& en,const at::Tensor& dense){
 head_energy_kernel<<<h.size(0),640,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()),en.data_ptr<float>(),h.size(0),h.stride(0),dense.data_ptr<int>());
}
void head_screen_simt(const at::Tensor& h,const at::Tensor& hi,const at::Tensor& s,const at::Tensor& en,
 at::Tensor& scr,at::Tensor& sig,at::Tensor& low,at::Tensor& up,const at::Tensor& dense,double z,int64_t rpw){
 auto st=at::cuda::getCurrentCUDAStream();
 #define HCASE(M) case M:screen_rpw<M>(rpw,h,hi,s,en,scr,sig,low,up,dense,z,st);break;
 switch(h.size(0)){HCASE(1) HCASE(2) HCASE(3) HCASE(4) HCASE(5) HCASE(6) HCASE(7) HCASE(8) default:TORCH_CHECK(false,"head M must be 1..8");}
 #undef HCASE
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void head_threshold1(const at::Tensor& low,at::Tensor& keys,const at::Tensor& karg,const at::Tensor& dense){
 head_threshold1_kernel<<<dim3(64,low.size(0)),256,0,at::cuda::getCurrentCUDAStream()>>>(low.data_ptr<float>(),reinterpret_cast<uint32_t*>(keys.data_ptr<int>()),low.size(1),karg.data_ptr<int>(),dense.data_ptr<int>());
}
void head_threshold2(const at::Tensor& keys,at::Tensor& tau,at::Tensor& counts,at::Tensor& overflow,const at::Tensor& karg,const at::Tensor& dense){
 head_threshold2_kernel<<<tau.numel(),256,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint32_t*>(keys.data_ptr<int>()),tau.data_ptr<float>(),counts.data_ptr<int>(),overflow.data_ptr<int>(),karg.data_ptr<int>(),dense.data_ptr<int>());
}
void head_compact(const at::Tensor& upper,const at::Tensor& tau,at::Tensor& ids,at::Tensor& counts,at::Tensor& overflow,at::Tensor& out,const at::Tensor& dense){
 head_compact_kernel<<<dim3((upper.size(1)+255)/256,upper.size(0)),256,0,at::cuda::getCurrentCUDAStream()>>>(
 upper.data_ptr<float>(),tau.data_ptr<float>(),ids.data_ptr<int>(),counts.data_ptr<int>(),overflow.data_ptr<int>(),
 reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),upper.size(1),ids.size(1),dense.data_ptr<int>());
}

void head_screen_mma(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&,const at::Tensor&,double);
void head_screen(const at::Tensor& h,const at::Tensor& hi,const at::Tensor& s,const at::Tensor& en,
 at::Tensor& scr,at::Tensor& sig,at::Tensor& low,at::Tensor& up,const at::Tensor& dense,double z,int64_t rpw){
 static const bool mma=[] {const char* e=getenv("QWENFAST_HEAD_SCREEN_MMA");return !e||atoi(e)!=0;}();
 if(mma)head_screen_mma(h,hi,s,en,scr,sig,low,up,dense,z);
 else head_screen_simt(h,hi,s,en,scr,sig,low,up,dense,z,rpw);
}
