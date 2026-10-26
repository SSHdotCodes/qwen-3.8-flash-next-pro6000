#pragma once
#include <cuda_runtime.h>
// Exact token rounding: first round state*decay, then FMA(normalized K, D).
// D is the FP32 (raw V - dot(decayed state,K))*beta saved during verify.
__device__ __forceinline__ float wpgdn_replay_update(float state,float k,float d,float decay){
 float rounded,updated;
 // Explicit PTX preserves subnormals despite the extension's --use_fast_math.
 asm("mul.rn.f32 %0,%1,%2;":"=f"(rounded):"f"(state),"f"(decay));
 asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(updated):"f"(k),"f"(d),"f"(rounded));
 return updated;
}
