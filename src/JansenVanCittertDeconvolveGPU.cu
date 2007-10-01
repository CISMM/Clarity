#include <stdio.h>

#define BLOCKS 16
#define THREADS_PER_BLOCK 128


__global__ void JansenVanCittertCUDAKernel(int n, float* in, float inMax,
                                           float invMaxSq, float* i_k,
                                           float* o_k, float* i_kNext) {
   const int tid     = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
   const int threadN = __mul24(blockDim.x, gridDim.x);

   for (int j = tid; j < n; j += threadN) {
      float diff = o_k[j] - inMax;
      float gamma = 1.0f - ((diff * diff) * invMaxSq);
      float val = i_k[j] + (gamma * (in[j] - o_k[j]));
      i_kNext[j] = max(val, 0.0f);
   }
}


extern "C"
void
JansenVanCittertDeconvolveKernelGPU(int nx, int ny, int nz,
                                    float* in, float inMax, float invMaxSq,
                                    float* i_k, float* o_k, float* i_kNext) {
   int n = nz*ny*nx;
   dim3 grid(BLOCKS);
   dim3 block(THREADS_PER_BLOCK);

   JansenVanCittertCUDAKernel<<<grid, block>>>(n, in, inMax, invMaxSq, 
      i_k, o_k, i_kNext);

}
