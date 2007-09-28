#include <stdio.h>


__global__ void MaximumLikelihoodCUDAKernel(int nx, int ny, int nz) {
   int x = nx;
}


extern "C"
void
MaximumLikelihoodDeconvolveKernelGPU(int nx, int ny, int nz,
                                    float* in, float inMax, float invMaxSq,
                                    float* i_k, float* o_k, float* i_kNext) {
   printf("MaximumLikelihoodDeconvolveGPU\n");
   MaximumLikelihoodCUDAKernel<<<dim3(1), dim3(1), 256>>>(nx, ny, nz);
}