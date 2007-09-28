#include <stdio.h>


__global__ void JansenVanCittertCUDAKernel(int nx, int ny, int nz) {
   int x = nx;
}


extern "C"
void
JansenVanCittertDeconvolveKernelGPU(int nx, int ny, int nz,
                                    float* in, float inMax, float invMaxSq,
                                    float* i_k, float* o_k, float* i_kNext) {
   printf("JansenVanCittertDeconvolveGPU\n");
   JansenVanCittertCUDAKernel<<<dim3(1), dim3(1), 256>>>(nx, ny, nz);
}