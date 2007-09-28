#include <stdio.h>


__global__ void WienerCUDAKernel(int nx, int ny, int nz) {
   int x = nx;
}


extern "C"
void
WienerDeconvolveKernelGPU(int nx, int ny, int nz, float* inFT, float* psfFT, 
                    float* result, float sigma, float epsilon) {
   printf("WienerDeconvolveGPU\n");
   WienerCUDAKernel<<<dim3(1), dim3(1), 256>>>(nx, ny, nz);
}
