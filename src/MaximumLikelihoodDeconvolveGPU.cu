#include <stdio.h>

#define BLOCKS 16
#define THREADS_PER_BLOCK 128


__global__ void MaximumLikelihoodDivideDeviceKernel(int n, float *out, 
                                                    float *numerator, float *divisor) {
   const int tid     = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
   const int threadN = __mul24(blockDim.x, gridDim.x);

   for (int j = tid; j < n; j += threadN) {
      float div = divisor[j];
      float num = numerator[j];
      if (div < 0.00001f)
         out[j] = 0.0f;
      else
         out[j] = num / div;
   }
}


extern "C"
void
MaximumLikelihoodDivideKernelGPU(int nx, int ny, int nz, float* out, float* a, float* b) {
   int n = nz*ny*nx;
   dim3 grid(BLOCKS);
   dim3 block(THREADS_PER_BLOCK);

   MaximumLikelihoodDivideDeviceKernel<<<grid, block>>>(n, out, a, b);
}


__global__ void MaximumLikelihoodMultiplyDeviceKernel(int n, float *out, float kappa, 
                                                      float *a, float *b) {
   const int tid     = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
   const int threadN = __mul24(blockDim.x, gridDim.x);

   for (int j = tid; j < n; j += threadN) {
      out[j] = kappa * a[j] * b[j];
   }
}


extern "C"
void
MaximumLikelihoodMultiplyKernelGPU(int nx, int ny, int nz, 
                                   float *out, float kappa, float *a, float *b) {
   int n = nz*ny*nx;
   dim3 grid(BLOCKS);
   dim3 block(THREADS_PER_BLOCK);

   MaximumLikelihoodMultiplyDeviceKernel<<<grid, block>>>(n, out, kappa, a, b);
}
