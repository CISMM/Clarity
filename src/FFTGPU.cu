#include <cuda.h>
#include <stdio.h>

#include "ComplexCUDA.h"

#define BLOCKS 16
#define THREADS_PER_BLOCK 128


__global__ void ModulateCUDAKernel(int n, float scale, Complex* inFT, Complex* psfFT, Complex* outFT) {
   const int tid     = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
   const int threadN = __mul24(blockDim.x, gridDim.x);

   for (int voxelID = tid; voxelID < n; voxelID += threadN) {
      outFT[voxelID] = complexMulAndScale(inFT[voxelID], psfFT[voxelID], scale);
   }

}


extern "C"
void
Clarity_Modulate_KernelGPU(int nx, int ny, int nz, float* inFT,
                           float* psfFT, float* outFT) {
   int n = nz*ny*(nx/2 + 1);
   dim3 grid(BLOCKS);
   dim3 block(THREADS_PER_BLOCK);
   float scale = 1.0f / ((float) nx*ny*nz);

   ModulateCUDAKernel<<<grid, block>>>(n, scale, (Complex*)inFT, 
      (Complex*)psfFT, (Complex*)outFT);

   cudaError result = cudaThreadSynchronize();
   if (result != cudaSuccess) {
      fprintf(stderr, "CUDA error: %s in file '%s' in line %i : %s.\n",
              "ModulateCUDAKernel failed", __FILE__, __LINE__, cudaGetErrorString(result));

   }
}
