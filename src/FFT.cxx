#include "FFT.h"

#include "Complex.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <fftw3.h>

extern bool gCUDACapable;

#ifdef TIME
#include <iostream>
#include "Stopwatch.h"
#endif


ClarityResult_t
Clarity_Complex_Malloc(void** buffer, size_t size, int nx, int ny, int nz) {
   ClarityResult_t result = CLARITY_SUCCESS;

   size_t totalSize = size*2*nz*ny*(nx/2 + 1);
   if (gCUDACapable) {
      cudaError_t cudaResult = cudaMalloc(buffer, totalSize);
      if (cudaResult != cudaSuccess) {
         return CLARITY_DEVICE_OUT_OF_MEMORY;
      }
   } else {
      *buffer = fftwf_malloc(totalSize);
      if (*buffer == NULL) {
         result = CLARITY_OUT_OF_MEMORY;
      }
   }

   return result;
}


ClarityResult_t
Clarity_Real_Malloc(void** buffer, size_t size, int nx, int ny, int nz) {
   ClarityResult_t result = CLARITY_SUCCESS;

   size_t totalSize = size*nz*ny*nx;
   if (gCUDACapable) {
      cudaError_t cudaResult = cudaMalloc(buffer, totalSize);
      if (cudaResult != cudaSuccess) {
         return CLARITY_DEVICE_OUT_OF_MEMORY;
      }
   } else {
      *buffer = fftwf_malloc(sizeof(size)*nz*ny*nx);
      if (*buffer == NULL) {
         result = CLARITY_OUT_OF_MEMORY;
      }
   }

   return result;
}


ClarityResult_t
Clarity_CopyToDevice(int nx, int ny, int nz, size_t size, void* dst, void* src) {
   if (!gCUDACapable)
      return CLARITY_INVALID_OPERATION;

   size_t totalSize = size*nx*ny*nz;
   cudaError_t cudaResult = cudaMemcpy(dst, src, totalSize,
      cudaMemcpyHostToDevice);
   if (cudaResult != cudaSuccess) {
      return CLARITY_INVALID_OPERATION;
   }

   return CLARITY_SUCCESS;
}


ClarityResult_t
Clarity_CopyFromDevice(int nx, int ny, int nz, size_t size, void* dst, void* src) {
   if (!gCUDACapable)
      return CLARITY_INVALID_OPERATION;

   size_t totalSize = size*nx*ny*nz;
   cudaError_t cudaResult = cudaMemcpy(dst, src, totalSize,
      cudaMemcpyDeviceToHost);
   if (cudaResult != cudaSuccess) {
      return CLARITY_INVALID_OPERATION;
   }

   return CLARITY_SUCCESS;
}


ClarityResult_t
Clarity_Real_MallocCopy(void** buffer, size_t size, int nx, int ny, int nz, void* src) {
   if (!gCUDACapable)
      return CLARITY_INVALID_OPERATION;

   ClarityResult_t result = CLARITY_SUCCESS;
   result = Clarity_Real_Malloc(buffer, size, nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      return result;
   }

   result = Clarity_CopyToDevice(nx, ny, nz, size, *buffer, src);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(*buffer);
      return result;
   }

   return result;
}


void
Clarity_Free(void* buffer) {
   if (buffer == NULL)
      return;
   if (gCUDACapable) {
      cudaFree(buffer);
   } else {
      fftwf_free(buffer);
   }
}


ClarityResult_t
Clarity_FFT_R2C_float(int nx, int ny, int nz, float* in, float* out) {

   if (gCUDACapable) {
      cufftHandle plan;
      cufftResult cufftResult = cufftPlan3d(&plan, nz, ny, nx, CUFFT_R2C);
      if (cufftResult != CUFFT_SUCCESS) {
         return CLARITY_FFT_FAILED;
      }
#ifdef TIME
      Stopwatch timer;
      timer.Start();
#endif
      cufftResult = cufftExecR2C(plan, (cufftReal*)in, (cufftComplex*)out);
#ifdef TIME
      timer.Stop();
      std::cout << "R2C: " << timer << std::endl;
#endif

      cufftDestroy(plan);
      if (cufftResult != CUFFT_SUCCESS) {
         return CLARITY_FFT_FAILED;
      }
   } else {
      fftwf_complex* outComplex = (fftwf_complex*) out;

      // Holy smokes, I wasted a lot of time just to find that FFTW
      // expects data arranged not in the normal way in medical/biological
      // imaging. Dimension sizes need to be reversed.
      fftwf_plan plan = fftwf_plan_dft_r2c_3d(nz, ny, nx, in, outComplex,
         FFTW_ESTIMATE);
      if (plan == NULL) {
         return CLARITY_FFT_FAILED;
      }
#ifdef TIME
      Stopwatch timer;
      timer.Start();
#endif
      fftwf_execute(plan);
#ifdef TIME
      timer.Stop();
      std::cout << "R2C: " << timer << std::endl;
#endif

      fftwf_destroy_plan(plan);
   }

   return CLARITY_SUCCESS;
}


ClarityResult_t
Clarity_FFT_C2R_float(int nx, int ny, int nz, float* in, float* out) {
   int numVoxels = nx*ny*nz;

   if (gCUDACapable) {
      cufftHandle plan;
      cufftResult cufftResult = cufftPlan3d(&plan, nz, ny, nx, CUFFT_C2R);
      if (cufftResult != CUFFT_SUCCESS) {
         return CLARITY_FFT_FAILED;
      }
      cufftResult = cufftExecC2R(plan, (cufftComplex*)in, (cufftReal*)out);
      cufftDestroy(plan);
      if (cufftResult != CUFFT_SUCCESS) {
         return CLARITY_FFT_FAILED;
      }
   } else {
      fftwf_complex* inComplex = (fftwf_complex*) in;

      // Holy smokes, I wasted a lot of time just to find that FFTW expects
      // data arranged not in the normal way in medical/biological imaging.
      // Dimension sizes need to be reversed.
      fftwf_plan plan = fftwf_plan_dft_c2r_3d(nz, ny, nx, inComplex, out, 
         FFTW_ESTIMATE);
      if (plan == NULL) {
         return CLARITY_FFT_FAILED;
      }
      fftwf_execute(plan);
      fftwf_destroy_plan(plan);
   }

   return CLARITY_SUCCESS;
}


extern "C"
void
Clarity_Modulate_KernelGPU(int nx, int ny, int nz, float* inFT,
                           float* otf, float* outFT);


void
Clarity_Modulate_KernelCPU(int nx, int ny, int nz, float* inFT,
                           float* otf, float* outFT) {
   int numVoxels = nz*ny*(nx/2 + 1);
   float scale = 1.0f / ((float) nz*ny*nx);
#pragma omp parallel for
   for (int i = 0; i < numVoxels; i++) {
      ComplexMultiplyAndScale(inFT + (2*i), otf + (2*i), scale, outFT + (2*i));
   }
}


void
Clarity_Modulate(int nx, int ny, int nz, float* in, float* otf, float* out) {
   if (gCUDACapable) {
      Clarity_Modulate_KernelGPU(nx, ny, nz, in, otf, out);
   } else {
      Clarity_Modulate_KernelCPU(nx, ny, nz, in, otf, out);
   }
}


ClarityResult_t
Clarity_Convolve_OTF(int nx, int ny, int nz, float* in, float* otf, float* out) {
   ClarityResult_t result = CLARITY_SUCCESS;

   float* inFT = NULL;
   result = Clarity_Complex_Malloc((void**) &inFT, sizeof(float), nx, ny, nz);
   if (result == CLARITY_OUT_OF_MEMORY) {
      return result;
   }

   result = Clarity_FFT_R2C_float(nx, ny, nz, in, inFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT);
      return result;
   }

   Clarity_Modulate(nx, ny, nz, inFT, otf, inFT);

   result = Clarity_FFT_C2R_float(nx, ny, nz, inFT, out);
   Clarity_Free(inFT);

   return result;
}


ClarityResult_t
Clarity_ConvolveInternal(int nx, int ny, int nz, float* in, float* psf, float* out) {
   ClarityResult_t result = CLARITY_SUCCESS;
   float* inFT = NULL;
   result = Clarity_Complex_Malloc((void**) &inFT, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) { 
      return result;
   }
   result = Clarity_FFT_R2C_float(nx, ny, nz, in, inFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT);
      return result;
   }

   float* psfFT = NULL;
   result = Clarity_Complex_Malloc((void**) &psfFT, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT);
      return result;
   }
   result = Clarity_FFT_R2C_float(nx, ny, nz, psf, psfFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT); Clarity_Free(psfFT);
      return result;
   }

   // Modulate the two transforms
   Clarity_Modulate(nx, ny, nz, inFT, psfFT, inFT);
   Clarity_Free(psfFT);

   result = Clarity_FFT_C2R_float(nx, ny, nz, inFT, out);
   Clarity_Free(inFT);

   return result;
}