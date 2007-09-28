#include "Clarity.h"
#include "Complex.h"
#include "FFT.h"

#include <stdlib.h>
#include <omp.h>

extern bool gCUDACapable;

#ifdef TIME
#include <iostream>
#include "Stopwatch.h"

static Stopwatch totalTimer("JansenVanCittert filter (total time)");
static Stopwatch transferTimer("JansenVanCittert filter (transfer time)");
#endif

float
Clarity_GetImageMax(float *inImage, int numVoxels) {
   float max = inImage[0];
#pragma omp parallel
   {
      int numThreads = omp_get_num_threads();
      float *threadMax = new float[numThreads];
      int tid = omp_get_thread_num();
      threadMax[tid] = max;

#pragma omp for
      for (int i = 0; i < numVoxels; i++) {
         float val = inImage[i];
         if (val > threadMax[tid]) threadMax[tid] = val;
      }

      for (int i = 0; i < numThreads; i++) {
         if (threadMax[i] > max) max = threadMax[i];
      }
      delete[] threadMax;
   }

   return max;
}

extern "C"
void
JansenVanCittertDeconvolveKernelGPU(int nx, int ny, int nz,
                                    float* in, float inMax, float invMaxSq,
                                    float* i_k, float* o_k, float* i_kNext);


void
JansenVanCittertDeconvolveKernelCPU(int nx, int ny, int nz,
                                    float* in, float inMax, float invMaxSq,
                                    float* i_k, float* o_k, float* i_kNext) {
   int numVoxels = nx*ny*nz;

#pragma omp parallel for
   for (int j = 0; j < numVoxels; j++) {
      float diff = o_k[j] - inMax;
      float gamma = 1.0f - ((diff * diff) * invMaxSq);
      float val = i_k[j] + (gamma * (in[j] - o_k[j]));
      if (val < 0.0f) val = 0.0f;
      i_kNext[j] = val;
   }

}


void
JansenVanCittertDeconvolveKernel(int nx, int ny, int nz,
                                 float* in, float inMax, float invMaxSq,
                                 float* i_k, float* o_k, float* i_kNext) {
   if (gCUDACapable) {
      JansenVanCittertDeconvolveKernelGPU(nx, ny, nz, in, inMax, invMaxSq, 
                                          i_k, o_k, i_kNext);
   } else {
      JansenVanCittertDeconvolveKernelCPU(nx, ny, nz, in, inMax, invMaxSq, 
                                          i_k, o_k, i_kNext);
   }
}


ClarityResult_t 
Clarity_JansenVanCittertDeconvolve(float* outImage, float* inImage, float* psfImage, 
                                   int nx, int ny, int nz, unsigned iterations) {
   int numVoxels = nx*ny*nz;
   ClarityResult_t result = CLARITY_SUCCESS;

   // Find maximum value in the input image.
   float max = Clarity_GetImageMax(inImage, numVoxels);

#ifdef TIME
   // We'll start timing here to exclude the on-CPU maximum calculation
   totalTimer.Start();
#endif

   float A = 0.5f * max;
   float invASq = 1.0f / (A * A);

   // Fourier transform of PSF.
   float* psfFT = NULL;
   result = Clarity_R2C_Malloc((void**) &psfFT, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      return result;
   }
   result = Clarity_FFT_R2C_3D_float(nx, ny, nz, psfImage, psfFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT);
      return result;
   }

   // Set up the array holding the current guess.
   float* iPtr = NULL;
   result = Clarity_C2R_Malloc((void**)&iPtr, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT);
      return CLARITY_OUT_OF_MEMORY;
   }

   // Storage for convolution of current guess with the PSF.
   float* oPtr = NULL;
   result = Clarity_C2R_Malloc((void**) &oPtr, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(iPtr);
      return result;
   }

   // Iterate
   for (unsigned k = 0; k < iterations; k++) {
      if (k == 0)
         result = Clarity_Convolve_OTF(nx, ny, nz, inImage, psfFT, oPtr);
      else
         result = Clarity_Convolve_OTF(nx, ny, nz, iPtr, psfFT, oPtr);
      if (result != CLARITY_SUCCESS) {
         break;
      }

      if (k < iterations - 1) {
         JansenVanCittertDeconvolveKernel(nx, ny, nz, inImage,
            A, invASq, iPtr, oPtr, iPtr);
      } else {
         JansenVanCittertDeconvolveKernel(nx, ny, nz, inImage,
            A, invASq, iPtr, oPtr, outImage);
      }
   }

   Clarity_Free(psfFT); Clarity_Free(oPtr); Clarity_Free(iPtr);

#ifdef TIME
   totalTimer.Stop();
   std::cout << totalTimer << std::endl;
   std::cout << transferTimer << std::endl;
   totalTimer.Reset();
   transferTimer.Reset();
#endif

   return CLARITY_SUCCESS;
}
