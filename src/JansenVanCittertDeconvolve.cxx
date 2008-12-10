#include "Clarity.h"

#include <iostream>
#include <stdlib.h>
#include <omp.h>

#include "Convolve.h"
#include "FFT.h"
#include "Memory.h"

extern bool g_CUDACapable;

#ifdef TIME
#include <iostream>
#include "Stopwatch.h"

static Stopwatch totalTimer("JansenVanCittert filter (total time)");
static Stopwatch transferTimer("JansenVanCittert filter (transfer time)");
#endif

float
Clarity_GetImageMax(
   float *inImage, int numVoxels) {

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


void
JansenVanCittertDeconvolveKernelCPU(
   int nx, int ny, int nz, float* in, float inMax, float invMaxSq,
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


ClarityResult_t 
Clarity_JansenVanCittertDeconvolveCPU(
   float* outImage, float* inImage, float* psfImage, 
   int nx, int ny, int nz, float max, unsigned iterations) {
   
   ClarityResult_t result = CLARITY_SUCCESS;
   float A = 0.5f * max;
   float invASq = 1.0f / (A * A);

   // Fourier transform of PSF.
   float* psfFT = NULL;
   result = Clarity_Complex_Malloc((void**) &psfFT, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      return result;
   }
   result = Clarity_FFT_R2C_float(nx, ny, nz, psfImage, psfFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT);
      return result;
   }

   // Set up the array holding the current guess.
   float* iPtr = NULL;
   result = Clarity_Real_Malloc((void**)&iPtr, sizeof(float), 
      nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT);
      return CLARITY_OUT_OF_MEMORY;
   }

   // Storage for convolution of current guess with the PSF.
   float* oPtr = NULL;
   result = Clarity_Real_Malloc((void**) &oPtr, sizeof(float), 
      nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(iPtr);
      return result;
   }

   // First iteration.
   result = Clarity_Convolve_OTF(nx, ny, nz, inImage, psfFT, oPtr);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(oPtr); Clarity_Free(iPtr);
      return result;
   }
   if (iterations > 1) {
      JansenVanCittertDeconvolveKernelCPU(nx, ny, nz, inImage, 
         A, invASq, inImage, oPtr, iPtr);
   } else {
      JansenVanCittertDeconvolveKernelCPU(nx, ny, nz, inImage, 
         A, invASq, inImage, oPtr, outImage);
   }

   // Iterate
   for (unsigned k = 1; k < iterations; k++) {
      result = Clarity_Convolve_OTF(nx, ny, nz, iPtr, psfFT, oPtr);
      if (result != CLARITY_SUCCESS) break;

      if (k < iterations - 1) {
         JansenVanCittertDeconvolveKernelCPU(nx, ny, nz, inImage,
            A, invASq, iPtr, oPtr, iPtr);
      } else {
         JansenVanCittertDeconvolveKernelCPU(nx, ny, nz, inImage,
            A, invASq, iPtr, oPtr, outImage);
      }
   }

   Clarity_Free(psfFT); Clarity_Free(oPtr); Clarity_Free(iPtr);

   return result;
}

#ifdef BUILD_WITH_CUDA

#include "JansenVanCittertDeconvolveGPU.h"


ClarityResult_t 
Clarity_JansenVanCittertDeconvolveGPU(
   float* outImage, float* inImage, float* psfImage, 
   int nx, int ny, int nz, float max, unsigned iterations) {

   ClarityResult_t result = CLARITY_SUCCESS;
   float A = 0.5f * max;
   float invASq = 1.0f / (A * A);

   // Copy over PSF and take its Fourier transform.
   float* psf = NULL;
   result = Clarity_Real_MallocCopy((void**) &psf, sizeof(float), 
      nx, ny, nz, psfImage);
   if (result != CLARITY_SUCCESS) {
      return result;
   }
   float* psfFT = NULL;
   result = Clarity_Complex_Malloc((void**) &psfFT, sizeof(float), 
      nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psf);
      return result;
   }
   result = Clarity_FFT_R2C_float(nx, ny, nz, psf, psfFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psf); Clarity_Free(psfFT);
      return result;
   }
   Clarity_Free(psf);

   // Copy over image.
   float* in = NULL;
   result = Clarity_Real_MallocCopy((void**) &in, sizeof(float), 
      nx, ny, nz, inImage);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT);
      return result;
   }

   // Set up the array holding the current guess.
   float* iPtr = NULL;
   result = Clarity_Real_Malloc((void**)&iPtr, sizeof(float), 
      nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(in);
      return result;
   }

   // Storage for convolution of current guess with the PSF.
   float* oPtr = NULL;
   result = Clarity_Real_Malloc((void**)&oPtr, sizeof(float), 
      nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(in); Clarity_Free(iPtr);
      return result;
   }

   // First iteration.
   result = Clarity_Convolve_OTF(nx, ny, nz, in, psfFT, oPtr);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(in);
      Clarity_Free(iPtr); Clarity_Free(oPtr);
      return result;
   }
   JansenVanCittertDeconvolveKernelGPU(nx, ny, nz, in, A, invASq, 
      in, oPtr, iPtr);

   // Iterate
   for (unsigned k = 1; k < iterations; k++) {
      result = Clarity_Convolve_OTF(nx, ny, nz, iPtr, psfFT, oPtr);
      if (result != CLARITY_SUCCESS) {
         break;
      }
      JansenVanCittertDeconvolveKernelGPU(nx, ny, nz, in, A, 
         invASq, iPtr, oPtr, iPtr);
   }

   // Copy result from device.
   result = Clarity_CopyFromDevice(nx, ny, nz, sizeof(float), 
      outImage, iPtr);

   Clarity_Free(psfFT); Clarity_Free(in);
   Clarity_Free(iPtr); Clarity_Free(oPtr);

   return result;
}

#endif // BUILD_WITH_CUDA

ClarityResult_t 
Clarity_JansenVanCittertDeconvolve(
   float* outImage, float* inImage, float* psfImage, 
   Clarity_Dim3 dim, unsigned iterations) {

   ClarityResult_t result;

   // Find maximum value in the input image.
   float max = Clarity_GetImageMax(inImage, dim.x*dim.y*dim.z);

#ifdef TIME
   // We'll start timing here to exclude the on-CPU maximum calculation
   totalTimer.Start();
#endif

#ifdef BUILD_WITH_CUDA
   if (g_CUDACapable) {
      result = Clarity_JansenVanCittertDeconvolveGPU(outImage, 
         inImage, psfImage, dim.x, dim.y, dim.z, max, iterations);
   } else
#endif // BUILD_WITH_CUDA
   {
      result = Clarity_JansenVanCittertDeconvolveCPU(outImage, 
         inImage, psfImage, dim.x, dim.y, dim.z, max, iterations);
   }

#ifdef TIME
   totalTimer.Stop();
   std::cout << totalTimer << std::endl;
   std::cout << transferTimer << std::endl;
   totalTimer.Reset();
   transferTimer.Reset();
#endif

   return result;
}
