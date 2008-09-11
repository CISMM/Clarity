#include "Clarity.h"
#include "MaximumLikelihoodDeconvolve.h"
#include "Complex.h"
#include "FFT.h"

#include <cstdio>
#include <cstdlib>
#include <omp.h>

extern bool gCUDACapable;

#ifdef TIME
#include <iostream>
#include "Stopwatch.h"

static Stopwatch totalTimer("BlindMaximumLikelihood filter (total time)");
static Stopwatch transferTimer("BlindMaximumLikelihood filter (transfer time)");
#endif


ClarityResult_t 
Clarity_BlindMaximumLikelihoodDeconvolveCPU(
   float* outImage, float* inImage, float* psfImage,
   int nx, int ny, int nz, unsigned iterations) {

   ClarityResult_t result = CLARITY_SUCCESS;

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
   result = Clarity_Real_Malloc((void**)&iPtr, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT);
      return CLARITY_OUT_OF_MEMORY;
   }

   // Storage for two intermediate arrays
   float* s1 = NULL;
   result = Clarity_Real_Malloc((void**) &s1, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(iPtr); 
      return result;
   }

   // Storage for convolution of current guess with the PSF.
   float* s2 = NULL;
   result = Clarity_Real_Malloc((void**) &s2, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(iPtr); Clarity_Free(s1);
      return result;
   }

   // Iterate
   for (unsigned k = 0; k < iterations; k++) {
      float* currentGuess = (k == 0 ? inImage : iPtr);
      float* newGuess     = (k == iterations-1 ? outImage : iPtr);

      result = Clarity_MaximumLikelihoodUpdateCPU(nx, ny, nz, inImage, 
         currentGuess, psfFT, s1, s2, newGuess);
      if (result != CLARITY_SUCCESS) {
         Clarity_Free(psfFT); Clarity_Free(iPtr); 
         Clarity_Free(s1); Clarity_Free(s2);
         return result;
      }
   }

   Clarity_Free(psfFT); Clarity_Free(iPtr);
   Clarity_Free(s1);    Clarity_Free(s2);

   return result;
}

#ifdef BUILD_WITH_CUDA

extern "C"
void
MaximumLikelihoodDivideKernelGPU(int nx, int ny, int nz,
                                 float* out, float *a, float *b);

extern "C"
void
MaximumLikelihoodMultiplyKernelGPU(int nx, int ny, int nz, float *out, float kappa, 
                                   float *a, float *b);


ClarityResult_t 
Clarity_BlindMaximumLikelihoodDeconvolveGPU(float* outImage, float* inImage, 
                                            float* psfImage, int nx, int ny, int nz, 
                                            unsigned iterations) {
   ClarityResult_t result = CLARITY_SUCCESS;

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
   result = Clarity_Real_Malloc((void**)&iPtr, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(in);
      return result;
   }

   // Storage for intermediate arrays
   float* s1 = NULL;
   result = Clarity_Real_Malloc((void**) &s1, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(in); Clarity_Free(iPtr);
      return result;
   }
   float* s2 = NULL;
   result = Clarity_Real_Malloc((void**)&s2, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(in); Clarity_Free(iPtr); 
      Clarity_Free(s1);
      return result;
   }

   // Iterate
   for (unsigned k = 0; k < iterations; k++) {
      float* currentGuess = (k == 0 ? inImage : iPtr);
      float* newGuess     = (k == iterations-1 ? outImage : iPtr);

      result = Clarity_MaximumLikelihoodUpdateGPU(nx, ny, nz, inImage, 
         currentGuess, psfFT, s1, s2, newGuess);
      if (result != CLARITY_SUCCESS) {
         Clarity_Free(psfFT); Clarity_Free(in); Clarity_Free(iPtr); 
         Clarity_Free(s1); Clarity_Free(s2);
         return result;
      }

      //// 1. Convolve current guess with h
      //result = Clarity_Convolve_OTF(nx, ny, nz, currentGuess, psfFT, oPtr);
      //if (result != CLARITY_SUCCESS) break;

      //// 2. Point-wise divide with current guess (i/guess)
      //MaximumLikelihoodDivideKernelGPU(nx, ny, nz, midPtr, in, oPtr);

      //// 3. Convolve result with h
      //result = Clarity_Convolve_OTF(nx, ny, nz, midPtr, psfFT, oPtr);
      //if (result != CLARITY_SUCCESS) break;

      //// 4. Point-wise multiply by current guess.
      //float kappa = 1.0f;
      //MaximumLikelihoodMultiplyKernelGPU(nx, ny, nz, iPtr, kappa, currentGuess, oPtr);
   }

   // Copy result from device.
   result = Clarity_CopyFromDevice(nx, ny, nz, sizeof(float), outImage, iPtr);

   Clarity_Free(psfFT); Clarity_Free(in); Clarity_Free(iPtr);
   Clarity_Free(s1);    Clarity_Free(s2);

   return result;
}

#endif // BUILD_WITH_CUDA


ClarityResult_t 
Clarity_BlindMaximumLikelihoodDeconvolve(float* outImage, float* inImage, float* psfImage, 
                                         int nx, int ny, int nz, unsigned iterations) {
   int numVoxels = nx*ny*nz;
   ClarityResult_t result = CLARITY_SUCCESS;

#ifdef TIME
   totalTimer.Start();
#endif

#ifdef BUILD_WITH_CUDA
   if (gCUDACapable) {
      result = Clarity_BlindMaximumLikelihoodDeconvolveGPU(outImage, inImage, psfImage, 
                                                           nx, ny, nz, iterations);
   } else
#endif // BUILD_WITH_CUDA
   {
      result = Clarity_BlindMaximumLikelihoodDeconvolveCPU(outImage, inImage, psfImage, 
                                                           nx, ny, nz, iterations);
   }

#ifdef TIME
   totalTimer.Stop();
   std::cout << totalTimer << std::endl;
   std::cout << transferTimer << std::endl;
   totalTimer.Reset();
   transferTimer.Reset();
#endif

   return CLARITY_SUCCESS;
}
