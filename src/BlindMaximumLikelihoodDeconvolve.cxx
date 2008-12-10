#include "Clarity.h"

#include <cstdio>
#include <cstdlib>
#include <omp.h>

#include "ComputePrimitives.h"
#include "MaximumLikelihoodDeconvolve.h"
#include "FFT.h"
#include "Memory.h"

extern bool g_CUDACapable;

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
   result = Clarity_Real_Malloc((void**)&iPtr, sizeof(float), 
      nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT);
      return CLARITY_OUT_OF_MEMORY;
   }

   // Storage for two intermediate arrays
   float* s1 = NULL;
   result = Clarity_Real_Malloc((void**) &s1, sizeof(float), 
      nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(iPtr); 
      return result;
   }

   // Storage for convolution of current guess with the PSF.
   float* s2 = NULL;
   result = Clarity_Real_Malloc((void**) &s2, sizeof(float), 
      nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(iPtr); Clarity_Free(s1);
      return result;
   }

   // Compute original energy in the image
   float energy;
   Clarity_ReduceSum(&energy, inImage, nx*ny*nz);

   // Iterate
   for (unsigned k = 0; k < iterations; k++) {
      float* currentGuess = (k == 0 ? inImage : iPtr);
      float* newGuess     = (k == iterations-1 ? outImage : iPtr);

      // Update the image.
      result = Clarity_MaximumLikelihoodUpdate(nx, ny, nz, 
         inImage, energy, currentGuess, psfFT, s1, s2, newGuess);
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

#include "MaximumLikelihoodDeconvolveGPU.h"

ClarityResult_t 
Clarity_BlindMaximumLikelihoodDeconvolveGPU(
   float* outImage, float* inImage, float* psfImage, 
   int nx, int ny, int nz, unsigned iterations) {

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
   result = Clarity_Real_Malloc((void**)&iPtr, sizeof(float), 
      nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(in);
      return result;
   }

   // Storage for intermediate arrays
   float* s1 = NULL;
   result = Clarity_Real_Malloc((void**)&s1, sizeof(float), 
      nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(in); Clarity_Free(iPtr);
      return result;
   }
   float* s2 = NULL;
   result = Clarity_Real_Malloc((void**)&s2, sizeof(float), 
      nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(in); Clarity_Free(iPtr); 
      Clarity_Free(s1);
      return result;
   }

   // Compute original energy in the image
   float energy;
   Clarity_ReduceSum(&energy, in, nx*ny*nz);

   // Iterate
   for (unsigned k = 0; k < iterations; k++) {
      float* currentGuess = (k == 0 ? in : iPtr);
      float* newGuess     = iPtr;

      result = Clarity_MaximumLikelihoodUpdate(nx, ny, nz, 
         in, energy, currentGuess, psfFT, s1, s2, newGuess);
      if (result != CLARITY_SUCCESS) {
         Clarity_Free(psfFT); Clarity_Free(in); 
         Clarity_Free(iPtr); 
         Clarity_Free(s1); Clarity_Free(s2);
         return result;
      }
   }

   // Copy result from device.
   result = Clarity_CopyFromDevice(nx, ny, nz, sizeof(float), 
      outImage, iPtr);

   Clarity_Free(psfFT); Clarity_Free(in); Clarity_Free(iPtr);
   Clarity_Free(s1);    Clarity_Free(s2);

   return result;
}

#endif // BUILD_WITH_CUDA


ClarityResult_t 
Clarity_BlindMaximumLikelihoodDeconvolve(
   float* outImage, float* inImage, float* psfImage, 
   Clarity_Dim3 dim, unsigned iterations) {

   int numVoxels = dim.x*dim.y*dim.z;
   ClarityResult_t result = CLARITY_SUCCESS;

#ifdef TIME
   totalTimer.Start();
#endif

#ifdef BUILD_WITH_CUDA
   if (g_CUDACapable) {
      result = Clarity_BlindMaximumLikelihoodDeconvolveGPU(
         outImage, inImage, psfImage, dim.x, dim.y, dim.z, iterations);
   } else
#endif // BUILD_WITH_CUDA
   {
      result = Clarity_BlindMaximumLikelihoodDeconvolveCPU(
         outImage, inImage, psfImage, dim.x, dim.y, dim.z, iterations);
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
