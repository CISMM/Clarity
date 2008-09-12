#include "Clarity.h"

#include <stdlib.h>
#include <omp.h>

#include "ComputePrimitives.h"
#include "Convolve.h"
#include "FFT.h"
#include "MaximumLikelihoodDeconvolve.h"
#include "Memory.h"

extern bool g_CUDACapable;

#ifdef TIME
#include <iostream>
#include "Stopwatch.h"

static Stopwatch totalTimer("MaximumLikelihood filter (total time)");
static Stopwatch transferTimer("MaximumLikelihood filter (transfer time)");
#endif


ClarityResult_t
Clarity_MaximumLikelihoodUpdate(
   int nx, int ny, int nz, float* in, float energy,
   float* currentGuess, float* otf, float* s1, float* s2, 
   float* newGuess) {

   ClarityResult_t result = CLARITY_SUCCESS;

   // 1. Convolve current guess with h
   result = Clarity_Convolve_OTF(nx, ny, nz, currentGuess, otf, s1);	  
   if (result != CLARITY_SUCCESS) return result;

   // 2. Point-wise divide with current guess (i/guess)
   int numVoxels = nx*ny*nz;
   Clarity_DivideArraysComponentWise(s1, in, s1, 0.0f, numVoxels);

   // 3. Convolve result with h
   result = Clarity_Convolve_OTF(nx, ny, nz, s1, otf, s2);
   if (result != CLARITY_SUCCESS) return result;

   // 4. Point-wise multiply by current guess.
   Clarity_MultiplyArraysComponentWise(newGuess, currentGuess, 
      s2, numVoxels);

   // 5. Compute energy preservation scaling factor.
   float newEnergy = 0.0f;
   Clarity_ReduceSum(&newEnergy, newGuess, nx*ny*nz);

   // 6. Rescale to normalize energy.
   float scale = energy / newEnergy;
   Clarity_ScaleArray(newGuess, newGuess, numVoxels, scale);

   return result;
}


ClarityResult_t 
Clarity_MaximumLikelihoodDeconvolveCPU(
   float* outImage, float* inImage, float* psfImage, 
   int nx, int ny, int nz, unsigned iterations) {

   ClarityResult_t result = CLARITY_SUCCESS;

   // Fourier transform of PSF.
   float* psfFT = NULL;
   result = Clarity_Complex_Malloc((void**) &psfFT, 
      sizeof(float), nx, ny, nz);
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

   // Storage for intermediate array
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
   int numVoxels = nx*ny*nz;	 
   for (unsigned k = 0; k < iterations; k++) {
	   float* currentGuess = (k == 0 ? inImage : iPtr);
      float* newGuess     = (k == iterations-1 ? outImage : iPtr);

      result = Clarity_MaximumLikelihoodUpdate(nx, ny, nz, 
         inImage, energy, currentGuess, psfFT, s1, s2, newGuess);
      if (result != CLARITY_SUCCESS) {
         Clarity_Free(psfFT); Clarity_Free(iPtr); 
         Clarity_Free(s1); Clarity_Free(s2);
         return result;
      }
   }

   Clarity_Free(psfFT); Clarity_Free(iPtr);
   Clarity_Free(s1); Clarity_Free(s2);

   return result;
}

#ifdef BUILD_WITH_CUDA

#include "MaximumLikelihoodDeconvolveGPU.h"


ClarityResult_t 
Clarity_MaximumLikelihoodDeconvolveGPU(
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
Clarity_MaximumLikelihoodDeconvolve(
   float* outImage, float* inImage, float* psfImage, 
   int nx, int ny, int nz, unsigned iterations) {

   int numVoxels = nx*ny*nz;
   ClarityResult_t result = CLARITY_SUCCESS;

#ifdef TIME
   totalTimer.Start();
#endif

#ifdef BUILD_WITH_CUDA
   if (g_CUDACapable) {
      result = Clarity_MaximumLikelihoodDeconvolveGPU(
         outImage, inImage, psfImage, nx, ny, nz, iterations);
   } else
#endif // BUILD_WITH_CUDA
   {
      result = Clarity_MaximumLikelihoodDeconvolveCPU(
         outImage, inImage, psfImage, nx, ny, nz, iterations);
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
