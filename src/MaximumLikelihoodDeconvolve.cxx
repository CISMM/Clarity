#include "Clarity.h"
#include "Complex.h"
#include "FFT.h"

#include <stdlib.h>
#include <omp.h>

extern bool gCUDACapable;

#ifdef TIME
#include <iostream>
#include "Stopwatch.h"

static Stopwatch totalTimer("MaximumLikelihood filter (total time)");
static Stopwatch transferTimer("MaximumLikelihood filter (transfer time)");
#endif


extern "C"
void
MaximumLikelihoodDeconvolveKernelGPU(int nx, int ny, int nz,
                                    float* in, float inMax, float invMaxSq,
                                    float* i_k, float* o_k, float* i_kNext);


void
MaximumLikelihoodDeconvolveKernelCPU(int nx, int ny, int nz,
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
MaximumLikelihoodDeconvolveKernel(int nx, int ny, int nz,
                                 float* in, float inMax, float invMaxSq,
                                 float* i_k, float* o_k, float* i_kNext) {
   if (gCUDACapable) {
      MaximumLikelihoodDeconvolveKernelGPU(nx, ny, nz, in, inMax, invMaxSq, 
                                           i_k, o_k, i_kNext);
   } else {
      MaximumLikelihoodDeconvolveKernelCPU(nx, ny, nz, in, inMax, invMaxSq, 
                                           i_k, o_k, i_kNext);
   }
}


ClarityResult_t 
Clarity_MaximumLikelihoodDeconvolve(float* outImage, float* inImage, float* psfImage, 
                                   int nx, int ny, int nz, unsigned iterations) {
   int numVoxels = nx*ny*nz;
   ClarityResult_t result = CLARITY_SUCCESS;

#ifdef TIME
   // We'll start timing here to exclude the on-CPU maximum calculation
   totalTimer.Start();
#endif

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

   // Storage for intermediate array
   float* midPtr = NULL;
   result = Clarity_C2R_Malloc((void**) &midPtr, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(midPtr); 
      return result;
   }

   // Storage for convolution of current guess with the PSF.
   float* oPtr = NULL;
   result = Clarity_C2R_Malloc((void**) &oPtr, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(midPtr); Clarity_Free(iPtr);
      return result;
   }

   // Iterate
   for (unsigned k = 0; k < iterations; k++) {
	   float *currentGuess = (k==0) ? inImage : iPtr;

	  // 1. convolve h with current guess (iPtr, or inImages)
	  result = Clarity_Convolve_OTF(nx, ny, nz, currentGuess, psfFT, oPtr);	  
      
      if (result != CLARITY_SUCCESS) {
         break;
      }

	  // 2. pointwise divide with current guess(i/guess)
	  int numVoxels = nx*ny*nz;	  
	  #pragma omp parallel for
	  for (int j = 0; j < numVoxels; j++) {
		  //midPtr[j] = inImage[j] / oPtr[j];
		  if (oPtr[j] < .00001)
			  midPtr[j] = 0.0f;
		  else
			  midPtr[j] = inImage[j] / oPtr[j];
	  }

	  // 3. Convolve result with h
	  result = Clarity_Convolve_OTF(nx, ny, nz, midPtr, psfFT, oPtr);
	  if (result != CLARITY_SUCCESS) {
		  break;
	  }

	  // 4. pointwise multiply by current guess
	  float kappa = 1.0;
	  if (k < iterations - 1) {
		  #pragma omp parallel for
		  for (int j = 0; j < numVoxels; j++) {
		       iPtr[j] = kappa * currentGuess[j] * oPtr[j];
          }
	  } else {
		  #pragma omp parallel for
		  for (int j = 0; j < numVoxels; j++) {
		       outImage[j] = kappa * currentGuess[j] * oPtr[j];
          }
	  }
   }

   Clarity_Free(psfFT); Clarity_Free(midPtr); Clarity_Free(oPtr); Clarity_Free(iPtr);

#ifdef TIME
   totalTimer.Stop();
   std::cout << totalTimer << std::endl;
   std::cout << transferTimer << std::endl;
   totalTimer.Reset();
   transferTimer.Reset();
#endif

   return CLARITY_SUCCESS;
}
