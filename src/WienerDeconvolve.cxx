#include "Clarity.h"

#include "FFT.h"
#include "fftw3.h"

#include <iostream>

ClarityResult_t 
Clarity_WienerDeconvolve(float* outImage, float* inImage, int nx, int ny, int nz, float* psfImage) {
   int numVoxels = nx*ny*nz;

   ClarityResult_t result;

   float *inFT = (float *) malloc(sizeof(float)*numVoxels*2);
   if (inFT == NULL) {
      return CLARITY_OUT_OF_MEMORY;
   }

   // Forward Fourier transform of input image.
   result = fftf_r2c_3d(nx, ny, nz, inImage, inFT);
   if (result != CLARITY_SUCCESS) {
      free(inFT);    
      return result;
   }

   // Transform PSF.

   // Estimate power spectrum of input image.

   // Estimate power spectrum of noise.

   // Apply Wiener filter

   // Inverse Fourier transform of result.
   result = fftf_c2r_3d(nx, ny, nz, inFT, outImage);

   fftwf_free(inFT);

   return CLARITY_SUCCESS;
}
