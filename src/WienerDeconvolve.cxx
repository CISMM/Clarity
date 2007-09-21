#include "Clarity.h"

#include "Complex.h"
#include "FFT.h"
#include "fftw3.h"

#include <iostream>

ClarityResult_t 
Clarity_WienerDeconvolve(float* outImage, float* inImage, float* psfImage, 
                         int nx, int ny, int nz) {
   int numVoxels = nx*ny*nz;

   ClarityResult_t result;

   float* inFT = (float *) malloc(sizeof(float) * numVoxels * 2);
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
   float* psfFT = (float *) malloc(sizeof(float) * numVoxels * 2);
   if (psfFT == NULL) {
      free(inFT);
      return CLARITY_OUT_OF_MEMORY;
   }

   result = fftf_r2c_3d(nx, ny, nz, psfImage, psfFT);
   if (result != CLARITY_SUCCESS) {
      free(inFT);
      free(psfFT);
      return result;
   }

#if 1
   // Quick test of convolution by multiplication
   for (int i = 0; i < numVoxels; i++) {
      ComplexMultiply(inFT + (2*i), psfFT + (2*i), inFT + (2*i));
   }
#endif

   float sigma = 3.0f;

   // Apply Wiener filter

   // Inverse Fourier transform of result.
   result = fftf_c2r_3d(nx, ny, nz, inFT, outImage);

   fftwf_free(inFT);
   fftwf_free(psfFT);

   return CLARITY_SUCCESS;
}
