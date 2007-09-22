#include "Clarity.h"

#include "Complex.h"
#include "FFT.h"
#include "fftw3.h"
#include "math.h"

#include <iostream>

ClarityResult_t 
Clarity_WienerDeconvolve(float* outImage, float* inImage, float* psfImage, 
                         int nx, int ny, int nz, float noiseStdDev, float epsilon) {
   int numVoxels = nx*ny*nz;
   ClarityResult_t result;

   // Forward Fourier transform of input image.
   float* inFT = (float *) malloc(sizeof(float) * numVoxels * 2);
   if (inFT == NULL) {
      return CLARITY_OUT_OF_MEMORY;
   }
   result = fftf_r2c_3d(nx, ny, nz, inImage, inFT);
   if (result != CLARITY_SUCCESS) {
      free(inFT);    
      return result;
   }

   // Fourier transform of PSF.
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

#if 0
   // Quick test of convolution by multiplication
   for (int i = 0; i < numVoxels; i++) {
      ComplexMultiply(inFT + (2*i), psfFT + (2*i), inFT + (2*i));
   }
#else

   float sigma = noiseStdDev;

   // Apply Wiener filter
   // Reference: J.S. Lim, "Two dimensional signal and image processing",
   // Prentice Hall, 1990- pg.560 Eq. (9. 73)
   for (int i = 0; i < numVoxels; i++) {
      // Create inverse filter if we can
      float H[2] = {psfFT[2*i+0], psfFT[2*i+1]};
      float absHSquared = ComplexMagnitudeSquared(H);
      if (absHSquared <= epsilon) {
         inFT[2*i + 0] = 0.0f;
         inFT[2*i + 1] = 0.0f;
         continue;
      }
      
      float inverseFilter[2];
      float absH = sqrt(absHSquared);
      ComplexInverse(H, inverseFilter);

      // Wiener filter
      float inFTpower = ComplexMagnitudeSquared(inFT+(2*i));
      float wiener[2];
      ComplexMultiply(inverseFilter, inFTpower / (inFTpower + (sigma*sigma)), wiener);      

      // Now multiply and invert
      ComplexMultiply(inFT+(2*i), wiener, inFT+(2*i));
   }
#endif

   // Inverse Fourier transform of result.
   result = fftf_c2r_3d(nx, ny, nz, inFT, outImage);

   fftwf_free(inFT);
   fftwf_free(psfFT);

   return CLARITY_SUCCESS;
}
