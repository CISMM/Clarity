#include "Clarity.h"

#include "Complex.h"
#include "FFT.h"
#include "fftw3.h"
#include "math.h"

#include <iostream>

//#define CONVOLUTION

ClarityResult_t 
Clarity_WienerDeconvolve(float* outImage, float* inImage, float* psfImage, 
                         int nx, int ny, int nz, float noiseStdDev, float epsilon) {
   int numVoxels = nx*ny*nz;
   ClarityResult_t result = CLARITY_SUCCESS;

   // Forward Fourier transform of input image.
   float* inFT = NULL;
   result = Clarity_R2C_Malloc((void**) &inFT, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      return CLARITY_OUT_OF_MEMORY;
   }
   result = Clarity_FFT_R2C_3D_float(nx, ny, nz, inImage, inFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT);
      return result;
   }

   // Fourier transform of PSF.
   float* psfFT = NULL;
   result = Clarity_R2C_Malloc((void**) &psfFT, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT);
      return CLARITY_OUT_OF_MEMORY;
   }
   result = Clarity_FFT_R2C_3D_float(nx, ny, nz, psfImage, psfFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT); Clarity_Free(psfFT);
      return result;
   }

#ifdef CONVOLUTION
   // Quick test of convolution
   Clarity_Convolve_OTF(nx, ny, nz, inImage, psfFT, outImage);
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

#ifndef CONVOLUTION
   // Inverse Fourier transform of result.
   result = Clarity_FFT_R2C_3D_float(nx, ny, nz, inFT, outImage);
#endif
   Clarity_Free(inFT);
   Clarity_Free(psfFT);

   return CLARITY_SUCCESS;
}
