#include "Clarity.h"

ClarityResult_t
Clarity_SmoothedJansenVanCittertDeconvolve(
   float* outImage, float* inImage, float* psfImage,
   Clarity_Dim3 dim, unsigned iterations, 
   unsigned smoothInterval, float smoothSigma[3]) {

   // Temporary code to produce something.
   int size = dim.x * dim.y * dim.z;
   for (int i = 0; i < size; i++) {
      outImage[i] = 0.0f;
   }

   return CLARITY_SUCCESS;
}
