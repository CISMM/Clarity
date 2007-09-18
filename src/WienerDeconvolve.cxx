#include "Clarity.h"


ClarityResult_t 
Clarity_WienerDeconvolve(float* outImage, float* inImage, float* psfImage, int dimensions[3]) {

   // Temporary code to produce something for checking whether VTK hookup works.
   int size = dimensions[0] * dimensions[1] * dimensions[2];
   for (int i = 0; i < size; i++) {
      outImage[i] = 0.0f;
   }

   return CLARITY_SUCCESS;
}
