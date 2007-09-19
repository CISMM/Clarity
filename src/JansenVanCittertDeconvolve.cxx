#include "Clarity.h"

ClarityResult_t 
Clarity_JansenVanCittertDeconvolve(float* outImage, float* inImage, int nx, int ny, int nz, float* psfImage, 
                                   unsigned iterations) {

   // Temporary code to produce something for checking whether VTK hookup works.
   int size = nx * ny * nz;
   for (int i = 0; i < size; i++) {
      outImage[i] = 0.0f;
   }

   return CLARITY_SUCCESS;
}
