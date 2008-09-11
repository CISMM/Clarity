#include "Clarity.h"

#include "math.h"
#include "omp.h"

static float IDIVFunctional(
   float* g, float* gHat, float* sHat, float alpha, 
   int nx, int ny, int nz) {

   float sum = 0.0f;
   int numVoxels = nx*ny*nz;

#pragma omp parallel for reduction(+:sum)
   for (int i = 0; i < numVoxels; i++) {
      sum += (g[i]*log(g[i]/gHat[i])) + gHat[i] - g[i] + (alpha*sHat[i]*sHat[i]);
   }
   return sum;
}


static void IDIVGradient(
   float* g, float* gHat, float* sHat, float* flippedPSFtf, 
   float alpha, float* gradient) {


}



ClarityResult_t
Clarity_IDivergenceDeconvolve(
   float* outImage, float* inImage, float* psfImage, 
   int nx, int ny, int nz) {

   // Temporary code to produce something for checking whether VTK hookup works.
   int size = nx*ny*nz;
   for (int i = 0; i < size; i++) {
      outImage[i] = 0.0f;
   }

   return CLARITY_SUCCESS;
}
