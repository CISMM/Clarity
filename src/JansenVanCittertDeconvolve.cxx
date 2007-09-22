#include "Clarity.h"
#include "Complex.h"
#include "FFT.h"

#include "omp.h"

#include <iostream>

float
Clarity_GetImageMax(float *inImage, int numVoxels) {
   float max = inImage[0];
#pragma omp parallel
   {
      int numThreads = omp_get_num_threads();
      float *threadMax = new float[numThreads];
      int tid = omp_get_thread_num();
      threadMax[tid] = max;

#pragma omp for
      for (int i = 0; i < numVoxels; i++) {
         float val = inImage[i];
         if (val > threadMax[tid]) threadMax[tid] = val;
      }

      for (int i = 0; i < numThreads; i++) {
         if (threadMax[i] > max) max = threadMax[i];
      }
      delete[] threadMax;
   }

   return max;
}


ClarityResult_t 
Clarity_JansenVanCittertDeconvolve(float* outImage, float* inImage, float* psfImage, 
                                   int nx, int ny, int nz, unsigned iterations) {
   int numVoxels = nx*ny*nz;
   ClarityResult_t result;

   // Find maximum value in the input image.
   float max = Clarity_GetImageMax(inImage, numVoxels);
   float A = 0.5f * max;
   float invASq = 1.0f / (A * A);

   // Fourier transform of PSF.
   float* psfFT = (float *) malloc(sizeof(float) * numVoxels * 2);
   if (psfFT == NULL) {
      return CLARITY_OUT_OF_MEMORY;
   }
   result = fftf_r2c_3d(nx, ny, nz, psfImage, psfFT);
   if (result != CLARITY_SUCCESS) {
      free(psfFT);
      return result;
   }

   // Set up the array holding the current guess and copy initial
   // image into it.
   float* iPtr = (float *) malloc(sizeof(float) * numVoxels);
   if (iPtr == NULL) {
      free(psfFT);
      return CLARITY_OUT_OF_MEMORY;
   }
#pragma omp parallel for
   for (int j = 0; j < numVoxels; j++) {
      iPtr[j] = inImage[j];
   }

   // Stores convolution of current guess with the PSF
   float* oPtr = (float *) malloc(sizeof(float) * numVoxels);
   if (oPtr == NULL) {
      free(psfFT); free(iPtr);
      return CLARITY_OUT_OF_MEMORY;
   }

   // Temporary storage for results of point-wise multiplication
   float* convTmp = (float *) malloc(sizeof(float) * numVoxels * 2);
   if (convTmp == NULL) {
      free(psfFT); free(iPtr); free(oPtr);
      return CLARITY_OUT_OF_MEMORY;
   }

   // Iterate
   for (unsigned k = 0; k < iterations; k++) {      
      result = fftf_r2c_3d(nx, ny, nz, iPtr, convTmp);
      if (result != CLARITY_SUCCESS) {
         free(psfFT); free(iPtr); free(oPtr);
         return result;
      }

#pragma omp parallel for
      for (int j = 0; j < numVoxels; j++) {
         ComplexMultiply(convTmp+(2*j), psfFT+(2*j), convTmp+(2*j));
      }

      result = fftf_c2r_3d(nx, ny, nz, convTmp, oPtr);
      if (result != CLARITY_SUCCESS) {
         free(psfFT); free(iPtr); free(oPtr);
         return result;
      }

      if (k < iterations - 1) {
#pragma omp parallel for
         for (int j = 0; j < numVoxels; j++) {
            float diff = oPtr[j] - A;
            float gamma = 1.0f - ((diff * diff) * invASq);
            float val = iPtr[j] + (gamma * (inImage[j] - oPtr[j]));
            if (val < 0.0f) val = 0.0f;
            iPtr[j] = val;
         }
      } else {
#pragma omp parallel for
         for (int j = 0; j < numVoxels; j++) {
            float diff = oPtr[j] - A;
            float gamma = 1.0f - ((diff * diff) * invASq);
            float val = iPtr[j] + (gamma * (inImage[j] - oPtr[j]));
            if (val < 0.0f) val = 0.0f;
            outImage[j] = val;
         }
      }
   }

   free(psfFT); free(oPtr);

   return CLARITY_SUCCESS;
}
