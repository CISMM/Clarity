#include "Clarity.h"

#include <omp.h>
#include <fftw3.h>
#include <math.h>

#include "Complex.h"
#include "FFT.h"

//#define CONVOLUTION

#ifdef TIME
#include <iostream>
#include "Stopwatch.h"

static Stopwatch totalTimer("Wiener filter (total time)");
static Stopwatch transferTimer("Wiener filter (transfer time)");
#endif

extern bool gCUDACapable;

extern "C"
void
WienerDeconvolveKernelGPU(int nx, int ny, int nz, float* inFT, float* psfFT, 
                          float* result, float sigma, float epsilon);


void
WienerDeconvolveKernelCPU(int nx, int ny, int nz, float* inFT, float* psfFT, 
                          float* result, float sigma, float epsilon) {
   int numVoxels = nx*ny*nz;

   // From Sibarita, "Deconvolution Microscopy"
#pragma omp parallel for
   for (int i = 0; i < numVoxels; i++) {
      float H[2] = {psfFT[2*i + 0], psfFT[2*i + 1]};
      float HConj[2];
      ComplexConjugate(H, HConj);
      float HMagSquared = ComplexMagnitudeSquared(H);
      ComplexMultiply(HConj, 1.0f / (HMagSquared + epsilon), HConj);
      ComplexMultiply(HConj, inFT + (2*i), result + (2*i));
   }
}


void
WienerDeconvolveKernel(int nx, int ny, int nz, float* inFT, float* psfFT, 
                       float* result, float sigma, float epsilon) {
   if (gCUDACapable) {
      WienerDeconvolveKernelGPU(nx, ny, nz, inFT, psfFT, result, sigma, epsilon);
   } else {
      WienerDeconvolveKernelCPU(nx, ny, nz, inFT, psfFT, result, sigma, epsilon);
   }
}


ClarityResult_t 
Clarity_WienerDeconvolve(float* outImage, float* inImage, float* psfImage, 
                         int nx, int ny, int nz, float noiseStdDev, float epsilon) {

#ifdef TIME
   totalTimer.Start();
#endif
                            
   int numVoxels = nx*ny*nz;
   ClarityResult_t result = CLARITY_SUCCESS;

   // Forward Fourier transform of input image.
   float* inFT = NULL;
   result = Clarity_Complex_Malloc((void**) &inFT, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      return CLARITY_OUT_OF_MEMORY;
   }
   result = Clarity_FFT_R2C_float(nx, ny, nz, inImage, inFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT);
      return result;
   }

   // Fourier transform of PSF.
   float* psfFT = NULL;
   result = Clarity_Complex_Malloc((void**) &psfFT, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT);
      return CLARITY_OUT_OF_MEMORY;
   }
   result = Clarity_FFT_R2C_float(nx, ny, nz, psfImage, psfFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT); Clarity_Free(psfFT);
      return result;
   }

#ifdef CONVOLUTION
   Clarity_Convolve_OTF(nx, ny, nz, inImage, psfFT, outImage);
#else
   WienerDeconvolveKernel(nx, ny, nz, inFT, psfFT, inFT, noiseStdDev, epsilon);
#endif

#ifndef CONVOLUTION
   // Inverse Fourier transform of result.
   result = Clarity_FFT_C2R_3D_float(nx, ny, nz, inFT, outImage);
#endif
   Clarity_Free(inFT);
   Clarity_Free(psfFT);

#ifdef TIME
   totalTimer.Stop();
   std::cout << totalTimer << std::endl;
   std::cout << transferTimer << std::endl;
   totalTimer.Reset();
   transferTimer.Reset();
#endif

   return CLARITY_SUCCESS;
}
