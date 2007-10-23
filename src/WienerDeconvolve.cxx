#include "Clarity.h"

#include <iostream>
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


void
WienerDeconvolveKernelCPU(int nx, int ny, int nz, float* inFT, float* psfFT, 
                          float* result, float sigma, float epsilon) {
   int numVoxels = nz*ny*(nx/2 + 1);
   float scale = 1.0f / ((float) nz*ny*nx);

   // From Sibarita, "Deconvolution Microscopy"
#pragma omp parallel for
   for (int i = 0; i < numVoxels; i++) {
      float H[2] = {psfFT[2*i + 0], psfFT[2*i + 1]};
      float HConj[2];
      ComplexConjugate(H, HConj);
      float HMagSquared = ComplexMagnitudeSquared(H);
      ComplexMultiply(HConj, 1.0f / (HMagSquared + epsilon), HConj);
      ComplexMultiplyAndScale(HConj, inFT + (2*i), scale, result + (2*i));
   }
}


ClarityResult_t
Clarity_WienerDeconvolveCPU(float* outImage, float* inImage, float* psfImage,
                            int nx, int ny, int nz, float noiseStdDev, float epsilon) {
   ClarityResult_t result = CLARITY_SUCCESS;

   // Forward Fourier transform of input image.
   float* inFT = NULL;
   result = Clarity_Complex_Malloc((void**) &inFT, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      return result;
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

   WienerDeconvolveKernelCPU(nx, ny, nz, inFT, psfFT, inFT, noiseStdDev, epsilon);

   result = Clarity_FFT_C2R_float(nx, ny, nz, inFT, outImage);

   Clarity_Free(inFT);
   Clarity_Free(psfFT);

   return result;
}


#ifdef BUILD_WITH_CUDA

extern "C"
void
WienerDeconvolveKernelGPU(int nx, int ny, int nz, float* inFT, float* psfFT, 
                          float* outFT, float sigma, float epsilon);


ClarityResult_t
Clarity_WienerDeconvolveGPU(float* outImage, float* inImage, float* psfImage,
                            int nx, int ny, int nz, float noiseStdDev, float epsilon) {
   ClarityResult_t result = CLARITY_SUCCESS;

   // Send PSF image.
   float* psf = NULL;
   result = Clarity_Real_MallocCopy((void**) &psf, sizeof(float),
      nx, ny, nz, psfImage);
   if (result != CLARITY_SUCCESS) {
      return result;
   }

   // Fourier transform of PSF.
   float* psfFT = NULL;
   result = Clarity_Complex_Malloc((void**) &psfFT, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psf);
      return result;
   }
   result = Clarity_FFT_R2C_float(nx, ny, nz, psf, psfFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psf); Clarity_Free(psfFT);
      return result;
   }
   Clarity_Free(psf); // Don't need this anymore.

   // Send input image.
   float* in = NULL;
   result = Clarity_Real_MallocCopy((void**) &in, sizeof(float), 
      nx, ny, nz, inImage);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT);
      return result;
   }

   // Forward Fourier transform of input image.
   float* inFT = NULL;
   result = Clarity_Complex_Malloc((void**) &inFT, sizeof(float), nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(in);
      return result;
   }
   result = Clarity_FFT_R2C_float(nx, ny, nz, in, inFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(psfFT); Clarity_Free(in); Clarity_Free(inFT);
      return result;
   }

   // Apply Wiener filter
   WienerDeconvolveKernelGPU(nx, ny, nz, inFT, psfFT, inFT, 
      noiseStdDev, epsilon);

   result = Clarity_FFT_C2R_float(nx, ny, nz, inFT, in);
   
   // Read back
   result = Clarity_CopyFromDevice(nx, ny, nz, sizeof(float), outImage, in);

   return result;
}

#endif // BUILD_WITH_CUDA


#ifdef CONVOLUTION

ClarityResult_t 
Clarity_WienerDeconvolve(float* outImage, float* inImage, float* psfImage, 
                         int nx, int ny, int nz, float noiseStdDev, float epsilon) {
   return Clarity_Convolve(nx, ny, nz, inImage, psfImage, outImage);
}

#else

ClarityResult_t 
Clarity_WienerDeconvolve(float* outImage, float* inImage, float* psfImage, 
                         int nx, int ny, int nz, float noiseStdDev, float epsilon) {

#ifdef TIME
   totalTimer.Start();
#endif

#ifdef BUILD_WITH_CUDA
   if (gCUDACapable) {
      return Clarity_WienerDeconvolveGPU(outImage, inImage, psfImage,
         nx, ny, nz, noiseStdDev, epsilon);
   } else
#endif // BUILD_WITH_CUDA
   {
      return Clarity_WienerDeconvolveCPU(outImage, inImage, psfImage,
         nx, ny, nz, noiseStdDev, epsilon);
   }

#ifdef TIME
   totalTimer.Stop();
   std::cout << totalTimer << std::endl;
   std::cout << transferTimer << std::endl;
   totalTimer.Reset();
   transferTimer.Reset();
#endif

   return CLARITY_SUCCESS;
}

#endif // CONVOLUTION

