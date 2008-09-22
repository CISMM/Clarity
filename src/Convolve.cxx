#include "Clarity.h"

#include "Complex.h"
#include "Convolve.h"
#include "FFT.h"
#include "Memory.h"

extern bool g_CUDACapable;

#ifdef TIME
#include <iostream>
#include "Stopwatch.h"

static Stopwatch totalTimer("JansenVanCittert filter (total time)");
static Stopwatch transferTimer("JansenVanCittert filter (transfer time)");
#endif

ClarityResult_t
Clarity_Convolve(
   int nx, int ny, int nz, float* inImage, float* kernel,
   float* outImage) {

#ifdef TIME
   totalTimer.Start();
#endif
   int numVoxels = nx*ny*nz;
   ClarityResult_t result = CLARITY_SUCCESS;

   // Copy over the input image and PSF.
#ifdef BUILD_WITH_CUDA
   if (g_CUDACapable) {
      float* in;
      float* psf;
      
      result = Clarity_Real_MallocCopy((void**) &in, sizeof(float), 
         nx, ny, nz, inImage);
      if (result != CLARITY_SUCCESS) {
         return result;
      }
      result = Clarity_Real_MallocCopy((void **) &psf, 
         sizeof(float), nx, ny, nz, kernel);
      if (result != CLARITY_SUCCESS) {
         Clarity_Free(in);
         return result;
      }
      Clarity_ConvolveInternal(nx, ny, nz, in, psf, in);
      result = Clarity_CopyFromDevice(nx, ny, nz, sizeof(float),
         outImage, in);
      Clarity_Free(in); Clarity_Free(psf);

   } else 
#endif // BUILD_WITH_CUDA
   {

      Clarity_ConvolveInternal(nx, ny, nz, inImage, kernel, 
         outImage);
   }

#ifdef TIME
   totalTimer.Stop();
   std::cout << totalTimer << std::endl;
   std::cout << transferTimer << std::endl;
   totalTimer.Reset();
   transferTimer.Reset();
#endif

   return result;
}


ClarityResult_t
Clarity_Convolve_OTF(
   int nx, int ny, int nz, float* in, float* otf, float* out) {

   ClarityResult_t result = CLARITY_SUCCESS;

   float* inFT = NULL;
   result = Clarity_Complex_Malloc((void**) &inFT, sizeof(float), 
      nx, ny, nz);
   if (result == CLARITY_OUT_OF_MEMORY) {
      return result;
   }

   result = Clarity_FFT_R2C_float(nx, ny, nz, in, inFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT);
      return result;
   }

   Clarity_Modulate(nx, ny, nz, inFT, otf, inFT);

   result = Clarity_FFT_C2R_float(nx, ny, nz, inFT, out);
   Clarity_Free(inFT);

   return result;
}


ClarityResult_t
Clarity_ConvolveInternal(
   int nx, int ny, int nz, float* in, float* psf, float* out) {

   ClarityResult_t result = CLARITY_SUCCESS;
   float* inFT = NULL;
   result = Clarity_Complex_Malloc((void**) &inFT, sizeof(float), 
      nx, ny, nz);
   if (result != CLARITY_SUCCESS) { 
      return result;
   }
   result = Clarity_FFT_R2C_float(nx, ny, nz, in, inFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT);
      return result;
   }

   float* psfFT = NULL;
   result = Clarity_Complex_Malloc((void**) &psfFT, sizeof(float),
      nx, ny, nz);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT);
      return result;
   }
   result = Clarity_FFT_R2C_float(nx, ny, nz, psf, psfFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT); Clarity_Free(psfFT);
      return result;
   }

   // Modulate the two transforms
   Clarity_Modulate(nx, ny, nz, inFT, psfFT, inFT);
   Clarity_Free(psfFT);

   result = Clarity_FFT_C2R_float(nx, ny, nz, inFT, out);
   Clarity_Free(inFT);

   return result;
}


#ifdef BUILD_WITH_CUDA
extern "C"
void
Clarity_Modulate_KernelGPU(
   int nx, int ny, int nz, float* inFT, float* otf, float* outFT);
#endif


void
Clarity_Modulate_KernelCPU(
   int nx, int ny, int nz, float* inFT, float* otf, float* outFT) {
   int numVoxels = nz*ny*(nx/2 + 1);
   float scale = 1.0f / ((float) nz*ny*nx);
#pragma omp parallel for
   for (int i = 0; i < numVoxels; i++) {
      ComplexMultiplyAndScale(inFT + (2*i), otf + (2*i), scale, outFT + (2*i));
   }
}


void
Clarity_Modulate(
   int nx, int ny, int nz, float* in, float* otf, float* out) {

#ifdef BUILD_WITH_CUDA
   if (g_CUDACapable) {
      Clarity_Modulate_KernelGPU(nx, ny, nz, in, otf, out);
   } else
#endif // BUILD_WITH_CUDA
   {
      Clarity_Modulate_KernelCPU(nx, ny, nz, in, otf, out);
   }
}
