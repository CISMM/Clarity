#include "FFT.h"

#include "Complex.h"
#include "fftw3.h"

extern bool gCUDACapable;

/** Planning method for FFTW. */
static unsigned g_fftw3_planner_flags = FFTW_MEASURE;

ClarityResult_t
Clarity_Complex_Malloc(void** buffer, size_t size, int nx, int ny, int nz) {
   ClarityResult_t result = CLARITY_SUCCESS;

   *buffer = fftwf_malloc(sizeof(size)*2*nz*ny*(nx/2 + 1));
   if (*buffer == NULL) {
      result = CLARITY_OUT_OF_MEMORY;
   }

   return result;
}


ClarityResult_t
Clarity_Real_Malloc(void** buffer, size_t size, int nx, int ny, int nz) {
   ClarityResult_t result = CLARITY_SUCCESS;

   *buffer = fftwf_malloc(sizeof(size)*nz*ny*nx);
   if (*buffer == NULL) {
      result = CLARITY_OUT_OF_MEMORY;
   }

   return result;
}


void
Clarity_Free(void* buffer) {
   if (buffer == NULL)
      return;
   if (gCUDACapable) {
   } else {
      fftwf_free(buffer);
   }
}


ClarityResult_t
Clarity_FFT_R2C_float(int nx, int ny, int nz, float* in, float* out) {
   int numVoxels = nx*ny*nz;

   fftwf_complex* outComplex = (fftwf_complex*) out;

   // Holy smokes, I wasted a lot of time just to find that FFTW expects
   // data arranged not in the normal way in medical/biological imaging.
   // Dimension sizes need to be reversed.
   fftwf_plan plan = fftwf_plan_dft_r2c_3d(nz, ny, nx, in, outComplex,
      FFTW_ESTIMATE);
   if (plan == NULL) {
      return CLARITY_FFT_FAILED;
   }
   fftwf_execute(plan);
   fftwf_destroy_plan(plan);

   return CLARITY_SUCCESS;
}


ClarityResult_t
Clarity_FFT_C2R_float(int nx, int ny, int nz, float* in, float* out) {
   int numVoxels = nx*ny*nz;

   fftwf_complex* inComplex = (fftwf_complex*) in;

   // Holy smokes, I wasted a lot of time just to find that FFTW expects
   // data arranged not in the normal way in medical/biological imaging.
   // Dimension sizes need to be reversed.
   fftwf_plan plan = fftwf_plan_dft_c2r_3d(nz, ny, nx, inComplex, out, 
      FFTW_ESTIMATE);
   if (plan == NULL) {
      return CLARITY_FFT_FAILED;
   }
   fftwf_execute(plan);
   fftwf_destroy_plan(plan);

   float multiplier = 1.0f / ((float) numVoxels);
#pragma omp parallel for
   for (int i = 0; i < numVoxels; i++) {
      out[i] *= multiplier;
   }

   return CLARITY_SUCCESS;
}


ClarityResult_t
Clarity_Convolve_OTF(int nx, int ny, int nz, float* in, float* otf, float* out) {
   ClarityResult_t result = CLARITY_SUCCESS;

   float* inFT = NULL;
   result = Clarity_Complex_Malloc((void**) &inFT, sizeof(float), nx, ny, nz);
   if (result == CLARITY_OUT_OF_MEMORY) {
      return result;
   }

   result = Clarity_FFT_R2C_float(nx, ny, nz, in, inFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT);
      return result;
   }

   int numVoxels = nz*ny*(nx/2 + 1);
#pragma omp parallel for
   for (int i = 0; i < numVoxels; i++) {
      ComplexMultiply(inFT + (2*i), otf + (2*i), inFT + (2*i));
   }

   result = Clarity_FFT_C2R_float(nx, ny, nz, inFT, out);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(&inFT);
      return result;
   }

   Clarity_Free(&inFT);

   return result;
}