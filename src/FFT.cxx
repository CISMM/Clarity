#include "FFT.h"

#include "Complex.h"
#include "fftw3.h"

/** Planning method for FFTW. */
static unsigned g_fftw3_planner_flags = FFTW_MEASURE;

ClarityResult_t
Clarity_R2C_Malloc(void** buffer, size_t size, int nx, int ny, int nz) {
   ClarityResult_t result = CLARITY_SUCCESS;

   *buffer = fftwf_malloc(sizeof(size)*nx*ny*nz*2);
   if (*buffer == NULL) {
      result = CLARITY_OUT_OF_MEMORY;
   }

   return result;
}


ClarityResult_t
Clarity_C2R_Malloc(void** buffer, size_t size, int nx, int ny, int nz) {
   ClarityResult_t result = CLARITY_SUCCESS;

   *buffer = fftwf_malloc(sizeof(size)*nx*ny*nz);
   if (*buffer == NULL) {
      result = CLARITY_OUT_OF_MEMORY;
   }

   return result;
}


void
Clarity_Free(void* buffer) {
   fftwf_free(buffer);
}


ClarityResult_t
Clarity_FFT_R2C_3D_float(int nx, int ny, int nz, float* in, float* out) {
   int numVoxels = nx*ny*nz;

   fftwf_complex* inComplex = 
      (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*numVoxels);
   if (inComplex == NULL) {
      return CLARITY_OUT_OF_MEMORY;
   }

   fftwf_complex* outComplex = (fftwf_complex*) out;

#pragma omp parallel for
   for (int i = 0; i < numVoxels; i++) {
      inComplex[i][0] = in[i];
      inComplex[i][1] = 0.0f;
   }

   // Holy smokes, I wasted a lot of time just to find that FFTW expects
   // data arranged not in the normal way in medical/biological imaging.
   // Dimension sizes need to be reversed.
   fftwf_plan plan = fftwf_plan_dft_3d(nz, ny, nx, inComplex, outComplex,
      FFTW_FORWARD, FFTW_ESTIMATE);
   if (plan == NULL) {
      fftwf_free(inComplex);
      return CLARITY_FFT_FAILED;
   }
   fftwf_execute(plan);
   fftwf_destroy_plan(plan);
   fftwf_free(inComplex);

   return CLARITY_SUCCESS;
}


ClarityResult_t
Clarity_FFT_C2R_3D_float(int nx, int ny, int nz, float* in, float* out) {
   int numVoxels = nx*ny*nz;

   fftwf_complex* outComplex =
      (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*numVoxels);
   if (outComplex == NULL) {
      return CLARITY_OUT_OF_MEMORY;
   }

   fftwf_complex* inComplex = (fftwf_complex*) in;

   // Holy smokes, I wasted a lot of time just to find that FFTW expects
   // data arranged not in the normal way in medical/biological imaging.
   // Dimension sizes need to be reversed.
   fftwf_plan plan = fftwf_plan_dft_3d(nz, ny, nx, inComplex, outComplex,
      FFTW_BACKWARD, FFTW_ESTIMATE);
   if (plan == NULL) {
      fftwf_free(outComplex);
      return CLARITY_FFT_FAILED;
   }
   fftwf_execute(plan);
   fftwf_destroy_plan(plan);

#pragma omp parallel for
   float multiplier = 1.0f / ((float) numVoxels);
   for (int i = 0; i < numVoxels; i++) {
      out[i] = outComplex[i][0] * multiplier;
   }

   fftwf_free(outComplex);

   return CLARITY_SUCCESS;
}


ClarityResult_t
Clarity_Convolve_OTF(int nx, int ny, int nz, float* in, float* otf, float* out) {
   ClarityResult_t result = CLARITY_SUCCESS;
   int numVoxels = nx*ny*nz;

   float* inFT = NULL;
   result = Clarity_R2C_Malloc((void**) &inFT, sizeof(float), nx, ny, nz);
   if (result == CLARITY_OUT_OF_MEMORY) {
      return result;
   }

   result = Clarity_FFT_R2C_3D_float(nx, ny, nz, in, inFT);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT);
      return result;
   }

#pragma parallel for
   for (int i = 0; i < numVoxels; i++) {
      ComplexMultiply(inFT + (2*i), otf + (2*i), inFT + (2*i));
   }

   result = Clarity_FFT_C2R_3D_float(nx, ny, nz, inFT, out);
   if (result != CLARITY_SUCCESS) {
      Clarity_Free(inFT);
      return result;
   }

   Clarity_Free(inFT);

   return result;
}