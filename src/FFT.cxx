#include "FFT.h"

#include "fftw3.h"

/** Planning method for FFTW. */
static unsigned g_fftw3_planner_flags = FFTW_MEASURE;


ClarityResult_t
fftf_r2c_3d(int nx, int ny, int nz, float* in, float* out) {
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
fftf_c2r_3d(int nx, int ny, int nz, float* in, float* out) {
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
