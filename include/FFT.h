#ifndef __FFT_H_
#define __FFT_H_

#include "Clarity.h"

/** 3D foward FFT function. */
ClarityResult_t fftf_r2c_3d(int nx, int ny, int nz, float* in, float* out);

/** 3D inverse FFT function. */
ClarityResult_t fftf_c2r_3d(int nx, int ny, int nz, float* in, float* out);


#endif // __FFT_H_
