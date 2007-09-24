#ifndef __FFT_H_
#define __FFT_H_

#include "Clarity.h"

/** Allocates memory for the complex result of a real-to-complex 
* Fourier transform. */
ClarityResult_t
Clarity_R2C_Malloc(void** buffer, size_t size, int nx, int ny, int nz);

/** Allocates memory for the real result of a complex-to-real
* Fourier transform. */
ClarityResult_t
Clarity_C2R_Malloc(void** buffer, size_t size, int nx, int ny, int nz);

void
Clarity_Free(void* buffer);

/** 3D foward FFT function. */
ClarityResult_t
Clarity_FFT_R2C_3D_float(int nx, int ny, int nz, float* in, float* out);

/** 3D inverse FFT function. */
ClarityResult_t
Clarity_FFT_C2R_3D_float(int nx, int ny, int nz, float* in, float* out);

/** Convolution function with pre-Fourier-transformed kernel. */
ClarityResult_t
Clarity_Convolve_OTF(int nx, int ny, int nz, float* in,
                 float* otf, float* out);

#endif // __FFT_H_
