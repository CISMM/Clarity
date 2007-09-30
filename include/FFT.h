#ifndef __FFT_H_
#define __FFT_H_

#include "Clarity.h"

/** Allocates memory for the complex result of a real-to-complex 
* Fourier transform. */
ClarityResult_t
Clarity_Complex_Malloc(void** buffer, size_t size, int nx, int ny, int nz);


/** Allocates memory for a real image. */
ClarityResult_t
Clarity_Real_Malloc(void** buffer, size_t size, int nx, int ny, int nz);


/** Copies a real-valued image to the FFT device. */
ClarityResult_t
Clarity_CopyToDevice(int nx, int ny, int nz, size_t size, 
                     void* dst, void* src);


/** Allocates memory for and copies a real image to the device. */
ClarityResult_t
Clarity_Real_MallocCopy(void** buffer, size_t size, int nx, int ny, int nz, void* src);


/** Copies a real-valued image from the FFT device. */
ClarityResult_t
Clarity_CopyFromDevice(int nx, int ny, int nz, size_t size,
                       void* dst, void* src);


/** Frees memory allocated with Clarity_Real_Malloc and 
* Clarity_Complex_Malloc. */
void
Clarity_Free(void* buffer);


/** 3D forward FFT function. */
ClarityResult_t
Clarity_FFT_R2C_float(int nx, int ny, int nz, float* in, float* out);


/** 3D inverse FFT function. */
ClarityResult_t
Clarity_FFT_C2R_float(int nx, int ny, int nz, float* in, float* out);


/** Per-pixel modulation of one transformed image with another. */
void
Clarity_Modulate(int nx, int ny, int nz, float* in, float* otf, float* out);


/** Convolution function with pre-Fourier-transformed kernel. */
ClarityResult_t
Clarity_Convolve_OTF(int nx, int ny, int nz, float* in,
                 float* otf, float* out);

/** Convolution function with padded and cyclically-shifted PSF. */
ClarityResult_t
Clarity_ConvolveInternal(int nx, int ny, int nz, 
                         float* in, float* psf, float* out);


#endif // __FFT_H_
