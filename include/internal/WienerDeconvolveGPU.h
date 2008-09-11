#ifndef __WIENER_DECONVOLVE_GPU_H_
#define __WIENER_DECONVOLVE_GPU_H_

#include "ComplexCUDA.h"

/**
 * Configures and launches the device function for Wiener
 * filter deconvolution.
 *
 * @param nx      Size in x-dimension of inFT, psfFT, and outFT.
 * @param ny      Size in y-dimension of inFT, psfFT, and outFT.
 * @param nz      Size in z-dimension of inFT, psfFT, and outFT.
 * @param inFT    Fourier transform of the input image.
 * @param psfFT   Fourier transform of the PSF.
 * @param outFT   Fourier transform of the result of the Wiener
 *                filter.
 * @param epsilon Smoothing factor.
 */
extern "C"
void
WienerDeconvolveKernelGPU(
   int nx, int ny, int nz, float* inFT, float* psfFT, 
   float* outFT, float epsilon);


#endif // __WIENER_DECONVOLVE_GPU_H_