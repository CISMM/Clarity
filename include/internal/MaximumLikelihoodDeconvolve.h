#ifndef __MAXIMUM_LIKELIHOOD_DECONVOLVE_H_
#define __MAXIMUM_LIKELIHOOD_DECONVOLVE_H_

#include "Clarity.h"


/**
 * Update step of the maximum likelihood algorithm.
 * 
 * @param nx           X-dimension of in, currentGuess, otf, newGuess.
 * @param ny           Y-dimension of in, currentGuess, otf, newGuess.
 * @param nz           Z-dimension of in, currentGuess, otf, newGuess.
 * @param in           Original real-valued image.
 * @param currentGuess Current guess of the uncorrupted image.
 * @param otf          Fourier transform of the convolution kernel.
 * @param s1           Temporary storage buffer big enough to store
 *                     real-valued image of dimensions nx*ny*nz.
 * @param s2           Temporary storage buffer the size of s1.
 * @param newGuess     Real-valued result of the function corresponding to
 *                     the next best guess of the uncorrupted image.
 */
ClarityResult_t
Clarity_MaximumLikelihoodUpdate(
   int nx, int ny, int nz, float* in, float energy,
   float* currentGuess, float* otf, float* s1, float* s2,
   float* newGuess);


#endif // __MAXIMUM_LIKELIHOOD_DECONVOLVE_H_
