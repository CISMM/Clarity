#ifndef __MAXIMUM_LIKELIHOOD_DECONVOLVE_H_
#define __MAXIMUM_LIKELIHOOD_DECONVOLVE_H_

#include "Clarity.h"

ClarityResult_t
Clarity_MaximumLikelihoodUpdateCPU(int nx, int ny, int nz, float* in,
                                   float* currentGuess, float* otf, 
                                   float* s1, float* s2, float* newGuess);

ClarityResult_t
Clarity_MaximumLikelihoodUpdateGPU(int nx, int ny, int nz, float* in,
                                   float* currentGuess, float* otf, 
                                   float* s1, float* s2, float* newGuess);


#endif // __MAXIMUM_LIKELIHOOD_DECONVOLVE_H_