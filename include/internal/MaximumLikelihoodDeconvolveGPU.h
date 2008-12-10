#ifndef __MAXIMUM_LIKELIHOOD_DECONVOLVE_GPU_H_
#define __MAXIMUM_LIKELIHOOD_DECONVOLVE_GPU_H_


/**
 * Function that calls point-wise division routine for real-valued
 * images on the GPU.
 * 
 * @param nx  X-dimension of out, a, and b.
 * @param ny  Y-dimension of out, a, and b.
 * @param nz  Z-dimension of out, a, and b.
 * @param out Buffer storing resulting real-valued image of size nx*ny*nz.
 * @param a   First real-valued image.
 * @param b   Second real-value image.
 */
extern "C"
void
MaximumLikelihoodDivideKernelGPU(
   int nx, int ny, int nz, float* out, float *a, float *b);


/**
 * Function that calls point-wise multiplication and scaling of two
 * real-valued images.
 * 
 * @param nx    X-dimension of out, a, and b.
 * @param ny    Y-dimension of out, a, and b.
 * @param nz    Z-dimension of out, a, and b.
 * @param out   Buffer storing resulting real-valued image of size nx*ny*nz.
 * @param kappa Scaling factor to be applied to result of point-wise
 *              multiplication.
 * @param a     First real-valued image.
 * @param b     Second real-valued image.
 */
extern "C"
void
MaximumLikelihoodMultiplyKernelGPU(
   int nx, int ny, int nz, float *out, float kappa, 
   float *a, float *b);


/**
 * Update step of the maximum likelihood algorithm on the GPU.
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
Clarity_MaximumLikelihoodUpdateGPU(
   int nx, int ny, int nz, float* in, float* currentGuess, 
   float* otf, float* s1, float* s2, float* newGuess);

#endif // __MAXIMUM_LIKELIHOOD_DECONVOLVE_GPU_H_
