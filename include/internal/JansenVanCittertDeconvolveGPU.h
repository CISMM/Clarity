#ifndef __JANSEN_VAN_CITTERT_DECONVOLVE_H_
#define __JANSEN_VAN_CITTERT_DECONVOLVE_H_

/**
 * Function to invoke computation kernel on the GPU.
 * 
 * @param nx       X-dimension of in, i_k, o_k, and i_kNext.
 * @param ny       Y-dimension of in, i_k, o_k, and i_kNext.
 * @param nz       Z-dimension of in, i_k, o_k, and i_kNext.
 * @param in       Real input image.
 * @param inMax    Half the maximum value in the input image.
 * @param invMaxSq Inverse of half the maximum squared value in the input
 *                 image.
 * @param i_k      Current guess of the uncorrupted image.
 * @param o_k      Storage pointer.
 * @param i_kNext  Next guess of the corrupted image.
 */
extern "C"
void
JansenVanCittertDeconvolveKernelGPU(
   int nx, int ny, int nz, float* in, float inMax, float invMaxSq,
   float* i_k, float* o_k, float* i_kNext);


#endif // __JANSEN_VAN_CITTERT_DECONVOLVE_H_
