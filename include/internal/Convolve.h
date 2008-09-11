#ifndef __CONVOLVE_H_
#define __CONVOLVE_H_

/**
 * Convolves two images of equal dimensions.
 * 
 * @param nx       X-dimension of inImage, kernel, and outImage.
 * @param ny       Y-dimension of inImage, kernel, and outImage.
 * @param nz       Z-dimension of inImage, kernel, and outImage.
 * @param inImage  Real image to convolve.
 * @param kernel   Convolution kernel.
 * @param outImage Resulting real image.
 */
ClarityResult_t
Clarity_Convolve(
   int nx, int ny, int nz, float* inImage, 
   float* kernel, float* outImage);


/** 
 * Convolution function with pre-Fourier-transformed kernel,
 * sometimes called the optical transfer function (OTF). 
 * 
 * @param nx  X-dimension of in, otf, and out.
 * @param ny  Y-dimension of in, otf, and out.
 * @param nz  Z-dimension of in, otf, and out.
 * @param in  Real input image.
 * @param otf Optical transfer function (Fourier transform of
 *            convolution kernel.
 * @param out Resulting real image.
 * 
 */
ClarityResult_t
Clarity_Convolve_OTF(
   int nx, int ny, int nz, float* in, float* otf, float* out);


/** 
 * Internal convolution function.
 * 
 * @param nx     X-dimension of in, kernel, and out.
 * @param ny     Y-dimension of in, kernel, and out.
 * @param nz     Z-dimension of in, kernel, and out.
 * @param in     Real input image.
 * @param kernel Real convolution kernel.
 * @param out    Resulting real image.
 */
ClarityResult_t
Clarity_ConvolveInternal(
   int nx, int ny, int nz, float* in, float* kernel, float* out);


/** Per-pixel modulation of one transformed image with another.
 *
 * @param nx  X-dimension of in, otf, and out.
 * @param ny  Y-dimension of in, otf, and out.
 * @param nz  Z-dimension of in, otf, and out.
 * @param in  Complex input image.
 * @param otf Modulating complex optical transfer function.
 * @param out Output of modulation.
 */
void
Clarity_Modulate(
   int nx, int ny, int nz, float* in, float* otf, float* out);


#endif // __CONVOLVE_H_