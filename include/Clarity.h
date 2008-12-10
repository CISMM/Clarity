/* note: changes to this file must be propagated to clarity.h.in so that 
   Clarity can be compiled using CMake and autoconf. */

#ifndef __CLARITY_LIB_H_
#define __CLARITY_LIB_H_

#ifdef __cplusplus
# if defined(WIN32) && defined(CLARITY_SHARED_LIB)
#  ifdef Clarity_EXPORTS
#   define C_FUNC_DEF extern "C" __declspec(dllexport)
#  else
#   define C_FUNC_DEF extern "C" __declspec(dllimport)
#  endif
# else
#  define C_FUNC_DEF extern "C"
# endif
#else
# if defined(WIN32) && defined(CLARITY_SHARED_LIB)
#  ifdef Clarity_EXPORTS
#   define C_FUNC_DEF __declspec(dllexport)
#  else
#   define C_FUNC_DEF __declspec(dllimport)
#  endif
# else
#  define C_FUNC_DEF
# endif
#endif

#ifndef NULL
#define NULL 0L
#endif

/** Enumerates the number and type of errors that 
   the Clarity library may produce. */
typedef enum {
   CLARITY_FFT_FAILED,            /** Fast Fourier transform routine 
                                   *  failed to execute. */
   CLARITY_OUT_OF_MEMORY,         /** Host system ran out of memory while 
                                   *  executing the function. */
   CLARITY_DEVICE_OUT_OF_MEMORY,  /** Computational accelerator ran out of
                                   *  memory while executing the function. */
   CLARITY_INVALID_OPERATION,     /** Operation is invalid for the arguments
                                   *  passed to it. */
   CLARITY_INVALID_ARGUMENT,      /** One or more of the arguments was invalid. */
   CLARITY_SUCCESS                /** Function executed successfully. */
} ClarityResult_t;


/*****************************/
/***** TYPES *****************/
/*****************************/

/** Type for specifying image 3D image dimensions. */
typedef struct _Clarity_Dim3_t {
   int x;
   int y;
   int z;
} Clarity_Dim3;


/** Creates a Clarity_Dim from a three-element integer array
 *  representing dimensions in x, y, and z.
 *
 * @param The three-element array.
 * @return The Clarity_Dim object.
 */
C_FUNC_DEF
Clarity_Dim3
Clarity_Dim3FromArray(int dimArray[3]);


/*****************************/
/***** UTILITY FUNCTIONS *****/
/*****************************/

/**
 * Clients should call this function prior to calling any other Clarity
 * function. Initializes underlying libraries and sets the number of
 * threads to the number of cores on the system.
 */
C_FUNC_DEF
ClarityResult_t
Clarity_Register();


/**
 * Clients should call this function when finished with the Clarity
 * library. It cleans up and releases resources used by the Clarity
 * library.
 */
C_FUNC_DEF
ClarityResult_t
Clarity_UnRegister();


/**
 * Sets the number of threads that should be used by the Clarity library.
 * Usually, you want this to be the same as the number of cores on the
 * CPU. By default, Clarity runs on a number of threads equal to the number
 * of cores on the CPU on which it is running.
 *
 * @param n Number of threads on which to run.
 */
C_FUNC_DEF
ClarityResult_t
Clarity_SetNumberOfThreads(unsigned n);


/**
 * Utility function to create a new image of a desired size containing the shifted
 * contents of the input image. Useful for padding and shifting convolution
 * kernels. 
 * 
 * @param dst       Destination buffer for shifted result. Buffer must be allocated
 *                  by the caller.
 * @param dstDim    Dimensions of the destination buffer.
 * @param src       Source buffer for image data to be shiftd.
 * @param srcDim    Dimensions of the source buffer.
 * @param shift     Three-element array corresponding the spatial shift 
                    in x, y, and z. Shifting operates cyclically across image 
                    boundaries.
 * @param fillValue Value to which pixels in parts of the new image not
 *                  corresponding to a shifted pixel get set.
 */
C_FUNC_DEF
ClarityResult_t
Clarity_ImagePadSpatialShift(float *dst, Clarity_Dim3 dstDim, 
                             float *src, Clarity_Dim3 srcDim,
                             int shift[3], float fillValue);

/**
 * Utility function to clip out a portion of an image. Useful for truncating the 
 * result of a convolution of a padded image.
 *
 * @param dst    Destination buffer for clipped result. Buffer must be allocated
 *               by the caller.
 * @param dstDim Dimensions of the destination buffer. Implicitly represents a 
                 coordinate in the source image. The clipped image corresponds to 
                 cropping the source image from the origin to coordinate (x, y, z).
 * @param src    Source buffer image to clip. Assumed to be larger or equal in size
                 in all three dimensions to the clipped image.
 * @param srcDim Dimensions of the source buffer.
 */
C_FUNC_DEF
ClarityResult_t
Clarity_ImageClip(float *dst, Clarity_Dim3 dstDim, float *src, Clarity_Dim3 srcDim);


/***********************************/
/***** DECONVOLUTION FUNCTIONS *****/
/***********************************/

/**
 * Applies a Wiener filter for deconvolution.
 *
 * @param outImage    Caller-allocated buffer holding result of Wiener filter.
 *                    Dimensions of this buffer are given by dim.
 * @param inImage     Image to be deconvolved. Dimensions of this buffer are given by dim.
 * @param psfImage    Image of the point-spread function of the system that produced
 *                    the image in the outImage parameter.
 * @param dim         Dimension of outImage, inImage, and psfImage.
 * @param epsilon     Constant standing in place of the ratio between power spectra of
 *                    noise and the power spectra of the underlying image, which are
 *                    unknown parameters. In practice, acts as a smoothing factor.
 *                    Typically set in the range 0.001 to 0.1.
 */
C_FUNC_DEF
ClarityResult_t 
Clarity_WienerDeconvolve(float* outImage, float* inImage, float* psfImage,
                         Clarity_Dim3 dim, float epsilon);


/**
 * Class Jansen-van Cittert formulation for constrained iterative deconvolution.
 * 
 * @param outImage   Caller-allocated buffer holding result of Wiener filter.
 *                   Dimensions of this buffer are given by dim.
 * @param inImage    Image to be deconvolved. Dimensions of this buffer are
 *                   given by dim.
 * @param psfImage   Image of the point-spread function of the system that produced
 *                   the image in the outImage parameter.
 * @param dim        Dimension of outImage, inImage, and psfImage.
 * @param iterations Number of algorithm iterations to run.
 */
C_FUNC_DEF
ClarityResult_t 
Clarity_JansenVanCittertDeconvolve(float* outImage, float* inImage, float* psfImage,
                                   Clarity_Dim3 dim, unsigned iterations);


/**
 * WARNING: This function's implementation is temporary and produces undefined results.
 *
 * Implementation of the Jansen-van Cittert formulation for constrained iterative
 * deconvolution that applies a smoothing step every few iterations to reduce noise
 * amplification.
 * 
 * @warning          Implementation incomplete.
 * @param outImage   Caller-allocated buffer holding result of this filter.
 *                   Dimensions of this buffer are given by dim.
 * @param inImage    Image to be deconvolved. Dimensions of this buffer are
 *                   given by dim.
 * @param psfImage   Image of the point-spread function of the system that produced
 *                   the image in the outImage parameter.
 * @param dim        Dimension of outImage, inImage, and psfImage.
 * @param iterations Number of algorithm iterations to run.
 */
C_FUNC_DEF
ClarityResult_t 
Clarity_SmoothedJansenVanCittertDeconvolve(float* outImage, float* inImage, float* psfImage,
                                           Clarity_Dim3 dim, unsigned iterations,
                                           unsigned smoothInterval, float smoothSigma[3]);


/**
 * WARNING: This function's implementation is temporary and produces undefined results.
 *
 * Unimplemented, but promising, deconvolution method based on the paper:
 * J. Markham and J.A. Conchello, Fast maximum-likelihood image-restoration algorithms
 * for three-dimensional fluorescence microscopy, J. Opt. Soc. Am. A, Vol. 18, No. 5,
 * May 2001.
 *
 * @param outImage   Caller-allocated buffer holding result of Wiener filter.
 *                   Dimensions of this buffer are given by dim.
 * @param inImage    Image to be deconvolved. Dimensions of this buffer are
 *                   given by dim.
 * @param psfImage   Image of the point-spread function of the system that produced
 *                   the image in the outImage parameter.
 * @param dim        Dimension of outImage, inImage, and psfImage.
 */
C_FUNC_DEF
ClarityResult_t
Clarity_IDivergenceDeconvolve(float* outImage, float* inImage, float* psfImage, 
                              Clarity_Dim3 dim);


/**
 * Maximum-likelihood deconvolution method cited in the paper:
 * J.B. Sibarita, Deconvolution microscopy, Adv. Biochem. Engin./Biotechnology (2005) 95: 201-243.
 *
 * @param outImage   Caller-allocated buffer holding result of Wiener filter.
 *                   Dimensions of this buffer are given by dim.
 * @param inImage    Image to be deconvolved. Dimensions of this buffer are
 *                   given by dim.
 * @param psfImage   Image of the point-spread function of the system that produced
 *                   the image in the outImage parameter.
 * @param dim        Dimension of outImage, inImage, and psfImage.
 * @param iterations Number of algorithm iterations to run.
 */
C_FUNC_DEF
ClarityResult_t
Clarity_MaximumLikelihoodDeconvolve(float* outImage, float* inImage, float* psfImage,
									         Clarity_Dim3 dim, unsigned iterations);

/**
 * Blind maximum-likelihood deconvolution method cited in the paper:
 * J.B. Sibarita, Deconvolution microscopy, Adv. Biochem. Engin./Biotechnology (2005) 95: 201-243.
 *
 * @param outImage   Caller-allocated buffer holding result of Wiener filter.
 *                   Dimensions of this buffer are given by dim.
 * @param inImage    Image to be deconvolved. Dimensions of this buffer are
 *                   given by dim.
 * @param psfImage   Image of the point-spread function of the system that produced
 *                   the image in the outImage parameter.
 * @param dim        Dimension of outImage, inImage, and psfImage.
 * @param iterations Number of algorithm iterations to run.
 */
C_FUNC_DEF
ClarityResult_t
Clarity_BlindMaximumLikelihoodDeconvolve(float* outImage, float* inImage, float* psfImage,
                                         Clarity_Dim3 dim, unsigned iterations);


/*************************/
/* CONVOLUTION FUNCTIONS */
/*************************/

/** Convolution function for a real image and a pre-padded and 
 * cyclically-shifted kernel.
 * 
 * @param dim        Dimension of outImage, inImage, and psfImage.
 * @param inImage    Image to be convolved. Dimensions of this buffer are given by
                     dim.
 * @param kernel     Convolution kernel.
 * @param outImage   Caller-allocated buffer to store results of the convolution.
 */
C_FUNC_DEF
ClarityResult_t
Clarity_Convolve(Clarity_Dim3 dim, float* inImage, float* kernel, 
                 float* outImage);


#endif // __CLARITY_LIB_H_
