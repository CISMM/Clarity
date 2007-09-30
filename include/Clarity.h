/* note: changes to this file must be propagated to clarity.h.in so that 
   Clarity can be compiled using CMake and autoconf. */

#ifndef VERDICT_INC_LIB
#define VERDICT_INC_LIB

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

/** Enumerates the number and type of errors that 
   the Clarity library may produce. */
typedef enum {
   CLARITY_FFT_FAILED,
   CLARITY_OUT_OF_MEMORY,
   CLARITY_DEVICE_OUT_OF_MEMORY,
   CLARITY_INVALID_OPERATION,
   CLARITY_SUCCESS
} ClarityResult_t;


/*****************************/
/***** UTILITY FUNCTIONS *****/
/*****************************/
C_FUNC_DEF ClarityResult_t
Clarity_Register();


C_FUNC_DEF ClarityResult_t
Clarity_UnRegister();


C_FUNC_DEF ClarityResult_t
Clarity_SetNumberOfThreads(unsigned n);


/***********************************/
/***** DECONVOLUTION FUNCTIONS *****/
/***********************************/

C_FUNC_DEF ClarityResult_t 
Clarity_WienerDeconvolve(float* outImage, float* inImage, float* psfImage,
                         int nx, int ny, int nz, float noiseStdDev, float epsilon);


C_FUNC_DEF ClarityResult_t 
Clarity_JansenVanCittertDeconvolve(float* outImage, float* inImage, float* psfImage,
                                   int nx, int ny, int nz, unsigned iterations);


C_FUNC_DEF ClarityResult_t
Clarity_SmoothedJansenVanCittertDeconvolve(float* outImage, float* inImage, float* psfImage,
                                           int nx, int ny, int nz, unsigned iterations, 
                                           unsigned smoothInterval, float smoothSigma[3]);


C_FUNC_DEF ClarityResult_t
Clarity_IDivergenceDeconvolve(float* outImage, float* inImage, float* psfImage, 
                              int nx, int ny, int nz);

C_FUNC_DEF ClarityResult_t
Clarity_MaximumLikelihoodDeconvolve(float* outImage, float* inImage, float* psfImage,
									int nx, int ny, int nz, unsigned iterations);


/*************************/
/* CONVOLUTION FUNCTIONS */
/*************************/
/** Convolution function for a real image and a pre-padded and 
* cyclically-shifted kernel. */
C_FUNC_DEF ClarityResult_t
Clarity_Convolve(int nx, int ny, int nz, float* inImage, float* psfImage, 
                 float* outImage);


#endif // VERDICT_INC_LIB
