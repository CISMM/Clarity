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
   CLARITY_SUCCESS
} ClarityResult_t;


/*****************************/
/***** UTILITY FUNCTIONS *****/
/*****************************/
C_FUNC_DEF ClarityResult_t
Clarity_Init();


C_FUNC_DEF ClarityResult_t
Clarity_Destroy();


C_FUNC_DEF ClarityResult_t
Clarity_SetNumberOfThreads(int n);


/***********************************/
/***** DECONVOLUTION FUNCTIONS *****/
/***********************************/

C_FUNC_DEF ClarityResult_t 
Clarity_WienerDeconvolve(float* outImage, float* inImage, float* psfImage, int dimensions[3]);


C_FUNC_DEF ClarityResult_t 
Clarity_JansenVanCittertDeconvolve(float* outImage, float* inImage, float* psfImage, int dimensions[3], 
                                   unsigned iterations);


C_FUNC_DEF ClarityResult_t
Clarity_SmoothedJansenVanCittertDeconvolve(float* outImage, float* inImage, float* psfImage, 
                                           int dimensions[3], unsigned iterations, 
                                           unsigned smoothInterval, float smoothSigma);


C_FUNC_DEF ClarityResult_t
Clarity_IDivergenceDeconvolve(float* outImage, float* inImage, float* psfImage, int dimensions[3]);

#endif // VERDICT_INC_LIB
