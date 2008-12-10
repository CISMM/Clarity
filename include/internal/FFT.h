#ifndef __FFT_H_
#define __FFT_H_

#include "Clarity.h"


/** 
 * 3D forward FFT function. 
 * 
 * @param nx  X-dimension of in and out.
 * @param ny  Y-dimension of in and out.
 * @param nz  Z-dimension of in and out.
 * @param in  Input real image.
 * @param out Output complex image without redundant coefficients.
 */
C_FUNC_DEF
ClarityResult_t
Clarity_FFT_R2C_float(
   int nx, int ny, int nz, float* in, float* out);


/** 
 * 3D inverse FFT function.
 * 
 * @param nx  X-dimension of in and out.
 * @param ny  Y-dimension of in and out.
 * @param nz  Z-dimension of in and out.
 * @param in  Input complex image without redundant coefficients.
 * @param out Output real image.
 */
C_FUNC_DEF
ClarityResult_t
Clarity_FFT_C2R_float(
   int nx, int ny, int nz, float* in, float* out);


#endif // __FFT_H_
