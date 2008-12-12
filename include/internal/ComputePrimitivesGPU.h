#ifndef __COMPUTE_PRIMITIVES_GPU_H_
#define __COMPUTE_PRIMITIVES_GPU_H_


/**
 * Sums the elements of an array on the GPU.
 *
 * @param result Single-element return parameter containing 
  *              result of reduction.
 * @param buffer Array of values to sum.
 * @param n      Number of float values to sum.
 */
extern "C"
void
Clarity_ReduceSumGPU(float* result, float* buffer, int n);

/**
 * Multiplies two arrays together component-wise.
 *
 * @param result The multiplied array.
 * @param a      First input array.
 * @param b      Second input array.
 * @param n      Length of arrays.
 */
extern "C"
void
Clarity_MultiplyArraysComponentWiseGPU(
   float* result, float* a, float* b, int n);


/**
 * Divides two arrays together component-wise.
 *
 * @param result The multiplied array.
 * @param a      First input array.
 * @param b      Second input array.
 * @param value  Value for result if denominator is zero.
 * @param n      Length of arrays.
 */
extern "C"
void
Clarity_DivideArraysComponentWiseGPU(
   float* result, float* a, float* b, float value, int n);


/**
 * Scales an array of real values by a constant.
 *
 * @param result The scaled array.
 * @param a      Array to scale.
 * @param n      Length of array.
 * @param scale  Scale factor.
 */
extern "C"
void
Clarity_ScaleArrayGPU(
   float* result, float* a, int n, float scale);


#endif // __COMPUTE_PRIMITIVES_GPU_H_
