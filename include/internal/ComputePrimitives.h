#ifndef __COMPUTE_PRIMITIVES_H_
#define __COMPUTE_PRIMITIVES_H_

/**
 * Sums the elements of an array.
 *
 * @param result Single-element return parameter containing 
  *              result of reduction.
 * @param buffer Array of values to sum.
 * @param n      Number of float values to sum.
 */
ClarityResult_t
Clarity_ReduceSum(
   float* result, float* buffer, int n);

/**
 * Multiplies two arrays together component-wise.
 *
 * @param result The multiplied array.
 * @param a      First input array.
 * @param b      Second input array.
 * @param n      Length of arrays.
 */
ClarityResult_t
Clarity_MultiplyArraysComponentWise(
   float* result, float* a, float* b, int n);


/**
 * Divides two arrays together component-wise.
 *
 * @param result The multiplied array.
 * @param a      First input array whose elements are 
 *               numerators in the division.
 * @param b      Second input array whose elements are
 *               denominators in the division.
 * @param value  Value for result if denominator is zero.
 * @param n      Length of arrays.
 */
ClarityResult_t
Clarity_DivideArraysComponentWise(
   float* result, float* a, float* b, float value, int n);

/**
 * Scales an array of real values by a constant.
 *
 * @param result The scaled array.
 * @param a      Array to scale.
 * @param n      Length of array.
 * @param scale  Scale factor.
 */
ClarityResult_t
Clarity_ScaleArray(
   float* result, float* a, int n, float scale);


#endif // __COMPUTE_PRIMITIVES_H_