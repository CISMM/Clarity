#ifndef __Complex_h_
#define __Complex_h_

/**
 * Returns conjugate of a complex number.
 * 
 * @param c1     Ccomplex number to conjugate.
 * @param result Return parameter storing the conjugate.
 *               Can be the same as c1.
 */
inline void ComplexConjugate(float* c1, float* result) {
   result[0] =  c1[0];
   result[1] = -c1[1];
}


/**
 * Adds two complex numbers.
 * 
 * @param c1     First complex number.
 * @param c2     Second complex number. Can be the same as the first.
 * @param result Storage for result of addition.
 *               Can be the same as c1, c2, or both.
 */
inline void ComplexAdd(float* c1, float* c2, float* result) {
   result[0] = c1[0] + c2[0];
   result[1] = c1[1] + c2[1];
}


/**
 * Multiplies two complex numbers.
 * 
 * @param c1     First complex number.
 * @param c2     Second complex number.
 * @param result Storage for result of multiplication.
 *               Can be the same as c1, c2, or both.
 */
inline void ComplexMultiply(float* c1, float* c2, float* result) {
   // (a + bi) * (c + di) = (ac - bd) + i(ad + bc)
   float a = c1[0];
   float b = c1[1];
   float c = c2[0];
   float d = c2[1];
   result[0] = a*c - b*d;
   result[1] = a*d + b*c;
}


/**
 * Multiplies a complex number with a real number.
 * 
 * @param c      Complex number to multiply.
 * @param real   Real multiplier.
 * @param result Storage for result of multiplication.
 */
inline void ComplexMultiply(float* c, float real, float* result) {
   result[0] = c[0]*real;
   result[1] = c[1]*real;
}


/**
 * Multiplies two complex numbers and scales the result by a real number.
 * 
 * @param c1     First complex number.
 * @param c2     Second complex number.
 * @param scale  Real scalar multiplier.
 * @param result Storage for result of the operation.
 */
inline void ComplexMultiplyAndScale(float* c1, float* c2, float scale, float* result) {
   float a = c1[0];
   float b = c1[1];
   float c = c2[0];
   float d = c2[1];
   result[0] = scale*(a*c - b*d);
   result[1] = scale*(a*d + b*c);
}


/**
 * Computes squared magnitude of a complex number, storing real result as a
 * complex number (zero imaginary component).
 * 
 * @param c		 Complex number whose magnitude should be taken.
 * @param result Complex number storing real result.
 */
inline void ComplexMagnitudeSquared(float* c, float* result) {
  // a^2 + b^2
  float a = c[0];
  float b = c[1];
  result[0] = a*a + b*b;
  result[1] = 0.0f;
}


/**
 * Computes squared magnitude of a complex number, returning real result.
 * 
 * @param c Complex number whose magnitude should be taken.
 * @return Squared magnitude.
 */
inline float ComplexMagnitudeSquared(float* c) {
  float a = c[0];
  float b = c[1];
  return ((a*a) + (b*b));
}


/**
 * Computes inverse of complex number.
 * 
 * @param c      Complex number whose inverse should be computed.
 * @param result Storage for the inverted number.
 */
inline void ComplexInverse(float* c, float* result) {
  float a = c[0];
  float b = c[1];
  float mag = a*a + b*b;
  result[0] = a / mag;
  result[1] = -b / mag;
}


#endif // __Complex_h_
