#ifndef __Complex_h_
#define __Complex_h_

inline void ComplexAdd(float *c1, float *c2, float *result) {
   result[0] = c1[0] + c2[0];
   result[1] = c1[1] + c2[1];
}


inline void ComplexMultiply(float *c1, float *c2, float *result) {
   // (a + bi) * (c + di) = (ac - bd) + i(ad + bc)
   float a = c1[0];
   float b = c1[1];
   float c = c2[0];
   float d = c2[1];
   result[0] = a*c - b*d;
   result[1] = a*d + b*c;
}


inline void ComplexMultiply(float *c, float real, float *result) {
   result[0] = c[0]*real;
   result[1] = c[1]*real;
}


inline void ComplexMagnitudeSquared(float *c, float *result) {
  // a^2 + b^2
  float a = c[0];
  float b = c[1];
  result[0] = a*a + b*b;
  result[1] = 0.0f;
}


inline float ComplexMagnitudeSquared(float *c) {
  float a = c[0];
  float b = c[1];
  return ((a*a) + (b*b));
}


inline void ComplexInverse(float *c, float *result) {
  float a = c[0];
  float b = c[1];
  float mag = a*a + b*b;
  result[0] = a / mag;
  result[1] = -b / mag;
}


#endif // __Complex_h_