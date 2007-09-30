typedef float2 Complex;

__device__ Complex complexConjugate(Complex a) {
   Complex t = {a.x, -a.y};
   return t;
}


__device__ Complex complexAdd(Complex a, Complex b) {
    Complex t = {a.x + b.x, a.y + b.y};
    return t;
}


__device__ Complex complexMul(Complex a, Complex b) {
    Complex t = {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
    return t;
}


__device__ Complex complexScale(Complex a, float c) {
    Complex t = {a.x * c, a.y * c};
    return t;
}


__device__ Complex complexMulAndScale(Complex a, Complex b, float c) {
    return complexScale(complexMul(a, b), c);
}


__device__ float complexMagnitudeSquared(Complex a) {
   return ((a.x*a.x) + (a.y*a.y));
}
