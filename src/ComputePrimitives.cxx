#include <cmath>
#include <cstdio>
#include <omp.h>

#include "Clarity.h"

#include "ComputePrimitives.h"
#include "ComputePrimitivesGPU.h"

extern bool g_CUDACapable;

ClarityResult_t
Clarity_ReduceSum(float* result, float* buffer, int n) {
   
  ClarityResult_t err = CLARITY_SUCCESS;

#ifdef BUILD_WITH_CUDA
  if (g_CUDACapable) {
    Clarity_ReduceSumGPU(result, buffer, n);
  } else
#endif // BUILD_WITH_CUDA
  {
    float sum = 0.0f;
#ifdef __GNUG__ // OpenMP in GCC is buggy with reductions, so we'll handle
                // the reduction serially.
    for (int i = 0; i < n; i++) {
      sum += buffer[i];
    }

#else
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
      sum += buffer[i];
    }
#endif // __GNUG__

    *result = sum;
  }

  return err;
}


ClarityResult_t
Clarity_MultiplyArraysComponentWise(
  float* result, float* a, float* b, int n) {

  ClarityResult_t err = CLARITY_SUCCESS;

#ifdef BUILD_WITH_CUDA
  if (g_CUDACapable) {
    Clarity_MultiplyArraysComponentWiseGPU(result, a, b, n);
  } else
#endif // BUILD_WITH_CUDA
  {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      result[i] = a[i] * b[i];
    }
  }

  return err;
}


ClarityResult_t
Clarity_DivideArraysComponentWise(
   float* result, float* a, float* b, float value, int n) {

   ClarityResult_t err = CLARITY_SUCCESS;

#ifdef BUILD_WITH_CUDA
   if (g_CUDACapable) {
      Clarity_DivideArraysComponentWiseGPU(result, a, b, value, n);
   } else
#endif // BUILD_WITH_CUDA
   {
#pragma omp parallel for
      for (int i = 0; i < n; i++) {
         if (fabs(b[i]) < 1e-5) {
            result[i] = value;
         } else {
            result[i] = a[i] / b[i];
         }
      }
   }

   return err;
}


ClarityResult_t
Clarity_ScaleArray(
   float* result, float* a, int n, float scale) {

   ClarityResult_t err = CLARITY_SUCCESS;

#ifdef BUILD_WITH_CUDA
   if (g_CUDACapable) {
      Clarity_ScaleArrayGPU(result, a, n, scale);
   } else
#endif // BUILD_WITH_CUDA
   {
#pragma omp parallel for
      for (int i = 0; i < n; i++) {
         result[i] = scale * a[i];
      }
   }

   return err;
}
