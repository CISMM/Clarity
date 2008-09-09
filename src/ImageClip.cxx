#include "Clarity.h"

// WARNING! Only the CPU side is provided here because padding and
// shifting is a low-frequency operation.
// Assumes adequate CPU side memory has been allocated in dst.
ClarityResult_t
Clarity_ImageClip(float *dst, int dstDim[3],
                  float *src, int srcDim[3]) {
   
   if (dst == NULL || src == NULL) {
      return CLARITY_INVALID_ARGUMENT;
   }

   for (int dk = 0; dk < dstDim[2]; dk++) {
      for (int dj = 0; dj < dstDim[1]; dj++) {
         for (int di = 0; di < dstDim[0]; di++) {
            int dIndex = (dk*dstDim[1]*dstDim[0]) + (dj*dstDim[0]) + di;
            int sIndex = (dk*srcDim[1]*srcDim[1]) + (dj*srcDim[0]) + di;
            dst[dIndex] = src[sIndex];
         }
      }
   }

   return CLARITY_SUCCESS;

}