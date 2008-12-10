#include "Clarity.h"

// WARNING! Only the CPU side is provided here because padding and
// shifting is a low-frequency operation.
// Assumes adequate CPU side memory has been allocated in dst.
ClarityResult_t
Clarity_ImageClip(float *dst, Clarity_Dim3 dstDim, 
                  float *src, Clarity_Dim3 srcDim) {
   
   if (dst == NULL || src == NULL) {
      return CLARITY_INVALID_ARGUMENT;
   }

   for (int dk = 0; dk < dstDim.z; dk++) {
      for (int dj = 0; dj < dstDim.y; dj++) {
         for (int di = 0; di < dstDim.x; di++) {
            int dIndex = (dk*dstDim.y*dstDim.x) + (dj*dstDim.x) + di;
            int sIndex = (dk*srcDim.y*srcDim.y) + (dj*srcDim.x) + di;
            dst[dIndex] = src[sIndex];
         }
      }
   }

   return CLARITY_SUCCESS;

}
