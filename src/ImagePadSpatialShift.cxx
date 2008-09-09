#include "Clarity.h"

#include <malloc.h>

// WARNING! Only the CPU side is provided here because padding and
// shifting is a low-frequency operation.
// Assumes adequate CPU side memory has been allocated in dst.
ClarityResult_t
Clarity_ImagePadSpatialShift(float *dst, int dstDim[3],
                             float *src, int srcDim[3],
                             int shift[3], float fillValue) {

   if (dst == NULL || src == NULL) {
      return CLARITY_INVALID_ARGUMENT;
   }
   
   for (int dk = 0; dk < dstDim[2]; dk++) {
      int sk = dk - shift[2];
      if (sk < 0) sk += dstDim[2];
      sk = sk % dstDim[2];
      bool withinK = sk >= 0 && sk < srcDim[2];
      for (int dj = 0; dj < dstDim[1]; dj++) {
         int sj = dj - shift[1];
         if (sj < 0) sj += dstDim[1];
         sj = sj % dstDim[1];
         bool withinJ = sj >= 0 && sj < srcDim[1];
         for (int di = 0; di < dstDim[0]; di++) {
            int si = di - shift[0];
            if (si < 0) si += dstDim[0];
            si = si % dstDim[0];
            bool withinI = si >= 0 && si < srcDim[0];
            int dIndex = (dk*dstDim[1]*dstDim[0]) + (dj*dstDim[0]) + di;
            if (withinI && withinJ && withinK) {
               int sIndex = (sk*srcDim[1]*srcDim[0]) + (sj*srcDim[0]) + si;
               dst[dIndex] = src[sIndex];
            } else {
               dst[dIndex] = fillValue;
            }
         }
      }
   }

   return CLARITY_SUCCESS;
}
