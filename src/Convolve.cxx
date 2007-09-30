#include "Clarity.h"

#include "FFT.h"

extern bool gCUDACapable;

ClarityResult_t
Clarity_Convolve(int nx, int ny, int nz, float* inImage, float* psfImage,
                 float* outImage) {
#ifdef TIME
   totalTimer.Start();
#endif
   int numVoxels = nx*ny*nz;
   ClarityResult_t result = CLARITY_SUCCESS;

   // Copy over the input image and PSF.
   if (gCUDACapable) {
      float* in;
      float* psf;
      
      result = Clarity_Real_MallocCopy((void**) &in, sizeof(float), 
         nx, ny, nz, inImage);
      if (result != CLARITY_SUCCESS) {
         return result;
      }
      result = Clarity_Real_MallocCopy((void **) &psf, sizeof(float), 
         nx, ny, nz, psfImage);
      if (result != CLARITY_SUCCESS) {
         Clarity_Free(in);
         return result;
      }
      Clarity_ConvolveInternal(nx, ny, nz, in, psf, in);
      result = Clarity_CopyFromDevice(nx, ny, nz, sizeof(float), outImage, in);
      Clarity_Free(in); Clarity_Free(psf);

   } else {

      Clarity_ConvolveInternal(nx, ny, nz, inImage, psfImage, outImage);
   }

#ifdef TIME
   totalTimer.Stop();
   std::cout << totalTimer << std::endl;
   std::cout << transferTimer << std::endl;
   totalTimer.Reset();
   transferTimer.Reset();
#endif

   return result;
}
