#include "Clarity.h"

#include "fftw3.h"
#include <iostream>
#include <omp.h>

#ifdef BUILD_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

/** How many clients are registered. */
static unsigned gRegisteredClients = 0;

/** Indicates that a CUDA-capable device is available. */
bool gCUDACapable = false;

ClarityResult_t
Clarity_Register() {
   if (gRegisteredClients <= 0) {
      fftwf_init_threads();
      int np = omp_get_num_procs();
      Clarity_SetNumberOfThreads(np);

#ifdef BUILD_WITH_CUDA
      int deviceCount = 0;
      cudaGetDeviceCount(&deviceCount);
      if (deviceCount >= 1) {
         cudaDeviceProp deviceProp;
         cudaGetDeviceProperties(&deviceProp, 0);
         std::cout << "CUDA device found: '" << deviceProp.name << "'" << std::endl;
         gCUDACapable = true;
      }
#endif
   }
   gRegisteredClients++;

   return CLARITY_SUCCESS;
}


ClarityResult_t
Clarity_UnRegister() {
   gRegisteredClients--;
   if (gRegisteredClients <= 0) {
      fftwf_cleanup_threads();
      gCUDACapable = false;
   }

   return CLARITY_SUCCESS;
}


C_FUNC_DEF ClarityResult_t
Clarity_SetNumberOfThreads(unsigned n) {
   omp_set_num_threads(n);
   int np = omp_get_num_procs();
   fftwf_plan_with_nthreads(np);

   return CLARITY_SUCCESS;
}