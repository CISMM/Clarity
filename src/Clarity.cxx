#include "Clarity.h"

#include "fftw3.h"
#include <iostream>
#include <omp.h>

static unsigned gRegisteredClients = 0;

ClarityResult_t
Clarity_Register() {
   if (gRegisteredClients <= 0) {
      int np = omp_get_num_procs();
      Clarity_SetNumberOfThreads(np);
      fftwf_init_threads();
      fftwf_plan_with_nthreads(np);
   }
   gRegisteredClients++;

   return CLARITY_SUCCESS;
}


ClarityResult_t
Clarity_UnRegister() {
   gRegisteredClients--;
   if (gRegisteredClients <= 0) {
      
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