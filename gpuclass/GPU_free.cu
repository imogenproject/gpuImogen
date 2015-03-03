#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif
#include "mex.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas.h"

#include "cudaCommon.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // wrapper for cudaFree().
  if((nlhs != 0) || (nrhs == 0)) mexErrMsgTxt("GPU_free: syntax is GPU_free(arbitrarily many GPU_Types, gpu tags, or ImogenArrays)");

  CHECK_CUDA_ERROR("Entering GPU_free()");
  MGArray t[nrhs];

  int worked = accessMGArrays(prhs, 0, nrhs-1, &t[0]);

  int i, j;

  for(i = 0; i < nrhs; i++) {
	  if(t[i].numSlabs < 1) continue; // This is a slab reference and was never actually allocated. Ignore it.
    for(j = 0; j < t[i].nGPUs; j++) {
      cudaSetDevice(t[i].deviceID[j]);
      CHECK_CUDA_ERROR("cudaSetDevice()");
      cudaError_t result = cudaFree(t[i].devicePtr[j]);
      CHECK_CUDA_ERROR("After GPU_free()");
    }
  }

return;
}
