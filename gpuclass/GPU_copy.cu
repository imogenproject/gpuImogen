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

// GPU_Tag = GPU_clone(GPU_type)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // At least 2 arguments expected
  // Input and result
  if((nlhs != 0) || (nrhs != 2)) { mexErrMsgTxt("Form: GPU_copy(to tag, from tag)"); }

  CHECK_CUDA_ERROR("entering GPU_copy");
  
  MGArray orig[2];
  int worked = accessMGArrays(prhs, 0, 1, &orig[0]);

  int j;
  int sub[6];
  int64_t dan;
  for(j = 0; j < orig[0].nGPUs; j++) { 
    calcPartitionExtent(&orig[0], j, &sub[0]);
    dan = sub[3]*sub[4]*sub[5];

    cudaSetDevice(orig[0].deviceID[j]);
    CHECK_CUDA_ERROR("setdevice");
    cudaMemcpy((void *)orig[0].devicePtr[j], (void*)orig[1].devicePtr[j], dan*sizeof(double), cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR("cudamemcpy");
  }

  return;
}
