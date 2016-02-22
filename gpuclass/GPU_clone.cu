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
  if((nlhs != 1) || (nrhs != 1)) { mexErrMsgTxt("Form: result_tag = GPU_clone(input tag)"); }

  if(CHECK_CUDA_ERROR("entering GPU_clone") != SUCCESSFUL) return;
  
  MGArray orig;
  int returnCode = MGA_accessMatlabArrays(prhs, 0, 0, &orig);

  if(returnCode != SUCCESSFUL) {
	  CHECK_IMOGEN_ERROR(returnCode);
	  return;
  }

  MGArray *nu = MGA_createReturnedArrays(plhs, 1, &orig);

  int j;
  int sub[6];
  int64_t dan;
  for(j = 0; j < orig.nGPUs; j++) { 
    calcPartitionExtent(&orig, j, &sub[0]);
    dan = sub[3]*sub[4]*sub[5];

    cudaSetDevice(orig.deviceID[j]);
    cudaMemcpy((void *)nu->devicePtr[j], (void*)orig.devicePtr[j], dan*sizeof(double), cudaMemcpyDeviceToDevice);
    returnCode = CHECK_CUDA_ERROR("GPU_clone: memcpy");
    if(returnCode != SUCCESSFUL) break;
  }

  return;
}
