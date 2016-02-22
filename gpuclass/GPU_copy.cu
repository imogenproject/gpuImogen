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

  int returnCode = CHECK_CUDA_ERROR("entering GPU_copy");
  if(returnCode != SUCCESSFUL) return;
  
  MGArray orig[2];
  returnCode = MGA_accessMatlabArrays(prhs, 0, 1, &orig[0]);
  if(returnCode != SUCCESSFUL) {
	  CHECK_IMOGEN_ERROR(returnCode);
	  return;
  }

  int j;
  int sub[6];
  int64_t dan;
  for(j = 0; j < orig[0].nGPUs; j++) { 
    calcPartitionExtent(&orig[0], j, &sub[0]);
    dan = sub[3]*sub[4]*sub[5];

    cudaSetDevice(orig[0].deviceID[j]);
    CHECK_CUDA_ERROR("setdevice");
    cudaMemcpyPeerAsync((void *)orig[0].devicePtr[j], orig[0].deviceID[j], (void*)orig[1].devicePtr[j], orig[1].deviceID[j], dan*sizeof(double));
    returnCode = CHECK_CUDA_ERROR("cudamemcpy");
    if(returnCode != SUCCESSFUL) break;
  }

  CHECK_IMOGEN_ERROR(returnCode);
  return;
}
