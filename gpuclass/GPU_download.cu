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

// host_array = GPU_download(gpu type)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // At least 2 arguments expected
  // Input and result
  if((nlhs != 1) || (nrhs != 1)) { mexErrMsgTxt("Form: host_array = GPU_download(GPU array)"); }

  CHECK_CUDA_ERROR("entering GPU_download");

  MGArray m;
  
  MGA_accessMatlabArrays(prhs, 0, 0, &m);

  int nd = 3;
  if(m.dim[2] == 1) {
    nd = 2;
    if(m.dim[1] == 1) {
      nd = 1;
    }
  }
  mwSize odims[3];
  odims[0] = m.dim[0];
  odims[1] = m.dim[1];
  odims[2] = m.dim[2];

  // Create output numeric array
  plhs[0] = mxCreateNumericArray(nd, odims, mxDOUBLE_CLASS, mxREAL);

  double *result = mxGetPr(plhs[0]);

  int itworked = MGA_downloadArrayToCPU(&m, &result, 0);

  return;
}
