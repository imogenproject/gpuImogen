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

// static paramaters

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // wrapper for cudaFree().
  if((nlhs != 1) || (nrhs != 0)) mexErrMsgTxt("GPU_memavail: syntax is freemem = GPU_memavail()");

  size_t freemem; size_t totalmem;

  cudaError_t fail =  cudaMemGetInfo(&freemem, &totalmem);

  if(fail != cudaSuccess) mexErrMsgTxt("GPU_memavail: mem info returned not-success.");

  mwSize dims[3]; dims[0] = dims[1] = dims[2] = 1;
  plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);

  double *d = mxGetPr(plhs[0]);
  *d = freemem;

  return;
}
