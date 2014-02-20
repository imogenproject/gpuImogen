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
  if((nlhs != 0) || (nrhs != 1)) mexErrMsgTxt("GPU_free: syntax is GPU_free(GPU_Type or gpu tag)");

//  if(mxGetClassID(prhs[0]) != mxINT64_CLASS) mexErrMsgTxt("GPU_free: passed a not-gpupointer");

  ArrayMetadata amd;
  double **a = getGPUSourcePointers(prhs, &amd, 0, 0);

  cudaCheckError("Before GPU_free()");
  cudaError_t result = cudaFree(a[0]);
  cudaCheckError("After GPU_free()");

  free(a);

  return;
}
