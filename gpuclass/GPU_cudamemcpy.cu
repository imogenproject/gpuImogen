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
  // At least 2 arguments expected
  // Input and result
  if ((nlhs == 1) & (nrhs==1)) {
    // double -> GPU array
    // It is very important that this all be right because all GPU stuff originates from here.
    double *from = mxGetPr(prhs[0]);

    mwSize dims[2]; dims[0] = 5; dims[1] = 1;
    plhs[0] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);

    int64_t *retptr = (int64_t *)mxGetData(plhs[0]);

    size_t Nel = mxGetNumberOfElements(prhs[0]);
    
    double *rmem;

    cudaError_t fail = cudaMalloc((void **)&rmem, Nel * sizeof(double));
    if(fail == cudaErrorMemoryAllocation) mexErrMsgTxt("GPU_cudamemcpy: H2D, Cuda unable to allocate memory. We're boned.");
    retptr[0] = (int64_t)rmem;

    fail = cudaMemcpy((void *)rmem, (void *)from, Nel * sizeof(double), cudaMemcpyHostToDevice);
    if(fail != cudaSuccess) mexErrMsgTxt("GPU_cudamemcpy: H2D, Copy to device failed.");

    // Given that we've succeeded, now tag the result with the array dimensions
    retptr[1] = mxGetNumberOfDimensions(prhs[0]); // r[1] = number of dims
    const mwSize *idims = mxGetDimensions(prhs[0]); // r[2:2+r[1]] = dimension size
    for(Nel = 0; Nel < retptr[1]; Nel++) retptr[2+Nel] = idims[Nel]; 
    for(; Nel < 3; Nel++) { retptr[2+Nel] = 1; } // or 1 if blank

    return;
  }

  if((nlhs == 1) & (nrhs == 2)) { 
    // GPU array -> double array or GPU array -> GPU array if numel(prhs[1]) == 1
    int ndims = mxGetNumberOfElements(prhs[1]);

    if(ndims > 1) {
      double *d = mxGetPr(prhs[1]);
      mwSize dims[3]; dims[0] = dims[1] = dims[2] = 1;
      int i; for(i = 0; i < ndims; i++) { dims[i] = (int)d[i]; }
    
      plhs[0] = mxCreateNumericArray(ndims, dims, mxDOUBLE_CLASS, mxREAL);

      int64_t *q = (int64_t *)mxGetData(prhs[0]);
      if(q[0] == 0) mexErrMsgTxt("GPU_cudamemcpy: D2H, wtf r u doin, gpu pointer is not initialized.");
      double *Dmem = (double *)q[0];
     
      cudaError_t fail = cudaMemcpy(mxGetPr(plhs[0]), Dmem, dims[0]*dims[1]*dims[2]*sizeof(double), cudaMemcpyDeviceToHost);
      if(fail != cudaSuccess) mexErrMsgTxt("GPU_cudamemcpy: D2H, Copy to host failed.");
    } else {  
      double *d = mxGetPr(prhs[1]);
      size_t numel = (size_t)d[0];

      mwSize dims[2]; dims[0] = 5; dims[1] = 1;
 
      plhs[0] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);
      int64_t *retptr = (int64_t *)mxGetData(plhs[0]);

      int64_t *from = (int64_t *)mxGetData(prhs[0]);
      if(from[0] == 0) mexErrMsgTxt("GPU_cudamemcpy: D2D, wtf r u doin, gpu pointer is not initialized");

      cudaError_t fail = cudaMalloc((void **)&retptr[0], numel * sizeof(double));
      if(fail == cudaErrorMemoryAllocation) mexErrMsgTxt("GPU_cudamemcpy: D2D, Cuda unable to allocate memory. We're boned.");

      fail = cudaMemcpy((void *)retptr[0], (void *)from[0], numel * sizeof(double), cudaMemcpyDeviceToDevice);
      if(fail != cudaSuccess) mexErrMsgTxt("GPU_cudamemcpy: D2D, Copy failed.");

      int i;
      for(i = 1; i < 5; i++) { retptr[i] = from[i]; } // copy metadata tags to destination array
    }

    return;
  }

  mexErrMsgTxt("syntax is one of:\n  GPU_MemPtr = GPU_cudamemcpy(double arr),\n  double = GPU_cudamemcpy(GPU_MemPtr, dims),\n  GPU_MemPtr = GPU_cudamemcpy(GPU_MemPtr src, numel in src)");

}
