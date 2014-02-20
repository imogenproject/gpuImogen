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
  // At least 2 arguments expected
  // Input and result
  if(nlhs != 1) { mexErrMsgTxt("GPU_cudamemcpy: Must have 1 return argument for copied array."); }

  cudaCheckError("entering GPU_cudamemcpy");

  int arg1isml = mxIsDouble(prhs[0]);

  if(nrhs == 1) {
    // Copy an input Matlab array to GPU, and return a gputag.
    if(arg1isml) {
      double *from = mxGetPr(prhs[0]);

      mwSize dims[2]; dims[0] = 5; dims[1] = 1;
      plhs[0] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);

      int64_t *retptr = (int64_t *)mxGetData(plhs[0]);
      size_t Nel = mxGetNumberOfElements(prhs[0]);
      double *rmem;

      cudaError_t fail = cudaMalloc((void **)&rmem, Nel * sizeof(double));
      if(fail == cudaErrorMemoryAllocation) mexErrMsgTxt("GPU_cudamemcpy: H2D, error attempting to allocate memory (cudaErrorMemoryAllocation). We've been had.");
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

    // Copy an input gpu array back to cpu if D2D is not specified by a second argument.
    if(arg1isml == 0) {
      ArrayMetadata amd;
      double **gpuarray = getGPUSourcePointers(prhs, &amd, 0, 0); 
      mwSize odims[3];
      int j;
      for(j = 0; j < amd.ndims; j++) { odims[j] = amd.dim[j]; }
      plhs[0] = mxCreateNumericArray(amd.ndims, odims, mxDOUBLE_CLASS, mxREAL);
      
      double *src = gpuarray[0];
      double *dst = mxGetPr(plhs[0]);

      cudaError_t fail = cudaMemcpy(dst, src, amd.numel*sizeof(double), cudaMemcpyDeviceToHost);
      if(fail != cudaSuccess) mexErrMsgTxt("GPU_cudamemcpy: D2H, Copy to host failed.");
  
      return;
      }
  } else if(nrhs == 2) {
    if(arg1isml == 0) {
      mwSize dout[2]; dout[0] = 5; dout[1] = 1;
      plhs[0] = mxCreateNumericArray(2, dout, mxINT64_CLASS, mxREAL);
      int64_t srctag[5]; getTagFromGPUType(prhs[0], &srctag[0]);
      int64_t dsttag[5]; getTagFromGPUType(plhs[0], &srctag[0]);

      int j;
      for(j = 1; j < 5; j++) { dsttag[j] = srctag[j]; }

      cudaError_t fail = cudaMalloc((void **)&dsttag[0], srctag[2]*srctag[3]*srctag[4]*sizeof(double));
      if(fail != cudaSuccess) mexErrMsgTxt("GPU_cudamemcpy: D2D, error attempting to allocate memory (cudaErrorMemoryAllocation). We've been had.");
  
      fail = cudaMemcpy((void *)dsttag[0], (void *)srctag[0], srctag[2]*srctag[3]*srctag[4]* sizeof(double), cudaMemcpyDeviceToDevice);
      if(fail != cudaSuccess) mexErrMsgTxt("GPU_cudamemcpy: D2D, Copy failed.");

      return;
      }
  }

  mexErrMsgTxt("syntax is one of:\n  GPU_MemPtr = GPU_cudamemcpy(double arr),\n  double = GPU_cudamemcpy(GPU_MemPtr, dims),\n  GPU_MemPtr = GPU_cudamemcpy(GPU_MemPtr src, numel in src)");

}
