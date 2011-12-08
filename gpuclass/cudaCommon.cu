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


/* Given the RHS, an array to return array size, and the set of array indexes to take *s from */
double **getGPUSourcePointers(const mxArray *prhs[], ArrayMetadata *metaReturn, int fromarg, int toarg)
{

  double **gpuPointers = (double **)malloc((1+toarg-fromarg) * sizeof(double *));
  int iter;

  mxClassID dtype;

  dtype = mxGetClassID(prhs[fromarg]);
  if(dtype != mxINT64_CLASS) mexErrMsgTxt("cudaCommon: fatal, tried to get gpu src pointer from something not a gpu tag.");

  int64_t *dims = (int64_t *)mxGetData(prhs[fromarg]);
  for(iter = 0; iter < 3; iter++) { metaReturn->dim[iter] = (int)dims[2+iter]; } // copy metadata out of first gpu*
  metaReturn->numel = metaReturn->dim[0]*metaReturn->dim[1]*metaReturn->dim[2];
  metaReturn->ndims = dims[1];

  for(iter = fromarg; iter <= toarg; iter++) {
     dtype = mxGetClassID(prhs[iter]);
    if(dtype != mxINT64_CLASS) {
      printf("For argument %i\n",iter);
      mexErrMsgTxt("cudaCommon: fatal, tried to get gpu src pointer from something not a gpu tag.");
      }

    dims = (int64_t *)mxGetData(prhs[iter]);
    gpuPointers[iter-fromarg] = (double *)dims[0];
  }

return gpuPointers;
}

/* Creates destination array that the kernels write to; Returns the GPU memory pointer, and assigns the LHS it's passed */
double **makeGPUDestinationArrays(int64_t *reference, mxArray *retArray[], int howmany)
{

double **rvals = (double **)malloc(howmany*sizeof(double *));
int i;
mwSize dims[2]; dims[0] = 5; dims[1] = 1;

int64_t *rv; size_t numel;

numel = reference[2]*reference[3]*reference[4];

for(i = 0; i < howmany; i++) {
  retArray[i] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);
  rv = (int64_t *)mxGetData(retArray[i]);

  cudaError_t fail = cudaMalloc((void **)&rv[0], numel*sizeof(double));
  if(fail != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(fail));
    mexErrMsgTxt("makeGPUDestinationArrays: I haz an cudaMalloc fail. And a sad.");
    }

  int q; for(q = 1; q < 5; q++) rv[q] = reference[q];
  rvals[i] = (double *)rv[0];
  }

return rvals;

}

