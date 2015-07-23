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

/* THIS FUNCTION
The cudaArrayAtomic function is meant to perform operations that operate elementwise
on single arrays. The only such functions yet encountered are in "control" functions where
we require that either density be kept to a minimum value, or that NaNs be replaced by 0s.
*/

__global__ void cukern_ArraySetMin(double *array, double min,    int n);
__global__ void cukern_ArraySetMax(double *array, double max,    int n);
__global__ void cukern_ArrayFixNaN(double *array, double fixval, int n);

#define BLOCKDIM 256

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // At least 2 arguments expected
  // Input and result
  if (nrhs!=3)
     mexErrMsgTxt("Wrong number of arguments. Expected form: cudaArrayAtomic(gputag, value, [1: set min, 2: set max, 3: NaN->value])");

  // Get GPU array pointers
  double val       = *mxGetPr(prhs[1]);
  int operation = (int)*mxGetPr(prhs[2]);

  MGArray phi;
  int worked = MGA_accessMatlabArrays(prhs, 0, 0, &phi);
  int j; int sub[6];

  CHECK_CUDA_ERROR("Entering cudaArrayAtomic");

  for(j = 0; j < phi.nGPUs; j++) {
    cudaSetDevice(phi.deviceID[j]);
    CHECK_CUDA_ERROR("Setting device.");

    switch(operation) {
      case 1: cukern_ArraySetMin<<<32, BLOCKDIM>>>(phi.devicePtr[j], val, phi.partNumel[j]); break;
      case 2: cukern_ArraySetMax<<<32, BLOCKDIM>>>(phi.devicePtr[j], val, phi.partNumel[j]); break;
      case 3: cukern_ArrayFixNaN<<<32, BLOCKDIM>>>(phi.devicePtr[j], val, phi.partNumel[j]); break;
    }
   CHECK_CUDA_LAUNCH_ERROR(256, 32, &phi, operation, "array min/max/nan sweeping");

  }

}

__global__ void cukern_ArraySetMin(double *array, double min, int n)
{
int x = threadIdx.x + blockDim.x * blockIdx.x;
int dx = blockDim.x * gridDim.x;

while(x < n) {
    if(array[x] < min) array[x] = min;
    x += dx;
    }

}

__global__ void cukern_ArraySetMax(double *array, double max, int n)
{
int x = threadIdx.x + blockDim.x * blockIdx.x;
int dx = blockDim.x * gridDim.x;

while(x < n) {
    if(array[x] > max) array[x] = max;
    x += dx;
    }

}

__global__ void cukern_ArrayFixNaN(double *array, double fixval, int n)
{
int x = threadIdx.x + blockDim.x * blockIdx.x;
int dx = blockDim.x * gridDim.x;

while(x < n) {
    if( isnan(array[x])) array[x] = fixval;
    x += dx;
    }

}

