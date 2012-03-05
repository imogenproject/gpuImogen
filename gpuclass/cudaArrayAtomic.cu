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

__global__ void cukern_ArraySetMin(double *array, double min,    int n);
__global__ void cukern_ArraySetMax(double *array, double max,    int n);
__global__ void cukern_ArrayFixNaN(double *array, double fixval, int n);

#define BLOCKDIM 256

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // At least 2 arguments expected
  // Input and result
  if (nrhs!=3)
     mexErrMsgTxt("Wrong number of arguments");

  // Get GPU array pointers
  double val       = *mxGetPr(prhs[1]);
  double operation = *mxGetPr(prhs[2]);

  ArrayMetadata amd;
  double **atomArray = getGPUSourcePointers(prhs, &amd, 0, 0);

  dim3 blocksize; blocksize.x = BLOCKDIM; blocksize.y = blocksize.z = 1;
  dim3 gridsize; gridsize.y = gridsize.z = 1;

  gridsize.x = amd.numel / BLOCKDIM;
  if(gridsize.x * BLOCKDIM < amd.numel) gridsize.x++;

  switch((int)operation) {
    case 1: cukern_ArraySetMin<<<gridsize, blocksize>>>(atomArray[0], val, amd.numel); break;
    case 2: cukern_ArraySetMax<<<gridsize, blocksize>>>(atomArray[0], val, amd.numel); break;
    case 3: cukern_ArrayFixNaN<<<gridsize, blocksize>>>(atomArray[0], val, amd.numel); break;
  }

cudaError_t epicFail = cudaGetLastError();
if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, (int)operation, "array min/max/nan sweeping");


}

__global__ void cukern_ArraySetMin(double *array, double min, int n)
{
int x = threadIdx.x + blockDim.x * blockIdx.x;
if(x >= n) return;

if(array[x] < min) array[x] = min;
}

__global__ void cukern_ArraySetMax(double *array, double max, int n)
{
int x = threadIdx.x + blockDim.x * blockIdx.x;
if(x >= n) return;

if(array[x] > max) array[x] = max;
}

__global__ void cukern_ArrayFixNaN(double *array, double fixval, int n)
{
int x = threadIdx.x + blockDim.x * blockIdx.x;
if(x >= n) return;

if (isnan( array[x] )) { array[x] = fixval; }

}

