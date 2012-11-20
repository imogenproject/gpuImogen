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

__global__ void cukern_applySpecial_fade(double *arr, double *linAddrs, double *consts, double *fadeCoeff, int nSpecials);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if( (nlhs != 0) || (nrhs != 7)) { mexErrMsgTxt("cudaStatics operator is cudaStatics(array, linearIndices, constants, fadeCoeffs, blockdim, permutation, offsetindex)"); }

  cudaCheckError("entering cudaSatics");

  ArrayMetadata ama, amf;

  int blockdim = (int)*mxGetPr(prhs[4]);
  double *perm = mxGetPr(prhs[5]);
  double *offsetcount = mxGetPr(prhs[6]);

  double **array = getGPUSourcePointers(prhs, &ama, 0, 0);
  double **fixies= getGPUSourcePointers(prhs, &amf, 1, 3);
 
  int offsetidx = 2*(perm[0]-1) + 1*(perm[1] > perm[2]);

  long int staticsOffset = (long int)offsetcount[2*offsetidx];
  int staticsNumel  = (int)offsetcount[2*offsetidx+1];

  dim3 griddim; griddim.x = staticsNumel / blockdim + 1;
  if(griddim.x > 32768) {
    griddim.x = 32768;
    griddim.y = staticsNumel/(blockdim*griddim.x) + 1;
    }

  cukern_applySpecial_fade<<<griddim, blockdim>>>(array[0], \
                                                 fixies[0] + staticsOffset, \
                                                 fixies[1] + staticsOffset, \
                                                 fixies[2] + staticsOffset, staticsNumel);

cudaError_t epicFail = cudaGetLastError();
if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blockdim, griddim, &ama, 0, "cuda statics application");

}

__global__ void cukern_applySpecial_fade(double *arr, double *linAddrs, double *consts, double *fadeCoeff, int nSpecials)
{
int myAddr = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x*blockIdx.y);
if(myAddr >= nSpecials) return;

double f = fadeCoeff[myAddr];
int xaddr = (int)linAddrs[myAddr];

arr[xaddr] = f*consts[myAddr] + (1.0-f)*arr[xaddr];

}


