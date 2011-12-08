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
  if( (nlhs != 0) || (nrhs != 6)) { mexErrMsgTxt("cudaStatics operator is cudaStatics(array, linearIndices, constants, fadeCoeffs, blockdim, permutation)"); }

  ArrayMetadata ama, amf;

  double *perm = mxGetPr(prhs[5]);

  double **array = getGPUSourcePointers(prhs, &ama, 0, 0);
  double **fixies= getGPUSourcePointers(prhs, &amf, 1, 3);
 
  int blockdim = (int)*mxGetPr(prhs[4]);

  int offsetidx = 2*(perm[0]-1) + 1*(perm[1] > perm[2]);

  dim3 griddim; griddim.x = amf.dim[0] / blockdim + 1;
  if(griddim.x > 32768) {
    griddim.x = 32768;
    griddim.y = amf.dim[0]/(blockdim*griddim.x) + 1;
    }

  cukern_applySpecial_fade<<<griddim, blockdim>>>(array[0], \
                                                 fixies[0] + offsetidx*amf.dim[0], \
                                                 fixies[1] + offsetidx*amf.dim[0], \
                                                 fixies[2] + offsetidx*amf.dim[0], amf.dim[0]);
}

__global__ void cukern_applySpecial_fade(double *arr, double *linAddrs, double *consts, double *fadeCoeff, int nSpecials)
{
int myAddr = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x*blockIdx.y);
if(myAddr >= nSpecials) return;

double f = fadeCoeff[myAddr];
int xaddr = (int)linAddrs[myAddr];

arr[xaddr] = f*consts[myAddr] + (1.0-f)*arr[xaddr];

}


