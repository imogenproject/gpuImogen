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
  if( (nlhs != 0) || (nrhs != 6)) { mexErrMsgTxt("compileStatics operator is compiled = cudaStatics(linearIndices, constants, fadeCoeffs)"); }

  ArrayMetadata amd;

  double *lind = mxGetPr(prhs[0]);
  double *lcon = mxGetPr(prhs[1]);
  double *lcoe = mxGetPr(prhs[2]);

  size_t numel = mxgetNumberOfelements(prhs[0]);

  int ndims = 2;
  mwSize dims[2]; dims[0] = numel; dims[1] = 1;

  plhs[0] = mxCreateNumericArray(2, );

  double **array = getGPUSourcePointers(prhs, &amd, 0, 2);

  int offsetidx = 2*(perm[0]-1) + 1*(perm[1] > perm[2]);
printf("offset idx: %i\n", offsetidx); fflush(stdout);

  dim3 griddim; griddim.x = amf.numel / blockdim + 1;
  if(griddim.x > 32768) {
    griddim.x = 32768;
    griddim.y = amf.numel/(blockdim*griddim.x) + 1;
    }

  cukern_applySpecial_fade<<<griddim, blockdim>>>(array[0], \
                                                 fixies[0] + offsetidx*amf.numel, \
                                                 fixies[1] + offsetidx*amf.numel, \
                                                 fixies[2] + offsetidx*amf.numel, amf.numel);
}

__global__ void cukern_applySpecial_fade(double *arr, double *linAddrs, double *consts, double *fadeCoeff, int nSpecials)
{
int myAddr = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x*blockIdx.y);
if(myAddr >= nSpecials) return;

double f = fadeCoeff[myAddr];
int xaddr = (int)linAddrs[myAddr];

arr[xaddr] = f*consts[myAddr] + (1.0-f)*arr[xaddr];

}


