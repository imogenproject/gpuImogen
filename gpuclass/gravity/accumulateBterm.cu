#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif

#include "mex.h"
#include "matrix.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas.h"

#include "cudaKernels.h"
#include "cudaCommon.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // At least 2 arguments expected
  // Input and result
  if (nrhs!=4)
     mexErrMsgTxt("Call form is accumulateBterm(source term, destination term, coefficient, preciditon term)");
  /* mex parameters are:
   0 Source term (that this applies the B operator to)
   1 Destination term (that this stores the result in)
   2 Precondition coefficient (one double)
   3 Precondition term (Accumulates successive scaled B operations
  */

  // Get GPU array pointers
  ArrayMetadata amd;

  double **srcdst = getGPUSourcePointers(prhs, &amd, 0, 1);
  double **accum  = getGPUSourcePointers(prhs, &amd, 2, 2);

  // Get some control variables sorted out
  int *dims    = amd.dim;

  dim3 gridsize;
  gridsize.x = dims[0]/EDGEDIM_BOP;
  gridsize.y = dims[1]/EDGEDIM_BOP;
  gridsize.z = 1;

  if(gridsize.x * EDGEDIM_BOP < dims[0]) gridsize.x++;
  if(gridsize.y * EDGEDIM_BOP < dims[1]) gridsize.y++;

  dim3 blocksize; blocksize.x = blocksize.y = EDGEDIM_BOP+2;
  blocksize.z = 1;

  int nx = dims[0];
  int ny = dims[1];
  int nz = dims[2];

  Laplacian_B_OperatorKernel<<<gridsize, blocksize>>>( srcdst[0], srcdst[1], *mxGetPr(prhs[2]), accum[0], nx, ny, nz, 4);

}
