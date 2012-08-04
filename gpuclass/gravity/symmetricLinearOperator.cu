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
  if ((nrhs!=3) && ((nlhs != 1) && (nrhs != 2)))
     mexErrMsgTxt("Wrong number of arguments: 1lhs + 2rhs or 3rhs acceptable");
  /* mex parameters are:
   0 Source data
   1 Destination data
   2 Coefficients [4]
  */

  ArrayMetadata amd;

  // Sort out source/dest stuff
  double **srcArray   = getGPUSourcePointers(prhs, &amd, 0,0);
  double **dstArray;
  double *opCoeffs;
  if(nrhs == 3) {
	dstArray    = getGPUSourcePointers(prhs, &amd, 1, 1);
	opCoeffs    = mxGetPr(prhs[2]);
  } else {
	//dstArray = create LHS
	// U = teh fu><X0r3dz, we no support diz
	opCoeffs    = mxGetPr(prhs[1]);
  }

  // Get some control variables sorted out
  int *dims  = amd.dim;

  dim3 gridsize;
  gridsize.x = dims[0]*dims[2]/64;
  gridsize.y = dims[1]/8;
  gridsize.z = 1;

  dim3 blocksize; blocksize.x = blocksize.y = 10; blocksize.z = 1;

  int nx = dims[0];
  int ny = dims[1];
  int nz = dims[2]/8 - 1;

  SymmetricOperatorKernel<<<gridsize, blocksize>>>(srcArray[0], dstArray[0], nx, ny, nz, opCoeffs[0], opCoeffs[1], opCoeffs[2], opCoeffs[3]);

}

