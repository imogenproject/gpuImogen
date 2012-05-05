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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // At least 2 arguments expected
  // Input and result
  //if ((nlhs == 1) & (nrhs==1)) {
    // double -> GPU array
//    double *from = mxGetPr(prhs[0]);

//    mwSize dims[2]; dims[0] = 5; dims[1] = 1;
//    plhs[0] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);

}
