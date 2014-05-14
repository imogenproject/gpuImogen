#include "stdio.h"

#include "mpi.h"

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

bool amLocked = mexIsLocked(); /* Our check for if this has been called already */

if(nlhs == 1) {
  mwSize dims[2]; dims[0] = 1; dims[1] = 1;
  plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  }

int x, y;
x = MPI_Initialized(&y);

if(amLocked || y) {
  if(nlhs == 1)  *(mxGetPr(plhs[0])) = 1.0;
  return;
  }

mexLock();
MPI_Init(NULL, NULL);

if(nlhs == 1) *(mxGetPr(plhs[0])) = 0.0;
}
