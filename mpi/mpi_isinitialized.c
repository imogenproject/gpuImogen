#include "stdio.h"

#include "mpi.h"

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

if((nlhs != 1) || (nrhs > 0)) {
  mexErrMsgTxt("Must call in the form TF = mpi_isinitialized()\n");
  return;
  }

mwSize dims[2]; dims[0] = 1; dims[1] = 1;
plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);

int x, y;
x = MPI_Initialized(&y);

double *a = mxGetPr(plhs[0]);

*a = y ? 1.0 : 0.0;
return;

}
