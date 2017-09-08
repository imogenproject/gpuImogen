#include "stdio.h"

#include "mpi.h"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if((nlhs != 1 ) || (nrhs != 0)) { mexErrMsgTxt("int32 N = mpi_myrank();"); }

   int bee;
   MPI_Comm_rank(MPI_COMM_WORLD, &bee);

   mwSize dims[2]; dims[0] = 1; dims[1] = 1;

   plhs[0] = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);

   int *d = (int *)mxGetData(plhs[0]);

   d[0] = bee;
}
