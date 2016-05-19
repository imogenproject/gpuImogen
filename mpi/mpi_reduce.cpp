#include "stdio.h"

#include "mpi.h"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
if((nrhs != 1) || (nlhs != 1)) mexErrMsgTxt("Call is recvd = mpi_reduce(sent)");

int gsize; MPI_Comm_size(MPI_COMM_WORLD, &gsize);

double *sbuf = mxGetPr(prhs[0]);
int numel = mxGetNumberOfElements(prhs[0]);

int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

mwSize dims[2]; dims[0] = numel*gsize; dims[1] = 1;
plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);

double *recv = mxGetPr(plhs[0]);

MPI_Allgather( sbuf, numel, MPI_DOUBLE, recv, numel, MPI_DOUBLE, MPI_COMM_WORLD);

}
