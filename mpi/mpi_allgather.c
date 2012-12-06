#include "stdio.h"

#include "mpi.h"
#include "mex.h"
#include "matrix.h"

#include "mpi_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
if((nrhs != 1) || (nlhs != 1)) mexErrMsgTxt("Call is recvd = mpi_allgather(sent)");

int gsize; MPI_Comm_size(MPI_COMM_WORLD, &gsize);

mxClassID id = mxGetClassID(prhs[0]);

MPI_Datatype dt = typeid_ml2mpi(id);

void *sbuf = mxGetData(prhs[0]);
int numel = mxGetNumberOfElements(prhs[0]);

int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

mwSize dims[2]; dims[0] = numel*gsize; dims[1] = 1;
plhs[0] = mxCreateNumericArray(2, dims, id, mxREAL);

void *recv = mxGetData(plhs[0]);

MPI_Allgather( sbuf, numel, dt, recv, numel, dt, MPI_COMM_WORLD); 


}
