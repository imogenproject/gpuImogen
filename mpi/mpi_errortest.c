#include "stdio.h"

#include "mpi.h"
#include "mex.h"
#include "matrix.h"

#include "parallel_halo_arrays.h"
#include "mpi_common.h"

pParallelTopology topoStructureToC(const mxArray *prhs); 

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
if((nrhs != 1) || (nlhs != 0)) mexErrMsgTxt("Call is  mpi_errortest(value); if any rank has value != 0, causes error\n");

MPI_Comm commune = MPI_COMM_WORLD;

double d = *mxGetPr(prhs[0]);

int fail;
int result;

fail = (d == 0) ? 0 : 1;

MPI_Allreduce(&fail, &result, 1, MPI_INT, MPI_MAX, commune);

if(result != 0) {
	mexErrMsgTxt("At least one rank passed an error. Causing interpreter error.");	
}

}

