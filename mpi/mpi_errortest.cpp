#include "stdio.h"

#include "mpi.h"
#include "mex.h"

#include "mpi_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
if((nrhs != 1) || (nlhs != 0)) {
mexErrMsgTxt("Call is  mpi_errortest(value); if any rank has value != 0, causes error\n");
}

MPI_Comm commune = MPI_COMM_WORLD;

double d = *mxGetPr(prhs[0]);

int fail;
int result;

fail = (d == 0) ? 0 : 1;

if(fail) {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	printf("Rank %i has an error: All ranks will now throw a mex error.\n", rank);
	}

MPI_Allreduce(&fail, &result, 1, MPI_INT, MPI_MAX, commune);

if(result != 0) {
	mexErrMsgTxt("At least one rank passed an error. All ranks throwing interpreter error.");
}

}

