#include "stdio.h"
#include "string.h"

#include "mpi.h"
#include "mex.h"

#include "mpi_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if((nrhs != 3) || (nlhs != 0)) mexErrMsgTxt("Call is max = mpi_dimreduce(array, dimension, topology)");

	ParallelTopology top;
	ParallelTopology *topology = &top;
	topoStructureToC(prhs[2], topology);

	int d = (int)*mxGetPr(prhs[1]); /* dimension we're calculating in*/

	long arrayNumel = mxGetNumberOfElements(prhs[0]);

	double *tmpstore = (double *)malloc(sizeof(double)*arrayNumel);
	double *sendbuf  = mxGetPr(prhs[0]);

	MPI_Status amigoingtodie;

	MPI_Comm dc = MPI_Comm_f2c(topology->dimcomm[d-1]);

	/* Perform the reduce */
	MPI_Allreduce(sendbuf, tmpstore, arrayNumel, MPI_DOUBLE, MPI_MAX, dc);

	MPI_Barrier(dc);

	memcpy(sendbuf, tmpstore, arrayNumel*sizeof(double));
	free(tmpstore);

	return;


}



