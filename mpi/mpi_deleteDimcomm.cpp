#include "stdio.h"

#include "mpi.h"
#include "mex.h"

#include "mpi_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	if(nrhs != 1) {
		mexErrMsgTxt("Require topo =mpi_deleteDimcomm(topo)");
	}

	ParallelTopology topo;
	topoStructureToC(prhs[0], &topo);
	topoDestroyDimensionalCommunicators(&topo);
	if(nlhs == 1) topoCToStructure(&topo, plhs);

	return; 
}
