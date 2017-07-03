#include "stdio.h"
#include "stdint.h"

#include "mpi.h"
#include "mex.h"

#include "mpi_common.h"

#ifndef MPIOPERATION
  #error "mpi_allreduce.cpp must be compiled with -DMPIOPERATION=x where x is an MPI_Op"
#endif


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
if((nrhs < 1) || (nrhs > 2) || (nlhs != 1)) mexErrMsgTxt("Call is min = mpi_min(array) or min = mpi_min(array, communicator)");

MPI_Comm commune = MPI_COMM_WORLD;
/*if(nrhs == 2) commune = (MPI_Comm)*mxGetPr(prhs[1]);*/

/* Grab some basic meta info about the input arrays */
/* God help you if they aren't all the same size. */
// FIXME: Accept a 'paranoid' flag to check if they are & avoid crashing the MPI runtime?

long numel = mxGetNumberOfElements(prhs[0]);
mwSize ndims = mxGetNumberOfDimensions(prhs[0]);
const mwSize *arrDims = mxGetDimensions(prhs[0]);
mxClassID arraytype = mxGetClassID(prhs[0]);

/* Create our output array containing the reduced values, on every host */
plhs[0] = mxCreateNumericArray(ndims, arrDims, arraytype, mxREAL);

MPI_Datatype mtype = typeid_ml2mpi(arraytype);

void *src = mxGetData(prhs[0]);
void *dst = mxGetData(plhs[0]);

if(mtype == MPI_BYTE) { // reduce hates mpi_byte for me...
	int *srcI = (int *)malloc(2*numel*sizeof(int));
	int *dstI = srcI + numel;

	int j;
	for(j = 0; j < numel; j++) {
		srcI[j] = (int)(( (uint8_t *)src)[j]);
	}

	MPI_Allreduce((void *)srcI, (void *)dstI, numel, MPI_INT, MPIOPERATION, commune);

	for(j = 0; j < numel; j++) {
		(( (uint8_t *)dst)[j]) = (uint8_t)dstI[j];
	}
	free(srcI);
} else {
	MPI_Allreduce(src, dst, numel, mtype, MPIOPERATION, commune);
}



}

