#include "stdio.h"

#include "mpi.h"
#include "mex.h"

#include "mpi_constants.h"

/* Wrapper for MPI_Send:
 * [variable, status] = mpi_recv(type_integer, srcRank, tag, communicator)
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

int srcRank;
int tag;
MPI_Comm theCommunicator;

// Poll input arguments & fill in defaults if missing
if(nrhs < 2) {
	printf("mpi_recv requires at least 2 arguments:\n\tmpi_recv(type integer, source rank [, tag [, communicator]]);\n");
	return;
}

if(nrhs == 4) {
	theCommunicator = MPI_Comm_f2c((int)*mxGetPr(prhs[3]));
} else {
	theCommunicator = MPI_COMM_WORLD;
}

if(nrhs >= 3) {
	tag = (int)*mxGetPr(prhs[2]);
} else {
	tag = 0;
}

srcRank = (int)*mxGetPr(prhs[1]);

// Yucky workaround avoids a (send meta) - (recv meta) - (send data) - (recv data)
int thetype = *mxGetPr(prhs[0]);

// Learn length in advance
MPI_Status whatWeGot;
int status = MPI_Probe(srcRank, tag, theCommunicator, &whatWeGot);

mxClassID varType;
MPI_Datatype mdt;

switch(thetype) {
	case ML_MPIDOUBLE:
		mdt = MPI_DOUBLE;
		varType = mxDOUBLE_CLASS;
		break;
	case ML_MPISINGLE:
		mdt = MPI_FLOAT;
		varType = mxSINGLE_CLASS;
		break;
	case ML_MPICHAR:
		mdt = MPI_BYTE;
		varType = mxCHAR_CLASS;
		break;
	case ML_MPIUINT16:
		varType = mxUINT16_CLASS;
		mdt = MPI_SHORT;
		break;
	case ML_MPIINT16:
		varType = mxINT16_CLASS;
		mdt = MPI_SHORT;
		break;
	case ML_MPIUINT32:
		varType = mxUINT32_CLASS;
		mdt = MPI_INT;
	case ML_MPIINT32:
		varType = mxINT32_CLASS;
		mdt = MPI_INT;
		break;
	case ML_MPIUINT64:
		varType = mxUINT64_CLASS;
		mdt = MPI_LONG;
		break;
	case ML_MPIINT64:
		varType = mxINT64_CLASS;
		mdt = MPI_LONG;
		break;
}

int numel;

status = MPI_Get_count(&whatWeGot, mdt, &numel);


mwSize dims[2];
dims[0] = numel;
dims[1] = 1;

plhs[0] = mxCreateNumericArray(2, dims, varType, mxREAL);

void *bufptr = mxGetData(plhs[0]);

int result = MPI_Recv(bufptr, numel, mdt, srcRank, tag, theCommunicator, &whatWeGot);

if(nlhs > 0) {
	dims[0] = 1;
	plhs[1] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
	double *r = mxGetPr(plhs[1]);
	r[0] = result;
}

}
