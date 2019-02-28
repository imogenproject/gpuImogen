#include "stdio.h"

#include "mpi.h"
#include "mex.h"

/* Wrapper for MPI_Send:
 * status = mpi_send(variable, destRank, tag, communicator)
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

int destRank;
int tag;
MPI_Comm theCommunicator;

// Poll input args & fillin defaults if possible
switch(nrhs) {
	case 4: break; // okay we're good
	case 3: theCommunicator = MPI_COMM_WORLD; printf("using default comm\n"); break;
	case 2: tag = 0; break;
	default:
		printf("Incorrect function call. Use:\n\tmpi_send(variable, destRank [,tag [, communicator] ] );\n");
		return; // shit!
		break;
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

destRank = (int)*mxGetPr(prhs[1]);

int numel = (int)mxGetNumberOfElements(prhs[0]);
mxClassID varType = mxGetClassID(prhs[0]);

void *bufptr;
MPI_Datatype mdt;

switch(varType) {
	case mxDOUBLE_CLASS:
		bufptr = (void *)mxGetPr(prhs[0]);
		mdt = MPI_DOUBLE;
		break;
	case mxSINGLE_CLASS:
		bufptr = (void *)mxGetData(prhs[0]);
		mdt = MPI_FLOAT;
		break;
	case mxCHAR_CLASS:
		bufptr = (void *)mxGetData(prhs[0]);
		mdt = MPI_BYTE;
		break;
	case mxINT16_CLASS:
	case mxUINT16_CLASS:
		bufptr = (void *)mxGetData(prhs[0]);
		mdt = MPI_SHORT;
		break;
	case mxINT32_CLASS:
	case mxUINT32_CLASS:
		bufptr = (void *)mxGetData(prhs[0]);
		mdt = MPI_INT;
		break;
	case mxINT64_CLASS:
	case mxUINT64_CLASS:
		bufptr = (void *)mxGetData(prhs[0]);
		mdt = MPI_LONG;
		break;
}


int result = MPI_Send(bufptr, numel, mdt, destRank, tag, MPI_COMM_WORLD);

if(nlhs > 0) {
	// return result in matlab format
	mwSize dims[2];
	dims[0] = dims[1] = 1;
	mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
	double *r = mxGetPr(plhs[0]);
	r[0] = result;
}

}
