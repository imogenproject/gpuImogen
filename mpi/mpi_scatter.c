#include "stdio.h"

#include "mpi.h"

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
if((nrhs != 2) || (nlhs != 1)) mexErrMsgTxt("Call is recvd = mpi_scatter(data, sending rank)");

int size; MPI_Comm_size(MPI_COMM_WORLD, &size);
int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int root = (int)*mxGetPr(prhs[1]);

int arrayinfo[6];
int j;

if(rank == root) {
  mwSize nd = mxGetNumberOfDimensions(prhs[0]);
  const mwSize *dims = mxGetDimensions(prhs[0]);
  arrayinfo[0] = nd; /* send number of dimensions */
  arrayinfo[1] = (int)mxGetClassID(prhs[0]); /* send  */
  for(j = 0; j < nd; j++) { arrayinfo[j+2] = (int)dims[j]; }
  }
/* Tell the world what dimensions we have
   I MUST TELL ZEE WORLD! */
MPI_Bcast(arrayinfo, 6, MPI_INTEGER, root, MPI_COMM_WORLD);

double *iobuf;

/* Create output array for everyone else, dupe array for root */
mwSize d[4]; 
for(j = 0; j < 4; j++) { d[j] = arrayinfo[j+2]; }
if(rank != root) {
  plhs[0] = mxCreateNumericArray((mwSize)arrayinfo[0], d, (mxClassID)arrayinfo[1], mxREAL);
  } else {
  plhs[0] = mxDuplicateArray(prhs[0]);
  }

/* Set tx/rx buffer for the broadcast */
if(rank == root) { iobuf = mxGetPr(prhs[0]); } else { iobuf = mxGetPr(plhs[0]); }

/* calc numel */
int numel = 1; for(j = 0; j < arrayinfo[0]; j++) { numel *= arrayinfo[j+2]; }

/* call the exchange of the data */
switch(mxGetClassID(plhs[0])) {
  case mxCHAR_CLASS:   MPI_Bcast(iobuf, numel, MPI_CHAR, root, MPI_COMM_WORLD); break;
  case mxSINGLE_CLASS: MPI_Bcast(iobuf, numel, MPI_FLOAT, root, MPI_COMM_WORLD); break;
  case mxDOUBLE_CLASS: MPI_Bcast(iobuf, numel, MPI_DOUBLE, root, MPI_COMM_WORLD); break;
  case mxINT32_CLASS:  MPI_Bcast(iobuf, numel, MPI_INT, root, MPI_COMM_WORLD); break;
  case mxUINT32_CLASS: MPI_Bcast(iobuf, numel, MPI_INT, root, MPI_COMM_WORLD); break;
  }

}
