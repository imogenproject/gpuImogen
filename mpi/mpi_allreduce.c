#include "stdio.h"

#include "mpi.h"
#include "mex.h"
#include "matrix.h"

#include "parallel_halo_arrays.h"
#include "mpi_common.h"


#ifndef MPIOPERATION
  #error "mpi_allreduce.c must be compiled with -DMPIOPERATION=x where x is an MPI_Op"
#endif

pParallelTopology topoStructureToC(const mxArray *prhs); 

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
if((nrhs < 1) || (nrhs > 2) || (nlhs != 1)) mexErrMsgTxt("Call is min = mpi_min(array) or min = mpi_min(array, communicator)");

MPI_Comm commune = MPI_COMM_WORLD;
/*if(nrhs == 2) commune = (MPI_Comm)*mxGetPr(prhs[1]);*/

/* Grab some basic meta info about the input arrays */
/* God help you if they aren't all the same size. */
long numel = mxGetNumberOfElements(prhs[0]);
mwSize ndims = mxGetNumberOfDimensions(prhs[0]);
const mwSize *arrDims = mxGetDimensions(prhs[0]);
mxClassID arraytype = mxGetClassID(prhs[0]);

/* Create our output array containing the reduced values, on every host */
plhs[0] = mxCreateNumericArray(ndims, arrDims, arraytype, mxREAL);

MPI_Datatype mtype = typeid_ml2mpi(arraytype);

void *src = mxGetData(prhs[0]);
void *dst = mxGetData(plhs[0]);

MPI_Allreduce(src, dst, numel, mtype, MPIOPERATION, commune);

}

pParallelTopology topoStructureToC(const mxArray *prhs)
{
mxArray *a;

pParallelTopology pt = (pParallelTopology)malloc(sizeof(ParallelTopology));

a = mxGetFieldByNumber(prhs,0,0);
pt->ndim = (int)*mxGetPr(a);
a = mxGetFieldByNumber(prhs,0,1);
pt->comm = (int)*mxGetPr(a);

int *val;
int i;

val = (int *)mxGetData(mxGetFieldByNumber(prhs,0,2));
for(i = 0; i < pt->ndim; i++) pt->coord[i] = val[i];

val = (int *)mxGetData(mxGetFieldByNumber(prhs,0,3));
for(i = 0; i < pt->ndim; i++) pt->neighbor_left[i] = val[i];

val = (int *)mxGetData(mxGetFieldByNumber(prhs,0,4));
for(i = 0; i < pt->ndim; i++) pt->neighbor_right[i] = val[i];

val = (int *)mxGetData(mxGetFieldByNumber(prhs,0,5));
for(i = 0; i < pt->ndim; i++) pt->nproc[i] = val[i];

for(i = pt->ndim; i < 4; i++) {
  pt->coord[i] = 0;
  pt->nproc[i] = 1;
  }

return pt;

}


