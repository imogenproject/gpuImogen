#include "stdio.h"

#include "mpi.h"
#include "mex.h"
#include "matrix.h"

#include "parallel_halo_arrays.h"
#include "mpi_common.h"

pParallelTopology topoStructureToC(const mxArray *prhs); 

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
if((nrhs < 1) || (nrhs > 2) || (nlhs != 1)) mexErrMsgTxt("Call is TF = mpi_all(array)");

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
void *src; void *dst;
src = mxGetData(prhs[0]);
dst = mxGetData(plhs[0]);

/* Safe to do on the Matlab data */
if(mpidatatypeIsInteger(mtype)) {
    MPI_Allreduce(src, dst, numel, mtype, MPI_LOR, commune);
} else {
/* Whelp, this just got slower */
    uint8_t *logicSrc = malloc(numel*2);
    uint8_t *logicDst = logicSrc + numel;

    long k;
    if(arraytype == mxDOUBLE_CLASS) {
        double *d = (double *)src;
        for(k = 0; k < numel; k++)
            logicSrc[k] = (d[k] != 0);
        MPI_Allreduce(logicSrc, logicDst, numel, MPI_BYTE, MPI_LOR, commune);

        d = (double *)dst;
        for(k = 0; k < numel; k++)
            d[k] = (double)logicDst[k];

    }
    if(arraytype == mxSINGLE_CLASS) {
        float *d  = (float *)src;
        for(k = 0; k < numel; k++)
            logicSrc[k] = (d[k] != 0);
        MPI_Allreduce(logicSrc, logicDst, numel, MPI_BYTE, MPI_LOR, commune);

        d = (float *)dst;
        for(k = 0; k < numel; k++)
            d[k] = (float)logicDst[k];
    }

   free(logicSrc);  
}


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


