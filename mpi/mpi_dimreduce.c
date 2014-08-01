#include "stdio.h"

#include "mpi.h"
#include "mex.h"

#include "parallel_halo_arrays.h"
#include "mpi_common.h"

pParallelTopology topoStructureToC(const mxArray *prhs); 

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
if((nrhs != 3) || (nlhs != 0)) mexErrMsgTxt("Call is max = mpi_dimreduce(array, dimension, topology)");

pParallelTopology topology = topoStructureToC(prhs[2]);

/* Reversed for silly Fortran memory ordering */
int d = (int)*mxGetPr(prhs[1]); /* dimension we're calculating in*/
int dmax = topology->nproc[d];

MPI_Comm commune = MPI_Comm_f2c(topology->comm);
int r0; MPI_Comm_rank(commune, &r0);


long arrayNumel = mxGetNumberOfElements(prhs[0]);

double *tmpstore = (double *)malloc(sizeof(double)*arrayNumel);
double *sendbuf  = mxGetPr(prhs[0]);

MPI_Status amigoingtodie;

/* FIXME: This is a temporary hack
   FIXME: The creation of these communicators should be done once,
   FIXME: by PGW, at start time. */
int dimprocs[dmax];
int proc0, procstep;
switch(d) { /* everything here is Wrong because fortran is Wrong */
  case 0: /* i0 = nx Y + nx ny Z, step = 1 -> nx ny */
	  /* x dimension: step = ny nz, i0 = z + nz y */
    proc0 = topology->coord[2] + topology->nproc[2]*topology->coord[1];
    procstep = topology->nproc[2]*topology->nproc[1];
    break;
  case 1: /* i0 = x + nx ny Z, step = nx */
	  /* y dimension: step = nz, i0 = z + nx ny x */
    proc0 = topology->coord[2] + topology->nproc[2]*topology->nproc[1]*topology->coord[0];
    procstep = topology->nproc[2];
    break;
  case 2: /* i0 = x + nx Y, step = nx ny */
	  /* z dimension: i0 = nz y + nz ny x, step = 1 */
    proc0 = topology->nproc[2]*(topology->coord[1] + topology->nproc[1]*topology->coord[0]);
    procstep = 1;
    break;
  }

int j;
for(j = 0; j < dmax; j++) {
  dimprocs[j] = proc0 + j*procstep;
  }

MPI_Group worldgroup, dimgroup;
MPI_Comm dimcomm;
/* r0 has our rank in the world group */
MPI_Comm_group(commune, &worldgroup);
MPI_Group_incl(worldgroup, dmax, dimprocs, &dimgroup);
/* Create communicator for this dimension */
MPI_Comm_create(commune, dimgroup, &dimcomm);

/* Perform the reduce */
MPI_Allreduce(sendbuf, tmpstore, arrayNumel, MPI_DOUBLE, MPI_MAX, dimcomm);

MPI_Barrier(dimcomm);
/* Clean up */
MPI_Group_free(&dimgroup);
MPI_Comm_free(&dimcomm);

memcpy(sendbuf, tmpstore, arrayNumel*sizeof(double));

return;

free(tmpstore);

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


