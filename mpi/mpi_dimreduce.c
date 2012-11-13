#include "stdio.h"

#include "mpi.h"
#include "mex.h"

#include "parallel_halo_arrays.h"

pParallelTopology topoStructureToC(const mxArray *prhs);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
if((nrhs != 3) || (nlhs != 0)) mexErrMsgTxt("Call is max = mpi_dimreduce(array, dimension, topology)");

pParallelTopology topology = topoStructureToC(prhs[2]);

/* Reversed for silly Fortran memory ordering */
int d = (int)*mxGetPr(prhs[1]); /* dimension we're calculating in*/
int delta0; /* to one proc over in my line of processors */

switch(d) {
  case 2: delta0 = 1; break;
  case 1: delta0 = topology->nproc[2]; break;
  case 0: delta0 = topology->nproc[2]*topology->nproc[1]; break;
/*  case 0: delta0 = 1; break;
  case 1: delta0 = topology->nproc[0]; break;
  case 2: delta0 = topology->nproc[0]*topology->nproc[1]; break; */
  }

int delta = delta0;
int n = topology->coord[d]; /*n = my topology.coord[d] in this dimension */
int dmax = topology->nproc[d];
MPI_Comm commune = MPI_Comm_f2c(topology->comm);
int r0; MPI_Comm_rank(commune, &r0);

int j = 2; /* the distance we're sending the next fold-in */

long numel = mxGetNumberOfElements(prhs[0]);

double *tmpstore = (double *)malloc(sizeof(double)*numel);
double *sendbuf  = mxGetPr(prhs[0]);

MPI_Status amigoingtodie;

while((j/2) < dmax) { /* as long as we aren't trying to send it past the end of the line.
                                                                   send it past the end of the line.
                                                                                the end of the line.
                                                                                    END   OF   LINE */

  /* Send iff we are in the middle of the foldin */
  if((n % j) == (j/2))
    {
/*   printf("fanin: %i here with n=%i at j=%i: would tx to %i\n", r0, n, j, r0-delta); */
    MPI_Send((void *)sendbuf, numel, MPI_DOUBLE, r0 - delta, 12345, commune); 
/*  send my array back by (j >> 1) */
    }
  if(((n % j) == 0) && (n + (j/2) < dmax ) ) {
/*    printf("fanin: %i here with n=%i at j=%i: would rx from %i\n", r0, n, j, r0+delta); */
    MPI_Recv((void *)tmpstore, numel, MPI_DOUBLE, r0 + delta, 12345, commune, &amigoingtodie); 
    /* store in my array the max of (mine, received) */
    long q; for(q = 0; q < numel; q++) { if(tmpstore[q] > sendbuf[q]) sendbuf[q] = tmpstore[q]; } 
    }
  j     *= 2;
  delta *= 2;


  }

MPI_Barrier(commune);

/* Sample sequence:
      n = 0  1  2  3  4  5  6  7  8  9
  
  m,i=0   0  1  0  1  0  1  0  1  0  1  
  i=0     R  T  R  T  R  T  R  T  R  T  j=2, delta = 1
  
  m,i=1   0  1  2  3  0  1  2  3  0  1
  i=1     R     T     R     T     x     j=4, delta = 2
  
  m,i=2   0  1  2  3  4  5  6  7  0  1  
  i=2     R           T           x     j=8, delta = 4
  
  m,i=3   0  1  2  3  4  5  6  7  8  9
  i=3     R                       T     j=16,delta = 8: end

   n=0 now posesses the global maximum. Now log fold-out instead. */

while(j >= 2) {
  if((n % j) == (j /2))
    {
/*    printf("fanout: %i here with n=%i at j=%i: would rx from %i\n", r0, n, j, r0-delta); */
    MPI_Recv((void *)sendbuf, numel, MPI_DOUBLE, r0 - delta, 12345, commune, &amigoingtodie); 
    }
  if(((n % j) == 0) && (n + (j/2) < dmax ) ) {
/*    printf("fanout: %i here with n=%i at j=%i: would tx to %i\n", r0, n, j, r0+delta);*/
    MPI_Send((void *)sendbuf, numel, MPI_DOUBLE, r0 + delta, 12345, commune); 
    }

  j = j/2;
  delta = delta / 2;
  }

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

