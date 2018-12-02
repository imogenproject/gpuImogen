#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif
#include "mex.h"

// CUDA
// #include "cuda.h"
// #include "cuda_runtime.h"
// #include "cublas.h"

#include "mpi_common.h"

void cpBuffer(int *dims, double *big, double *buf, int readbig);
void preloadBuffer(double *buf, int len);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

if(nrhs != 2) {
    printf("Require two input args (a double array and the parallel context)\n");
    return;
}

if(nlhs != 0) {
    printf("Not returning any errors, require 0 return args...\n");
    return;
}

double *array = mxGetPr(prhs[0]);
ParallelTopology theTopo;
int succeed = topoStructureToC(prhs[1], &theTopo);

int nd = mxGetNumberOfDimensions(prhs[0]);
int myrank;
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

// fix up sizes for us
const mwSize *d = mxGetDimensions(prhs[0]);
int dims[3];
int i;
for(i = 0; i < nd; i++) dims[i] = d[i];
if(nd < 3) dims[2] = 1;
if(nd < 2) dims[1] = 1;

printf("RANK %i: Input array dim = [%i %i %i]\n", myrank, dims[0], dims[1], dims[2]);

// num transverse
int nxy = dims[1]*dims[2];

//hardcode for test...
int haloDepth = 4;

int nalloc = haloDepth * 4 * nxy;
printf("RANK %i: Nalloc'd = 4 buffers = %i total elements\n", myrank, nalloc);

double *buf = (double *)malloc(nalloc * sizeof(double));

double *bufptrs[4];
bufptrs[0] = buf;
bufptrs[1] = buf + 4*nxy;
bufptrs[2] = buf + 2*4*nxy;
bufptrs[3] = buf + 3*4*nxy;

preloadBuffer(buf, 4*haloDepth*nxy); // preload with 0-65535

// do buffer reads:
// read left side
cpBuffer(&dims[0], array + 4, bufptrs[0], 1);
// read right side
cpBuffer(&dims[0], array + dims[0] - 8, bufptrs[1], 1);

// this deliberately leaves rank 0 left read and rank 1 right read un-filled
// and loaded with the padding garbage

int diditwork = mpi_exchangeHalos(&theTopo, 0, (void *)bufptrs[0], (void *)bufptrs[1], (void *)bufptrs[2], (void *)bufptrs[3], 4*nxy, MPI_DOUBLE);

if(1) {
printf("RANK %i, in haloExchange, leading contents of blocks after mpi-exchangeHalos:\n", myrank);
int j;
for(j = 0; j < 4; j++) {
printf("RANK %i: %i - %le %le %le %le\n", myrank, j, bufptrs[0][j], bufptrs[1][j], bufptrs[2][j], bufptrs[3][j]);
}
}

// write left side
cpBuffer(&dims[0], array, bufptrs[2], 0);
// write right side
 cpBuffer(&dims[0], array + dims[0] - 4, bufptrs[3], 0);

}

/* Dumps unique & deterministic values into the buffer so we know EXACTLY where any uninitalized things are hiding */
void preloadBuffer(double *buf, int len)
{
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int q;
for(q = 0; q < len; q++) { buf[q] = 100000 * rank + q; }
}


/* Emulates the MGA_wholeFaceToLinear() function for input host *big */
void cpBuffer(int *dims, double *big, double *buf, int readbig)
{
int i,j,k;

for(k = 0; k < dims[2]; k++) {
	for(j = 0; j < dims[1]; j++) {
		for(i = 0; i < 4; i++) {
			if(readbig) {
				buf[i+4*(j+dims[1]*k)]       = big[i+dims[0]*(j+dims[1]*k)];
//printf("copying from buf=%lx of dimensions [%i %i x] at [%i %i %i] to linear[%i]\n", buf, dims[0], dims[1], i, j, k, i+4*(j+dims[1]*k));
			} else {
				big[i+dims[0]*(j+dims[1]*k)] = buf[i+4*(j+dims[1]*k)];
//printf("copying from linear[%i] to buf=%lx of dimensions [%i %i x] at [%i %i %i]\n", i+4*(j+dims[1]*k), buf, dims[0], dims[1], i, j, k);
			}
		}
	}
}

}
