#include "stdio.h"

#include "mpi.h"

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

   MPI_Init(NULL, NULL);

   int size, bee;
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &bee);

   printf("bee %i/%i: ready\n", bee, size);


}
