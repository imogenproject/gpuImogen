#include "stdio.h"

#include "mpi.h"

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

if((nlhs != 1 ) || (nrhs != 0)) { mexErrMsgTxt("call is q = mpi_basicinfo(), q=[size rank hostnamehash]"); }

   int size, bee;
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &bee);

   mwSize dims[2]; dims[0] = 3; dims[1] = 1;

   plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);

   double *d = mxGetPr(plhs[0]);
   d[0] = size; d[1] = bee;

   char *hn = calloc(255, sizeof(char));
   gethostname(hn,255);
   int i;

   int *d0 = (int *)hn; int hash = 0;
   /* calculate a simple 4-byte hash of the hostname */
   for(i = 0; i < 64; i++) { hash ^= *d0++; } 
  
   d[2] = (double)hash;
   free(hn);
}
