#include "stdio.h"
#include "unistd.h"

#include "mpi.h"

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

if(nrhs > 1) { mexErrMsgTxt("call is debugSpin([array of ranks to spin, or empty defaults to all]);"); }

   int myrank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   /* Default to spin, but check for match w/rank # if we get an argument */
   int doIspin = 1;
   if(nrhs > 0) {
	doIspin = 0;
        double *d = mxGetPr(prhs[0]);
	mwSize N = mxGetNumberOfElements(prhs[0]);
	int i;

	for(i = 0; i < N; i++) {
            if( (mwSize)(d[i]) == myrank) { doIspin = 1; break; } 
        }
    }

    pid_t P = getpid();

    if(doIspin) {
        printf("Rank %i IS ENTERING DEBUG SPINWAIT: ATTACH gdb/cuda-gdb TO PROCESS %i; break debugSpin.c:37\n", myrank, (int)P);
    } else {
        printf("PROCESS %i IS NOT ENTERING DEBUG SPINWAIT\n", myrank);
    }

    while(doIspin) {
        sleep(1);
    }

    return;

}
