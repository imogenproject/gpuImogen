#include "stdio.h"

#include "mpi.h"

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

bool amLocked = mexIsLocked(); /* Our check for if this has been called already */

if(amLocked) {
  printf("WARNING: mpi_init() called, but has already been previously called.\n");
  return;
  }

mexLock();

MPI_Init(NULL, NULL);
}
