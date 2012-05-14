#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif
#include "mex.h"

/* This implements halo_alloc for the MPI parallelization scheme */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

if(nlhs != 2) mexErrMsgTxt("Require return to be [array halo_meta]\n");
if(nrhs != 3) mexErrMsgTxt("Require args to be (ndims, dims[ndims], halo[ndims])\n");

int ndims = (int)*mxGetPr(prhs[0]);
mwSize dims[ndims];
mwSize halo[ndims];

int q;
double *dimsin = mxGetPr(prhs[1]);
double *haloin = mxGetPr(prhs[2]);

for(q = 0; q < ndims; q++) {
  dims[q] = (mwSize)*dimsin;
  halo[q] = (mwSize)*haloin;

  dimsin++; haloin++;
  }

plhs[0] = mxCreateNumericArray(ndims, dims, mxDOUBLE_CLASS, mxREAL);

int haloND = 1;
mwSize halolen = 8;

plhs[1] = mxCreateNumericArray(haloND, &halolen, mxINT32_CLASS, mxREAL);

}
