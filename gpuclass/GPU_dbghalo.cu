#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif
#include "mex.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas.h"

#include "cudaCommon.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	  // Input and result
	  if((nlhs != 1) || (nrhs != 4)) { mexErrMsgTxt("Form: halo = GPU_dbghalo(GPU_Type, direction, depth, side)"); }

	  int returnCode = CHECK_CUDA_ERROR("entering GPU_test");
	  if(returnCode != SUCCESSFUL) return;
	  
	  int dir = (int)*mxGetPr(prhs[1]);
	  int depth=(int)*mxGetPr(prhs[2]);
	  int side =(int)*mxGetPr(prhs[3]);

	  MGArray thetag;
	  returnCode = MGA_accessMatlabArrays(prhs, 0, 0, &thetag);
	  if(returnCode != SUCCESSFUL) {
		  CHECK_IMOGEN_ERROR(returnCode);
		  return;
	  }

	  MGArray skel = thetag;
	  skel.dim[dir-1] = depth;

	  MGArray *haloinfo;
	  returnCode = MGA_allocArrays(&haloinfo, 1, &skel);

	  returnCode = MGA_wholeFaceToLinear(&thetag, dir, side, 0, depth, &haloinfo->devicePtr[0]);

	  MGA_returnOneArray(plhs, haloinfo);

	  free(haloinfo);

	  CHECK_IMOGEN_ERROR(returnCode);
	  return;
}
