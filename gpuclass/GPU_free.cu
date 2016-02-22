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
	// wrapper for cudaFree().
	if((nlhs != 0) || (nrhs == 0)) mexErrMsgTxt("GPU_free: syntax is GPU_free(arbitrarily many GPU_Types, gpu tags, or ImogenArrays)");

	int returnCode = CHECK_CUDA_ERROR("Entering GPU_free()");
	if(returnCode != SUCCESSFUL)
		return;

	MGArray t[nrhs];

	returnCode = MGA_accessMatlabArrays(prhs, 0, nrhs-1, &t[0]);
	if(returnCode != SUCCESSFUL) {
		CHECK_IMOGEN_ERROR(returnCode);
		return;
	}

	int i;
	for(i = 0; i < nrhs; i++) {
		returnCode = MGA_delete(t+i);
		if(returnCode != SUCCESSFUL) break;
	}

	if(returnCode != SUCCESSFUL) CHECK_IMOGEN_ERROR(returnCode);

	return;
}
