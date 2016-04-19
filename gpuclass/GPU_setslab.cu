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
	// At least 2 arguments expected
	// Input and result
	if((nlhs != 1) || (nrhs != 3)) { mexErrMsgTxt("Form: slab reference = GPU_setslab(gputag, which slab # to set, source array)."); }

	CHECK_CUDA_ERROR("entering GPU_getslab");

	MGArray m;

	MGA_accessMatlabArrays(prhs, 0, 0, &m);

	int x = (int)*mxGetPr(prhs[1]);
	int sub[6];
	int i;
	MGArray slab = m;

	for(i = 0; i < m.nGPUs; i++) {
		calcPartitionExtent(&m, i, &sub[0]);
		// number of bytes per slab
		int64_t slabsize = sub[3]*sub[4]*sub[5] * sizeof(double);
		// round up to make a pleasantly CUDA-aligned amount
		int64_t slabpitch = ROUNDUPTO(slabsize, 256);
		slabpitch /= sizeof(double);

		slab.devicePtr[i] += x*slabpitch;
		slab.numSlabs = -x;
	}

	MGA_returnOneArray(plhs, &slab);

	mxClassID dtype = mxGetClassID(prhs[2]);
        if(dtype == mxDOUBLE_CLASS) {
		MGA_uploadMatlabArrayToGPU(prhs[2], &slab, -1);
	} else {
		MGArray src;
		MGA_accessMatlabArrays(prhs, 2, 2, &src);

		int j;
		for(j = 0; j < 3; j++) {
			if(slab.dim[j] != src.dim[j])
				DROP_MEX_ERROR("Cannot upload array: Source gpu array size incompatible with destination size.");
		}

		/* Otherwise, fire off copying! */
		for(j = 0; j < src.nGPUs; j++) {
			cudaSetDevice(src.deviceID[j]);
			cudaMemcpyPeerAsync(slab.devicePtr[j], slab.deviceID[j], src.devicePtr[j], src.deviceID[j], src.partNumel[j]*sizeof(double));
		}
	}
	return;
}
