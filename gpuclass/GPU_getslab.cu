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
	if((nlhs != 1) || (nrhs != 2)) { mexErrMsgTxt("Form: slab reference = GPU_getslab(gputag, which to get)."); }

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
			int64_t slabpitch = slabsize / 256; slabpitch += (256*slabpitch < slabsize); slabpitch *= 256;
			slabpitch /= sizeof(double);

			slab.devicePtr[i] += x*slabpitch;
			slab.numSlabs = -x;
		}

		MGA_returnOneArray(plhs, &slab);
	return;
}
