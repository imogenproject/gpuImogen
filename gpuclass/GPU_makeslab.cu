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

// GPU_Tag = GPU_upload(host_array[double], device IDs[integers], [integer halo dim, integer partition direction])

// GPU_slabs(gputag, how many slabs)
// tag = GPU_slabs(gputag, which slab to return)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// At least 2 arguments expected
	// Input and result
	if((nlhs != 1) || (nrhs != 2)) { mexErrMsgTxt("Form: newtag = GPU_makeslab(original gputag, how many to have)."); }

	CHECK_CUDA_ERROR("entering GPU_makeslab");

	MGArray m;

	accessMGArrays(prhs, 0, 0, &m);

	int x = (int)*mxGetPr(prhs[1]);
	int sub[6];
		// Do the allocate-and-copy dance since we don't have a cudaRealloc that I know of
		int i;
		double *newblock;

		for(i = 0; i < m.nGPUs; i++) {
			cudaSetDevice(m.deviceID[i]);
			CHECK_CUDA_ERROR("setdevice");

			calcPartitionExtent(&m, i, &sub[0]);
			// number of bytes per slab
			int64_t slabsize = sub[3]*sub[4]*sub[5] * sizeof(double);
			// round up to make a pleasantly CUDA-aligned amount
			int64_t slabpitch = slabsize / 256; slabpitch += (256*slabpitch < slabsize); slabpitch *= 256;

			cudaMalloc((void **)&newblock, slabpitch*x);
		 	CHECK_CUDA_ERROR("malloc");
			cudaMemcpy((void *)newblock, (void *)m.devicePtr[i], slabsize, cudaMemcpyDeviceToDevice);
			CHECK_CUDA_ERROR("cudamemcpy");
			cudaFree((void *)m.devicePtr[i]);
			CHECK_CUDA_ERROR("free");

			m.devicePtr[i] = newblock;
			m.numSlabs = x;
		}
returnAnMGArray(plhs, &m);

	return;
}
