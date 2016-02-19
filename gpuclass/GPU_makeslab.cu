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

	int worked = CHECK_CUDA_ERROR("entering GPU_makeslab");
	if(worked != SUCCESSFUL) { DROP_MEX_ERROR("Aborting mission, CUDA api reported error entering GPU_makeslab"); }

	MGArray m;

	MGA_accessMatlabArrays(prhs, 0, 0, &m);

	int x = (int)*mxGetPr(prhs[1]);
	int sub[6];
	// Do the allocate-and-copy dance since we don't have a cudaRealloc that I know of
	int i;
	double *newblock;

	for(i = 0; i < m.nGPUs; i++) {
		cudaSetDevice(m.deviceID[i]);
		worked = CHECK_CUDA_ERROR("setdevice");
		if(worked != SUCCESSFUL) break;

		calcPartitionExtent(&m, i, &sub[0]);
		// number of bytes per slab
		int64_t slabsize = sub[3]*sub[4]*sub[5] * sizeof(double);
		// round up to make a pleasantly CUDA-aligned amount
		int64_t slabpitch = slabsize / 256; slabpitch += (256*slabpitch < slabsize); slabpitch *= 256;

		cudaMalloc((void **)&newblock, slabpitch*x);
		worked = CHECK_CUDA_ERROR("malloc");
		if(worked != SUCCESSFUL) break;

		cudaMemcpy((void *)newblock, (void *)m.devicePtr[i], slabsize, cudaMemcpyDeviceToDevice);
		worked = CHECK_CUDA_ERROR("cudamemcpy");
		if(worked != SUCCESSFUL) break;

		cudaFree((void *)m.devicePtr[i]);
		CHECK_CUDA_ERROR("free");
		if(worked != SUCCESSFUL) break;

		m.devicePtr[i] = newblock;
		m.numSlabs = x;
	}

	if(worked != SUCCESSFUL) { DROP_MEX_ERROR("GPU_makeslab failed during creation."); }

	MGA_returnOneArray(plhs, &m);

	return;
}
