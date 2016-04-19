#include "cudaCommon.h"
#include "mex.h"

#warning "WARNING: COMPILING cudaFluidStep() WITH DEBUG ENABLED. cudaFluidStep will require an output argument to dump to!"

// If defined, the code runs the Euler prediction step and copies wStepValues back to the Matlab fluid data arrays
// If not defined, it runs the RK2 predictor/corrector step
#define DBG_FIRSTORDER

// If not debugging the 1st order step, flips on debugging of the 2nd order step
#ifndef DBG_FIRSTORDER
#define DBG_SECONDORDER
#else
#warning "WARNING: Compiling cudaFluidStep to take 1st order time steps [dump wStep array straight to output]"
#endif

#define DBG_NUMARRAYS 6

#ifdef DBG_FIRSTORDER
#define DBGSAVE(n, x) if(thisThreadDelivers) { Qout[((n)+6)*DEV_SLABSIZE] = (x); }
#else
#define DBGSAVE(n, x) if(thisThreadDelivers) {  Qin[((n)+6)*DEV_SLABSIZE] = (x); }
#endif

// Assuming debug has been put on the wStepValues array, download it to a Matlab array
void returnDebugArray(MGArray *ref, int x, double **dbgArrays, mxArray *plhs[])
{
	CHECK_CUDA_ERROR("entering returnDebugArray");

	MGArray m = *ref;

	int nd = 3;
	if(m.dim[2] == 1) {
		nd = 2;
		if(m.dim[1] == 1) {
			nd = 1;
		}
	}
	nd = 4;
	mwSize odims[4];
	odims[0] = m.dim[0];
	odims[1] = m.dim[1];
	odims[2] = m.dim[2];
	odims[3] = x;

	// Create output numeric array
	plhs[0] = mxCreateNumericArray(nd, odims, mxDOUBLE_CLASS, mxREAL);

	double *result = mxGetPr(plhs[0]);

	// Create a sacrificial MGA
	MGArray scratch = ref[0];
	// wStepValues is otherwise identical so overwrite the new one's pointers
	int j, k;
	for(j = 0; j < scratch.nGPUs; j++) scratch.devicePtr[j] = dbgArrays[j];

	for(k = 0; k < x; k++) {
		// download
		MGA_downloadArrayToCPU(&scratch, &result, 0);
		// move over and repeat
		result += scratch.numel;
		for(j = 0; j < scratch.nGPUs; j++) { scratch.devicePtr[j] += scratch.slabPitch[j] / 8; }

	}
	return;
}
