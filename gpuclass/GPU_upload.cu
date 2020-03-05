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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// At least 2 arguments expected
	// Input and result
	if((nlhs != 1) || (nrhs < 1)) { mexErrMsgTxt("Form: result_tag = GPU_upload(host array [, device list [, (halo [,partition direct [, clone_partitions]])]])"); }

	CHECK_CUDA_ERROR("entering GPU_upload");

	MGArray m;

	// Default to no halo, X partition, add exterior halo
	m.haloSize = 0;
	m.partitionDir = PARTITION_X;
	m.addExteriorHalo = 1;
	m.vectorComponent = 0; // default, poke using GPU_Type.updateVectorComponent(n)
	int forceClone = 0;

	if(nrhs >= 3) {
		int a = mxGetNumberOfElements(prhs[2]);
		double *d = mxGetPr(prhs[2]);

		if(a >= 1) {
			m.haloSize = (int)*d;
			if(m.haloSize < 0) {
				printf("WARNING: Halo size %i is being clamped to zero.\n", m.haloSize);
				m.haloSize = 0;
			}

		}
		if(a >= 2) {
			m.partitionDir = (int)d[1];
			if((m.partitionDir < 1) || (m.partitionDir > 3)) m.partitionDir = PARTITION_X;
		}
		if(a >= 3) {
			// addExteriorHalo should be false iff #procs(partition direction) > 1
			m.addExteriorHalo = (int)d[2];
		}
		if(a >= 4) {
			forceClone = (int)d[3];
		}
	}

	// Default to circular boundary conditions
	m.mpiCircularBoundaryBits = 63;

	// With any new upload, assume this is the XYZ orientation
	m.permtag = 1;
	MGA_permtagToNums(m.permtag, &m.currentPermutation[0]);

	// Default to entire array on current device
	m.nGPUs = 1;
	cudaGetDevice(&m.deviceID[0]);
	// But of course we may partition it otherwise
	if(nrhs >= 2) {
		int j;
		double *g = mxGetPr(prhs[1]);
		m.nGPUs = mxGetNumberOfElements(prhs[1]);
		for(j = 0; j < m.nGPUs; j++) {
			m.deviceID[j] = (int)g[j];
			m.devicePtr[j] = 0x0;
		}
	}

	double *hmem = mxGetPr(prhs[0]);
	int nd = mxGetNumberOfDimensions(prhs[0]);
	if(nd > 3) mexErrMsgTxt("Array dimensionality > 3 unsupported.");
	const mwSize *idims = mxGetDimensions(prhs[0]);
	int i;
	for(i = 0; i < nd; i++) { m.dim[i] = idims[i]; }
	for(;      i < 3; i++) { m.dim[i] = 1; }

	// If we are already cloning, multiply the size in the partition direct by #GPUs
	if(forceClone) {
		m.dim[m.partitionDir-1] = m.dim[m.partitionDir-1] * m.nGPUs;
	}

	// If the size in the partition direction is 1, clone it instead
	if((m.dim[m.partitionDir-1] == 1) && (m.nGPUs > 1)) {
		m.haloSize = 0;
		m.dim[m.partitionDir-1] = m.nGPUs;
		forceClone = 1;
	}

	m.numel = m.dim[0]*m.dim[1]*m.dim[2];
	int sub[6];
	for(i = 0; i < m.nGPUs; i++) {
		calcPartitionExtent(&m, i, &sub[0]);
		m.partNumel[i] = sub[3]*sub[4]*sub[5];
	}
	m.numSlabs = 1;

	MGArray *dest = MGA_createReturnedArrays(plhs, 1, &m);

	int worked;
	if(forceClone) {
		worked = MGA_uploadArrayToGPU(hmem, dest, 0);
	} else {
		worked = MGA_uploadArrayToGPU(hmem, dest, -1);
	}
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) {
		mexErrMsgTxt("Attempt to upload Matlab array to GPU was unsuccessful.\n");
		return;
	}
	if(forceClone) {
	    worked = MGA_distributeArrayClones(dest, 0);
	    if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) {
	    	mexErrMsgTxt("Redistribution of cloned array failed!\n");
	    }
	}


	return;
}
