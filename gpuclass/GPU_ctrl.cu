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

// static paramaters

void printAboutDevice(cudaDeviceProp info, int devno)
{
printf("Information on device %i (%s):\n", devno, info.name);
printf("\tCompute capability: %i.%i\n", info.major, info.minor);
printf("\tGlobal mem        = %iMB\n", info.totalGlobalMem/1048576);
printf("\tConstant mem      = %iKB\n", info.totalConstMem/1024);
printf("\tShared mem/block  = %iKB\n", info.sharedMemPerBlock/1024);
printf("\tRegisters/block   = %i\n", info.regsPerBlock);
printf("\tWarp size         = %i\n", info.warpSize);
printf("\tMemory pitch      = %i\n", info.memPitch);
printf("\tMax threads/block = %i\n", info.maxThreadsPerBlock);
printf("\tMax block dim     = %ix%ix%i\n", info.maxThreadsDim[0], info.maxThreadsDim[1], info.maxThreadsDim[2]);
printf("\tMax grid dim      = %ix%ix%i\n", info.maxGridSize[0], info.maxGridSize[1], info.maxGridSize[2]);
printf("\tClock rate        = %iMhz\n",(int)((float)info.clockRate / 1000.0));
printf("\t# of multiprocs	= %i\n", info.multiProcessorCount);
printf("\tExecution timeout? %s\n", info.kernelExecTimeoutEnabled == 1 ? "yes" : "no");
printf("\tSupports host mmap? %s\n", info.canMapHostMemory == 1? "yes" : "no");
//				size_t textureAlignment;
//				int deviceOverlap;
//				int integrated;
//				int computeMode;

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if(nrhs < 1) mexErrMsgTxt("Error: needs arguments; GPU_ctrl('help') for help.\n");

	int nDevices;
	cudaGetDeviceCount(&nDevices);
	CHECK_CUDA_ERROR("GPU_ctrl(): initialization\n");

	mexLock(); // It would be Bad if this disappeared on us at any point.

	mxClassID argtype = mxGetClassID(prhs[0]);

	if(argtype != mxCHAR_CLASS) mexErrMsgTxt("See GPU_ctrl('help').\n");

	int slen = mxGetNumberOfElements(prhs[0]);
	char c[slen+1];
	int stat = mxGetString(prhs[0], &c[0],	slen+1);

	if(strcmp(c, "help") == 0) {
		printf("%=== GPU_ctrl commands ===%\n GPU_ctrl('help'): This message\n\
 GPU_ctrl('info, [device #s]'): print device information (about specific devices)\n\
 GPU_ctrl('peers', [1/0]): Print matrix of cudaDeviceCanAccessPeer results, 1 or 0 to enable/disable it.\n\
 GPU_ctrl('reset'): If CUDA crashes, reset it so we can reinitialize without restarting Matlab.\n\
 m = GPU_ctrl('memory'): m[i,:] = [free, total] on device i.\n");
		return;
	}
	if(strcmp(c, "info") == 0) {
		if(nlhs > 0) mexErrMsgTxt("Error: No return value from GPU_ctrl('info')\n");
		int nDevices;
		cudaGetDeviceCount(&nDevices);
		cudaDeviceProp devprops;
		int q;
		for(q = 0; q < nDevices; q++) {
			cudaGetDeviceProperties(&devprops, q);
			printAboutDevice(devprops, q);
		}
	}
	if(strcmp(c, "peers") == 0) {
		// print accessibility matrix
		int nDevices;
		cudaGetDeviceCount(&nDevices);

		// turn it on/off if given to
		if(nrhs > 1) {
			mxArray *getGM[1];
			int stupid = mexCallMATLAB(1, &getGM[0], 0, NULL, "GPUManager.getInstance");

			double *f = mxGetPr(prhs[1]);
			int doit = (int)*f;
			// FIXME: loop over only gm.deviceList devices
			int u, v, IseeU;
			for(u = 0; u < nDevices; u++) {
				for(v = 0; v < nDevices; v++) {
					if(u == v) continue;
					cudaDeviceCanAccessPeer(&IseeU, u, v);
					if(IseeU) {
						cudaSetDevice(u);
						if(doit == 1) cudaDeviceEnablePeerAccess(v, 0);
						if(doit == 0) cudaDeviceDisablePeerAccess(v);
					}
				}
			}
		} else { // We have no request to turn it on/off, just gather information
			int IseeU;
			int u,v;

			int justprint = (nlhs == 0);
			double *outputMatrix = NULL;
			if(justprint) {
				printf("Device accessibility: symmetric matrix M_ij = device i can access device j\n");
				printf("	| ");
				for(u = 0; u < nDevices; u++) { printf("%i ", u); }
				printf("\n");
			} else {
				mwSize matSize = nDevices;
				plhs[0] = mxCreateDoubleMatrix(matSize, matSize, mxREAL);
				outputMatrix = mxGetPr(plhs[0]);
			}

			for(u = 0; u < nDevices; u++) {
				if(justprint) printf("%i | ", u);
				for(v = 0; v < nDevices; v++) {
					cudaDeviceCanAccessPeer(&IseeU, u, v);
					if(justprint) {
						if(u == v) { printf("x "); } else { printf("%i ", IseeU); }
					} else {
						outputMatrix[u+nDevices*v] = (double)IseeU;
					}
				}
				printf("\n");
			}
		}
	}
	if(strcmp(c, "reset") == 0) {
		mxArray *getGM[1];
		int stupid = mexCallMATLAB(1, &getGM[0], 0, NULL, "GPUManager.getInstance");
		// FIXME: device reset needs to be actually implemented.


	}
	if(strcmp(c, "memory") == 0) {
		size_t freemem; size_t totalmem;

		cudaError_t fail;

		mwSize dims[3];
		dims[0] = nDevices;
		dims[1] = 2;
		dims[2] = 1;
		plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
		double *d = mxGetPr(plhs[0]);

		int i;
		for(i = 0; i < nDevices; i++) {
			cudaSetDevice(i);
			CHECK_CUDA_ERROR("cudaSetDevice()");
			fail =  cudaMemGetInfo(&freemem, &totalmem);
			CHECK_CUDA_ERROR("cudaMemGetInfo()");

			d[i] = freemem; d[i+nDevices] = totalmem;

		}
	}

}
