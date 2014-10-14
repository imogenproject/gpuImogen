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
printf("\tGlobal mem				= %iMB\n", info.totalGlobalMem/1048576);
printf("\tConstant mem			= %iKB\n", info.totalConstMem/1024);
printf("\tShared mem/block	= %iKB\n", info.sharedMemPerBlock/1024);
printf("\tRegisters/block	 = %i\n", info.regsPerBlock);
printf("\tWarp size				 = %i\n", info.warpSize);
printf("\tMemory pitch			= %i\n", info.memPitch);
printf("\tMax threads/block = %i\n", info.maxThreadsPerBlock);
printf("\tMax block dim		 = %ix%ix%i\n", info.maxThreadsDim[0], info.maxThreadsDim[1], info.maxThreadsDim[2]);
printf("\tMax grid dim			= %ix%ix%i\n", info.maxGridSize[0], info.maxGridSize[1], info.maxGridSize[2]);
printf("\tClock rate				= %iMhz\n",(int)((float)info.clockRate / 1000.0));
printf("\t# of multiprocs	 = %i\n", info.multiProcessorCount);
printf("\tExecution timeout? %s\n", info.kernelExecTimeoutEnabled == 1 ? "yes" : "no");
printf("\tSupports host mmap? %s\n", info.canMapHostMemory == 1? "yes" : "no");
//				size_t textureAlignment;
//				int deviceOverlap;
//				int integrated;
//				int computeMode;

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if((nlhs != 0) || (nrhs < 1)) mexErrMsgTxt("Error: needs arguments; GPU_ctrl('help') for help.\n");

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
		printf("%=== GPU_ctrl commands ===%\n GPU_ctrl('help'): This message\n GPU_ctrl('info, [device #s]'): print device information (about specific devices)\n GPU_ctrl('peers', [1/0]): Print matrix of cudaDeviceCanAccessPeer results, 1 or 0 to enable/disable it.\n GPU_ctrl('reset'): If CUDA crashes, reset it so we can reinitialize without restarting Matlab.\n");
		return;
	}
	if(strcmp(c, "info") == 0) {
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

		int IseeU;
		int u,v;
		printf("Device accessibility: M_ij = device i can access device j\n");
		printf("	| ");
		for(u = 0; u < nDevices; u++) { printf("%i ", u); }
		printf("\n");

		for(u = 0; u < nDevices; u++) {
			printf("%i | ", u);
			for(v = 0; v < nDevices; v++) {
				cudaDeviceCanAccessPeer(&IseeU, u, v);
				if(u == v) { printf("x "); } else { printf("%i ", IseeU); }
			}
			printf("\n");
		}

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

		}

	}
	if(strcmp(c, "reset") == 0) {
		 mxArray *getGM[1];
		 int stupid = mexCallMATLAB(1, &getGM[0], 0, NULL, "GPUManager.getInstance");
// FIXME: device reset needs to be actually implemented.


	}

}
