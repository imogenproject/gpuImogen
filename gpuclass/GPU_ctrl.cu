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

// static parameters

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
	int returnCode = CHECK_CUDA_ERROR("GPU_ctrl(): cudaGetDeviceCount doesn't even work.\nIf occurring apropos of nothing, this appears to occur if Matlab was open,\nhad acquired a context, and the system was ACPI suspended.\n");
	if(returnCode != SUCCESSFUL) DROP_MEX_ERROR("GPU_ctrl failed at entry.");

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
		cudaError_t goofed;

		// turn it on/off if given to
		if(nrhs > 1) {
			mxArray *getGM[1];
			int stupid = mexCallMATLAB(1, &getGM[0], 0, NULL, "GPUManager.getInstance");

			double *f = mxGetPr(prhs[1]);
			int doit = (int)*f;
			// FIXME: loop over only gm.deviceList devices
			int u, v, IseeU;
			mxArray *mxDevList = mxGetProperty(getGM[0], 0, "deviceList");
			double *devList = mxGetPr(mxDevList);
			int numList = mxGetNumberOfElements(mxDevList);

			for(u = 0; u < numList; u++) {
				for(v = 0; v < numList; v++) {
					if(u == v) continue;
					int uhat = (int)devList[u];
					int vhat = (int)devList[v];

					cudaDeviceCanAccessPeer(&IseeU, uhat, vhat);
					if(IseeU) {
						cudaSetDevice(uhat);
						if(doit == 1) cudaDeviceEnablePeerAccess(vhat, 0);
						if(doit == 0) cudaDeviceDisablePeerAccess(vhat);

						goofed = cudaGetLastError();
						if(goofed == cudaErrorPeerAccessAlreadyEnabled) {
							printf("Oops: Peer access apparently already on. Returning...\n");
							return;
						}
						if(goofed == cudaErrorPeerAccessNotEnabled) {
							printf("Oops: Peer access already disabled. Returning...\n");
							return;
						}
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
		int nDevices;
		cudaGetDeviceCount(&nDevices);
		int i;
		for(i = 0; i < nDevices; i++) {
			printf("Resetting device %i... ", i);
			cudaSetDevice(i);
			cudaDeviceReset();
			printf("Done.\n");
			returnCode = CHECK_CUDA_ERROR("device reset");
			if(returnCode != SUCCESSFUL) break;
			}
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
			returnCode = CHECK_CUDA_ERROR("cudaSetDevice()");
			if(returnCode != SUCCESSFUL) break;

			cudaMemGetInfo(&freemem, &totalmem);
			returnCode = CHECK_CUDA_ERROR("cudaMemGetInfo()");
			if(returnCode != SUCCESSFUL) break;

			d[i] = freemem; d[i+nDevices] = totalmem;
		}
	}
	if(strcmp(c,"createStreams") == 0) { /* FIXME lines 186-217 are new/untested and are about as safe as letting grandma drive your your Shelby GTX */
		if(nrhs < 1) { printf("Must receive list of devices to get new streams: streams_ptr = GPU_ctrl('createStreams',[0 1]) e.g."); return; }
		if(nlhs < 1) { printf("Must be able to return cudaStream_t *: streams_ptr = GPU_ctrl('createStreams',[0 1]) e.g."); return; }
		double *d = mxGetPr(prhs[1]);

		int i;
		int imax = mxGetNumberOfElements(prhs[1]);

		mwSize dims[2];
		dims[0] = 1;
		dims[1] = 1;

		plhs[0] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);
		int64_t *out = (int64_t *)mxGetData(plhs[0]);

		cudaStream_t *pstream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * imax);
		for(i = 0; i < imax; i++) {
			cudaSetDevice((int)d[i]);
			cudaStreamCreate(pstream + i);
		}

		out[0] = (int64_t)pstream;

	}
	if(strcmp(c,"destroyStreams") == 0) {
		if(nrhs < 2) { printf("Must receive GPUManager.cudaStreamsPtr and # of streams in it."); return; }
		cudaStream_t *streams = (cudaStream_t *)mxGetData(prhs[0]);
		int x = (int)*mxGetPr(prhs[1]);
		int i;
		for(i = 0; i < x; i++) {
			cudaStreamDestroy(*streams);
			streams++;
		}
	}

	if(returnCode != SUCCESSFUL) {
		DROP_MEX_ERROR("Operation of GPU_ctrl failed: Causing interpreter error.");
	}

}
