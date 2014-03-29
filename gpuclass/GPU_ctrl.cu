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
printf("\t# of multiprocs   = %i\n", info.multiProcessorCount);
printf("\tExecution timeout? %s\n", info.kernelExecTimeoutEnabled == 1 ? "yes" : "no");
printf("\tSupports host mmap? %s\n", info.canMapHostMemory == 1? "yes" : "no");
//        size_t textureAlignment;
//        int deviceOverlap;
//        int integrated;
//        int computeMode;

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if(nlhs != 0) mexErrMsgTxt("Error: syntax is one of\n  GPU_ctrl('info', [devices #s]),\n  GPU_ctrl() (init device 0)\n  GPU_ctrl(device #)\n  GPU_ctrl('start', [device #]) (default device 0)\n  GPU_ctrl('switch', device #)\n  GPU_ctrl('exit')");

  if(nrhs == 0) { // GPU_ctrl() -> activate device 0 by default
    bool amLocked = mexIsLocked();

    if(amLocked) {
      mexErrMsgTxt("Error: GPU_ctrl() considered initializing default device, but a device has already been initialized.");
      return;
      }

    int nDevices;
    cudaGetDeviceCount(&nDevices);
    CHECK_CUDA_ERROR("GPU_ctrl(): initialization\n");

    mexLock(); // It would be Bad if this disappeared on us at any point.
    cudaSetDevice(0);

    return;
    }

  if(nrhs == 1) {
    // Could be GPU_ctrl(device #),
    //          GPU_ctrl('exit');
    mxClassID argtype = mxGetClassID(prhs[0]);

    if(argtype == mxCHAR_CLASS) { // 'exit'
      int slen = mxGetNumberOfElements(prhs[0]);
      char c[slen+1];
      int stat = mxGetString(prhs[0], &c[0],  slen+1);
//printf("string: %c%c%c%c\n", c[0], c[1], c[2], c[3]);

      if(strncmp(c,"exit",4) == 0) {
        bool amLocked = mexIsLocked();
        if(amLocked) {
          mexUnlock();
          cudaDeviceReset();
        } else {
          mexErrMsgTxt("GPU_ctrl('exit') attempted but no device is initialized.\n");
        }
      }

      if(strncmp(c,"info",4) == 0) {
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        cudaDeviceProp devprops;
        int q;
        for(q = 0; q < nDevices; q++) {
          cudaGetDeviceProperties(&devprops, q);
          printAboutDevice(devprops, q);
          }
        
        }

    }

    // Attempting to set device;
    // Are we initializing for the first time?  -> mexLock()
    // Is the device # given not acceptable?    -> mexError
    // Is the device # given already active?    -> print warning
    if(argtype == mxDOUBLE_CLASS) {
      if(mxGetNumberOfElements(prhs[0]) != 1) mexErrMsgTxt("GPU_ctrl(#) called with more than one number.");

      if(mexIsLocked() == false) mexLock();

      int nDevices;
      cudaGetDeviceCount(&nDevices);
      CHECK_CUDA_ERROR("GPU_ctrl(#): initializing chosen device");

      int currentDev; cudaGetDevice(&currentDev);
      int requestDev = (int)*mxGetPr(prhs[0]);

      if(requestDev >= nDevices) {
        printf("WARNING: Attempted to activate device %i but only %i devices; Activating device 0.\n", requestDev, nDevices);
        requestDev = 0;
        }

      if(requestDev != currentDev) cudaSetDevice(requestDev);
      }

    return;
    }

  // Possible syntaxes?
  if(nrhs == 2) {
    
    }

  mexErrMsgTxt("No sensible command to GPU_ctrl.\n");

}
