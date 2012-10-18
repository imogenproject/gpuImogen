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
  // wrapper for cudaFree().
  if((nlhs != 0) || (nrhs > 1)) mexErrMsgTxt("Error: syntax is GPU_init(optional auto device #)");

  int numDevices; cudaError_t fail = cudaGetDeviceCount(&numDevices);
  if(fail != cudaSuccess) mexErrMsgTxt("Failure to clear previous error messages. Bleh!");

  int deviceNum = 9999;
  if(nrhs == 1) {
    if(mxGetNumberOfElements(prhs[0]) != 1) mexErrMsgTxt("GPU_init: device # argument, but it is not a scalar.");
    deviceNum = (int)*mxGetPr(prhs[0]);
    }

  if(deviceNum < numDevices) { // Requested device w/o theatrics
    fail = cudaSetDevice(deviceNum);
    if(fail == cudaErrorInvalidDevice)      mexErrMsgTxt("GPU_init: selected device DNE");
    if(fail == cudaErrorSetOnActiveProcess) mexErrMsgTxt("GPU_init: already initialized, dingus");

//    cudaDeviceProp devinfo;
//    cudaGetDeviceProperties(&devinfo, deviceNum);
//    printAboutDevice(devinfo, deviceNum);
    return;
  }

  printf("%i Devices on this system supporting CUDA:\n", numDevices);

  int i;
  for(i = 0; i < numDevices; i++) {
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
      printAboutDevice(prop, i);
    }
  }

  if(numDevices == 1) {
    fail = cudaSetDevice(0);
    if(fail != cudaSuccess) mexErrMsgTxt("GPU_init: Unable to select only CUDA device. Oh my.");

    printf("Selected only available CUDA device automatically.\n");
    return;
  }

  i = -1;
  printf("Please select a device from 0 to %i.\n", numDevices-1);
  while((i < 0) || (i >= numDevices)) {
    mxArray   *new_number, *str;
    double out;

    str = mxCreateString("Device no: ");
    mexCallMATLAB(1,&new_number,1,&str,"input");
    out = mxGetScalar(new_number);
    mxDestroyArray(new_number);
    mxDestroyArray(str);

    i = (int)out;
  }

  cudaSetDevice(i);

  return;
}
