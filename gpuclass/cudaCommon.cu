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


/* Given the RHS, an array to return array size, and the set of array indexes to take *s from */
double **getGPUSourcePointers(const mxArray *prhs[], ArrayMetadata *metaReturn, int fromarg, int toarg)
{

  double **gpuPointers = (double **)malloc((1+toarg-fromarg) * sizeof(double *));
  int iter;

  mxClassID dtype;

  dtype = mxGetClassID(prhs[fromarg]);
  if(dtype != mxINT64_CLASS) mexErrMsgTxt("cudaCommon: fatal, tried to get gpu src pointer from something not a gpu tag.");

  int64_t *dims = (int64_t *)mxGetData(prhs[fromarg]);
  for(iter = 0; iter < 3; iter++) { metaReturn->dim[iter] = (int)dims[2+iter]; } // copy metadata out of first gpu*
  metaReturn->numel = metaReturn->dim[0]*metaReturn->dim[1]*metaReturn->dim[2];
  metaReturn->ndims = dims[1];

  for(iter = fromarg; iter <= toarg; iter++) {
     dtype = mxGetClassID(prhs[iter]);
    if(dtype != mxINT64_CLASS) {
      printf("For argument %i\n",iter);
      mexErrMsgTxt("cudaCommon: fatal, tried to get gpu src pointer from something not a gpu tag.");
      }

    dims = (int64_t *)mxGetData(prhs[iter]);
    gpuPointers[iter-fromarg] = (double *)dims[0];
  }

return gpuPointers;
}

/* Creates destination array that the kernels write to; Returns the GPU memory pointer, and assigns the LHS it's passed */
double **makeGPUDestinationArrays(int64_t *reference, mxArray *retArray[], int howmany)
{

double **rvals = (double **)malloc(howmany*sizeof(double *));
int i;
mwSize dims[2]; dims[0] = 5; dims[1] = 1;

int64_t *rv; size_t numel;

numel = reference[2]*reference[3]*reference[4];

for(i = 0; i < howmany; i++) {
  retArray[i] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);
  rv = (int64_t *)mxGetData(retArray[i]);

  cudaError_t fail = cudaMalloc((void **)&rv[0], numel*sizeof(double));
  if(fail != cudaSuccess) {
    printf("On array %i/%i: %s\n", i+1, howmany, cudaGetErrorString(fail));
    cudaCheckError("In makeGPUDestinationArrays: malloc failed and I am sad.");
    }

  int q; for(q = 1; q < 5; q++) rv[q] = reference[q];
  rvals[i] = (double *)rv[0];
  }

//size_t tot,fre;

//cuMemGetInfo(&fre, &tot);
//printf("Now free: %u\n", fre);

return rvals;

}

void cudaLaunchError(cudaError_t E, dim3 blockdim, dim3 griddim, ArrayMetadata *a, int i, char *srcname)
{
if(E == cudaSuccess) return;

printf("Severe CUDA failure in %s: %s -> %s\n", srcname, errorName(E), cudaGetErrorString(E));
printf("Array info: dims=<%i %i %i>, numel=%i. I was passed the integer %i.\n", a->dim[0], a->dim[1], a->dim[2], a->numel, i);
printf("Block and grid dimensions: <%i %i %i>, <%i %i %i>\n", blockdim.x, blockdim.y, blockdim.z, griddim.x, griddim.y, griddim.z);
mexErrMsgTxt("Forcing program halt due to CUDA error");

}

void cudaCheckError(char *where)
{
cudaError_t epicFail = cudaGetLastError();
if(epicFail == cudaSuccess) return;

printf("Encountered error at %s: %s -> %s\n", where, errorName(epicFail), cudaGetErrorString(epicFail));
mexErrMsgTxt("Forcing program halt due to pre-existing CUDA error");
}

void printdim3(char *name, dim3 dim)
{
printf("dim3 %s is [%i %i %i]\n", name, dim.x, dim.y, dim.z);
}

void printgputag(char *name, int64_t *tag)
{
printf("gputag %s is [*=%lu dims=%lu size=(%lu %lu %lu)]\n", name, tag[0], tag[1], tag[2], tag[3], tag[4]);
}

const char *errorName(cudaError_t E)
{
/* Written the stupid way because nvcc is a flaming retarded shitcock that claims these are all "case inaccessible" if it's done with a switch.

Fuck you, dumb lying braindead cockbite. */

  if(E == cudaSuccess) { static const char err[]="cudaSuccess"; return err; }
  if(E == cudaErrorMissingConfiguration) { static const char err[]="cudaErrorMissingConfiguration"; return err;  }
  if(E == cudaErrorMemoryAllocation) { static const char err[]="cudaErrorMemoryAllocation"; return err; }
  if(E == cudaErrorInitializationError) { static const char err[]="cudaErrorInitializationError"; return err; }
  if(E == cudaErrorLaunchFailure) { static const char err[]="cudaErrorLaunchFailure"; return err; }
  if(E == cudaErrorPriorLaunchFailure) { static const char err[]="cudaerrorPriorLaunchFailure"; return err; }
  if(E == cudaErrorLaunchTimeout) { static const char err[]="cudaErrorLaunchTimeout"; return err; }
  if(E == cudaErrorLaunchOutOfResources) { static const char err[]="cudaErrorLaunchOutOfResources"; return err; }
  if(E == cudaErrorInvalidDeviceFunction) { static const char err[]="cudaErrorInvalidDeviceFunction"; return err; }
  if(E == cudaErrorInvalidConfiguration) { static const char err[]="cudaErrorInvalidDeviceConfiguration"; return err; }
  if(E == cudaErrorInvalidDevice) { static const char err[]="cudaErrorInvalidDevice"; return err; }
  if(E == cudaErrorInvalidValue) { static const char err[]="cudaErrorInvalidValue"; return err; }
  if(E == cudaErrorInvalidPitchValue) { static const char err[]="cudaErrorInvalidPitchValue"; return err; }
  if(E == cudaErrorInvalidSymbol) { static const char err[]="cudaErrorInvalidSymbol"; return err; }
  if(E == cudaErrorMapBufferObjectFailed) { static const char err[]="cudaErrorMapBufferObjectFailed"; return err; }
  if(E == cudaErrorUnmapBufferObjectFailed) { static const char err[]="cudaErrorUnmapBufferObjectFailed"; return err; }
  if(E == cudaErrorInvalidHostPointer) { static const char err[]="cudaErrorInvalidHostPointer"; return err; }
  if(E == cudaErrorInvalidDevicePointer) { static const char err[]="cudaerrorInvalidDevicePointer"; return err; }
  if(E == cudaErrorInvalidTexture) { static const char err[]="cudaErrorInvalidTexture"; return err; }
  if(E == cudaErrorInvalidTextureBinding) { static const char err[]="cudaErrorInvalidTextureBinding"; return err; }
  if(E == cudaErrorInvalidChannelDescriptor) { static const char err[]="cudaErrorInvalidChannelDescriptor"; return err; }
  if(E == cudaErrorInvalidMemcpyDirection) { static const char err[]="cudaErrorInvalidMemcpyDirection"; return err; }
  if(E == cudaErrorAddressOfConstant) { static const char err[]="cudaErrorAddressOfConstant"; return err; }
  if(E == cudaErrorTextureFetchFailed) { static const char err[]="cudaErrorTextureFetchFailed"; return err; }
  if(E == cudaErrorTextureNotBound) { static const char err[]="cudaErrorTextureNotBound"; return err; }
  if(E == cudaErrorSynchronizationError) { static const char err[]="cudaErrorSynchronizationError"; return err; }
  if(E == cudaErrorInvalidFilterSetting) { static const char err[]="cudaErrorInvalidFilterSetting"; return err; }
  if(E == cudaErrorInvalidNormSetting) { static const char err[]="cudaErrorInvalidNormSetting"; return err; }
  if(E == cudaErrorMixedDeviceExecution) { static const char err[]="cudaErrorMixedDeviceExecution"; return err; }
  if(E == cudaErrorCudartUnloading) { static const char err[]="cudaErrorCudartUnloading"; return err; }
  if(E == cudaErrorUnknown) { static const char err[]="cudaErrorUnknown"; return err; }
  if(E == cudaErrorNotYetImplemented) { static const char err[]="cudaErrorNotYetImplemented"; return err; }
  if(E == cudaErrorMemoryValueTooLarge) { static const char err[]="cudaErrorMemoryValueTooLarge"; return err; }
  if(E == cudaErrorInvalidResourceHandle) { static const char err[]="cudaErrorInvalidResourcehandle"; return err; }
  if(E == cudaErrorNotReady) { static const char err[]="cudaErrorNotReady"; return err; }
  if(E == cudaErrorInsufficientDriver) { static const char err[]="cudaErrorInsufficientDriver"; return err; }
  if(E == cudaErrorSetOnActiveProcess) { static const char err[]="cudaErrorSetOnActiveProcess"; return err; }
  if(E == cudaErrorInvalidSurface) { static const char err[]="cudaErrorInvalidSurface"; return err; }
  if(E == cudaErrorNoDevice) { static const char err[]="cudaErrorNoDevice"; return err; }
  if(E == cudaErrorECCUncorrectable) { static const char err[]="cudaErrorECCUncorrectable"; return err; }
  if(E == cudaErrorSharedObjectSymbolNotFound) { static const char err[]="cudaErroSharedObjectSymbolNotFound"; return err; }
  if(E == cudaErrorSharedObjectInitFailed) { static const char err[]="cudaErroSharedObjectInitFailed"; return err; }
  if(E == cudaErrorUnsupportedLimit) { static const char err[]="cudaErrorUnsupportedLimit"; return err; }
  if(E == cudaErrorDuplicateVariableName) { static const char err[]="cudaErrorDuplicateVariableName"; return err; }
  if(E == cudaErrorDuplicateTextureName) { static const char err[]="cudaErrorDuplicateTextureName"; return err; }
  if(E == cudaErrorDuplicateSurfaceName) { static const char err[]="cudaErrorDuplicateSurfaceName"; return err; }
  if(E == cudaErrorDevicesUnavailable) { static const char err[]="cudaErrorDevicesUnavailable"; return err; }
  if(E == cudaErrorInvalidKernelImage) { static const char err[]="cudaErrorInvalidKernelImage"; return err; }
  if(E == cudaErrorNoKernelImageForDevice) { static const char err[]="cudaErrorNoKernelImageForDevice"; return err; }
  if(E == cudaErrorIncompatibleDriverContext) { static const char err[]="cudaErrorIncompatibleDriverContext"; return err; }
  if(E == cudaErrorPeerAccessAlreadyEnabled) { static const char err[]="cudaErrorPeerAccessAlreadyEnabled"; return err; }
  if(E == cudaErrorPeerAccessNotEnabled) { static const char err[]="cudaErrorPeerAccessNotEnabled"; return err; }
  if(E == cudaErrorDeviceAlreadyInUse) { static const char err[]="cudaErrorDeviceAlreadyInUse"; return err; }
  if(E == cudaErrorProfilerDisabled) { static const char err[]="cudaErrorProfilerDisabled"; return err; }
  if(E == cudaErrorProfilerNotInitialized) { static const char err[]="CudaErrorProfilerNotInitialized"; return err; }
  if(E == cudaErrorProfilerAlreadyStarted) { static const char err[]="cudaErrorProfilerAlreadyStarted"; return err; }
  if(E == cudaErrorProfilerAlreadyStopped) { static const char err[]="cudaErrorProfilerAlreadyStopped"; return err; }
//  if(E == cudaErrorAssert) { static const char err[]="cudaErrorAssert"; return err; }
//  if(E == cudaErrorTooManyPeers) { static const char err[]="cudaErrorTooManyPeers"; return err; }
//  if(E == cudaErrorHostMemoryAlreadyRegistered) { static const char err[]="cudaErrorHostMemoryAlreadyRegistered"; return err; }
//  if(E == cudaErrorHostMemoryNotRegistered) { static const char err[]="cudaErrorHostMemoryNotRegistered"; return err; }
//  if(E == cudaErrorOperatingSystem) { static const char err[]="cudaErrorOperatingsystem"; return err; }
  if(E == cudaErrorStartupFailure ) { static const char err[]="cudaErrorStartupFailure "; return err; }

return NULL;
}
