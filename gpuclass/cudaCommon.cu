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

/* Helper function;
   If it's passed a 5x1 mxINT64 it passes through,
   If it's passed a class with classname "GPU_Type" it returns the GPU_MemPtr property's array */
void getTagFromGPUType(const mxArray *gputype, int64_t *rettag)
{
mxClassID dtype = mxGetClassID(gputype);

/* Handle gpu tags straight off */
if(dtype == mxINT64_CLASS) {
  int64_t *x = (int64_t *)mxGetData(gputype);
  int j;
  for(j = 0; j < 5; j++) rettag[j] = x[j];
  return;
  }

mxArray *tag;

const char *cname = mxGetClassName(gputype);

/* If we were passed a GPU_Type, retreive the GPU_MemPtr element */
if(strcmp(cname, "GPU_Type") == 0) {
  tag = mxGetProperty(gputype, 0, "GPU_MemPtr");
  } else { /* Assume it's an ImogenArray or descendent and retreive the gputag property */
  tag = mxGetProperty(gputype, 0, "gputag");
  }

/* We made a fair effort. There is no dishonor in surrendering now. */
if(tag == NULL) {
  mexErrMsgTxt("cudaCommon: fatal, tried to get gpu src pointer from something not a gpu tag, GPU_Type class, or Imogen array");
  }

/* Copy data out */
int64_t *t = (int64_t *)mxGetData(tag);
int j; for(j = 0; j < 5; j++) rettag[j] = t[j]; 

}

void arrayMetadataToTag(ArrayMetadata *meta, int64_t *tag)
{
if(meta == NULL) {
  mexErrMsgTxt("arrayMetadataToTag: Fatal: meta was null");
  }
if(tag == NULL) {
  mexErrMsgTxt("arrayMetadataToTag: Fatal: tag was null");
  }

/* tag[0] = original pointer; */
tag[1] = meta->ndims;
tag[2] = meta->dim[0];
tag[3] = meta->dim[1];
tag[4] = meta->dim[2];

return;
}

/* Given the RHS, an array to return array size, and the set of array indexes to take *s from */
double **getGPUSourcePointers(const mxArray *prhs[], ArrayMetadata *metaReturn, int fromarg, int toarg)
{

  double **gpuPointers = (double **)malloc((1+toarg-fromarg) * sizeof(double *));
  int iter;

  int64_t tag[5]; getTagFromGPUType(prhs[fromarg], &tag[0]);
/*  printf("tag: %li %li %li %li %li\n", tag[0], tag[1], tag[2], tag[3], tag[4]);*/

  for(iter = 0; iter < 3; iter++) { metaReturn->dim[iter] = (int)tag[2+iter]; } // copy metadata out of first gpu*
  metaReturn->numel = metaReturn->dim[0]*metaReturn->dim[1]*metaReturn->dim[2];
  metaReturn->ndims = tag[1];

  for(iter = fromarg; iter <= toarg; iter++) {
    getTagFromGPUType(prhs[iter], &tag[0]);
    gpuPointers[iter-fromarg] = (double *)tag[0];
  }

return gpuPointers;
}

/* Creates destination array that the kernels write to; Returns the GPU memory pointer, and assigns the LHS it's passed */
double **makeGPUDestinationArrays(ArrayMetadata *amdRef, mxArray *retArray[], int howmany)
{

double **rvals = (double **)malloc(howmany*sizeof(double *));

int i;
mwSize dims[2]; dims[0] = 5; dims[1] = 1;

int64_t *rv;

for(i = 0; i < howmany; i++) {
  retArray[i] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);
  rv = (int64_t *)mxGetData(retArray[i]);

  cudaError_t fail = cudaMalloc((void **)&rv[0], amdRef->numel*sizeof(double));
  if(fail != cudaSuccess) {
    CHECK_CUDA_ERROR("In makeGPUDestinationArrays: malloc failed and I am sad.");
    }

  rv[1] = amdRef->ndims;
  rv[2] = amdRef->dim[0];
  rv[3] = amdRef->dim[1];
  rv[4] = amdRef->dim[2];
  rvals[i] = (double *)rv[0];
  }

//size_t tot,fre;

//cuMemGetInfo(&fre, &tot);
//printf("Now free: %u\n", fre);

return rvals;

}

// Takes the array at prhs[target] and overwrites its gputag with one matching newdims
// and returns the pointer to the new array
// You must have previously fetched the original memory pointer or it will be lost
double *replaceGPUArray(const mxArray *prhs[], int target, int *newdims)
{
mxClassID dtype = mxGetClassID(prhs[target]);
if(dtype != mxINT64_CLASS) mexErrMsgTxt("cudaCommon: fatal, tried to get gpu src pointer from something not a gpu tag.");


double *ret;
cudaError_t fail = cudaMalloc((void **)&ret, newdims[0]*newdims[1]*newdims[2]*sizeof(double));
if(fail != cudaSuccess) {
  printf("On array replace: %s\n", cudaGetErrorString(fail));
  CHECK_CUDA_ERROR("In replaceGPUArray: malloc failed and I am sad.");
  }

int64_t *tag = (int64_t *)mxGetData(prhs[target]);
tag[0] = (int64_t)ret;
if(newdims[2] > 1) { tag[1] = 3; } else { if(newdims[1] > 1) { tag[1] = 2; } else { tag[1] = 1; } }
tag[2] = newdims[0]; tag[3] = newdims[1]; tag[4] = newdims[2];

return ret;
}

// Just grab in.fieldA.fieldB
// Or in.fieldA if fieldB is blank
mxArray *derefXdotAdotB(const mxArray *in, char *fieldA, char *fieldB)
{
	if(fieldA == NULL) mexErrMsgTxt("In derefAdotBdotC: fieldA null!");

	mxArray *A; mxArray *B;
	mxClassID t0 = mxGetClassID(in);

	int snum = strlen("Failed to read field fieldA in X.A.B") + strlen(fieldA) + strlen(fieldB) + 10;
	char *estring = (char *)calloc(snum, sizeof(char));

	if(t0 == mxSTRUCT_CLASS) { // Get structure field from A
		A = mxGetField(in, 0, fieldA);

		if(A == NULL) {
			sprintf(estring,"Failed to get X.%s", fieldA);
			mexErrMsgTxt(estring);
		}
	} else { // Get field struct A and fail if it doesn't work
		A = mxGetProperty(in, 0, fieldA);

		if(A == NULL) {
			sprintf(estring,"Failed to get X.%s", fieldA);
			mexErrMsgTxt(estring);
		}
	}

	if(fieldB != NULL) {
		t0 = mxGetClassID(A);
		if(t0 == mxSTRUCT_CLASS) {
			B = mxGetField(A, 0, fieldB);
		} else {
			B = mxGetProperty(A, 0, fieldB);
		}
	
		sprintf(estring,"Failed to get X.%s.%s", fieldA, fieldB);
		if(B == NULL) mexErrMsgTxt(estring);
	
		return B;
	} else {
		return A;
	}
}

// Two utility extensions of the deref above, to grab either the
// first element of a presumed double array or the first N elements
double derefXdotAdotB_scalar(const mxArray *in, char *fieldA, char *fieldB)
{
mxArray *u = derefXdotAdotB(in, fieldA, fieldB);

if(u != NULL) return *mxGetPr(u);

return NAN;
}

void derefXdotAdotB_vector(const mxArray *in, char *fieldA, char *fieldB, double *x, int N)
{
mxArray *u = derefXdotAdotB(in, fieldA, fieldB);

int Nmax = mxGetNumberOfElements(u);
N = (N > Nmax) ? Nmax : N;

double *d = mxGetPr(u);
int i;

if(d != NULL) {
	for(i = 0; i < N; i++) { x[i] = d[i]; } // Give it the d.
} else {
	for(i = 0; i < N; i++) { x[i] = NAN; }
}

}

void getTiledLaunchDims(int *dims, dim3 *tileDim, dim3 *halo, dim3 *blockdim, dim3 *griddim)
{
	blockdim->x = tileDim->x + halo->x;
	blockdim->y = tileDim->y + halo->y;
	blockdim->z = tileDim->z + halo->z;

	griddim->x = dims[0] / tileDim->x; griddim->x += ((griddim->x * tileDim->x) < dims[0]);
	griddim->y = dims[1] / tileDim->y; griddim->y += ((griddim->y * tileDim->y) < dims[1]);
	griddim->z = dims[2] / tileDim->z; griddim->z += ((griddim->z * tileDim->z) < dims[2]);
}

void checkCudaLaunchError(cudaError_t E, dim3 blockdim, dim3 griddim, ArrayMetadata *a, int i, char *srcname, char *fname, int lname)
{
if(E == cudaSuccess) return;

printf("CUDA error fired in at %s:%i: error %s -> %s\n", fname, lname, errorName(E), cudaGetErrorString(E));
printf("Description passed: %s\n", srcname);
printf("Array info: dims=<%i %i %i>; numel=%i. Received the integer %i.\n", a->dim[0], a->dim[1], a->dim[2], a->numel, i);
printf("Block and grid dims: <%i %i %i>, <%i %i %i>\n", blockdim.x, blockdim.y, blockdim.z, griddim.x, griddim.y, griddim.z);
mexErrMsgTxt("Forcing stop due to CUDA error");
}

void checkCudaError(char *where, char *fname, int lname)
{
cudaError_t epicFail = cudaGetLastError();
if(epicFail == cudaSuccess) return;

printf("cudaCheckError was non-success when polled at %s (%s:%i): %s -> %s\n", where, fname, lname, errorName(epicFail), cudaGetErrorString(epicFail));
mexErrMsgTxt("Forcing stop due to CUDA error");
}

void printdim3(char *name, dim3 dim)
{ printf("dim3 %s is [%i %i %i]\n", name, dim.x, dim.y, dim.z); }

void printgputag(char *name, int64_t *tag)
{ printf("gputag %s is [*=%lu dims=%lu size=(%lu %lu %lu)]\n", name, tag[0], tag[1], tag[2], tag[3], tag[4]); }

#define NOM(x) if(E == x) { static const char err[]=#x; return err; }

const char *errorName(cudaError_t E)
{
/* Written the stupid way because nvcc is idiotically claims these are all "case inaccessible" if it's done with a switch.

WRONG, asshole! */
// OM...
NOM(cudaSuccess)
NOM(cudaErrorMissingConfiguration)
NOM(cudaErrorMemoryAllocation)
NOM(cudaErrorInitializationError)
NOM(cudaErrorLaunchFailure)
NOM(cudaErrorPriorLaunchFailure)
NOM(cudaErrorLaunchTimeout)
NOM(cudaErrorLaunchOutOfResources)
NOM(cudaErrorInvalidDeviceFunction)
NOM(cudaErrorInvalidConfiguration)
NOM(cudaErrorInvalidDevice)
NOM(cudaErrorInvalidValue)
NOM(cudaErrorInvalidPitchValue)
NOM(cudaErrorInvalidSymbol)
NOM(cudaErrorMapBufferObjectFailed)
NOM(cudaErrorUnmapBufferObjectFailed)
NOM(cudaErrorInvalidHostPointer)
NOM(cudaErrorInvalidDevicePointer)
NOM(cudaErrorInvalidTexture)
NOM(cudaErrorInvalidTextureBinding)
NOM(cudaErrorInvalidChannelDescriptor)
NOM(cudaErrorInvalidMemcpyDirection)
NOM(cudaErrorAddressOfConstant)
NOM(cudaErrorTextureFetchFailed)
NOM(cudaErrorTextureNotBound)
NOM(cudaErrorSynchronizationError)
NOM(cudaErrorInvalidFilterSetting)
NOM(cudaErrorInvalidNormSetting)
NOM(cudaErrorMixedDeviceExecution)
NOM(cudaErrorCudartUnloading)
NOM(cudaErrorUnknown)
NOM(cudaErrorNotYetImplemented)
NOM(cudaErrorMemoryValueTooLarge)
NOM(cudaErrorInvalidResourceHandle)
NOM(cudaErrorNotReady)
NOM(cudaErrorInsufficientDriver)
NOM(cudaErrorSetOnActiveProcess)
NOM(cudaErrorInvalidSurface)
NOM(cudaErrorNoDevice)
NOM(cudaErrorECCUncorrectable)
NOM(cudaErrorSharedObjectSymbolNotFound)
NOM(cudaErrorSharedObjectInitFailed)
NOM(cudaErrorUnsupportedLimit)
NOM(cudaErrorDuplicateVariableName)
NOM(cudaErrorDuplicateTextureName)
NOM(cudaErrorDuplicateSurfaceName)
NOM(cudaErrorDevicesUnavailable)
NOM(cudaErrorInvalidKernelImage)
NOM(cudaErrorNoKernelImageForDevice)
NOM(cudaErrorIncompatibleDriverContext)
NOM(cudaErrorPeerAccessAlreadyEnabled)
NOM(cudaErrorPeerAccessNotEnabled)
NOM(cudaErrorDeviceAlreadyInUse)
NOM(cudaErrorProfilerDisabled)
NOM(cudaErrorProfilerNotInitialized)
NOM(cudaErrorProfilerAlreadyStarted)
NOM(cudaErrorProfilerAlreadyStopped)
/*cudaErrorAssert
cudaErrorTooManyPeers
cudaErrorHostMemoryAlreadyRegistered
cudaErrorHostMemoryNotRegistered
cudaErrorOperatingSystem*/
NOM(cudaErrorStartupFailure)
// ... NOM, ASSHOLE!
return NULL;
}

