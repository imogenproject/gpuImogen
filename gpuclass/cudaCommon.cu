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

#define SYNCBLOCK 16

bool sanityCheckTag(const mxArray *tag)
{
int64_t *x = (int64_t *)mxGetData(tag);

int tagsize = mxGetNumberOfElements(tag);

// This cannot possibly be valid
if(tagsize < 6) return false;

int nx = x[0];
int ny = x[1];
int nz = x[2];

// Null array OK
if((nx == 0) && (ny == 0) && (nz == 0) && (tagsize == 6)) return true;

if((nx < 0) || (ny < 0) || (nz < 0)) return false;

int halo         = x[3];
int partitionDir = x[4];
int nDevs        = x[5];

// Some basic does-this-make-sense
if(nDevs < 1) return false;
if(nDevs > MAX_GPUS_USED) return false;
if(halo < 0) return false;
if((partitionDir < 1) || (partitionDir > 3)) return false;

// Require there be exactly the storage required
int requisiteNumel = 6 + 2*nDevs;
if(tagsize != requisiteNumel) return false;

int j;
x += 6;
// CUDA device #s are nonnegative, and it is nonsensical that there would be over 16 of them.
for(j = 0; j < nDevs; j++) {
    if((x[2*j] < 0) || (x[2*j] >= 16)) {
        return false;
        }
    }

return true;
}

// Fills sub with [x0 y0 z0 nx ny nz] of partition P of multi-GPU array m
void calcPartitionExtent(MGArray *m, int P, int *sub)
{
if(P >= m->nGPUs) mexErrMsgTxt("Fatal: request for partition # > # of GPUs used");

int direct = m->partitionDir - 1; // zero-indexed direction

int i;
// We get the whole array transverse to the partition direction
for(i = 0; i < 3; i++) {
  if(i == direct) continue;
  sub[i  ] = 0;
  sub[3+i] = m->dim[i];
}

sub += direct;
int alpha = m->dim[direct] / m->nGPUs;

// "raw" offset of P*alpha, extent of alpha
sub[0] = P*alpha;
sub[3] = alpha;
// Rightmost partition takes up any remainder slack
if(P == (m->nGPUs-1)) sub[3] = m->dim[direct] - P*alpha;

// If not the leftmost, move start left & add halo
if(P > 0)            { sub[0] -= m->haloSize; sub[3] += m->haloSize; }
// If not rightmost, extend halo right
if(P < (m->nGPUs-1))   sub[3] += m->haloSize;

}

// This does the "ugly" work of deciding what was passed and getting a hold of the raw data pointer
int64_t *getGPUTypeTag(const mxArray *gputype)
{
mxClassID dtype = mxGetClassID(gputype);

/* Handle gpu tags straight off */
if(dtype == mxINT64_CLASS) {
  bool sanity = sanityCheckTag(gputype);
  if(sanity == false) mexErrMsgTxt("cudaCommon: fatal, passed tag failed sanity test.");
  return  (int64_t *)mxGetData(gputype);
  }

mxArray *tag;
const char *cname = mxGetClassName(gputype);

/* If we were passed a GPU_Type, retreive the GPU_MemPtr element */
if(strcmp(cname, "GPU_Type") == 0) {
  tag = mxGetProperty(gputype, 0, "GPU_MemPtr");
  } else { /* Assume it's an ImogenArray or descendent and retreive the gputag property */
  tag = mxGetProperty(gputype, 0, "gputag");
  }

/* We have done all that duty required, there is no dishonor in surrendering */
if(tag == NULL) {
  mexErrMsgTxt("cudaCommon: fatal, tried to get gpu src pointer from something not a gpu tag, GPU_Type class, or Imogen array");
  }

bool sanity = sanityCheckTag(tag);
if(sanity == false)  mexErrMsgTxt("cudaCommon: fatal, passed tag failed sanity test.");
return (int64_t *)mxGetData(tag);

}

// SERDES routines
void deserializeTagToMGArray(int64_t *tag, MGArray *mg)
{
int k;
mg->numel = 1;

for(k = 0; k < 3; k++) {
    mg->dim[k] = tag[k];
    mg->numel *= mg->dim[k];
    }

mg->haloSize     = tag[3];
mg->nGPUs        = tag[4];
mg->partitionDir = tag[5];

int sub[6];

tag += 6;
for(k = 0; k < mg->nGPUs; k++) {
    mg->deviceID[k]  = (int)tag[2*k];
    mg->devicePtr[k] = (double *)tag[2*k+1];
    // Many elementwise funcs only need numel, so avoid having to do this every time
    calcPartitionExtent(mg, k, sub);
    mg->partNumel[k] = sub[3]*sub[4]*sub[5];
    }
for(; k < MAX_GPUS_USED; k++) {
    mg->deviceID[k]  = -1;
    mg->devicePtr[k] = 0x0;
    mg->partNumel[k] = 0;
    }

return;
}

/* Serialized tag form:
   [ Nx
     Ny
     Nz
     halo size on shared edges
     Direction to parition in [1 = x, 2 = y, 3 = x]
     N = # of GPU paritions
     device ID 0
     device memory* 0
     device ID 1
     device memory* 1
     (...)
     device ID N-1
     device memory* N-1]
*/
void serializeMGArrayToTag(MGArray *mg, int64_t *tag)
{
tag[0] = mg->dim[0];
tag[1] = mg->dim[1];
tag[2] = mg->dim[2];
tag[3] = mg->haloSize;
tag[4] = mg->partitionDir;
tag[5] = mg->nGPUs;
int k;
for(k = 0; k < mg->nGPUs; k++) {
    tag[6+2*k]   = (int64_t)mg->deviceID[k];
    tag[6+2*k+1] = (int64_t)mg->devicePtr[k];
    }

return;
}

// Helpers to easily access/create multiple arrays
int accessMGArrays(const mxArray *prhs[], int idxFrom, int idxTo, MGArray *mg)
{
int i;
prhs += idxFrom;

int64_t *tag;

for(i = 0; i < (idxTo + 1 - idxFrom); i++) {
    tag = getGPUTypeTag(prhs[i]);
    deserializeTagToMGArray(tag, &mg[i]);
    }

return 0;
}

MGArray *allocMGArrays(int N, MGArray *skeleton)
{

// Do some preliminaries,
MGArray *m = (MGArray *)malloc(N*sizeof(MGArray));

int i;
int j;

// clone skeleton,
for(i = 0; i < N; i++) {
    m[i] = *skeleton;

    // allocate new memory
    for(j = 0; j < skeleton->nGPUs; j++) {
        cudaSetDevice(m[i].deviceID[j]);
        m[i].devicePtr[j] = 0x0;
        cudaMalloc((void **)&m[i].devicePtr[j], m[i].partNumel[j]*sizeof(double));
        CHECK_CUDA_ERROR("createMGArrays: malloc");
    }
}

return m;
}

MGArray *createMGArrays(mxArray *plhs[], int N, MGArray *skeleton)
{
MGArray *m = allocMGArrays(N, skeleton);

int i;

mwSize dims[2]; dims[0] = 6+2*skeleton->nGPUs; dims[1] = 1;
int64_t *r;

// create Matlab arrays holding serialized form,
for(i = 0; i < N; i++) {
    plhs[i] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);
    r = (int64_t *)mxGetData(plhs[i]);
    serializeMGArrayToTag(m+i, r);
    }

// send back the MGArray structs.
return m;
}

// Necessary when non-point operations have been performed in the partition direction
void exchangeMGArrayHalos(MGArray *a, int n)
{
int i, j;
dim3 blocksize, gridsize;

for(i = 0; i < n; i++) {
    for(j = 0; j < a->nGPUs-1; j++) {
        switch(a->partitionDir) {
            case 1: {
                cudaSetDevice(a->deviceID[j]);
                blocksize.x = a->haloSize;
                blocksize.y = SYNCBLOCK;
                blocksize.z = 1;
                gridsize.x  = a->dim[1]/SYNCBLOCK; gridsize.x += (gridsize.x*SYNCBLOCK < a->dim[1]);
                gridsize.y = gridsize.z = 1;
                cudaMGHaloSyncX<<<gridsize, blocksize>>>(a->devicePtr[j], a->devicePtr[j+1], a->dim[0], a->dim[1], a->dim[2], a->haloSize);
            } break;
            case 2: {
                cudaSetDevice(a->deviceID[j]);
                blocksize.x = blocksize.y = SYNCBLOCK;
                blocksize.z = 1;
                gridsize.x = a->dim[0]/SYNCBLOCK; gridsize.x += (gridsize.x*SYNCBLOCK < a->dim[0]);
                gridsize.y = a->dim[2]/SYNCBLOCK; gridsize.y += (gridsize.y*SYNCBLOCK < a->dim[2]);
                cudaMGHaloSyncY<<<gridsize, blocksize>>>(a->devicePtr[j], a->devicePtr[j+1], a->dim[0], a->dim[1], a->dim[2], a->haloSize);
            } break;
            case 3: {
                int sub[6];
                calcPartitionExtent(a, j, sub);
                size_t halotile = a->dim[0]*a->dim[1];
                size_t byteblock = halotile*a->haloSize*sizeof(double);

                size_t L_halo = (sub[5] - a->haloSize)*halotile;
                size_t L_src  = (sub[5]-2*a->haloSize)*halotile;

		// Fill right halo with left's source
                cudaMemcpy((void *)a->devicePtr[j+1],
                           (void *)(a->devicePtr[j] + L_src), byteblock, cudaMemcpyDeviceToDevice);
                // Fill left halo with right's source
                cudaMemcpy((void *)(a->devicePtr[j] + L_halo),
                           (void *)(a->devicePtr[j+1]+halotile*a->haloSize), byteblock, cudaMemcpyDeviceToDevice);

            } break;

        }
    }

    a++;
}


}

// expect invocation with [h BLKy 1] threads and [ny/BLKy 1 1].rp blocks
__global__ void cudaMGHaloSyncX(double *L, double *R, int nx, int ny, int nz, int h)
{
int xL = nx - h + threadIdx.x;
int xR = threadIdx.x;

int jmp = nx*(threadIdx.y + blockIdx.x*blockDim.x);
if(jmp >= nx*ny) return;

L += jmp;
R += jmp;
jmp = nx*ny;

int i;
for(i = 0; i < nz; i++) {
    L[jmp+xL] = R[jmp+xR+3];
    R[jmp+xR] = L[jmp+xL-3];
    L += jmp; R += jmp;
}

}

/* Expect invocation with [BLKx BLKz 1] threads and [nx/BLKx nz/BLKz 1].rp blocks */
__global__ void cudaMGHaloSyncY(double *L, double *R, int nx, int ny, int nz, int h)
{
int x0 = threadIdx.x + blockIdx.x*blockDim.x;
int z0 = threadIdx.y + blockIdx.y*blockDim.y;

if((x0 >= nx) || (z0 >= nz)) return;

L += (x0 + nx*(ny-h+ny*z0)); // To the plus y extent
R += (x0 + nx*ny*z0);        // to the minus y extent

int i;
for(i = 0; i < h; i++) {
    L[i*nx]     = R[(i+3)*nx];
    L[(i-3)*nx] = R[i*nx];
}

}

// This is more easily done with a simple memcpy
//__global__ void cudaMGHaloSyncZ(double *L, double *R, int nx, int ny, int nz, int h)

/* Given a list of indices into the global array, returns a * to a new array listing only */
// that subset which exists on partition i of the given MGArray.
//int partitionIndexTranslate(void)


/********************************************************************/
/* Below this line is almost all obsolete ***************************/
/********************************************************************/

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

void checkCudaLaunchError(cudaError_t E, dim3 blockdim, dim3 griddim, MGArray *a, int i, char *srcname, char *fname, int lname)
{
if(E == cudaSuccess) return;

printf("Caught CUDA error at %s:%i: error %s -> %s\n", fname, lname, errorName(E), cudaGetErrorString(E));
printf("Code's description of what it just did: %s\n", srcname);

if(a == NULL) {
  printf("No MGArray passed.\n");
  mexErrMsgTxt("Forcing program stop.");
  return;
  }

printf("Information about rx'd MGArray*:\n");

char pdStrings[4] = "XYZ";
printf("\tdim = <%i %i %i>\n\thalo size = %i\n\tpartition direction=%c\n", a->dim[0], a->dim[1], a->dim[2], a->haloSize, pdStrings[a->partitionDir-1]);
printf("Array partitioned across %i devices: [", a->nGPUs);
int u;
for(u = 0; u < a->nGPUs; u++) {
  printf("%i%s", a->deviceID[u], u==(a->nGPUs-1) ? "]\n" : ", ");
  }

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


