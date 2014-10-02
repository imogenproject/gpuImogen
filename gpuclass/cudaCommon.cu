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
if(halo < 0) { // check it is sane to clone
	if(halo != PARTITION_CLONED) return false; // if it's actually marked as cloned and not just FUBAR

	if(x[partitionDir-1] != 1) return false;
}
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
if(P >= m->nGPUs) {
	char bugstring[256];
	sprintf(bugstring, "Fatal: Requested partition %i but only %i GPUs in use.", P, m->nGPUs);
	mexErrMsgTxt(bugstring);
}

// If the array is marked as cloned, the "partition" is simply the whole array
if(m->haloSize == PARTITION_CLONED) {
	sub[0] = sub[1] = sub[2] = 0;
	sub[3] = m->dim[0];
	sub[4] = m->dim[1];
	sub[5] = m->dim[2];
	return;
}

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
int i;
mg->numel = 1;

mg->dim[0] = tag[GPU_TAG_DIM0];
mg->numel *= mg->dim[0];
mg->dim[1] = tag[GPU_TAG_DIM1];
mg->numel *= mg->dim[1];
mg->dim[2] = tag[GPU_TAG_DIM2];
mg->numel *= mg->dim[2];

mg->haloSize     = tag[GPU_TAG_HALO];
mg->partitionDir = tag[GPU_TAG_PARTDIR];
mg->nGPUs        = tag[GPU_TAG_NGPUS];

int sub[6];

tag += 6;
for(i = 0; i < mg->nGPUs; i++) {
    mg->deviceID[i]  = (int)tag[2*i];
    mg->devicePtr[i] = (double *)tag[2*i+1];
    // Many elementwise funcs only need numel, so avoid having to do this every time
    calcPartitionExtent(mg, i, sub);
    mg->partNumel[i] = sub[3]*sub[4]*sub[5];
    }
for(; i < MAX_GPUS_USED; i++) {
    mg->deviceID[i]  = -1;
    mg->devicePtr[i] = 0x0;
    mg->partNumel[i] = 0;
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
tag[GPU_TAG_DIM0] = mg->dim[0];
tag[GPU_TAG_DIM1] = mg->dim[1];
tag[GPU_TAG_DIM2] = mg->dim[2];
tag[GPU_TAG_HALO] = mg->haloSize;
tag[GPU_TAG_PARTDIR] = mg->partitionDir;
tag[GPU_TAG_NGPUS] = mg->nGPUs;
int i;
for(i = 0; i < mg->nGPUs; i++) {
    tag[6+2*i]   = (int64_t)mg->deviceID[i];
    tag[6+2*i+1] = (int64_t)mg->devicePtr[i];
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

int sub[6];

// clone skeleton,
for(i = 0; i < N; i++) {
    m[i]       = *skeleton;
    // but all "derived" qualities need to be reset
    m[i].numel = m[i].dim[0]*m[i].dim[1]*m[i].dim[2];

    // allocate new memory
    for(j = 0; j < skeleton->nGPUs; j++) {
        cudaSetDevice(m[i].deviceID[j]);
        m[i].devicePtr[j] = 0x0;

        // Check this, because the user may have merely set .haloSize = PARTITION_CLONED
        calcPartitionExtent(m+i, j, sub);
        m[i].partNumel[j] = sub[3]*sub[4]*sub[5];

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

void pullMGAPointers( MGArray *m, int N, int i, double **dst)
{
	int x;
	for(x = 0; x < N; x++) { dst[x] = m[x].devicePtr[i]; }
}

template<MGAReductionOperator OP>
__global__ void cudaClonedReducer(double *a, double *b, int numel);
template<MGAReductionOperator OP>
__global__ void cudaClonedReducerQuad(double *a, double *b, double *c, double *d, int numel);

// Reduce a cloned array in the cloned direction via (sum, product, max, min)
// Then emits the result back to the input arrays
// so if *a is { [x] [y] [z] } on 3 devices,
// this would end with *a = { [w] [w] [w] } where w is the reduction of [x y z].
// if target is not null, puts the output data in target instead
int reduceClonedMGArray(MGArray *a, MGAReductionOperator op)
{
	if(a->haloSize != PARTITION_CLONED) return 0;
	// Copying shit over and then reducing gains us nothing over letting UVA handle
	// it transparently, so we're just gonna run with it.
	int i;

	dim3 gridsize; gridsize.x = 32; gridsize.y = gridsize.z = 1;
	dim3 blocksize; blocksize.x = 256; blocksize.y = blocksize.z = 1;

	switch(a->nGPUs) {
	case 1: break; // nofin to do
	case 2: // single reduce
		cudaSetDevice(a->deviceID[0]);
		CHECK_CUDA_ERROR("cudaSetDevice()");
		switch(op) {
		case OP_SUM: cudaClonedReducer<OP_SUM><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->partNumel[0]); break;
		case OP_PROD: cudaClonedReducer<OP_PROD><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->partNumel[0]); break;
		case OP_MAX: cudaClonedReducer<OP_MAX><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->partNumel[0]); break;
		case OP_MIN: cudaClonedReducer<OP_MIN><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->partNumel[0]); break;
		}
		CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, "clone reduction for 2 GPUs");
		break;
	case 3: // two reduces, serially
		cudaSetDevice(a->deviceID[0]);
		CHECK_CUDA_ERROR("cudaSetDevice()");
		switch(op) {
				case OP_SUM: cudaClonedReducer<OP_SUM><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->partNumel[0]); break;
				case OP_PROD: cudaClonedReducer<OP_PROD><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->partNumel[0]); break;
				case OP_MAX: cudaClonedReducer<OP_MAX><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->partNumel[0]); break;
				case OP_MIN: cudaClonedReducer<OP_MIN><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->partNumel[0]); break;
				}
		CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, "clone reduction for 3 GPUs, first call");
		cudaSetDevice(a->deviceID[0]);
		CHECK_CUDA_ERROR("cudaSetDevice()");
		switch(op) {
				case OP_SUM: cudaClonedReducer<OP_SUM><<<32, 256>>>(a->devicePtr[0], a->devicePtr[2], a->partNumel[0]); break;
				case OP_PROD: cudaClonedReducer<OP_PROD><<<32, 256>>>(a->devicePtr[0], a->devicePtr[2], a->partNumel[0]); break;
				case OP_MAX: cudaClonedReducer<OP_MAX><<<32, 256>>>(a->devicePtr[0], a->devicePtr[2], a->partNumel[0]); break;
				case OP_MIN: cudaClonedReducer<OP_MIN><<<32, 256>>>(a->devicePtr[0], a->devicePtr[2], a->partNumel[0]); break;
				}
		CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, "clone reduction for 3 GPUs, second call");
		break;
	case 4: // three reduces, 2 parallel then 1 more
		cudaSetDevice(a->deviceID[0]);
		CHECK_CUDA_ERROR("cudaSetDevice()");
		switch(op) {
				case OP_SUM: cudaClonedReducerQuad<OP_SUM><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->devicePtr[2], a->devicePtr[3], a->partNumel[0]); break;
				case OP_PROD: cudaClonedReducerQuad<OP_PROD><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->devicePtr[2], a->devicePtr[3], a->partNumel[0]); break;
				case OP_MAX: cudaClonedReducerQuad<OP_MAX><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->devicePtr[2], a->devicePtr[3], a->partNumel[0]); break;
				case OP_MIN: cudaClonedReducerQuad<OP_MIN><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->devicePtr[2], a->devicePtr[3], a->partNumel[0]); break;
				}

		CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, "clone reduction for 4 GPUs using quadreducer");
		break;
	default: return -1;
	}

	// Now drop the result back to the other cloned partitions
	for(i = 1; i < a->nGPUs; i++) {
		cudaMemcpy((void *)a->devicePtr[i], (void *)a->devicePtr[0], a->partNumel[i]*sizeof(double), cudaMemcpyDeviceToDevice);
		CHECK_CUDA_ERROR("Copying after cloned partition reduce.");
	}


	return 0;
}

template<MGAReductionOperator OP>
__global__ void cudaClonedReducer(double *a, double *b, int numel)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int step_i = blockDim.x*gridDim.x;

	for(; i < numel; i+= step_i) {
		switch(OP) {
		case OP_SUM: { a[i] += b[i]; } break;
		case OP_PROD: { a[i] *= b[i]; } break;
		case OP_MAX: {a[i] = (a[i] > b[i]) ? a[i] : b[i]; } break;
		case OP_MIN: {a[i] = (a[i] < b[i]) ? a[i] : b[i]; } break;
		}
	}
}

template<MGAReductionOperator OP>
__global__ void cudaClonedReducerQuad(double *a, double *b, double *c, double *d, int numel)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int step_i = blockDim.x*gridDim.x;

	double u, v, w, x;

	for(; i < numel; i+= step_i) {
		u = a[i]; v = b[i];
		w = c[i]; x = d[i];

		switch(OP) {
		case OP_SUM: { a[i] = u + v + w + x; } break;
		case OP_PROD: { a[i] = u*v*w*x; } break;
		case OP_MAX: {
			u = (u > v) ? u : v;
			w = (w > x) ? w : x;
			a[i] = (u > w) ? u : w; } break;
		case OP_MIN: {
			u = (u < v) ? u : v;
			w = (w < x) ? w : x;
			a[i] = (u < w) ? u : w; } break;
		}
	}
}

// Necessary when non-point operations have been performed in the partition direction
void exchangeMGArrayHalos(MGArray *a, int n)
{
int i, j;
dim3 blocksize, gridsize;

for(i = 0; i < n; i++) {
	// Skip this if it's a cloned partition
	if(a->haloSize == PARTITION_CLONED) { a++; continue; }

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

// The Z halo exchange function is simply done by a pair of memcopies

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

/* Given a list of indices into the global array, returns a * to a new array listing only */
// that subset which exists on partition i of the given MGArray.
//int partitionIndexTranslate(void)

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

void dropMexError(char *excuse, char *infile, int atline)
{
char *turd = (char *)malloc(strlen(excuse) + strlen(infile) + 32);

sprintf(turd, "The crap hit the fan: %s\nLocation was %s:%i", excuse, infile, atline);

mexErrMsgTxt(turd);

free(turd);

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


