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
#include "mpi_common.h"

/* THIS FUNCTION ACCEPTS THE FOLLOWING FLAGS VIA -D:
 * -DALLOCFREE_DEBUG
 *     Causes every call to MGA_allocArrays and MGA_delete to emit a message to stdout;
 *     A poor man's valgrind.
 */

#define SYNCBLOCK 16

__global__ void cudaMGHaloSyncX_p2p(double *L, double *R, int nxL, int nxR, int ny, int nz, int h);
__global__ void cudaMGHaloSyncY_p2p(double *L, double *R, int nx, int nyL, int nyR, int nz, int h);

template<int lr_rw>
__global__ void cudaMGA_haloXrw(double *phi, double *linear, int nx, int ny, int nz, int h);

template<int lr_rw>
__global__ void cudaMGA_haloYrw(double *phi, double *linear, int nx, int ny, int nz, int h);

/* Given an mxArray* that points to a GPU tag (specifically the uint64_t array, not the more
 * general types tolerated by the higher-level functions), checks whether it can pass
 * muster as a MGArray (i.e. one packed by serializeMGArrayToTag).
 */
bool sanityCheckTag(const mxArray *tag)
{
	int64_t *x = (int64_t *)mxGetData(tag);

	int tagsize = mxGetNumberOfElements(tag);

	// This cannot possibly be valid
	if(tagsize < GPU_TAG_LENGTH) return false;

	int nx = x[GPU_TAG_DIM0];
	int ny = x[GPU_TAG_DIM1];
	int nz = x[GPU_TAG_DIM2];

	// Null array OK
	if((nx == 0) && (ny == 0) && (nz == 0) && (tagsize == GPU_TAG_LENGTH)) return true;

	if((nx < 0) || (ny < 0) || (nz < 0)) return false;

	int halo         = x[GPU_TAG_HALO];
	int partitionDir = x[GPU_TAG_PARTDIR];
	int nDevs        = x[GPU_TAG_NGPUS];

	int permtag      = x[GPU_TAG_DIMPERMUTATION];

	int circlebits   = x[GPU_TAG_CIRCULARBITS];

	// Some basic does-this-make-sense
	if(nDevs < 1) {
		printf((const char *)"Tag indicates less than one GPU in use.\n");
		return false;
	}
	if(nDevs > MAX_GPUS_USED) {
		printf((const char *)"Tag indicates %i GPUs in use, current config only supports %i.\n", nDevs, MAX_GPUS_USED);
		return false;
	}
	if(halo < 0) { // not reasonable.
		return false;
	}

	if((permtag < 1) || (permtag > 6)) {
		if(permtag == 0) {
			// meh
		} else {
			printf((const char *)"Permutation tag is %i: Valid values are 1 (XYZ), 2 (XZY), 3 (YXZ), 4 (YZX), 5 (ZXY), 6 (ZYX)\n");
			return false;
		}
	}

	if((circlebits < 0) || (circlebits > 63)) {
		printf((const char *)"halo sharing bits have value %i, valid range is 0-63!\n", circlebits);
		return false;

	}

	if((partitionDir < 1) || (partitionDir > 3)) {
		printf((const char *)"Indicated partition direction of %i is not 1, 2, or 3.\n", partitionDir);
		return false;
	}

	// Require there be enough additional elements to hold the physical device pointers & cuda device IDs
	int requisiteNumel = GPU_TAG_LENGTH + 2*nDevs;
	if(tagsize != requisiteNumel) {
		printf((const char *)"Tag length is %i: Must be %i base + 2*nDevs = %i\n", tagsize, GPU_TAG_LENGTH, requisiteNumel);
		return false;
	}

	int j;
	x += GPU_TAG_LENGTH;
	// CUDA device #s are nonnegative, and it is nonsensical that there would be over 16 of them.
	for(j = 0; j < nDevs; j++) {
		if((x[2*j] < 0) || (x[2*j] >= MAX_GPUS_USED)) {
			return false;
		}
	}

	return true;
}

/* Write [x0 y0 z0 nx ny nz] to the first 6 elements of sub for partition P of the MGArray
 * pointed to by m.
 * DO NOT use ANY other function to compute MGA extents!!!
 */
void calcPartitionExtent(MGArray *m, int P, int *sub)
{
	if(P >= m->nGPUs) {
		char bugstring[256];
		sprintf(bugstring, (const char *)"Fatal: Requested partition %i but only %i GPUs in use.", P, m->nGPUs);
		mexErrMsgTxt(bugstring);
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

	// MultiGPU operation requires halos on both sides of the partitions for FDing operations
	if(m->nGPUs > 1) {
		if((m->addExteriorHalo != 0) || (P > 0)) {
			sub[0] -= m->haloSize; sub[3] += m->haloSize;
		}
		if((m->addExteriorHalo != 0) || (P < (m->nGPUs-1))) {
			sub[3] += m->haloSize;
		}
	}

}

/* Given an mxArray *X, it searches "all the places you'd expect" any function in Imogen to have
 * stored the uint64_t pointer to the tag itself. Specifically, if X is a:
 *   uint64_t class: returns mxGetData(X)
 *   GPU_Type class: returns mxGetData(mxGetProperty(X, 0, "GPU_MemPtr"));
 *   FluidArray    : returns mxGetData(mxGetProperty(X, 0, "gputag"));
 */
int getGPUTypeTag(const mxArray *gputype, int64_t **tagPointer)
{
	return getGPUTypeTagIndexed(gputype, tagPointer, 0);
}

/* Behaves as getGPUTypeTag, but can fetch outside of index zero. */
int getGPUTypeTagIndexed(const mxArray *gputype, int64_t **tagPointer, int mxarrayIndex)
{
	if(tagPointer == NULL) {
		PRINT_FAULT_HEADER;
		printf("input tag pointer was null!\n");
		PRINT_FAULT_FOOTER;
		return ERROR_NULL_POINTER;
	}
	tagPointer[0] = NULL;

	mxClassID dtype = mxGetClassID(gputype);

	/* Handle gpu tags straight off */
	if(dtype == mxINT64_CLASS) {
		bool sanity = sanityCheckTag(gputype);
		if(sanity == false) {
			PRINT_FAULT_HEADER;
			printf((const char *)"Failure to access GPU tag: Sanity check failed.\n");
			PRINT_FAULT_FOOTER;
			return ERROR_GET_GPUTAG_FAILED;
		}
		tagPointer[0] = (int64_t *)mxGetData(gputype);
		return SUCCESSFUL;
	}

	mxArray *tag;
	const char *cname = mxGetClassName(gputype);

	/* If we were passed a GPU_Type, retreive the GPU_MemPtr element */
	if(strcmp(cname, (const char *)"GPU_Type") == 0) {
		tag = mxGetProperty(gputype, mxarrayIndex, (const char *)"GPU_MemPtr");
	} else { /* Assume it's an ImogenArray or descendant and retrieve the gputag property */
		tag = mxGetProperty(gputype, mxarrayIndex, (const char *)"gputag");
	}

	/* We have done all that duty required, there is no dishonor in surrendering */
	if(tag == NULL) {
		PRINT_FAULT_HEADER;
		printf((const char *)"getGPUTypeTag was called with something that is not a gpu tag, or GPU_Type class, or ImogenArray class\nArgument order wrong?\n");
		PRINT_FAULT_FOOTER;
		return ERROR_CRASH;
	}

	bool sanity = sanityCheckTag(tag);
	if(sanity == false) {
		PRINT_FAULT_HEADER;
		printf((const char *)"Failure to access GPU tag: Sanity check failed.\n");
		PRINT_FAULT_FOOTER;
		return ERROR_GET_GPUTAG_FAILED;
	}
	tagPointer[0] = (int64_t *)mxGetData(tag);

	return SUCCESSFUL;
}

cudaStream_t *getGPUTypeStreams(const mxArray *fluidarray) {
	mxArray *streamptr  = mxGetProperty(fluidarray, 0, (const char *)"streamptr");

	return (cudaStream_t *)(*((int64_t *)mxGetData(streamptr)) );
}

// SERDES: Unpacks a uint64_t vector into an MGArray
int deserializeTagToMGArray(int64_t *tag, MGArray *mg)
{
	if(tag == NULL) {
			PRINT_FAULT_HEADER;
			printf((const char *)"input tag pointer was null!\n");
			PRINT_FAULT_FOOTER;
			return ERROR_NULL_POINTER;
		}
	int i;
	mg->numel = 1;

	mg->dim[0] = tag[GPU_TAG_DIM0];
	mg->numel *= mg->dim[0];
	mg->dim[1] = tag[GPU_TAG_DIM1];
	mg->numel *= mg->dim[1];
	mg->dim[2] = tag[GPU_TAG_DIM2];
	mg->numel *= mg->dim[2];
	mg->numSlabs = tag[GPU_TAG_DIMSLAB];

	mg->haloSize     = tag[GPU_TAG_HALO];
	mg->partitionDir = tag[GPU_TAG_PARTDIR];
	mg->nGPUs        = tag[GPU_TAG_NGPUS];

	mg->addExteriorHalo = tag[GPU_TAG_EXTERIORHALO];

	mg->permtag = tag[GPU_TAG_DIMPERMUTATION];
    MGA_permtagToNums(mg->permtag, &mg->currentPermutation[0]);

    mg->circularBoundaryBits = tag[GPU_TAG_CIRCULARBITS];

	int sub[6];

	tag += GPU_TAG_LENGTH;
	for(i = 0; i < mg->nGPUs; i++) {
		mg->deviceID[i]  = (int)tag[2*i];
		mg->devicePtr[i] = (double *)tag[2*i+1];
		// Many elementwise funcs only need numel, so avoid having to do this every time
		calcPartitionExtent(mg, i, sub);
		mg->partNumel[i] = sub[3]*sub[4]*sub[5];
		mg->slabPitch[i] = ROUNDUPTO(mg->partNumel[i]*sizeof(double), 256);
	}
	for(; i < MAX_GPUS_USED; i++) {
		mg->deviceID[i]  = -1;
		mg->devicePtr[i] = 0x0;
		mg->partNumel[i] = 0;
		mg->slabPitch[i] = 0;
	}

	return SUCCESSFUL;
}

// SERDES: Packs an MGArray into a uint64_t vector
void serializeMGArrayToTag(MGArray *mg, int64_t *tag)
{
	tag[GPU_TAG_DIM0]    = mg->dim[0];
	tag[GPU_TAG_DIM1]    = mg->dim[1];
	tag[GPU_TAG_DIM2]    = mg->dim[2];
	tag[GPU_TAG_DIMSLAB] = mg->numSlabs;
	tag[GPU_TAG_HALO]    = mg->haloSize;
	tag[GPU_TAG_PARTDIR] = mg->partitionDir;
	tag[GPU_TAG_NGPUS]   = mg->nGPUs;
	tag[GPU_TAG_EXTERIORHALO]   = mg->addExteriorHalo;
	tag[GPU_TAG_DIMPERMUTATION] = mg->permtag;
	tag[GPU_TAG_CIRCULARBITS] = mg->circularBoundaryBits;

	int i;
	for(i = 0; i < mg->nGPUs; i++) {
		tag[GPU_TAG_LENGTH+2*i]   = (int64_t)mg->deviceID[i];
		tag[GPU_TAG_LENGTH+2*i+1] = (int64_t)mg->devicePtr[i];
	}

	return;
}

/* Converts the MGArray's .permtag element which uniquely specifies the current in-memory layout
 * into p[3] such that p[i] gives the 'physical' direction which lies in the i direction
 * in memory, for convenience.
 * */
void MGA_permtagToNums(int permtag, int *p)
{
	if(p == NULL) return;

	switch(permtag) {
	case 1: p[0] = 1; p[1] = 2; p[2] = 3; break;
	case 2: p[0] = 1; p[1] = 3; p[2] = 2; break;
	case 3: p[0] = 2; p[1] = 1; p[2] = 3; break;
	case 4: p[0] = 2; p[1] = 3; p[2] = 1; break;
	case 5: p[0] = 3; p[1] = 1; p[2] = 2; break;
	case 6: p[0] = 3; p[1] = 2; p[2] = 1; break;
	}

}

/* Reverse of permtagToNums: Routines which rotate memory and alter the currentPermutation[]
 * should use this to make sure that .permtag, which actually represents it, is updated too
 */
int MGA_numsToPermtag(int *nums)
{
	if(nums == NULL) return -1;

	switch(nums[0]) {
	case 1: { // x first
		if((nums[1] == 2) && (nums[2] == 3)) return 1; // XYZ
		if((nums[1] == 3) && (nums[2] == 2)) return 2; // XZY
	} break;
	case 2: { // y first
		if((nums[1] == 1) && (nums[2] == 3)) return 3; // YXZ
		if((nums[1] == 3) && (nums[2] == 1)) return 4; // YXZ
	} break;
	case 3: { // z first
		if((nums[1] == 1) && (nums[2] == 2)) return 5; // ZXY
		if((nums[1] == 2) && (nums[2] == 1)) return 6; // ZYX
	} break;
	}

return 0;
}

int MGA_dir2memdir(int *perm, int dir)
{
	if(perm == NULL) return -1;

	if(perm[0] == dir) return 1;
	if(perm[1] == dir) return 2;
	if(perm[2] == dir) return 3;

	return -1;
}

/* Facilitates access to MGArrays stored in Imogen's Matlab structures:
 * the mxArray pointers prhs[i] for i spanning idxFrom to idxTo inclusive
 * are decoded into mg[i - idxFrom]. Such that if the Matlab call is
 *    matlabFoo(1, 2, gpuA, gpuB, gpuC)
 * then foo(const mxArray *prhs[], ...) should use
 *    MGA_accessMatlabArrays(prhs, 2, 4, x)
 * with the result that x[0] = gpuA, x[1] = gpuB, x[2] = gpuC.
 */
int MGA_accessMatlabArrays(const mxArray *prhs[], int idxFrom, int idxTo, MGArray *mg)
{

	int i;
	int returnCode = SUCCESSFUL;
	prhs += idxFrom;

	int64_t *tag;

	for(i = 0; i < (idxTo + 1 - idxFrom); i++) {
			returnCode = getGPUTypeTag(prhs[i], &tag);

			if(returnCode == SUCCESSFUL)
				returnCode = deserializeTagToMGArray(tag, &mg[i]);

			mg[i].matlabClassHandle = prhs[i]; // FIXME: This is a craptastic hack
			// I am too lazy to implement boundary condition data storage properly...
			mg[i].mlClassHandleIndex = 0;

			if(returnCode != SUCCESSFUL) break;
		}

	return CHECK_IMOGEN_ERROR(returnCode);
}

/* Facilitates access to vector GPU array arguments from Matlab. Such that if
 *   matlab>> x = [gpuA gpuB gpuC];
 *   matlab>> matlabFoo(1, x, stuff)
 * Then foo(const mxArray *prhs, ...) should use
 *   MGA_accessMatlabArrayVector(prhs[1], 0, 2, z)
 * with the result that z[0] = gpuA, z[1] = gpuB, z[2] = gpuC.
 */
int MGA_accessMatlabArrayVector(const mxArray *m, int idxFrom, int idxTo, MGArray *mg)
{

	int i;
	int returnCode = SUCCESSFUL;

	int64_t *tag;

	for(i = 0; i < (idxTo + 1 - idxFrom); i++) {
			returnCode = getGPUTypeTagIndexed(m, &tag, i);

			if(returnCode == SUCCESSFUL)
				returnCode = deserializeTagToMGArray(tag, &mg[i]);

			mg[i].matlabClassHandle = m; // FIXME: This is a craptastic hack
			// I am too lazy to implement boundary condition data storage properly...
			mg[i].mlClassHandleIndex = i;

			if(returnCode != SUCCESSFUL) break;
		}

	return CHECK_IMOGEN_ERROR(returnCode);
}

/* Given a pointer to a template array 'skeleton', returns a vector of N MGArrays whose
 * size and partitioning match that of skeleton. If skeleton is a slab referent, new arrays
 * are real arrays the size of one slab (numSlabs = 1).
 */
MGArray *MGA_allocArrays(int N, MGArray *skeleton)
{
	// Do some preliminaries,
	MGArray *m = (MGArray *)malloc(N*sizeof(MGArray));

	int i;
	int j;

	int sub[6];

	/* If we are passed a slab array (e.g. the second slab of a 5-slab set),
	 * allocate this array to be a single-slab array (i.e. assume that unless
	 * explicitly stated otherwise, "make new a new array like skeleton" means
	 * one slab element, not the whole thing.
	 */
	int nActualSlabs = skeleton->numSlabs;
	if(nActualSlabs <= 0) nActualSlabs = 1;

	// clone skeleton,
	for(i = 0; i < N; i++) {
		m[i]       = *skeleton;
		m[i].numSlabs = nActualSlabs;

		// but all "derived" qualities need to be reset
		m[i].numel = m[i].dim[0]*m[i].dim[1]*m[i].dim[2];

		// allocate new memory
		for(j = 0; j < skeleton->nGPUs; j++) {
			cudaSetDevice(m[i].deviceID[j]);
			m[i].devicePtr[j] = 0x0;

			// Check this, because the user may have merely set .haloSize = PARTITION_CLONED
			calcPartitionExtent(m+i, j, sub);
			m[i].partNumel[j] = sub[3]*sub[4]*sub[5];
			m[i].slabPitch[j] = ROUNDUPTO(m[i].partNumel[j]*sizeof(double), 256);

			/* Differs if we have slabs... */
			int64_t num2alloc = m[i].partNumel[j] * sizeof(double);
			if(m[i].numSlabs > 1) num2alloc = m[i].slabPitch[j];

			cudaMalloc((void **)&m[i].devicePtr[j], num2alloc);
			CHECK_CUDA_ERROR("MGA_createReturnedArrays: cudaMalloc");
		}
	}

#ifdef ALLOCFREE_DEBUG
printf((const char *)"============= MGA_allocArrays invoked\n");
printf((const char *)"Creating %i arrays\n", N);
printf((const char *)"Array ptr: %x\n", m);
for(i = 0; i < N; i++) {
	for(j = 0; j < m[i].nGPUs; j++) printf((const char *)"	Pointer %i: %x\n", m[i].deviceID[j], m[i].devicePtr[j]);
}
#endif

	return m;
}

int MGA_duplicateArray(MGArray **dst, MGArray *src)
{
	int status = SUCCESSFUL;
	if((src == NULL) || (dst == NULL)) {
		PRINT_FAULT_HEADER;
		printf("Null src argument passed to MGA_duplicateArray\n");
		PRINT_FAULT_FOOTER;
		return ERROR_NULL_POINTER;
	}
	if(dst[0] == NULL) {
		dst[0] = MGA_allocArrays(1, src);
	}

	int i;
	for(i = 0; i < src->nGPUs; i++) {
		cudaSetDevice(src->deviceID[i]);
		if(CHECK_CUDA_ERROR("Array duplicator:cudaSetDevice") != SUCCESSFUL) {
					status = ERROR_CRASH;
					break;
				}

		cudaError_t unhappy = cudaMemcpy((void *)dst[0]->devicePtr[i], (void *)src->devicePtr[i], src->partNumel[i]*sizeof(double), cudaMemcpyDeviceToDevice);
		if(CHECK_CUDA_ERROR("Array duplicator:cudaMemcpy") != SUCCESSFUL) {
			status = ERROR_CRASH;
			break;
		}
	}
	return status;
}

/* A convenient wrapper for returning data to Matlab:
 * Creates arrays in the style of MGA_allocArrays, but serializes them into the
 * first N elements of plhs[] as well before returning the C vector pointer.
 */
MGArray *MGA_createReturnedArrays(mxArray *plhs[], int N, MGArray *skeleton)
{
	MGArray *m = MGA_allocArrays(N, skeleton);

	int i;

	mwSize dims[2]; dims[0] = GPU_TAG_LENGTH+2*skeleton->nGPUs; dims[1] = 1;
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

/* Does exactly what it says: Accepts one MGArray pointer and
 * writes one serialized representation to plhs[0].
 */
void MGA_returnOneArray(mxArray *plhs[], MGArray *m)
{
	mwSize dims[2]; dims[0] = GPU_TAG_LENGTH+2*m->nGPUs; dims[1] = 1;
	int64_t *r;

	// create Matlab arrays holding serialized form,
	plhs[0] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);
	r = (int64_t *)mxGetData(plhs[0]);
	serializeMGArrayToTag(m, r);
}

/* Deallocates an MGArray by freeing all its devicePtr[] entries.
 * If it is passed a slab reference (numSlabs < 1), does nothing */
int MGA_delete(MGArray *victim)
{
	if(victim == NULL) {
		PRINT_FAULT_HEADER;
		printf("MGA_delete passed a null MGA to delete!\n");
		PRINT_FAULT_FOOTER;
		return ERROR_NULL_POINTER;
	}
	if(victim->numSlabs < 1) return SUCCESSFUL; // Ignore attempts to deallocate slab refs, this lets us pretend they're "normal"

	int returnCode = SUCCESSFUL;
	int j = 0;

#ifdef ALLOCFREE_DEBUG
printf((const char *)"MGA_delete invoked ==============\n");
printf((const char *)"Victim *: %x\n", victim);
for(j = 0; j < victim->nGPUs; j++) {
	printf((const char *)"	Device: %i, ptr %x\n", victim->deviceID[j], victim->devicePtr[j]);
}
fflush(stdout);
#endif

	for(j = 0; j<victim->nGPUs; j++){
		cudaSetDevice(victim->deviceID[j]);
		returnCode = CHECK_CUDA_ERROR((const char *)"In MGA_delete, setting device");

		cudaFree(victim->devicePtr[j]);
		if(returnCode == SUCCESSFUL) returnCode = CHECK_CUDA_ERROR((const char *)"In MGA_delete after free");

		if(returnCode != SUCCESSFUL) break;
	}

	return CHECK_IMOGEN_ERROR(returnCode);
}

/* Some routines still run in the mold of "we were passed N arrays so expect N pointers"
 * and this makes it simple enough to do this.
 */
void pullMGAPointers( MGArray *m, int N, int i, double **dst)
{
	int x;
	for(x = 0; x < N; x++) { dst[x] = m[x].devicePtr[i]; }
}

int3 makeInt3(int x, int y, int z) {
	int3 a; a.x = x; a.y = y; a.z = z; return a; }
int3 makeInt3(int *b) {
	int3 a; a.x = b[0]; a.y = b[1]; a.z = b[2]; return a; }
dim3 makeDim3(unsigned int x, unsigned int y, unsigned int z) {
	dim3 a; a.x = x; a.y = y; a.z = z; return a; }
dim3 makeDim3(unsigned int *b) {
	dim3 a; a.x = b[0]; a.y = b[1]; a.z = b[2]; return a; }
dim3 makeDim3(int *b) {
	dim3 a; a.x = (unsigned int)b[0]; a.y = (unsigned int)b[1]; a.z = (unsigned int)b[2]; return a; }

/* This function should only be used for debugging race conditions
 * It loops over q->deviceID[] and calls cudaDeviceSynchronize() for each. */
void MGA_sledgehammerSequentialize(MGArray *q)
{
	int i;
	for(i = 0; i < q->nGPUs; i++) {
		cudaSetDevice(q->deviceID[i]);
		cudaDeviceSynchronize();
	}
}


double cpu_reduceInitValue(MGAReductionOperator op)
{
	switch(op) {
	case MGA_OP_SUM:  return 0.0;
	case MGA_OP_PROD: return 1.0;
	case MGA_OP_MIN:  return 1e37;
	case MGA_OP_MAX:  return -1e37;
	}
	return NAN;
}

double cpu_reducePair(double A, double B, MGAReductionOperator op)
{
	switch(op) {
	case MGA_OP_SUM:  return A+B;
	case MGA_OP_PROD: return A*B;
	case MGA_OP_MIN:  return (A < B) ? A : B;
	case MGA_OP_MAX:  return (A > B) ? A : B;
	}
	return NAN;
}

// NVCC should optimize these to a single register load because they're called by a templated function only...
__device__ double cukern_reduceInitValue(MGAReductionOperator op)
{
	switch(op) {
	case MGA_OP_SUM:  return 0.0;
	case MGA_OP_PROD: return 1.0;
	case MGA_OP_MIN:  return 1e37;
	case MGA_OP_MAX:  return -1e37;
	}
	return NAN;
}

__device__ double cukern_reducePair(double A, double B, MGAReductionOperator op)
{
	switch(op) {
	case MGA_OP_SUM:  return A+B;
	case MGA_OP_PROD: return A*B;
	case MGA_OP_MIN:  return (A < B) ? A : B;
	case MGA_OP_MAX:  return (A > B) ? A : B;
	}
	return NAN;
}

template <MGAReductionOperator OPERATION>
__global__ void cukern_reduceScalar(double *phi, double *retvals, int n)
{
	unsigned int tix = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tix;

	__shared__ double W[256];

	double Wmax = -1e37;
	W[tix] = -1e37;
	if(tix == 0) retvals[blockIdx.x] = Wmax; // As a safety measure incase we return below

	if(x >= n) return; // If we're fed a very small array, this will be easy

	// Threads step through memory with a stride of (total # of threads), finphig the max in this space
	while(x < n) {
	  if(phi[x] > Wmax) Wmax = phi[x];
	  x += blockDim.x * gridDim.x;
	  }
	W[tix] = Wmax;

	x = 128;
	while(x > 16) {
		if(tix >= x) return;
		__syncthreads();
		W[tix] = cukern_reducePair(W[tix],W[tix+x], OPERATION);
		x=x/2;
	}

	__syncthreads();

	// We have one halfwarp (16 threads) remaining
	// Assume that warps behave SIMD-synchronously
	W[tix] = cukern_reducePair(W[tix],W[tix+16], OPERATION); if(tix >= 8) return;
	W[tix] = cukern_reducePair(W[tix],W[tix+8], OPERATION); if(tix >= 4) return;
	W[tix] = cukern_reducePair(W[tix],W[tix+4], OPERATION); if(tix >= 2) return;
	W[tix] = cukern_reducePair(W[tix],W[tix+2], OPERATION); if(tix) return;

	retvals[blockIdx.x] = cukern_reducePair(W[0],W[1], OPERATION);
}

/* MGA_localReduceScalar uses the GPU to compute the reduction of the array described by
 * 'in' to a single value and writes that to scalar[0].
 */
int MGA_localReduceScalar(MGArray *in, double *scalar, MGAReductionOperator operate)
{
	int returnCode;
	dim3 blocksize, gridsize;
	blocksize.x = 256; blocksize.y = blocksize.z = 1;

	gridsize.x = 32; // 8K threads ought to keep the bugger busy
	gridsize.y = gridsize.z =1;

	// Allocate gridsize elements of pinned memory per GPU
	// Results will be conveniently waiting on the CPU for us when we're done
	double *blockValues[in->nGPUs];

	int i;
	for(i = 0; i < in->nGPUs; i++) {
		cudaSetDevice(in->deviceID[i]);
		returnCode = CHECK_CUDA_ERROR((const char *)"calling cudaSetDevice()");
		if(returnCode != SUCCESSFUL) break;

		cudaMallocHost(&blockValues[i], gridsize.x * sizeof(double));
		returnCode = CHECK_CUDA_ERROR((const char *)"cudaMallocHost");
		if(returnCode != SUCCESSFUL) break;
		switch(operate) {
				case MGA_OP_SUM:  cukern_reduceScalar<MGA_OP_SUM><<<gridsize, blocksize>>>(in->devicePtr[i], blockValues[i], in->partNumel[i]); break;
				case MGA_OP_PROD: cukern_reduceScalar<MGA_OP_PROD><<<gridsize, blocksize>>>(in->devicePtr[i], blockValues[i], in->partNumel[i]); break;
				case MGA_OP_MAX:  cukern_reduceScalar<MGA_OP_MAX><<<gridsize, blocksize>>>(in->devicePtr[i], blockValues[i], in->partNumel[i]); break;
				case MGA_OP_MIN:  cukern_reduceScalar<MGA_OP_MIN><<<gridsize, blocksize>>>(in->devicePtr[i], blockValues[i], in->partNumel[i]); break;
				}
		returnCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, in, i, (const char *)"directionalMaxFinder(phi)");
		if(returnCode != SUCCESSFUL) break;
	}

	if(returnCode != SUCCESSFUL) return returnCode;

	// Since we get only 32*nGPUs elements back, not worth another kernel invocation
	double result = cpu_reduceInitValue(operate);
	int devCount = 0; // track which partition we're getting results from

	for(devCount = 0; devCount < in->nGPUs; devCount++) {
		cudaSetDevice(in->deviceID[devCount]);
		returnCode = CHECK_CUDA_ERROR((const char *)"cudaSetDevice()");
		if(returnCode != SUCCESSFUL) break;

		cudaDeviceSynchronize(); // FIXME: can use less restrictive form here?
		returnCode = CHECK_CUDA_ERROR((const char *)"cudaDeviceSynchronize()");
		if(returnCode != SUCCESSFUL) break;

		for(i = 0; i < gridsize.x; i++)
			result = cpu_reducePair(result, blockValues[devCount][i], operate);

		cudaFreeHost(blockValues[devCount]);
		returnCode = CHECK_CUDA_ERROR((const char *)"cudaFreeHost");
		if(returnCode != SUCCESSFUL) break;
	}

	scalar[0] = result;

	return returnCode;
}

/* The global reduction function accepts in addition a topology: After using the GPU to find
 * the local reduction, uses MPI to find the global value & stores it to scalar[0] for all
 * ranks participating in topology->comm.
 */
int MGA_globalReduceScalar(MGArray *in, double *scalar, MGAReductionOperator operate, ParallelTopology *topology)
{
	double nodeValue;
	int returnCode = MGA_localReduceScalar(in, &nodeValue, operate);
	if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;

	if(topology != NULL) {

		/* If parallel, now invoke MPI_Allreduce as well */
		MPI_Comm commune = MPI_Comm_f2c(topology->comm);
		int r0; MPI_Comm_rank(commune, &r0);
		int N; MPI_Comm_size(commune, &N);

		double globalValue;
		/* Perform the reduce */
		MPI_Allreduce((void *)&nodeValue, (void *)&globalValue, 1, MPI_DOUBLE, MGAReductionOperator_mga2mpi(operate), commune);

		scalar[0] = globalValue;
	} else {
		scalar[0] = nodeValue;
	}

	return SUCCESSFUL;
}

template <MGAReductionOperator OPERATION>
__global__ void cukern_ReduceX(double *phi, double *r, int nx);
template <MGAReductionOperator OPERATION>
__global__ void cukern_ReduceY(double *phi, double *r, int nx, int ny, int nz);
template <MGAReductionOperator OPERATION>
__global__ void cukern_ReduceZ(double *phi, double *r, int nx, int ny, int nz);

/* MGA_partitionReduceDimension is only meant to be called by MGA_localReduceDimension */
int MGA_partitionReduceDimension(MGArray *in, MGArray *out, MGAReductionOperator operate, int dir, int partition)
{
	int sub[6];
	calcPartitionExtent(in, partition, &sub[0]);

	dim3 blk, grid;

	cudaSetDevice(in->deviceID[partition]);

	/* If the partition already has size 1, just copy input to output. */
	if(sub[2+dir] == 1) {
		cudaMemcpyAsync(out->devicePtr[partition], in->devicePtr[partition], sizeof(double)*in->partNumel[partition], cudaMemcpyDeviceToDevice);
		return CHECK_CUDA_ERROR((const char *)"partition reduce shortcircuit via memcpy");
	}

	switch(dir) {
	case 1: {
		blk = makeDim3(32,1,1);
		grid = makeDim3(sub[4], sub[5], 1);
		cudaSetDevice(in->deviceID[partition]);
		switch(operate) {
		case MGA_OP_SUM:  cukern_ReduceX<MGA_OP_SUM> <<<blk, grid>>>(in->devicePtr[partition], out->devicePtr[partition], sub[3]); break;
		case MGA_OP_PROD: cukern_ReduceX<MGA_OP_PROD><<<blk, grid>>>(in->devicePtr[partition], out->devicePtr[partition], sub[3]); break;
		case MGA_OP_MAX:  cukern_ReduceX<MGA_OP_MAX> <<<blk, grid>>>(in->devicePtr[partition], out->devicePtr[partition], sub[3]); break;
		case MGA_OP_MIN:  cukern_ReduceX<MGA_OP_MIN> <<<blk, grid>>>(in->devicePtr[partition], out->devicePtr[partition], sub[3]); break;
		}
	} break;
	case 2: {
		blk = makeDim3(32, 8, 1);
		grid = makeDim3(ROUNDUPTO(sub[3],32)/32, ROUNDUPTO(sub[5],blk.y)/blk.y, 1);
		cudaSetDevice(in->deviceID[partition]);
		switch(operate) {
		case MGA_OP_SUM:  cukern_ReduceY<MGA_OP_SUM> <<<blk, grid>>>(in->devicePtr[partition], out->devicePtr[partition], sub[3],sub[4],sub[5]); break;
		case MGA_OP_PROD: cukern_ReduceY<MGA_OP_PROD><<<blk, grid>>>(in->devicePtr[partition], out->devicePtr[partition], sub[3],sub[4],sub[5]); break;
		case MGA_OP_MAX:  cukern_ReduceY<MGA_OP_MAX> <<<blk, grid>>>(in->devicePtr[partition], out->devicePtr[partition], sub[3],sub[4],sub[5]); break;
		case MGA_OP_MIN:  cukern_ReduceY<MGA_OP_MIN> <<<blk, grid>>>(in->devicePtr[partition], out->devicePtr[partition], sub[3],sub[4],sub[5]); break;
		}
	} break;
	case 3: {
		blk = makeDim3(16, 16, 1);
		grid = makeDim3(ROUNDUPTO(sub[3],blk.x)/blk.x, ROUNDUPTO(sub[4],blk.y)/blk.y, 1);
		cudaSetDevice(in->deviceID[partition]);
		switch(operate) {
		case MGA_OP_SUM:  cukern_ReduceZ<MGA_OP_SUM> <<<blk, grid>>>(in->devicePtr[partition], out->devicePtr[partition], sub[3],sub[4],sub[5]); break;
		case MGA_OP_PROD: cukern_ReduceZ<MGA_OP_PROD><<<blk, grid>>>(in->devicePtr[partition], out->devicePtr[partition], sub[3],sub[4],sub[5]); break;
		case MGA_OP_MAX:  cukern_ReduceZ<MGA_OP_MAX> <<<blk, grid>>>(in->devicePtr[partition], out->devicePtr[partition], sub[3],sub[4],sub[5]); break;
		case MGA_OP_MIN:  cukern_ReduceZ<MGA_OP_MIN> <<<blk, grid>>>(in->devicePtr[partition], out->devicePtr[partition], sub[3],sub[4],sub[5]); break;
		}
	} break;
	}

	return CHECK_CUDA_LAUNCH_ERROR(grid, blk, in, dir, (const char *)"Simple partition reduction function");

}

/* Invoke with blocks of 32 threads, and an [NY NZ 1] grid:
 * given size(phi) = [nx, gridDim.x, gridDim.y]
 * and   size(r)   = [gridDim.x, gridDim.y],
 * does
 *    r(blockIdx.x,blockIdx.y) <- OPERATION(phi(:,blockIdx.x,blockIdx.y)
 */
template <MGAReductionOperator OPERATION>
__global__ void cukern_ReduceX(double *phi, double *r, int nx)
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	int z = blockIdx.y;
	int ny = gridDim.x;

	__shared__ double W[32];

	double Q = cukern_reduceInitValue(OPERATION);

	if(x >= nx) return;

	phi += x + nx*(y+ny*z);

	while(x < nx) {
		Q = cukern_reducePair(Q, *phi, OPERATION);
		x += 32;
		phi += 32;
	}

	W[threadIdx.x] = Q;

	x = 16;
	int tix = threadIdx.x;
	if(tix >= x) return;

	/* This is relevant if in future block > 1 warp */
	while(x > 16) {
		if(tix >= x) return;
		__syncthreads();
		if(W[tix+x] > W[tix]) W[tix] = W[tix+x];
	        x=x/2;
	}

	__syncthreads();

	// We have one halfwarp (16 threads) remaining, proceed synchronously on assumption of warp-level SIMD synchronicity
	if(W[tix+16] > W[tix]) W[tix] = W[tix+16]; if(tix >= 8) return;
	if(W[tix+8] > W[tix]) W[tix] = W[tix+8]; if(tix >= 4) return;
	if(W[tix+4] > W[tix]) W[tix] = W[tix+4]; if(tix >= 2) return;
	if(W[tix+2] > W[tix]) W[tix] = W[tix+2]; if(tix) return;

	/* last guy out, please turn off the lights */
	r[y+ny*z] = (W[1] > W[0]) ? W[1] : W[0];
}

/* Invoke with blocks of [32 A] threads and [ceil(NX/32), ceil(NZ/A) 1 1 ] grid
 * given size(phi) = [nx ny nz]
 * and   size(r)   = [nx nz]
 * does
 *     r(x, z) <- OPERATION(phi(x,:,z))
 */
template <MGAReductionOperator OPERATION>
__global__ void cukern_ReduceY(double *phi, double *r, int nx, int ny, int nz)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = 0;
	int z = threadIdx.y + blockIdx.y*blockDim.y;

	if(x >= nx) return;
	if(z >= nz) return;
	phi += x + nx*ny*z;

	double Q = cukern_reduceInitValue(OPERATION);
	while(y < ny) {
		Q = cukern_reducePair(Q, phi[nx*ny], OPERATION);
		y++;
	}

	r[x+nx*z] = Q;
}

/* Invoke with [A B 1] block and [C D 1] grid such that
 *   AC >= nx
 *   BD >= ny
 * Given size(phi) = [nx ny nz] and size(r) = [nx ny 1], does
 *    r(x,y) <- OPERATION(phi(x,y,:))
 */
template <MGAReductionOperator OPERATION>
__global__ void cukern_ReduceZ(double *phi, double *r, int nx, int ny, int nz)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	if(x >= nx) return;
	if(y >= ny) return;
	phi += x + nx*y;
	r   += x + nx*y;

	double Q = cukern_reduceInitValue(OPERATION);

	int z = 0;
	int step = nx*ny;
	for(z = 0; z < nz; z++) {
		Q = cukern_reducePair(Q, *phi, OPERATION);
		phi += step;
	}

	*r = Q;
}

/* MGA_localReduceDimension operates such that
 *    out[...,dir=0,...] = REDUCE(in[...,:,...])
 *    out.dim[dir-1] will equal 1.
 * i.e. reduction to scalar is done to every vector of elements in the 'dir' direction.
 *
 * If the in->partitionDir == dir, the answer stored to partition 'partitionOnto.'
 *    If this is the case and 'redistribute' is set, this data is copied to all other partitions before return.
 * If in->partitionDir != dir, 'partitionOnto' and 'redistribute' are irrelevant.
 *
 * If out[0] is nonnull and the correct size, its contents are overwritten with the output array.
 * If out[0] is nonnull but is an incorrect size, ERROR_INVALID_ARGS occurs.
 * If out[0] is null, an output array of the correct size is allocated. */
int MGA_localReduceDimension(MGArray *in, MGArray **out, MGAReductionOperator operate, int dir, int partitionOnto, int redistribute)
{
	int returnCode = SUCCESSFUL;
	int i;

	MGArray clone = *in;

	// Determine what size the reduced array should be
	if(dir == in->partitionDir) {
		clone.haloSize = 0;
		clone.dim[dir-1] = in->nGPUs; // to flatten
	} else {
		clone.haloSize = 0;
		clone.dim[dir-1] = 1; //
	}

	// check or allocate
	if(out[0] != NULL) {
		if(out[0]->dim[0] != clone.dim[0]) returnCode = ERROR_INVALID_ARGS;
		if(out[0]->dim[1] != clone.dim[1]) returnCode = ERROR_INVALID_ARGS;
		if(out[0]->dim[2] != clone.dim[2]) returnCode = ERROR_INVALID_ARGS;
		if(returnCode != SUCCESSFUL) {
			PRINT_FAULT_HEADER;
			printf((const char *)"out[0] was not null, but the passed MGArray** is of inappropriate dimensions.\nCannot safely free it & overwrite: Must return error.\n");
			PRINT_FAULT_FOOTER;
			return returnCode;
		}
	} else {
		out[0] = MGA_allocArrays(1, &clone);
	}

	// Call per-partition reductions
	for(i = 0; i < in->nGPUs; i++) {
		returnCode = MGA_partitionReduceDimension(in, out[0], operate, dir, i);

		if(returnCode != SUCCESSFUL) {
			return CHECK_IMOGEN_ERROR(returnCode);
		}
	}

	if(dir == in->partitionDir) {
		// reduce across partitions
		returnCode = MGA_reduceAcrossDevices(out[0], operate, redistribute);
	}

	return CHECK_IMOGEN_ERROR(returnCode);
}

/* First uses MGA_localReduceDimension to perform an accelerated reduce locally.
 * Then the result is copied to host memory and the MPI communicator in topology->dimcomm
 * corresponding to the direction is used to find the reduction across nodes,
 * then results transferred back to the GPU.
 */
int MGA_globalReduceDimension(MGArray *in, MGArray **out, MGAReductionOperator operate, int dir, int partitionOnto, int redistribute, ParallelTopology *topology)
{
	int returnCode = SUCCESSFUL;

	MGArray clone = *in;

	// Determine what size the reduced array should be
	if(dir == in->partitionDir) {
		clone.haloSize = 0;
		clone.dim[dir-1] = in->nGPUs; // flatten
	} else {
		clone.haloSize = 0;
		clone.dim[dir-1] = 1;
	}

	// check or allocate
	if(out[0] != NULL) {
		if(out[0]->dim[0] != clone.dim[0]) returnCode = ERROR_INVALID_ARGS;
		if(out[0]->dim[1] != clone.dim[1]) returnCode = ERROR_INVALID_ARGS;
		if(out[0]->dim[2] != clone.dim[2]) returnCode = ERROR_INVALID_ARGS;
		if(returnCode != SUCCESSFUL) {
			PRINT_FAULT_HEADER;
			printf((const char *)"out[0] was not null, but the passed MGArray** is of inappropriate dimensions.\nCannot safely free it & overwrite: Must return error.\n");
			PRINT_FAULT_FOOTER;
			return returnCode;
		}
	} else {
		out[0] = MGA_allocArrays(1, &clone);
	}

	/* All ranks flatten to 1D in reduce dimension in parallel */
	returnCode = MGA_localReduceDimension(in, out, operate, dir, partitionOnto, 0);
	if(returnCode != SUCCESSFUL) { return CHECK_IMOGEN_ERROR(returnCode); }

	/* Skip parallel reduction if no topology is forthcoming */
	if(topology != NULL) {

		/* Reverse silly memory ordering */
		int d = dir - 1;
		int dmax = topology->nproc[d];

		MPI_Comm commune = MPI_Comm_f2c(topology->comm);
		int r0; MPI_Comm_rank(commune, &r0);

		double *readBuf = NULL;
		if(dir == out[0]->partitionDir) {
		    returnCode = MGA_downloadArrayToCPU(out[0], &readBuf, partitionOnto);
		} else {
			returnCode = MGA_downloadArrayToCPU(out[0], &readBuf, -1);
		}
		if(returnCode != SUCCESSFUL) { return CHECK_IMOGEN_ERROR(returnCode); }

		double *writeBuf= (double *)malloc(out[0]->numel*sizeof(double));
		if(writeBuf == NULL) {
			PRINT_FAULT_HEADER;
			printf((const char *)"Failed to allocate write buffer memory!\n");
			PRINT_FAULT_FOOTER;
			return ERROR_NULL_POINTER;
		}

		int numToReduce = (dir == out[0]->partitionDir) ? out[0]->partNumel[partitionOnto] : out[0]->numel;
		MPI_Comm dircom = MPI_Comm_f2c(topology->dimcomm[dir-1]);

		MPI_Allreduce((void *)readBuf, (void *)writeBuf, numToReduce, MPI_DOUBLE, MGAReductionOperator_mga2mpi(operate), dircom);

		MPI_Barrier(dircom);

		int upPart = (dir == out[0]->partitionDir) ? partitionOnto : -1;
		returnCode = MGA_uploadArrayToGPU(writeBuf, out[0], upPart);

		if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

		if(redistribute && (dir == out[0]->partitionDir)) returnCode = MGA_distributeArrayClones(out[0], partitionOnto);

		free(readBuf); free(writeBuf);
		// This leaks if we encounter an error
		// But if that's the case we're crashing anyway...

		return CHECK_IMOGEN_ERROR(returnCode);
	} else {
		return SUCCESSFUL;
	}

}


template<MGAReductionOperator OP>
__global__ void cukern_TwoElementwiseReduce(double *a, double *b, int numel);
template<MGAReductionOperator OP>
__global__ void cudaClonedReducerQuad(double *a, double *b, double *c, double *d, int numel);

/* Requiring that each partition have equal # of elements, computes
 * a->devicePtr[0][i] = REDUCTION(a->devicePtr[0][i], a->devicePtr[1][i], ..., a->devicePtr[a->nGPUs][i])
 * for i in 0 to a->partNumel[0].
 *
 * if redistribute == 1, a->devicePtr[0] data is copied to partitions 1 through a->nGPUs as well. */
int MGA_reduceAcrossDevices(MGArray *a, MGAReductionOperator operate, int redistribute)
{
	int i;

	// Check that this operation is acceptable
	for(i = 1; i < a->nGPUs; i++) {
		if(a->partNumel[i] != a->partNumel[0]) return ERROR_INVALID_ARGS;
	}

	int eachPartSize = a->partNumel[0];

	int returnCode = SUCCESSFUL;

	dim3 gridsize; gridsize.x = 32; gridsize.y = gridsize.z = 1;
	dim3 blocksize; blocksize.x = 256; blocksize.y = blocksize.z = 1;

	double *B; double *C;

	switch(a->nGPUs) {
	case 1: break; // Well this was a waste of time
	case 2: // reduce(A,B)->A
		cudaSetDevice(a->deviceID[0]);
		returnCode = CHECK_CUDA_ERROR((const char *)"cudaSetDevice()");
		if(returnCode != SUCCESSFUL) break;
		cudaMalloc((void **)&B, eachPartSize*sizeof(double));
		returnCode = CHECK_CUDA_ERROR((const char *)"cudaMalloc()");
		if(returnCode != SUCCESSFUL) break;
		cudaMemcpy((void *)B, (void*)a->devicePtr[1], eachPartSize*sizeof(double), cudaMemcpyDeviceToDevice);
		returnCode = CHECK_CUDA_ERROR((const char *)"cudaMalloc()");
		if(returnCode != SUCCESSFUL) break;

		switch(operate) {
		case MGA_OP_SUM: cukern_TwoElementwiseReduce<MGA_OP_SUM><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		case MGA_OP_PROD: cukern_TwoElementwiseReduce<MGA_OP_PROD><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		case MGA_OP_MIN: cukern_TwoElementwiseReduce<MGA_OP_MIN><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		case MGA_OP_MAX: cukern_TwoElementwiseReduce<MGA_OP_MAX><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		}

		returnCode = CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, "clone reduction for 2 GPUs");
		if(returnCode != SUCCESSFUL) break;
		cudaFree(B);
		returnCode = CHECK_CUDA_ERROR((const char *)"cudaFree()");
		if(returnCode != SUCCESSFUL) break;
		break;
	case 3: // reduce(A,B)->A; reduce(A, C)->A
		cudaSetDevice(a->deviceID[0]);
		returnCode = CHECK_CUDA_ERROR((const char *)"cudaSetDevice()");
		if(returnCode != SUCCESSFUL) break;
		cudaMalloc((void **)&B, eachPartSize*sizeof(double));
		returnCode = CHECK_CUDA_ERROR((const char *)"cuda malloc");
		if(returnCode != SUCCESSFUL) break;
		cudaMalloc((void **)&C, eachPartSize*sizeof(double));
		returnCode = CHECK_CUDA_ERROR((const char *)"cuda malloc");
		if(returnCode != SUCCESSFUL) break;

		cudaMemcpy((void *)B, (void *)a->devicePtr[1], eachPartSize*sizeof(double), cudaMemcpyDeviceToDevice);
		returnCode = CHECK_CUDA_ERROR((const char *)"cuda memcpy");
		if(returnCode != SUCCESSFUL) break;
		cudaMemcpy((void *)C, (void *)a->devicePtr[2], eachPartSize*sizeof(double), cudaMemcpyDeviceToDevice);
		returnCode = CHECK_CUDA_ERROR((const char *)"cuda memcpy");
		if(returnCode != SUCCESSFUL) break;

		switch(operate) {
		case MGA_OP_SUM: cukern_TwoElementwiseReduce<MGA_OP_SUM><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		case MGA_OP_PROD: cukern_TwoElementwiseReduce<MGA_OP_PROD><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		case MGA_OP_MIN: cukern_TwoElementwiseReduce<MGA_OP_MIN><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		case MGA_OP_MAX: cukern_TwoElementwiseReduce<MGA_OP_MAX><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		}

		returnCode = CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, (const char *)"clone reduction for 3 GPUs, first call");
		if(returnCode != SUCCESSFUL) break;

		switch(operate) {
		case MGA_OP_SUM: cukern_TwoElementwiseReduce<MGA_OP_SUM><<<32, 256>>>(a->devicePtr[0], C, eachPartSize); break;
		case MGA_OP_PROD: cukern_TwoElementwiseReduce<MGA_OP_PROD><<<32, 256>>>(a->devicePtr[0], C, eachPartSize); break;
		case MGA_OP_MIN: cukern_TwoElementwiseReduce<MGA_OP_MIN><<<32, 256>>>(a->devicePtr[0], C, eachPartSize); break;
		case MGA_OP_MAX: cukern_TwoElementwiseReduce<MGA_OP_MAX><<<32, 256>>>(a->devicePtr[0], C, eachPartSize); break;
		}

		returnCode = CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, (const char *)"clone reduction for 3 GPUs, second call");
		if(returnCode != SUCCESSFUL) break;

		cudaFree(B);
		returnCode = CHECK_CUDA_ERROR((const char *)"cudaFree");
		if(returnCode != SUCCESSFUL) break;
		cudaFree(C);
		returnCode = CHECK_CUDA_ERROR((const char *)"cudaFree");
		if(returnCode != SUCCESSFUL) break;

		break;
	case 4: // {reduce(A,B)->A, reduce(C,D)->C}; reduce(A,C)->A
		// FIXME: This is broken right now...
		mexErrMsgTxt((const char *)"This is broken soz.");

		// On device 0, allocate storage for device 1 and copy device 1 partition to device 0
		cudaSetDevice(a->deviceID[0]);
		CHECK_CUDA_ERROR((const char *)"cudaSetDevice()");
		cudaMalloc((void **)&B, eachPartSize*sizeof(double));
		CHECK_CUDA_ERROR((const char *)"cudaMalloc");
		cudaMemcpyAsync((void *)B, (void *)a->devicePtr[1], eachPartSize*sizeof(double), cudaMemcpyDeviceToDevice);
		returnCode = CHECK_CUDA_ERROR((const char *)"cuda memcpy");
		if(returnCode != SUCCESSFUL) break;

		// Launch (A,B)->A reduction on device 0
		switch(operate) {
		case MGA_OP_SUM: cukern_TwoElementwiseReduce<MGA_OP_SUM><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		case MGA_OP_PROD: cukern_TwoElementwiseReduce<MGA_OP_PROD><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		case MGA_OP_MIN: cukern_TwoElementwiseReduce<MGA_OP_MIN><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		case MGA_OP_MAX: cukern_TwoElementwiseReduce<MGA_OP_MAX><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		}
		returnCode = CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, "clone reduction for 4 GPUs, first call (A,B)->A");
		if(returnCode != SUCCESSFUL) break;

		// On device 2, allocate storage for device 3 and copy device 3 partition to device 2
		cudaSetDevice(a->deviceID[2]);
		CHECK_CUDA_ERROR((const char *)"cudaSetDevice()");
		cudaMalloc((void **)&C, eachPartSize*sizeof(double));
		CHECK_CUDA_ERROR((const char *)"cudaMalloc");
		cudaMemcpyAsync((void *)C, (void *)a->devicePtr[3], eachPartSize*sizeof(double), cudaMemcpyDeviceToDevice);
		returnCode = CHECK_CUDA_ERROR("cuda memcpy");
		if(returnCode != SUCCESSFUL) break;

		// Launch (C,D)->C reduction on device 2
		switch(operate) {
		case MGA_OP_SUM: cukern_TwoElementwiseReduce<MGA_OP_SUM><<<32, 256>>>(a->devicePtr[2], C, eachPartSize); break;
		case MGA_OP_PROD: cukern_TwoElementwiseReduce<MGA_OP_PROD><<<32, 256>>>(a->devicePtr[2], C, eachPartSize); break;
		case MGA_OP_MIN: cukern_TwoElementwiseReduce<MGA_OP_MIN><<<32, 256>>>(a->devicePtr[2], C, eachPartSize); break;
		case MGA_OP_MAX: cukern_TwoElementwiseReduce<MGA_OP_MAX><<<32, 256>>>(a->devicePtr[2], C, eachPartSize); break;
		}
		returnCode = CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, "clone reduction for 4 GPUs, second call (C,D)->C");
		if(returnCode != SUCCESSFUL) break;

		// Copy C -> A for the final reduction
		cudaSetDevice(a->deviceID[0]);

		cudaMemcpyAsync((void *)B, (void *)a->devicePtr[2], eachPartSize*sizeof(double), cudaMemcpyDeviceToDevice);
		returnCode = CHECK_CUDA_ERROR("cuda memcpy");
		if(returnCode != SUCCESSFUL) break;

		switch(operate) {
		case MGA_OP_SUM: cukern_TwoElementwiseReduce<MGA_OP_SUM><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		case MGA_OP_PROD: cukern_TwoElementwiseReduce<MGA_OP_PROD><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		case MGA_OP_MIN: cukern_TwoElementwiseReduce<MGA_OP_MIN><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		case MGA_OP_MAX: cukern_TwoElementwiseReduce<MGA_OP_MAX><<<32, 256>>>(a->devicePtr[0], B, eachPartSize); break;
		}
		returnCode = CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, "clone reduction for 4 GPUs, 3rd call (A,C)->A");
		if(returnCode != SUCCESSFUL) break;

		cudaFree(B);
		returnCode = CHECK_CUDA_ERROR("cudaFree");
		if(returnCode != SUCCESSFUL) break;

		cudaSetDevice(a->deviceID[2]);
		cudaFree(C);
		returnCode = CHECK_CUDA_ERROR("cudaFree");
		if(returnCode != SUCCESSFUL) break;

		break;
	default: return -1;
	}

	if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

	if(redistribute)
		returnCode = MGA_distributeArrayClones(a, 0);

	return CHECK_IMOGEN_ERROR(returnCode);
}

/* If partition sizes are equal (as typical e.g. post-reduction),
 * copies the array on partition partitionFrom to the others resulting in
 * identical copies on each device.
 */
int MGA_distributeArrayClones(MGArray *cloned, int partitionFrom)
{
	int returnCode = SUCCESSFUL;
	int j;

	for(j = 1; j < cloned->nGPUs; j++) {
		if(cloned->partNumel[j] != cloned->partNumel[0]) return ERROR_INVALID_ARGS;
	}

	cudaSetDevice(cloned->deviceID[partitionFrom]);

	for(j = 0; j < cloned->nGPUs; j++) {
		if(j == partitionFrom) continue;

		cudaMemcpyAsync(cloned->devicePtr[j], cloned->devicePtr[partitionFrom], sizeof(double)*cloned->partNumel[partitionFrom], cudaMemcpyDeviceToDevice);
		returnCode = CHECK_CUDA_ERROR("MGA_distributeArrayClones");
		if(returnCode != SUCCESSFUL) break;
	}

	return CHECK_IMOGEN_ERROR(returnCode);
}


template<MGAReductionOperator OP>
__global__ void cukern_TwoElementwiseReduce(double *a, double *b, int numel)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int step_i = blockDim.x*gridDim.x;

	for(; i < numel; i+= step_i) {
		a[i] = cukern_reducePair(a[i], b[i], OP);
	}
}

/* Synchronizes the halo regions between data partitions of a[0] to a[n-1].
 * Does nothing if a[i].haloSize == 0 or a[i].nGPUs == 1. */
int MGA_exchangeLocalHalos(MGArray *a, int n)
{
	int i, j, jn, jp;
	dim3 blocksize, gridsize;
	int returnCode = SUCCESSFUL;

	for(i = 0; i < n; i++) {
		// Can't do this if there are no halos
		if(a->haloSize == 0) { break; }
		// Or there's only one partition to begin with
		if(a->nGPUs == 1) { break; }

		double *buffs[a->nGPUs * 4];
		int sub[6];

		calcPartitionExtent(a, 0, &sub[0]);

		// Acquire sufficient RW linear buffers to R and W both sides
		int numTransverse = a->partNumel[0] / sub[2+a->partitionDir];
		int numHalo = a->haloSize * numTransverse;

		if(a->partitionDir != PARTITION_Z) {
			for(j = 0; j < a->nGPUs; j++) {
				cudaSetDevice(a->deviceID[j]);
				CHECK_CUDA_ERROR("cudaSetDevice()");
				cudaMalloc((void **)&buffs[4*j], 4*numHalo*sizeof(double));
				returnCode = CHECK_CUDA_ERROR("cudaMalloc");
				if(returnCode != SUCCESSFUL) break;
				buffs[4*j+1] = buffs[4*j] + 1*numHalo;
				buffs[4*j+2] = buffs[4*j] + 2*numHalo;
				buffs[4*j+3] = buffs[4*j] + 3*numHalo;
			}

			// Fetch current partition's halo to linear strips, letting jn denote next and jp denote previous
			for(j = 0; j < a->nGPUs; j++) {
				jn = (j+1) % a->nGPUs;
				jp = (j - 1 + a->nGPUs) % a->nGPUs;

				// If addExteriorHalo is set, we behave circularly
				// This is appropriate if e.g. we have only one MPI rank in the partitioned direction.

				// If there are N>1 MPI ranks in the U direction and we are partitioned in U,
				// We do not handle these boundaries & leave them to MPI (cudaHaloExchange)

				// At first glance full-circular here isn't a problem (after all, the MPI exchange will just overwrite
				// this, right?). However, IF cudaHaloExchange is involved, then outside-MGA things (i.e.
				// Matlab) will be aware of the halo because it was added by GIS, not MGA.
				// Then our use of the halo here will corrupt visible data.

				// In particular, it will corrupt the calculation of boundary conditions!
				if(a->addExteriorHalo || (j > 0)) {
					returnCode = MGA_partitionHaloToLinear(a, j, a->partitionDir, 0, 0, a->haloSize, &buffs[4*j+0]);
					if(returnCode != SUCCESSFUL) break;
				}
				if(a->addExteriorHalo || (j < (a->nGPUs-1))) {
					returnCode = MGA_partitionHaloToLinear(a, j, a->partitionDir, 1, 0, a->haloSize, &buffs[4*j+1]);
					if(returnCode != SUCCESSFUL) break;
				}
			}

//MGA_sledgehammerSequentialize(a);
			// Transfer linear strips
			for(j = 0; j < a->nGPUs; j++) {
				jn = (j+1) % a->nGPUs; jp = (j - 1 + a->nGPUs) % a->nGPUs;
				if(a->addExteriorHalo || (j > 0)) {
					cudaMemcpyPeer(buffs[4*jp+3], a->deviceID[jp], buffs[4*j], a->deviceID[j], numHalo * sizeof(double));
					returnCode = CHECK_CUDA_ERROR("cudaMemcpyPeer");
					if(returnCode != SUCCESSFUL) break;
				}
				if(a->addExteriorHalo || (j < (a->nGPUs-1))) {
					cudaMemcpyPeer(buffs[4*jn+2], a->deviceID[jn], buffs[4*j+1], a->deviceID[j], numHalo * sizeof(double));
					returnCode = CHECK_CUDA_ERROR("cudaMemcpyPeer");
					if(returnCode != SUCCESSFUL) break;
				}

			}
//MGA_sledgehammerSequentialize(a);
			// Dump the strips back to halo
			for(j = 0; j < a->nGPUs; j++) {
				jn = (j+1) % a->nGPUs; jp = (j - 1 + a->nGPUs) % a->nGPUs;
				if(a->addExteriorHalo || (j > 0)) {
					returnCode = MGA_partitionHaloToLinear(a, jp, a->partitionDir, 1, 1, a->haloSize, &buffs[4*jp+3]);
					if(returnCode != SUCCESSFUL) break;
				}
				if(a->addExteriorHalo || (j < (a->nGPUs-1))) {
					returnCode = MGA_partitionHaloToLinear(a, jn, a->partitionDir, 0, 1, a->haloSize, &buffs[4*jn+2]);
					if(returnCode != SUCCESSFUL) break;
				}
			}

			// Let go of temp memory
			for(j = 0; j < a->nGPUs; j++) {
				cudaSetDevice(a->deviceID[j]);
				CHECK_CUDA_ERROR("cudaSetDevice");
				cudaFree(buffs[4*j]);
				returnCode = CHECK_CUDA_ERROR("cudaFree");
				if(returnCode != SUCCESSFUL) break;
			}
			if(returnCode != SUCCESSFUL) break;

		} else {
			/* Z halos are delightful, we simply copy some already-linearly-contiguous blocks
			 * of memory back and forth. the partition halo call would *work* but we can short-circuit
			 * pointless copying this way.
			 */

			for(j = 0; j < a->nGPUs; j++) {
				cudaSetDevice(a->deviceID[j]);
				calcPartitionExtent(a, j, sub);
				jn = (j+1) % a->nGPUs; // Next partition

				size_t halotile = a->dim[0]*a->dim[1];
				size_t byteblock = halotile*a->haloSize*sizeof(double);

				size_t L_halo = (sub[5] - a->haloSize)*halotile;
				size_t L_src  = (sub[5]-2*a->haloSize)*halotile;

				// Fill right halo with left's source
				cudaMemcpy((void *)a->devicePtr[jn],
						(void *)(a->devicePtr[j] + L_src), byteblock, cudaMemcpyDeviceToDevice);
				returnCode = CHECK_CUDA_ERROR("cudaMemcpy");
				if(returnCode != SUCCESSFUL) break;

				// Fill left halo with right's source
				cudaMemcpy((void *)(a->devicePtr[j] + L_halo),
						(void *)(a->devicePtr[jn]+halotile*a->haloSize), byteblock, cudaMemcpyDeviceToDevice);
				returnCode = CHECK_CUDA_ERROR("cudaMemcpy");
				if(returnCode != SUCCESSFUL) break;

				cudaDeviceSynchronize();

			}

		}

		a++;

	}

	return CHECK_IMOGEN_ERROR(returnCode);
}


int MGA_wholeFaceHaloNumel(MGArray *a, int direction, int h)
{
if(a == NULL) DROP_MEX_ERROR("In MGA_faceHaloNumel sanity checks: a is NULL!\n");

int q = 0;

if(a->partitionDir == direction) {
	q = MGA_partitionHaloNumel(a, 0, direction, h);
} else {
	int i;
	for(i = 0; i < a->nGPUs; i++) {
		q += MGA_partitionHaloNumel(a, i, direction, h);
	}
}

return q;

}

/* From the MGArray pointed to by 'a', will read to CPU (writehalo == 0) or upload to gpu
 * (writehalo == 1) the lowest-index (rightside == 0) or highest-index (rightside == 1)
 * face of 'a' in the 'direction' direction (x=1,y=2,z=3) from linear[0].
 * If linear[0] is null, allocates storage into it.
 * If linear[0] is not null, it must point to sufficient storage.
 *
 * FIXME: This routine is potentially dangerous when called to fetch halos for outside-MGA purposes
 * FIXME: Reason: it assumes that ranks A and B have identical partitioning, in which case the metadata
 * FIXME: associated with MGA_partitionHaloToLinear output will be the same (i.e. partition i on rank B
 * FIXME: will have the same size, halo size and index permutation as partition i on rank A)
 */
int MGA_wholeFaceToLinear(MGArray *a, int direction, int rightside, int writehalo, int h, double **linear)
{

	int returnCode = SUCCESSFUL;
	if(direction == a->partitionDir) {
		int part = 0;
		if(rightside) part = a->nGPUs - 1;

		returnCode = MGA_partitionHaloToLinear(a, part, direction, rightside, writehalo, h, linear);
	} else { // Fetch all halo partitions
		int q = 0;
		int ctr;
		for(ctr = 0; ctr < a->nGPUs; ctr++) {
			double *ptmp = linear[0] + q;
			returnCode = MGA_partitionHaloToLinear(a, ctr, direction, rightside, writehalo, 3, &ptmp);
			if(returnCode != SUCCESSFUL) break;
			q += MGA_partitionHaloNumel(a, ctr, direction, 3);
		}
	}

	return CHECK_IMOGEN_ERROR(returnCode);
}


/* Determines how many linear memory elements will be required to store the given partition/direction
 * halo of 'a'
 */
int MGA_partitionHaloNumel(MGArray *a, int partition, int direction, int h)
{
	// Sanity checks!
	if(partition < 0) DROP_MEX_ERROR("MGA_partitionHaloNumel sanity checks: negative partition id!");
	if(a == NULL) DROP_MEX_ERROR("In MGA_partitionHaloNumel sanity checks: crap, a == NULL!");
	if(partition >= a->nGPUs) DROP_MEX_ERROR("In MGA_partitionHaloNumel sanity checks: crap, partition > # GPUs!");
	if(direction < 1) DROP_MEX_ERROR("In MGA_partitionHaloNumel sanity checks: direction < 1. Did you accidently use XYZ==012?");
	if(direction > 3) DROP_MEX_ERROR("In MGA_partitionHaloNumel sanity checks: direction > 3?");
	if(h < 0) DROP_MEX_ERROR("In MGA_partitionHaloNumel sanity checks: halo size h < 0?");

	int sub[6];
	calcPartitionExtent(a, partition, &sub[0]);

	int haloTransverse = a->partNumel[partition] / sub[2+direction];
	int haloNumel = haloTransverse * h;

	return haloNumel;
}


/* Fetches the indicated face of a partition's cube to a linear swatch of memory,
 * suitable for memcpy or MPI internode halo exchange
 */
int MGA_partitionHaloToLinear(MGArray *a, int partition, int direction, int right, int toHalo, int h, double **linear)
{
	int returnCode = SUCCESSFUL;
	cudaSetDevice(a->deviceID[partition]);
	CHECK_CUDA_ERROR("cudaSetDevice");

	int sub[6];
	calcPartitionExtent(a, partition, &sub[0]);

	int haloNumel = MGA_partitionHaloNumel(a, partition, direction, h);
	int haloTransverse = haloNumel / h;


	if(linear[0] == NULL) {
		cudaMalloc((void **)linear, 2*haloNumel*sizeof(double));
		returnCode = CHECK_CUDA_ERROR((const char *)"cudaMalloc()");
		if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
	}

	dim3 blocksize, gridsize;

	switch(direction) {
	case 1: {
		blocksize.x = a->haloSize;
		blocksize.y = SYNCBLOCK;
		blocksize.z = (sub[5] > 1) ? 8 : 1;

		gridsize.x  = ROUNDUPTO(a->dim[1], SYNCBLOCK)/SYNCBLOCK;
		gridsize.y  = 1; gridsize.z = 1;
		switch(right + 2*toHalo) {
		/* left read */
		case 0: cudaMGA_haloXrw<0><<<gridsize, blocksize>>>(a->devicePtr[partition] , *linear, sub[3], sub[4], sub[5], h); break;
		/* left write */
		case 1: cudaMGA_haloXrw<1><<<gridsize, blocksize>>>(a->devicePtr[partition] , *linear, sub[3], sub[4], sub[5], h); break;
		/* left write */
		case 2: cudaMGA_haloXrw<2><<<gridsize, blocksize>>>(a->devicePtr[partition] , *linear, sub[3], sub[4], sub[5], h); break;
		/* right write */
		case 3: cudaMGA_haloXrw<3><<<gridsize, blocksize>>>(a->devicePtr[partition] , *linear, sub[3], sub[4], sub[5], h); break;
		default: returnCode = ERROR_CRASH;
		}
		if(returnCode == ERROR_CRASH) break;
		returnCode = CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, right + 2*toHalo, (const char *)"cudaMGA_haloXrw");
		break;
	}
	case 2: {
		blocksize.x = blocksize.y = SYNCBLOCK;
		blocksize.z = 1;
		gridsize.x  = a->dim[0]/SYNCBLOCK; gridsize.x += (gridsize.x*SYNCBLOCK < a->dim[0]);
		gridsize.y  = a->dim[2]/SYNCBLOCK; gridsize.y += (gridsize.y*SYNCBLOCK < a->dim[2]);
		switch(right + 2*toHalo) {
		case 0: cudaMGA_haloYrw<0><<<gridsize, blocksize>>>(a->devicePtr[partition] , *linear, sub[3], sub[4], sub[5], h); break;
		case 1: cudaMGA_haloYrw<1><<<gridsize, blocksize>>>(a->devicePtr[partition] , *linear, sub[3], sub[4], sub[5], h); break;
		case 2: cudaMGA_haloYrw<2><<<gridsize, blocksize>>>(a->devicePtr[partition] , *linear, sub[3], sub[4], sub[5], h); break;
		case 3: cudaMGA_haloYrw<3><<<gridsize, blocksize>>>(a->devicePtr[partition] , *linear, sub[3], sub[4], sub[5], h); break;
		default: returnCode = ERROR_CRASH;
		}
		if(returnCode == ERROR_CRASH) break;
		returnCode = CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, right + 2*toHalo, (const char *)"cudaMGA_haloYrw");
		break;
	}

	case 3: {
		switch(right + 2*toHalo) {
		case 0: cudaMemcpy((void *)linear[0], (void *)(a->devicePtr[partition] + haloNumel),                   haloNumel*sizeof(double), cudaMemcpyDeviceToDevice); break;
		case 1: cudaMemcpy((void *)linear[0], (void *)(a->devicePtr[partition] + (sub[5]-2*h)*haloTransverse), haloNumel*sizeof(double), cudaMemcpyDeviceToDevice); break;
		case 2: cudaMemcpy((void *)linear[0], (void *)(a->devicePtr[partition]),                               haloNumel*sizeof(double), cudaMemcpyDeviceToDevice); break;
		case 3: cudaMemcpy((void *)linear[0], (void *)(a->devicePtr[partition] + (sub[5]-h)*haloTransverse),   haloNumel*sizeof(double), cudaMemcpyDeviceToDevice); break;
		default: returnCode = ERROR_CRASH;
		}
		if(returnCode == ERROR_CRASH) break;
		returnCode = CHECK_CUDA_ERROR("cudamemcpy");
		break;
	}

	}

return CHECK_IMOGEN_ERROR(returnCode);
}

/* expect invocation with [4*roundup(h/4) BLKy A] threads and [ny/BLKy B 1].rp blocks with "arbitrary" A and B
 * given thread index t.[xyz] block index b.[xyz] and grid size g.[xyz], then consider:
 * x0 = nxL - 2*h + t.x; x1 = t.x;
 * y0 = t.y + BLKy*b.y; z0 = t.z + A*b.y
 * copy from L[x0 + nxL*(y0 + ny*z0)] to R[x1 + nxR*(y0 + ny*z0)]
 * copy from R[x1 + h + nxR*(y0 + ny*z0)] to L[x0 + h + nxL*(y0 + ny*z0)]
 *
 * Extract common subfactors: jump L < L + nxL*(y0 + ny*z0) + x0, R < R + nxR*(y0 + ny*z0) + x1,
 * check y0 < ny, then equations simplify to
 * iterate (k = z0; k < nz; k+=blockIdx.z*blockDim.z)
 *    copy from L[0] to R[0]
 *    copy from R[h] to L[h]
 *    L += nxL*ny*g.y; R =+ nxR*ny*g.y;

 */
__global__ void cudaMGHaloSyncX_p2p(double *L, double *R, int nxL, int nxR, int ny, int nz, int h)
{
	int y0 = threadIdx.y + blockDim.y*blockIdx.x;
	if(y0 >= ny) return;
	int z0 = threadIdx.z + blockDim.z*blockIdx.y;

	/* This will generate unaligned addresses, yes I'm sorry, DEAL WITH IT */
	L += nxL*(y0 + ny*z0) + nxL - 2*h + threadIdx.x;
	R += nxR*(y0 + ny*z0) + threadIdx.x;

	int k;
	int hz = blockDim.z*gridDim.y;
	for(k = z0; k < nz; k+= hz) { /* This implicitly contains: if(z0 >= nz) { return; } */
		// read enough data, for sure
		R[0] = L[0];
		L[h] = R[h];

		L   += nxL*ny*hz;
		R   += nxR*ny*hz;
	}

}

// FIXME: And this ny on both sides, also goddamnit.
/* Expect invocation with [BLKx BLKz 1] threads and [nx/BLKx nz/BLKz 1].rp blocks */
__global__ void cudaMGHaloSyncY_p2p(double *L, double *R, int nx, int nyL, int nyR, int nz, int h)
{
	int x0 = threadIdx.x + blockIdx.x*blockDim.x;
	int z0 = threadIdx.y + blockIdx.y*blockDim.y;

	if((x0 >= nx) || (z0 >= nz)) return;

	L += (x0 + nx*(nyL-2*h + nyL*z0)); // To the plus y extent
	R += (x0 + nx*nyR*z0);        // to the minus y extent

	int i;
	for(i = 0; i < h; i++) {
		L[(i+h)*nx]     = R[(i+h)*nx];
		R[i*nx] = L[i*nx];
	}

}

/* FIXME: These kernels are NOT particularly efficient
 * FIXME: But they account for very little time vs actual compute kernels
 */

/* bit 0 = 0: left; bit 0 = 1: right
 * bit 1 = 0: read; bit 1 = 1: write to phi's halo
 */
template<int lr_rw>
__global__ void cudaMGA_haloXrw(double *phi, double *linear, int nx, int ny, int nz, int h)
{
	int y0 = threadIdx.y + blockDim.y*blockIdx.x;
	if(y0 >= ny) return;
	int z0 = threadIdx.z + blockDim.z*blockIdx.y;

	phi += nx*(y0 + ny*z0) + threadIdx.x;
	linear += threadIdx.x + h*(y0+ny*z0);

	switch(lr_rw) {
	case 0: /* left read   */ phi += h; break;
	case 1: /* right read  */ phi += nx - 2*h; break;
	case 2: /* left write  */ break;
	case 3: /* right write */ phi += nx - h; break;
	}

	int k;
	int hz = blockDim.z*gridDim.y;
	for(k = z0; k < nz; k+= hz) { /* This implicitly contains: if(z0 >= nz) { return; } */
		if(lr_rw & 2) {
			phi[0] = linear[0];
		} else {
			linear[0] = phi[0];
		}

		phi    += nx*ny*hz;
		linear += h*ny*hz;
	}

}

/* bit 0 = 0: left; bit 0 = 1: right
 * bit 1 = 0: read; bit 1 = 1: write to phi's halo
 */
template<int lr_rw>
__global__ void cudaMGA_haloYrw(double *phi, double *linear, int nx, int ny, int nz, int h)
{
	int x0 = threadIdx.x + blockIdx.x*blockDim.x;
	int z0 = threadIdx.y + blockIdx.y*blockDim.y;


	if((x0 >= nx) || (z0 >= nz)) return;

	phi    += x0 + nx*ny*z0;
	linear += x0 + nx*h*z0;

	switch(lr_rw) {
	case 0: /* left read   */ phi += nx*h; break;
	case 1: /* right read  */ phi += nx*(ny - 2*h); break;
	case 2: /* left write  */ break;
	case 3: /* right write */ phi += nx*(ny - h); break;
	}


	int i;
	for(i = 0; i < h; i++) {
		if(lr_rw & 2) {
			phi[0] = linear[0];
		} else {
			linear[0] = phi[0];
		}
		phi += nx;
		linear += nx;
	}

}

/* Copes the MGArray pointed to by g into the space at p[0].
 * If partitionFrom is nonnegative, fetches only that partition.
 * If partitionFrom is negative, returns the entire array.
 *
 * If p[0] == NULL, attempts to allocate sufficient storage, which is one of
 *     g->partNumel[partitionFrom] or g->numel.
 * If p[0] != NULL, p[0] must point to sufficient storage. */
// FIXME: this entire crapshow should just make a few calls to cudaMemcpy2D/3D
int MGA_downloadArrayToCPU(MGArray *g, double **p, int partitionFrom)
{
	int returnCode = SUCCESSFUL;

	int sub[6];
	long numelOut;
	if(partitionFrom >= 0) {
		numelOut = g->partNumel[partitionFrom];
	} else {
		numelOut = g->numel;
	}

	// Create output numeric array if passed NULL
	// If e.g. returning to MATLAB, it will have already been allocated for us.
	if(p[0] == NULL) {
		*p = (double *)malloc(numelOut * sizeof(double));
	}

	if(p[0] == NULL) {
		PRINT_FAULT_HEADER;
		printf("Host data pointer is null!\nFailed to allocate host storage!\n");
		PRINT_FAULT_FOOTER;
		return ERROR_NULL_POINTER;
	}

	int u, v, w, i;
	int64_t iT, iS;
	double *gmem[g->nGPUs];

	int fromPart, toPart;

	if(partitionFrom >= 0) { // we will fetch only this partition
		fromPart = partitionFrom;
		toPart = partitionFrom + 1;
	} else { // we will fetch all partitions
		fromPart = 0;
		toPart = g->nGPUs;
	}

	for(i = fromPart; i < toPart; i++) {
		gmem[i] = (double *)malloc(g->partNumel[i]*sizeof(double));
		if(gmem[i] == NULL) {
			PRINT_FAULT_HEADER;
			printf("FATAL: Unable to allocate download buffer for GPU array!\n");
			PRINT_FAULT_FOOTER;
			return ERROR_NULL_POINTER;
		}

		cudaError_t fail = cudaMemcpy((void *)gmem[i], (void *)g->devicePtr[i], g->partNumel[i]*sizeof(double), cudaMemcpyDeviceToHost);
		returnCode = CHECK_CUDA_ERROR("MGArray_downloadArrayToCPU");
		if(returnCode != SUCCESSFUL) break;
	}
	if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

	double *out = p[0];

	int3 ptSize, ptOffset, ptExtent, outOffset, outDims;

	if(partitionFrom >= 0) { // Specific partition: Out dim = that partition
		calcPartitionExtent(g, partitionFrom, &sub[0]);
		outDims.x = sub[3]; outDims.y = sub[4]; outDims.z = sub[5];

	} else {
		outDims.x = g->dim[0];
		outDims.y = g->dim[1];
		outDims.z = g->dim[2];
	}

	double *currentTarget;
	for(i = fromPart; i < toPart; i++) {
		calcPartitionExtent(g, i, &sub[0]);

		ptOffset.x  = sub[0]; ptOffset.y  = sub[1]; ptOffset.z  = sub[2];
		ptSize.x = sub[3]; ptSize.y = sub[4]; ptSize.z = sub[5];

		outOffset = ptOffset;
		ptExtent = ptSize;

		ptOffset.x = 0; ptOffset.y = 0; ptOffset.z = 0;

		currentTarget = gmem[i];

		if(g->nGPUs > 1) {
			// left halo removal
			if((g->addExteriorHalo != 0) || (i > 0)) {
				switch(g->partitionDir) {
				case PARTITION_X: ptExtent.x -= g->haloSize; outOffset.x += g->haloSize; ptOffset.x += g->haloSize; break;
				case PARTITION_Y: ptExtent.y -= g->haloSize; outOffset.y += g->haloSize; ptOffset.y += g->haloSize; break;
				case PARTITION_Z: ptExtent.z -= g->haloSize; outOffset.z += g->haloSize; ptOffset.z += g->haloSize; break;
				}
			}
			// right halo removal
			if((g->addExteriorHalo != 0) || (i < (g->nGPUs-1)))
				switch(g->partitionDir) {
				case PARTITION_X: ptExtent.x -= g->haloSize; break;
				case PARTITION_Y: ptExtent.y -= g->haloSize; break;
				case PARTITION_Z: ptExtent.z -= g->haloSize; break;
				}
		}

		// If we're fetching only 1 partition zap the offset
		if(partitionFrom >= 0) { outOffset.x = outOffset.y = outOffset.z = 0; }

		for(w = 0; w < ptExtent.z; w++) {
			for(v = 0; v < ptExtent.y; v++) {
				for(u = 0; u < ptExtent.x; u++) {
					iT = u + ptOffset.x + ptSize.x*(v + ptOffset.y + ptSize.y * (w + ptOffset.z));
					iS = u + outOffset.x + outDims.x*(v + outOffset.y + outDims.y * (w + outOffset.z));
					out[iS] = currentTarget[iT];
				}
			}
		}

		free(gmem[i]);
	}


	return SUCCESSFUL;
}

/* Given a pointer to the Matlab array m, checks that g points to an
 * MGArray of the same size as m:
 * if partitionTo is nonnegative, that that partition's extent equals
 * the size of m
 * if partitionTo is negative, that the g->dim equals the size of m.
 * and then initiates the transfer.
 */
int MGA_uploadMatlabArrayToGPU(const mxArray *m, MGArray *g, int partitionTo)
{

if(m == NULL) return -1;
if(g == NULL) return -1;

mwSize ndims = mxGetNumberOfDimensions(m);
if(ndims > 3) { DROP_MEX_ERROR((const char *)"Input array has more than 3 dimensions!"); }

const mwSize *arraydims = mxGetDimensions(m);

int j;
int failed = 0;

for(j = 0; j < ndims; j++) { 
	if(arraydims[j] != g->dim[j]) failed = 1;
}

if(failed) {
	PRINT_FAULT_HEADER;
	printf("Matlab array was %i dimensional, dims [", ndims);
	for(j = 0; j < ndims; j++) { printf("%i ", arraydims[j]); }
	printf("].\nGPU_Type target array was of size [%i %i %i] which is not the same. Not happy :(.\n", g->dim[0], g->dim[1], g->dim[2]);
	PRINT_FAULT_FOOTER;
	return ERROR_INVALID_ARGS;
	}

return CHECK_IMOGEN_ERROR(MGA_uploadArrayToGPU(mxGetPr(m), g, partitionTo));

}

/* Assuming that p points to an array whose size is compatible with either
 * the whole of g (partitionOnto < 0) or a specific partition of g (if
 * partitionOnto >= 0), transfers elements of p to g.
 */
int MGA_uploadArrayToGPU(double *p, MGArray *g, int partitionTo)
{
	int returnCode = SUCCESSFUL;
	int sub[6];

	// Create output numeric array if passed NULL
	// If e.g. returning to MATLAB, it will have already been allocated for us.
	if(p == NULL) {
		PRINT_FAULT_HEADER;
		printf("Host data pointer is null!");
		PRINT_FAULT_FOOTER;
		return ERROR_NULL_POINTER;
	}

	int u, v, w, i;
	int64_t iT, iS;
	double *gmem[g->nGPUs];

	int fromPart, toPart;

	if(partitionTo >= 0) { // Uploading to a single partition
		fromPart = partitionTo;
		toPart = partitionTo + 1;
	} else { // we will fetch all partitions
		fromPart = 0;
		toPart = g->nGPUs;
	}

	for(i = fromPart; i < toPart; i++) {
		gmem[i] = (double *)malloc(g->partNumel[i]*sizeof(double));
		if(gmem[i] == NULL) {
			PRINT_FAULT_HEADER;
			printf("Unable to allocate upload buffer!\n");
			PRINT_FAULT_FOOTER;
			return ERROR_NULL_POINTER;

		}
	}

	int3 ptSize, ptOff, partExtent, readOff;

	int *usedims;

	double *currentTarget;
	for(i = fromPart; i < toPart; i++) {
		calcPartitionExtent(g, i, &sub[0]);

		ptOff.x  = sub[0]; ptOff.y  = sub[1]; ptOff.z  = sub[2];
		ptSize.x = sub[3]; ptSize.y = sub[4]; ptSize.z = sub[5];

		readOff = ptOff;
		partExtent = ptSize;

		ptOff.x = 0; ptOff.y = 0; ptOff.z = 0;

		currentTarget = gmem[i];

		if(g->nGPUs > 1) {
			// left halo removal
			if((g->addExteriorHalo != 0) || (i > 0)) {
				switch(g->partitionDir) {
				case PARTITION_X: partExtent.x -= g->haloSize; readOff.x += g->haloSize; ptOff.x += g->haloSize; break;
				case PARTITION_Y: partExtent.y -= g->haloSize; readOff.y += g->haloSize; ptOff.y += g->haloSize;  break;
				case PARTITION_Z: partExtent.z -= g->haloSize; readOff.z += g->haloSize; ptOff.z += g->haloSize; break;
				}
			}
			// right halo removal
			if((g->addExteriorHalo != 0) || (i < (g->nGPUs-1)))
				switch(g->partitionDir) {
				case PARTITION_X: partExtent.x -= g->haloSize; break;
				case PARTITION_Y: partExtent.y -= g->haloSize; break;
				case PARTITION_Z: partExtent.z -= g->haloSize; break;
				}
		}

		// If we're fetching only 1 partition: zap the offset
		if(partitionTo >= 0) {
			readOff.x = readOff.y = readOff.z = 0;
			usedims = &sub[3];
		} else {
			usedims = &g->dim[0];
		}

		for(w = 0; w < partExtent.z; w++) {
			for(v = 0; v < partExtent.y; v++) {
				for(u = 0; u < partExtent.x; u++) {
					iT = u + ptOff.x + ptSize.x*(v + ptOff.y + ptSize.y*(w + ptOff.z));
					iS = u + readOff.x + usedims[0]*(v + readOff.y + usedims[1] * (w + readOff.z));
					currentTarget[iT] = p[iS];
				}
			}
		}

		cudaError_t fail = cudaMemcpy((void *)g->devicePtr[i], (void *)gmem[i], g->partNumel[i]*sizeof(double), cudaMemcpyHostToDevice);
		returnCode = CHECK_CUDA_ERROR((const char *)"MGArray_uploadArrayToGPU");
		if(returnCode != SUCCESSFUL) break;

		free(gmem[i]);
	}

	if(returnCode != SUCCESSFUL) {
		return CHECK_IMOGEN_ERROR(returnCode);
	}

	returnCode = MGA_exchangeLocalHalos(g, 1);
	return CHECK_IMOGEN_ERROR(returnCode);

}

/* Given a pointer to a FluidManager class, accesses the fluids stored
 * within it. */
int MGA_accessFluidCanister(const mxArray *canister, int fluidIdx, MGArray *fluid)
{
	/* Access the FluidManager canisters */
	mxArray *fluidPtrs[3];
	fluidPtrs[0] = mxGetProperty(canister, fluidIdx,(const char *)("mass"));
	if(fluidPtrs[0] == NULL) {
		PRINT_FAULT_HEADER;
		printf("Unable to fetch 'mass' property from canister\nNot a FluidManager class?\n");
		PRINT_FAULT_FOOTER;
		return ERROR_INVALID_ARGS;
	}
	fluidPtrs[1] = mxGetProperty(canister, fluidIdx,(const char *)("ener"));
	if(fluidPtrs[1] == NULL) {
		PRINT_FAULT_HEADER;
		printf("Unable to fetch 'mass' property from canister\nNot a FluidManager class?\n");
		PRINT_FAULT_FOOTER;
		return ERROR_INVALID_ARGS;
	}
	fluidPtrs[2] = mxGetProperty(canister, fluidIdx,(const char *)("mom"));
	if(fluidPtrs[2] == NULL) {
		PRINT_FAULT_HEADER;
		printf("Unable to fetch 'mass' property from canister\nNot a FluidManager class?\n");
		PRINT_FAULT_FOOTER;
		return ERROR_INVALID_ARGS;
	}

	int status = MGA_accessMatlabArrays((const mxArray **)&fluidPtrs[0], 0, 1, &fluid[0]);
	if(status != SUCCESSFUL) return CHECK_IMOGEN_ERROR(status);
    status = MGA_accessMatlabArrayVector(fluidPtrs[2], 0, 2, &fluid[2]);
    if(status != SUCCESSFUL) return CHECK_IMOGEN_ERROR(status);

    return SUCCESSFUL;
}

GeometryParams accessMatlabGeometryClass(const mxArray *geoclass)
{
	GeometryParams g;

	double v[3];

	g.Rinner = derefXdotAdotB_scalar(geoclass, "pInnerRadius", NULL);
	derefXdotAdotB_vector(geoclass, "d3h", NULL, &g.h[0], 3);

	int shapenum = derefXdotAdotB_scalar(geoclass, "pGeometryType", NULL);

	switch(shapenum) {
	case 1: g.shape = SQUARE; break;
	case 2: g.shape = CYLINDRICAL; break;
	// default: ?
	}
	derefXdotAdotB_vector(geoclass, "affine", NULL, &v[0], 3);
	g.x0 = v[0];
	g.y0 = v[1];
	g.z0 = v[2];


	return g;
}

/* A utility to ease access to Matlab structures/classes, fetches in.{fieldA}.{fieldB}
 * or in.{fieldA} if fieldB is blank and returns the resulting mxArray*. */
mxArray *derefXdotAdotB(const mxArray *in, const char *fieldA, const char *fieldB)
{
	if(fieldA == NULL) mexErrMsgTxt("In derefAdotBdotC: fieldA null!");

	mxArray *A; mxArray *B;
	mxClassID t0 = mxGetClassID(in);

	int snum = strlen("Failed to read field fieldA in X.A.B") + (fieldA != NULL ? strlen(fieldA) : 5) + (fieldB != NULL ? strlen(fieldB) : 5) + 10;
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

/* Fetches in.{fieldA}.{fieldB}, or in.{fieldA} if fieldB is NULL,
 * and returns the first double element of this.
 */
double derefXdotAdotB_scalar(const mxArray *in, const char *fieldA, const char *fieldB)
{
	mxArray *u = derefXdotAdotB(in, fieldA, fieldB);

	if(u != NULL) return *mxGetPr(u);

	return NAN;
}

/* Fetches in.{fieldA}.{fieldB}, or in.{fieldA} if fieldB is NULL,
 * and copies the first N elements of this into x[0, ..., N-1] if we get
 * a valid double *, or writes NANs if we do not.
 * If the Matlab array has fewer than N elements, truncates the copy.
 */
void derefXdotAdotB_vector(const mxArray *in, const char *fieldA, const char *fieldB, double *x, int N)
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

/* This should be checked after every GPU kernel launch: It provides detailed metadata feedback to
 * greatly facilitate debugging. If it does not return SUCCESSFUL, the function should
 * abort and return its return value: This effectively prints a compiled code backtrace.
 */
int checkCudaLaunchError(cudaError_t E, dim3 blockdim, dim3 griddim, MGArray *a, int i, const char *srcname, const char *fname, int lname)
{
	if(E == cudaSuccess) return SUCCESSFUL;

	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	printf("========== FAULT FROM CUDA API (%s:%i), RANK %i\n", fname, lname, myrank);

	printf("Caught CUDA error %s -> %s\n", errorName(E), cudaGetErrorString(E));
	printf("Code's description of what it just did: %s\n", srcname);
	printf("Rx'd the integer: %i\n", i);

	if(a == NULL) {
		PRINT_FAULT_HEADER;
		printf("CUDA reported a problem after kernel launch.\nBut no MGArray passed to error checker... ?!?!?!?\nReturning crash condition...\n");
		PRINT_FAULT_FOOTER;
		return ERROR_CRASH;
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

    PRINT_FAULT_FOOTER;

	return ERROR_CUDA_BLEW_UP;
}

void MGA_debugPrintAboutArray(MGArray *x)
{
	int n = x->nGPUs;
	printf("========== DEBUG INFORMATION ABOUT ARRAY\n");
	printf("  This array's address: %x\n", x);
	printf("===== Device information\n");
	printf("Array distributed onto    %i GPUs\n", x->nGPUs);
	int j;
	for(j = 0; j < n; j++) { printf("Device %i: [%i | %x]\n", j, x->deviceID[j], x->devicePtr[j]); }

	printf("Array's host-side extent: [%i %i %i]\n", x->dim[0], x->dim[1], x->dim[2]);
	printf("Array's host #elements  : %li", x->numel);
	if(x->numSlabs > 1) {
		printf("Array IS A REAL ALLOCATION with %i slabs.\n", x->numSlabs);
	} else {
		printf("Array IS A SLAB REFERENCE  index number %i\n", -x->numSlabs);
	}
	printf("Array's slab pitch      : [");
	for(j = 0; j < n; j++) { printf("%li ", x->slabPitch[j]); }
	printf("]\n");
	printf("Partition halo size     : %i\n", x->haloSize);
	printf("Partition direction     : %i\n", x->partitionDir);
	printf("Halo added to exterior? : %c\n", x->addExteriorHalo ? 'y' : 'n');
	printf("Permutation tag value   : %i\n", x->permtag);
	printf("Which represents        : [%i %i %i] stride ordering\n", x->currentPermutation[0], x->currentPermutation[1], x->currentPermutation[2]);
	printf("circularBoundaryBits    : %i\n", x->circularBoundaryBits);
	printf("Matlab source class index is %i\n", x->mlClassHandleIndex);
	printf("==========\n");

}

/* This should be polled after CUDA API calls. In the event of a problem, it provides detailed
 * metadata feedback to assist with debugging. If it returns unsuccessful, the function
 * should immediately cleanup & return its return value.
 */
int checkCudaError(const char *where, const char *fname, int lname)
{
	cudaError_t epicFail = cudaGetLastError();
	if(epicFail == cudaSuccess) return SUCCESSFUL;

	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	printf("========== FAULT FROM CUDA API (%s:%i), RANK %i\n", fname, lname, myrank);
	printf("cudaCheckError was non-success when polled at %s (%s:%i) by rank %i: %s -> %s\n", where, fname, lname, myrank, errorName(epicFail), cudaGetErrorString(epicFail));
	PRINT_FAULT_FOOTER;

	return ERROR_CUDA_BLEW_UP;
}

/* This function facilitates error feedback that assists with debugging.
 * Functions which can fail should return an error integer.
 * Every call to a function which can fail should use the CHECK_IMOGEN_ERROR() macro
 * and abort in the event it returns other than SUCCESSFUL;
 * This process will print a backtrace of where the error occurred, leading back
 * to the invoking mexFunction entry point.
 *
 * The mexFunction itself, upon detecting failure, should cause a mexError, which will
 * cause the invocation of a similar backtrace in the Matlab layer
 *
 * NOTE: Don't use this directly, use the CHECK_IMOGEN_ERROR(errorcode) macro which will automatically
 * have the correct filename, function name and line number.
 */
int checkImogenError(int errtype, const char *infile, const char *infunc, int atline)
{
	if(errtype == SUCCESSFUL) return SUCCESSFUL;
	int mpirank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

	const char *estring;

	switch(errtype)
	{
	case ERROR_INVALID_ARGS:              estring = "One or more invalid arguments caught."; break;
	case ERROR_CRASH:                     estring = "Evidently we can/will/must crash.\n"; break;
	case ERROR_NULL_POINTER:              estring = "Null pointer caught."; break;
	case ERROR_GET_GPUTAG_FAILED:         estring = "Attempt to get GPU tag failed."; break;
	case ERROR_DESERIALIZE_GPUTAG_FAILED: estring = "Deserialization of gputag -> MGArray failed."; break;
	case ERROR_CUDA_BLEW_UP:              estring = "CUDA API returned an oops; The Hindenburg will now burst into flames..."; break;
	}
	printf("Rank %i | In %s (%s:%i): %s\n", mpirank, infunc, infile, atline, estring);
	return errtype;
}

/* This function should be used the mexFunction entry points to signal Matlab of problems.
 * NOTE: Don't call this directly, use DROP_MEX_ERROR("string") to automatically
 * fill in the file and line numbers correctly.
 */
void dropMexError(const char *excuse, const char *infile, int atline)
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

/* Accepts an MPI_Op reduction type and returns an enum appropriate
 * for my templated accelerated reduce functions.
 */
MGAReductionOperator MGAReductionOperator_mpi2mga(MPI_Op mo)
{
	MGAReductionOperator op = MGA_OP_SUM;

	if(mo == MPI_SUM) op = MGA_OP_SUM;
	if(mo == MPI_PROD)op = MGA_OP_PROD;
	if(mo == MPI_MAX) op = MGA_OP_MAX;
	if(mo == MPI_MIN) op = MGA_OP_MIN;
	return op;
}
/* Accepts my enumerated reduce operators & converts them to MPI_Op types
 * for global operations.
 */
MPI_Op MGAReductionOperator_mga2mpi(MGAReductionOperator op)
{
	MPI_Op mo = MPI_SUM;

	switch(op) {
	case MGA_OP_SUM: mo = MPI_SUM; break;
	case MGA_OP_PROD:mo = MPI_PROD; break;
	case MGA_OP_MIN: mo = MPI_MIN; break;
	case MGA_OP_MAX: mo = MPI_MAX; break;
	}
return mo;
}


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



