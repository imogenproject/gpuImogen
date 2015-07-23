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

// Some basic does-this-make-sense
if(nDevs < 1) return false;
if(nDevs > MAX_GPUS_USED) return false;
if(halo < 0) { // check it is sane to clone
	if(halo != PARTITION_CLONED) return false; // if it's actually marked as cloned and not just FUBAR

	if(x[partitionDir-1] != 1) return false;
}
if((partitionDir < 1) || (partitionDir > 3)) return false;

// Require there be exactly the storage required
int requisiteNumel = GPU_TAG_LENGTH + 2*nDevs;
if(tagsize != requisiteNumel) return false;

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
  } else { /* Assume it's an ImogenArray or descendant and retrieve the gputag property */
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

cudaStream_t *getGPUTypeStreams(const mxArray *fluidarray) {
	mxArray *streamptr  = mxGetProperty(fluidarray, 0, "streamptr");

	return (cudaStream_t *)(*((int64_t *)mxGetData(streamptr)) );
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
mg->numSlabs = tag[GPU_TAG_DIMSLAB];

mg->haloSize     = tag[GPU_TAG_HALO];
mg->partitionDir = tag[GPU_TAG_PARTDIR];
mg->nGPUs        = tag[GPU_TAG_NGPUS];

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

return;
}

/* Serialized tag form:
   [ Nx
     Ny
     Nz
     Nslabs
     halo size on shared edges
     Direction to parition in [1 = x, 2 = y, 3 = x]
     # of GPU paritions being used
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
tag[GPU_TAG_DIMSLAB] = mg->numSlabs;
tag[GPU_TAG_HALO] = mg->haloSize;
tag[GPU_TAG_PARTDIR] = mg->partitionDir;
tag[GPU_TAG_NGPUS] = mg->nGPUs;
int i;
for(i = 0; i < mg->nGPUs; i++) {
    tag[GPU_TAG_LENGTH+2*i]   = (int64_t)mg->deviceID[i];
    tag[GPU_TAG_LENGTH+2*i+1] = (int64_t)mg->devicePtr[i];
    }

return;
}

// Helpers to easily access/create multiple arrays
int MGA_accessMatlabArrays(const mxArray *prhs[], int idxFrom, int idxTo, MGArray *mg)
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

return m;
}

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

void MGA_returnOneArray(mxArray *plhs[], MGArray *m)
{
mwSize dims[2]; dims[0] = GPU_TAG_LENGTH+2*m->nGPUs; dims[1] = 1;
int64_t *r;

// create Matlab arrays holding serialized form,
plhs[0] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);
r = (int64_t *)mxGetData(plhs[0]);
serializeMGArrayToTag(m, r);
}

void MGA_delete(MGArray *victim)
{
	if(victim->numSlabs < 1) return; // This is a slab reference and was never actually allocated. Ignore it.

	for(int j = 0; j<victim->nGPUs; j++){
		cudaSetDevice(victim->deviceID[j]);
		CHECK_CUDA_ERROR("in MGA_delete, setting device");
		cudaFree(victim->devicePtr[j]);
		CHECK_CUDA_ERROR("in MGA_delete, freeing");
	}
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

/* A node whose MGA partitions are of equal size may call this function
 * resulting in:
 *    in.devicePtr[partitionOnto][i] = REDUCE( { in->devicePtr[N][i] } ),
 * for N in (0, 1, ..., in->nGPus-1) and i in (0, ..., n->partNumel[0])
 * if(redistribute) { copies in->devicePtr[partitionOnto] back to the others as well. }
 */
int MGA_localElementwiseReduce(MGArray *in, MPI_Op operate, int dir, int partitionOnto, int redistribute)
{
	/* FIXME: not implemented */
	FATAL_NOT_IMPLEMENTED
	return -1;
}

/* A node whose MGAs have compatible sizes transverse to dir may call this
 * such that the array in 'dir' is reduced to size 1
 * If dir != in->partitionDir:
 *   each partition will simply be pancaked in parallel, partition scheme unchanged
 * If dir == in->partitionDir,
 *   Each partition is reduced in parallel
 *   Reduce is applied across devices to the 2D pancakes
 *   if(redistribute && (dir == in->partitionDir)) {
 *     The result is marked as PARTITION_CLONED and the reduction memcpy()ed back }
 */
int MGA_localPancakeReduce(MGArray *in, MGArray *out, MPI_Op operate, int dir, int partitionOnto, int redistribute)
{

	int i;
	int sub[6];

	MGArray clone = *in;

	clone.dim[dir] = 1;

	if(dir == in->partitionDir) {
		clone.haloSize = PARTITION_CLONED;
		out = MGA_allocArrays(1, &clone);

		for(i = 0; i < in->nGPUs; i++) {
			calcPartitionExtent(in, i, &sub[0]);
			if(sub[3+dir] > 1) {
				/* FIXME: reduce in->devicePtr[i] into out->devicePtr[i] */
				FATAL_NOT_IMPLEMENTED
			} else {
				cudaSetDevice(in->deviceID[i]);
				CHECK_CUDA_ERROR("cudaSetDevice");
				cudaMemcpy(out->devicePtr[i], in->devicePtr[i], in->partNumel[i]*sizeof(double), cudaMemcpyDeviceToDevice);
				CHECK_CUDA_ERROR("cudaMemcpy");
			}
		}

		MGA_reduceClonedArray(out, operate, redistribute);
		if(redistribute) MGA_distributeArrayClones(out, partitionOnto);
	} else {
		/* Reduce each device's array in parallel */
		for(i = 0; i < in->nGPUs; i++) {
			FATAL_NOT_IMPLEMENTED
			cudaSetDevice(in->deviceID[i]);
			/* FIXME: reduce in->devicePtr[i] to out->devicePtr[i] */
		}

	}

	return 0;

}

/* FIXME: Copied this flush out of cudaFluidStep, it should be checked for bullshit */
pParallelTopology topoStructureToC(const mxArray *prhs)
{
	mxArray *a;

	pParallelTopology pt = (pParallelTopology)malloc(sizeof(ParallelTopology));

	a = mxGetFieldByNumber(prhs,0,0);
	pt->ndim = (int)*mxGetPr(a);
	a = mxGetFieldByNumber(prhs,0,1);
	pt->comm = (int)*mxGetPr(a);

	int *val;
	int i;

	val = (int *)mxGetData(mxGetFieldByNumber(prhs,0,2));
	for(i = 0; i < pt->ndim; i++) pt->coord[i] = val[i];

	val = (int *)mxGetData(mxGetFieldByNumber(prhs,0,3));
	for(i = 0; i < pt->ndim; i++) pt->neighbor_left[i] = val[i];

	val = (int *)mxGetData(mxGetFieldByNumber(prhs,0,4));
	for(i = 0; i < pt->ndim; i++) pt->neighbor_right[i] = val[i];

	val = (int *)mxGetData(mxGetFieldByNumber(prhs,0,5));
	for(i = 0; i < pt->ndim; i++) pt->nproc[i] = val[i];

	for(i = pt->ndim; i < 4; i++) {
		pt->coord[i] = 0;
		pt->nproc[i] = 1;
	}

	return pt;

}

/* Every node in dir must call this:
 * Does MGA_localEmementwiseReduce per node to partitionOnto, copies this reduction to CPU memory,
 * then performs MPI_AllReduce in the given direction on it,
 * and copies that result back to partitionOnto's device memory.
 * if(redistribute) {
 *   result is memcpy()ed from partitionOnto to all others as well }
  */
int MGA_globalElementwiseReduce(MGArray *in, MPI_Op operate, int dir, int partitionOnto, int redistribute, const mxArray *topo)
{
	FATAL_NOT_IMPLEMENTED
	return 0;

}

/* Every node in dir must call this:
 * Does MGA_localpancakeReduce on each node to partitionOnto,
 * copies this to host memory
 * calls MPI_AllReduce,
 * copies result back to partitionOnto's device memory,
 *
 */
int MGA_globalPancakeReduce(MGArray *in, MGArray *out, MPI_Op operate, int dir, int partitionOnto, int redistribute, const mxArray *topo)
{
	pParallelTopology topology = topoStructureToC(topo);

	/* Reverse silly Fortran memory ordering */
	int d = 0;
	int dmax = topology->nproc[d];

	MPI_Comm commune = MPI_Comm_f2c(topology->comm);
	int r0; MPI_Comm_rank(commune, &r0);

	/* First step: Perform a local pancaking in the reduce dir */
	MGA_localPancakeReduce(in, out, operate, dir, partitionOnto, 0);

	double *readBuf;
	// This handles cloned or not-cloned automatically
	MGA_downloadArrayToCPU(out, &readBuf, partitionOnto);
	double *writeBuf= (double *)malloc(out->numel*sizeof(double));


	/* FIXME: This is a temporary hack
   FIXME: The creation of these communicators should be done once,
   FIXME: by PGW, at start time. */
	/* FIXME this fixme is as old as this crap from cudaWStep... */
	int dimprocs[dmax];
	int proc0, procstep;
	switch(d) { /* everything here is Wrong because fortran is Wrong */
	case 0: /* i0 = nx Y + nx ny Z, step = 1 -> nx ny */
		/* x dimension: step = ny nz, i0 = z + nz y */
		proc0 = topology->coord[2] + topology->nproc[2]*topology->coord[1];
		procstep = topology->nproc[2]*topology->nproc[1];
		break;
	case 1: /* i0 = x + nx ny Z, step = nx */
		/* y dimension: step = nz, i0 = z + nx ny x */
		proc0 = topology->coord[2] + topology->nproc[2]*topology->nproc[1]*topology->coord[0];
		procstep = topology->nproc[2];
		break;
	case 2: /* i0 = x + nx Y, step = nx ny */
		/* z dimension: i0 = nz y + nz ny x, step = 1 */
		proc0 = topology->nproc[2]*(topology->coord[1] + topology->nproc[1]*topology->coord[0]);
		procstep = 1;
		break;
	}
	int j;
	for(j = 0; j < dmax; j++) {
		dimprocs[j] = proc0 + j*procstep;
	}

	MPI_Group worldgroup, dimgroup;
	MPI_Comm dimcomm;
	/* r0 has our rank in the world group */
	MPI_Comm_group(commune, &worldgroup);
	MPI_Group_incl(worldgroup, dmax, dimprocs, &dimgroup);
	/* Create communicator for this dimension */
	MPI_Comm_create(commune, dimgroup, &dimcomm);

	/* Perform the reduce */
	MPI_Allreduce((void *)readBuf, (void *)writeBuf, out->partNumel[partitionOnto], MPI_DOUBLE, operate, dimcomm);

	MPI_Barrier(dimcomm);
	/* Clean up */
	MPI_Group_free(&dimgroup);
	MPI_Comm_free(&dimcomm);

	MGA_uploadArrayToGPU(writeBuf, out, partitionOnto);
	if(redistribute) MGA_distributeArrayClones(out, partitionOnto);

	free(readBuf); free(writeBuf);

	return 0;
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
int MGA_reduceClonedArray(MGArray *a, MPI_Op operate, int redistribute)
{
	if(a->haloSize != PARTITION_CLONED) return 0;

	dim3 gridsize; gridsize.x = 32; gridsize.y = gridsize.z = 1;
	dim3 blocksize; blocksize.x = 256; blocksize.y = blocksize.z = 1;

	double *B; double *C;

	switch(a->nGPUs) {
	case 1: break; // nofin to do
	case 2: // reduce(A,B)->A
		cudaSetDevice(a->deviceID[0]);
		CHECK_CUDA_ERROR("cudaSetDevice()");
		cudaMalloc((void **)&B, a->numel*sizeof(double));
		CHECK_CUDA_ERROR("cudaMalloc()");
		cudaMemcpy((void *)B, (void*)a->devicePtr[1], a->numel*sizeof(double), cudaMemcpyDeviceToDevice);
		CHECK_CUDA_ERROR("cudaMalloc()");

		if(operate == MPI_SUM) cudaClonedReducer<OP_SUM><<<32, 256>>>(a->devicePtr[0], B, a->partNumel[0]);
		if(operate == MPI_PROD) cudaClonedReducer<OP_PROD><<<32, 256>>>(a->devicePtr[0], B, a->partNumel[0]);
		if(operate == MPI_MAX) cudaClonedReducer<OP_MAX><<<32, 256>>>(a->devicePtr[0], B, a->partNumel[0]);
		if(operate == MPI_MIN) cudaClonedReducer<OP_MIN><<<32, 256>>>(a->devicePtr[0], B, a->partNumel[0]);

		CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, "clone reduction for 2 GPUs");
		cudaFree(B);
		CHECK_CUDA_ERROR("cudaFree()");
		break;
	case 3: // reduce(A,B)->A; reduce(A, C)->C
		cudaSetDevice(a->deviceID[0]);
		CHECK_CUDA_ERROR("cudaSetDevice()");
		cudaMalloc((void **)&B, a->numel*sizeof(double));
		CHECK_CUDA_ERROR("cuda malloc");
		cudaMalloc((void **)&C, a->numel*sizeof(double));
		CHECK_CUDA_ERROR("cuda malloc");

		cudaMemcpy((void *)B, (void *)a->devicePtr[1], a->numel*sizeof(double), cudaMemcpyDeviceToDevice);
		CHECK_CUDA_ERROR("cuda memcpy");
		cudaMemcpy((void *)C, (void *)a->devicePtr[2], a->numel*sizeof(double), cudaMemcpyDeviceToDevice);
		CHECK_CUDA_ERROR("cuda memcpy");

		if(operate == MPI_SUM) cudaClonedReducer<OP_SUM><<<32, 256>>>(a->devicePtr[0], B, a->partNumel[0]);
		if(operate == MPI_PROD) cudaClonedReducer<OP_PROD><<<32, 256>>>(a->devicePtr[0],B, a->partNumel[0]);
		if(operate == MPI_MAX) cudaClonedReducer<OP_MAX><<<32, 256>>>(a->devicePtr[0], B, a->partNumel[0]);
		if(operate == MPI_MIN) cudaClonedReducer<OP_MIN><<<32, 256>>>(a->devicePtr[0], B, a->partNumel[0]);

		CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, "clone reduction for 3 GPUs, first call");

		if(operate == MPI_SUM)  cudaClonedReducer<OP_SUM><<<32, 256>>>(a->devicePtr[0], C, a->partNumel[0]);
		if(operate == MPI_PROD) cudaClonedReducer<OP_PROD><<<32, 256>>>(a->devicePtr[0], C, a->partNumel[0]);
		if(operate == MPI_MAX)  cudaClonedReducer<OP_MAX><<<32, 256>>>(a->devicePtr[0], C, a->partNumel[0]);
		if(operate == MPI_MIN)  cudaClonedReducer<OP_MIN><<<32, 256>>>(a->devicePtr[0], C, a->partNumel[0]);

		CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, "clone reduction for 3 GPUs, second call");

		cudaFree(B);
		CHECK_CUDA_ERROR("cudaFree");
		cudaFree(C);
		CHECK_CUDA_ERROR("cudaFree");
		break;
	case 4: // {reduce(A,B)->A, reduce(C,D)->C}; reduce(A,C)->A
// FIXME: This is broken right now...
		mexErrMsgTxt("This is broken soz.");
		cudaSetDevice(a->deviceID[0]);
		CHECK_CUDA_ERROR("cudaSetDevice()");
		cudaMalloc((void **)&B ,a->partNumel[0]);
		CHECK_CUDA_ERROR("cudaMalloc");

		if(operate == MPI_SUM) cudaClonedReducerQuad<OP_SUM><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->devicePtr[2], a->devicePtr[3], a->partNumel[0]);
		if(operate == MPI_PROD) cudaClonedReducerQuad<OP_PROD><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->devicePtr[2], a->devicePtr[3], a->partNumel[0]);
		if(operate == MPI_MAX) cudaClonedReducerQuad<OP_MAX><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->devicePtr[2], a->devicePtr[3], a->partNumel[0]);
		if(operate == MPI_MIN) cudaClonedReducerQuad<OP_MIN><<<32, 256>>>(a->devicePtr[0], a->devicePtr[1], a->devicePtr[2], a->devicePtr[3], a->partNumel[0]);

		CHECK_CUDA_LAUNCH_ERROR(gridsize, blocksize, a, 2, "clone reduction for 4 GPUs using clonedReducerQuad");
		break;
	default: return -1;
	}

if(redistribute)
	MGA_distributeArrayClones(a, 0);

	return 0;
}

/* If partition sizes are equal (as typical e.g. post-reduction),
 * copies the array on partition partitionFrom to the others resulting in
 * identical copies on each device.
 */
int MGA_distributeArrayClones(MGArray *cloned, int partitionFrom)
{
if(cloned->partitionDir != PARTITION_CLONED) return 0;

int j;

for(j = 0; j < cloned->nGPUs; j++) {
	if(j == partitionFrom) continue;

	cudaMemcpy(cloned->devicePtr[j], cloned->devicePtr[partitionFrom], sizeof(double)*cloned->numel, cudaMemcpyDeviceToDevice);
	CHECK_CUDA_ERROR("MGA_distributeArrayClones");
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
void MGA_exchangeLocalHalos(MGArray *a, int n)
{
int i, j, jn;
dim3 blocksize, gridsize;

/* It is essential that ALL devices be absolutely synchronized before this operation starts
 * It is an observed fact that the cudaFree() after the fluid step kernels does NOT result in
 * sequential consistency between different devices.
 */
for(j = 0; j < a->nGPUs; j++) {
    	cudaSetDevice(a->deviceID[j]);
    	cudaDeviceSynchronize();
    }

for(i = 0; i < n; i++) {
	// Skip this if it's a cloned partition
	if(a->haloSize == PARTITION_CLONED) { a++; continue; }
	// Or there's only one partition to begin with
	if(a->nGPUs == 1) { a++; continue; }

    for(j = 0; j < a->nGPUs; j++) {
    	jn = (j+1) % a->nGPUs;

    	int subL[6], subR[6];
    	calcPartitionExtent(a, j, subL);
    	calcPartitionExtent(a, jn, subR);

        switch(a->partitionDir) {
            case 1: {
                cudaSetDevice(a->deviceID[j]);
                blocksize.x = a->haloSize;
                blocksize.y = SYNCBLOCK;
                blocksize.z = (subL[5] > 1) ? 8 : 1;

                gridsize.x  = a->dim[1]/SYNCBLOCK; gridsize.x += (gridsize.x*SYNCBLOCK < a->dim[1]);
                gridsize.y  = 1; gridsize.z = 1;
                cudaMGHaloSyncX<<<gridsize, blocksize>>>(a->devicePtr[j], a->devicePtr[jn], subL[3], subR[3], subL[4], subL[5], a->haloSize);
            } break;
            case 2: {
                cudaSetDevice(a->deviceID[j]);
                blocksize.x = blocksize.y = SYNCBLOCK;
                blocksize.z = 1;
                gridsize.x  = a->dim[0]/SYNCBLOCK; gridsize.x += (gridsize.x*SYNCBLOCK < a->dim[0]);
                gridsize.y  = a->dim[2]/SYNCBLOCK; gridsize.y += (gridsize.y*SYNCBLOCK < a->dim[2]);

                cudaMGHaloSyncY<<<gridsize, blocksize>>>(a->devicePtr[j], a->devicePtr[jn], a->dim[0], a->dim[1], a->dim[2], a->haloSize);
            } break;
            case 3: {

                size_t halotile = a->dim[0]*a->dim[1];
                size_t byteblock = halotile*a->haloSize*sizeof(double);

                size_t L_halo = (subL[5] - a->haloSize)*halotile;
                size_t L_src  = (subL[5]-2*a->haloSize)*halotile;

		// Fill right halo with left's source
                cudaMemcpy((void *)a->devicePtr[jn],
                           (void *)(a->devicePtr[j] + L_src), byteblock, cudaMemcpyDeviceToDevice);
                // Fill left halo with right's source
                cudaMemcpy((void *)(a->devicePtr[j] + L_halo),
                           (void *)(a->devicePtr[jn]+halotile*a->haloSize), byteblock, cudaMemcpyDeviceToDevice);

            } break;

        }
    }

    a++;
}
a--;
for(j = 0; j < a->nGPUs; j++) {
    	cudaSetDevice(a->deviceID[j]);
    	cudaDeviceSynchronize();
    }


}

// The Z halo exchange function is simply done by a pair of memcopies

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
__global__ void cudaMGHaloSyncX(double *L, double *R, int nxL, int nxR, int ny, int nz, int h)
{
	int y0 = threadIdx.y + blockDim.y*blockIdx.x;
	if(y0 >= ny) return;
	int z0 = threadIdx.z + blockDim.z*blockIdx.y;

	/* This will generate unaligned addresses, yes I'm sorry, DEAL WITH IT */
	L += nxL*(y0 + ny*z0) + nxL - 2*h + threadIdx.x;
	R += nxR*(y0 + ny*z0) + threadIdx.x;
	double alpha[2];

	int k;
	int hz = blockDim.z*gridDim.y;
	for(k = z0; k < nz; k+= hz) { /* This implicitly contains: if(z0 >= nz) { return; } */
		// read enough data, for sure
		alpha[0] = L[0];
		alpha[1] = R[h];

		if(threadIdx.x < h) {
			R[0] = alpha[0];
			L[h] = alpha[1];
		}

	    L   += nxL*ny*hz;
            R   += nxR*ny*hz;
	}

}


// FIXME: And this ny on both sides, also goddamnit.
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

/* Given an MGArray, allocates prod(g->dim) doubles at p and
 * copies it back to the cpu.
 * if(g->haloSize == PARTITION_CLONED), the partitionFrom-th device pointer is read*/
// FIXME: this entire serial crapshow should just make a few calls to cudaMemcpy2D/3D
int MGA_downloadArrayToCPU(MGArray *g, double **p, int partitionFrom)
{
	long numelOut = g->numel;

	// Create output numeric array if passed NULL
	// If e.g. returning to MATLAB, it will have already been allocated for us.
	if(p[0] == NULL)
		*p = (double *)malloc(numelOut * sizeof(double));

	int sub[6];
	int htrim[6];

	int u, v, w, i;
	int64_t iT, iS;
	double *gmem[g->nGPUs];

	if(g->haloSize == PARTITION_CLONED) {
		cudaSetDevice(g->deviceID[partitionFrom]);
		CHECK_CUDA_ERROR("cudaSetDevice()");
		cudaError_t fail = cudaMemcpy((void *)p, (void *)g->devicePtr[partitionFrom], numelOut*sizeof(double), cudaMemcpyDeviceToHost);
		CHECK_CUDA_ERROR("MGArray_downloadArrayToCPU");
		return 0;
	}

	/* Otherwise we need to do some actual work */
	/* Get all partitions streaming back */
	/* FIXME: look into making this asynchronous */
	for(i = 0; i < g->nGPUs; i++) {
		gmem[i] = (double *)malloc(g->partNumel[i]*sizeof(double));

		cudaError_t fail = cudaMemcpy((void *)gmem[i], (void *)g->devicePtr[i], g->partNumel[i]*sizeof(double), cudaMemcpyDeviceToHost);
		CHECK_CUDA_ERROR("MGArray_downloadArrayToCPU");
	}


	double *out = p[0];

	double *currentTarget;
	for(i = 0; i < g->nGPUs; i++) {
		// Get this partition's extent
		calcPartitionExtent(g, i, &sub[0]);
		// Trim the halo away when copying back to CPU
		for(u = 0; u < 6; u++) { htrim[u] = sub[u]; }
		if(i < (g->nGPUs-1)) { htrim[3+g->partitionDir-1] -= g->haloSize; }
		if(i > 0)           { htrim[g->partitionDir-1] += g->haloSize; htrim[3+g->partitionDir-1] -= g->haloSize; }

		currentTarget = gmem[i];

		// Copy into the output array
		// FIXME I can't get the mex compiler to use -fopenmp flags :(
#pragma omp parallel for private(u, v, w, iT, iS) default shared
		for(w = htrim[2]; w < htrim[2]+htrim[5]; w++)
			for(v = htrim[1]; v < htrim[1]+htrim[4]; v++)
				for(u = htrim[0]; u < htrim[0] + htrim[3]; u++) {
					iT = (u-sub[0]) + sub[3]*(v - sub[1]  + sub[4]*(w-sub[2]));
					iS = u+g->dim[0]*(v+g->dim[1]*w);

					out[iS] = currentTarget[iT];
				}
		free(currentTarget);

	}

	return 0;
}

int MGA_uploadArrayToGPU(double *p, MGArray *g, int partitionTo)
{
	int sub[6];
	int htrim[6];

	int u, v, w, i;
	int64_t iT, iS;
	double *gmem[g->nGPUs];

	if(g->haloSize == PARTITION_CLONED) {
		cudaSetDevice(g->deviceID[partitionTo]);
		CHECK_CUDA_ERROR("cudaSetDevice()");
		cudaError_t fail = cudaMemcpy((void *)g->devicePtr[partitionTo], (void *)p, g->numel*sizeof(double), cudaMemcpyHostToDevice);
		CHECK_CUDA_ERROR("MGArray_uploadArrayToGPU");
		return 0;
	}

	/* Otherwise we need to do some actual work */

	double *currentTarget;
	for(i = 0; i < g->nGPUs; i++) {
		gmem[i] = (double *)malloc(g->partNumel[i]*sizeof(double));
		// Get this partition's extent
		calcPartitionExtent(g, i, &sub[0]);
		// Trim the halo away when copying back to CPU
		for(u = 0; u < 6; u++) { htrim[u] = sub[u]; }
		if(i < (g->nGPUs-1)) { htrim[3+g->partitionDir-1] -= g->haloSize; }
		if(i > 0)           { htrim[g->partitionDir-1] += g->haloSize; htrim[3+g->partitionDir-1] -= g->haloSize; }

		currentTarget = gmem[i];

		// Copy into the output array
		// FIXME I can't get the mex compiler to use -fopenmp flags :(
#pragma omp parallel for private(u, v, w, iT, iS) default shared
		for(w = htrim[2]; w < htrim[2]+htrim[5]; w++)
			for(v = htrim[1]; v < htrim[1]+htrim[4]; v++)
				for(u = htrim[0]; u < htrim[0] + htrim[3]; u++) {
					iT = (u-sub[0]) + sub[3]*(v - sub[1]  + sub[4]*(w-sub[2]));
					iS = u+g->dim[0]*(v+g->dim[1]*w);

					currentTarget[iT] = p[iS];
				}


		cudaError_t fail = cudaMemcpy((void *)g->devicePtr[i], (void *)gmem[i], g->partNumel[i]*sizeof(double), cudaMemcpyHostToDevice);
		CHECK_CUDA_ERROR("MGArray_uploadArrayToGPU");
		free(currentTarget);
	}

        MGA_exchangeLocalHalos(g, 1);

	return 0;
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
printf("Rx'd integer: %i\n", i);

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

int myrank;
MPI_Commo_rank(MPI_COMM_WORLD, &myrank);

printf("cudaCheckError was non-success when polled at %s (%s:%i) by rank %i: %s -> %s\n", where, fname, lname, myrank, errorName(epicFail), cudaGetErrorString(epicFail));
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


