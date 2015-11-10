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

__global__ void cukern_ArrayTranspose2D(double *src, double *dst, int nx, int ny);
__global__ void cukern_ArrayExchangeXY(double *src, double *dst,   int nx, int ny, int nz);
__global__ void cukern_ArrayExchangeXZ(double *src, double *dst,   int nx, int ny, int nz);
__global__ void cukern_ArrayExchangeYZ(double *src, double *dst,   int nx, int ny, int nz);


#define BDIM 16

__global__ void cukern_dumbblit(double *src, double *dst, int nx, int ny, int nz);

__global__ void cukern_dumbblit(double *src, double *dst, int nx, int ny, int nz)
{
	//int myx = threadIdx.x + BDIM*blockIdx.x;
	//int myy = threadIdx.y + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
	//int myaddr = myx + nx*myy;

	//if((myx < nx) && (myy < ny)) dst[myaddr] = src[myaddr];
	return;

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int makeNew = 0;

        if(nrhs != 2) { mexErrMsgTxt("Input args must be cudaArrayRotateB(GPU array, dir to transpose with X)"); }
	switch(nlhs) {
		case 0: makeNew = 0; break;
		case 1: makeNew = 1; break;
		default: mexErrMsgTxt("cudaArrayRotate must return zero (in-place transpose) or one (new array) arguments."); break;
	}
	CHECK_CUDA_ERROR("entering cudaArrayRotateB");

	MGArray src;
	int worked = MGA_accessMatlabArrays(prhs, 0, 0, &src);

	/* This function will make the partition direction track the transposition of indices
	 * Such that if partitioning direction is X and a Y transpose is done, partition is in Y.
	 * The full matrix
	 * transpose =  XY | XZ | YZ |
	 *            +----+----+----+
	 * Part.    X | Y  | Z  | X  |
	 * initial  Y | X  | Y  | Z  | <- Output array will have partition in this direction
	 * direct   Z | Z  | X  | Y  |
	 */

	MGArray trans = src;

	int indExchange = (int)*mxGetPr(prhs[1]) - 1;
	int i, sub[6];

	int is3d = (src.dim[2] > 1);

	// Transpose XY or XZ
	switch(indExchange) {
	case 1: /* flip X <-> Y */
		trans.dim[0] = src.dim[1];
		trans.dim[1] = src.dim[0];

		if(src.partitionDir == PARTITION_X) { trans.partitionDir = PARTITION_Y; }
		if(src.partitionDir == PARTITION_Y) { trans.partitionDir = PARTITION_X; }
		break;
	case 2: /* flip X <-> Z */
		trans.dim[0] = src.dim[2];
		trans.dim[2] = src.dim[0];

		if(src.partitionDir == PARTITION_X) { trans.partitionDir = PARTITION_Z; }
		if(src.partitionDir == PARTITION_Z) { trans.partitionDir = PARTITION_X; }
		break;
	case 3: /* flip Y <-> Z */
		trans.dim[2] = src.dim[1];
		trans.dim[1] = src.dim[2];
		// Flip the partition direction if appropriate
		if(src.partitionDir == PARTITION_Y) { trans.partitionDir = PARTITION_Z; }
		if(src.partitionDir == PARTITION_Z) { trans.partitionDir = PARTITION_Y; }
		break;
	default:
		mexErrMsgTxt("Index to exchange is invalid!");
	}

	// Recalculate the partition sizes
	for(i = 0; i < trans.nGPUs; i++) {
		calcPartitionExtent(&trans, i, sub);
		trans.partNumel[i] = sub[3]*sub[4]*sub[5];
	}


	MGArray *nuClone;

	if(makeNew) {
		// allocate storage and return the newly transposed array
		nuClone = MGA_createReturnedArrays(plhs, 1, &trans);
	} else {
		// just allocate storage; Overwrite original tag
		nuClone = MGA_allocArrays(1, &trans);
		serializeMGArrayToTag(&trans, (int64_t *)mxGetData(prhs[0]));
	}

	dim3 blocksize, gridsize;

	for(i = 0; i < trans.nGPUs; i++) {
		cudaSetDevice(trans.deviceID[i]);
		CHECK_CUDA_ERROR("cudaSetDevice()");
		calcPartitionExtent(&src, i, sub);

		switch(indExchange) {
		case 1:
			blocksize.x = blocksize.y = BDIM; blocksize.z = 1;
			gridsize.x = ROUNDUPTO(src.dim[0], BDIM) / BDIM;
			gridsize.y = ROUNDUPTO(src.dim[1], BDIM) / BDIM;
			gridsize.z = 1;

			if(is3d) {
				cukern_ArrayExchangeXY<<<gridsize, blocksize>>>(src.devicePtr[i], nuClone->devicePtr[i], sub[3],sub[4],sub[5]);
			} else {
				cukern_ArrayTranspose2D<<<gridsize, blocksize>>>(src.devicePtr[i], nuClone->devicePtr[i], sub[3], sub[4]);
			}
			break;
		case 2:
			blocksize.x = blocksize.y = BDIM; blocksize.z = 1;
			gridsize.x = ROUNDUPTO(src.dim[0], BDIM) / BDIM;
			gridsize.y = ROUNDUPTO(src.dim[2], BDIM) / BDIM;
			gridsize.z = 1;

			cukern_ArrayExchangeXZ<<<gridsize, blocksize>>>(src.devicePtr[i], nuClone->devicePtr[i], sub[3], sub[4], sub[5]);
			break;

		case 3:
			blocksize.x = 32;
			blocksize.y = blocksize.z = 4;

			gridsize.x = ROUNDUPTO(sub[3], blocksize.x) / blocksize.x;
			gridsize.y = ROUNDUPTO(sub[4], blocksize.y) / blocksize.y;
			gridsize.z = 1;

			cukern_ArrayExchangeYZ<<<gridsize, blocksize>>>(src.devicePtr[i], nuClone->devicePtr[i], sub[3],sub[4],sub[5]);

			break;
		}
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &src, i, "array transposition");
	}

	// If performing transpose in place, move transposed data back to original array.
	if(makeNew == 0) {
		for(i = 0; i < trans.nGPUs; i++) {
			cudaSetDevice(trans.deviceID[i]);
			cudaMemcpyAsync(trans.devicePtr[i], nuClone->devicePtr[i], trans.partNumel[i]*sizeof(double), cudaMemcpyDeviceToDevice);
			CHECK_CUDA_ERROR("cudaMemcpy");
		}

		MGA_delete(nuClone);
	}

	CHECK_CUDA_ERROR("Departing cudaArrayRotateB");

}

__global__ void cukern_ArrayTranspose2D(double *src, double *dst, int nx, int ny)
{
	__shared__ double tmp[BDIM][BDIM];

	int myx = threadIdx.x + BDIM*blockIdx.x;
	int myy = threadIdx.y + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
	int myAddr = myx + nx*myy;

	if((myx < nx) && (myy < ny)) tmp[threadIdx.y][threadIdx.x] = src[myAddr];

	__syncthreads();

	//myx = threadIdx.x + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
	myAddr = myy + threadIdx.x - threadIdx.y;
	//myy = threadIdx.y + BDIM*blockIdx.x;
	myy  = myx + threadIdx.y - threadIdx.x;
	myx = myAddr;

	myAddr = myx + ny*myy;

	if((myx < ny) && (myy < nx)) dst[myAddr] = tmp[threadIdx.x][threadIdx.y];

}

__global__ void cukern_ArrayExchangeXY(double *src, double *dst, int nx, int ny, int nz)
{

	__shared__ double tmp[BDIM][BDIM+1];

	int myx = threadIdx.x + BDIM*blockIdx.x;
	int myy = threadIdx.y + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
	int mySrcAddr = myx + nx*myy;
	bool doRead = 0;
	bool doWrite = 0;

	if((myx < nx) && (myy < ny)) doRead = 1;

	myx = threadIdx.x + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
	myy = threadIdx.y + BDIM*blockIdx.x;
	int myDstAddr = myx + ny*myy;

	if((myx < ny) && (myy < nx)) doWrite = 1;

	for(myx = 0; myx < nz; myx++) {
		if(doRead) tmp[threadIdx.y][threadIdx.x] = src[mySrcAddr];
		mySrcAddr += nx*ny;
		__syncthreads();

		if(doWrite) dst[myDstAddr] = tmp[threadIdx.x][threadIdx.y];
		myDstAddr += nx*ny;
		__syncthreads();
	}

}

__global__ void cukern_ArrayExchangeXZ(double*src, double *dst, int nx, int ny, int nz)
{
	__shared__ double tmp[BDIM][BDIM+1];

	int myx = threadIdx.x + BDIM*blockIdx.x;
	int myz = threadIdx.y + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
	int mySrcAddr = myx + nx*ny*myz;
	bool doRead = 0;
	bool doWrite = 0;

	if((myx < nx) && (myz < nz)) doRead = 1;

	myx = threadIdx.x + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
	myz = threadIdx.y + BDIM*blockIdx.x;
	int myDstAddr = myx + nz*ny*myz;

	if((myx < nz) && (myz < nx)) doWrite = 1;

	for(myx = 0; myx < ny; myx++) {
		if(doRead) tmp[threadIdx.y][threadIdx.x] = src[mySrcAddr];
		mySrcAddr += nx;
		__syncthreads();

		if(doWrite) dst[myDstAddr] = tmp[threadIdx.x][threadIdx.y];
		myDstAddr += nz;
		__syncthreads();
	}


}

/* Assume we have threads that span X and input Y; step in input Z */
__global__ void cukern_ArrayExchangeYZ(double *src, double *dst,   int nx, int ny, int nz)
{
int myx = threadIdx.x + blockDim.x*blockIdx.x;
if(myx >= nx) return;

int inputY = threadIdx.y + blockDim.y*blockIdx.y;
if(inputY >= ny) return;

int inputZ;
int outputY, outputZ;

/* Because the natively aligned index doesn't change, no need to stream through shmem to maintain global r/w coalescence */
for(inputZ = threadIdx.z; inputZ < nz; inputZ += blockDim.z) {
	outputY = inputZ;
	outputZ = inputY;
	
	dst[myx + outputY*nx + outputZ*nx*nz] = 
	src[myx + inputY*nx + inputZ *nx*ny];

}


}
