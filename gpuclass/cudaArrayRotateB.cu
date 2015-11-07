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
__global__ void cukern_ArrayExchangeY(double *src, double *dst,   int nx, int ny, int nz);
__global__ void cukern_ArrayExchangeZ(double *src, double *dst,   int nx, int ny, int nz);

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

	dim3 blocksize; blocksize.x = blocksize.y = BDIM; blocksize.z = 1;
	dim3 gridsize;
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

	int indExchange = (int)*mxGetPr(prhs[1]);
	int i, sub[6];

	indExchange -= 1;
	int is3d = (src.dim[2] > 1);
	int newPartDirect;

	if(indExchange == 1) { newPartDirect = PARTITION_Y; } else { newPartDirect = PARTITION_Z; }

	gridsize.x = src.dim[0] / BDIM; if(gridsize.x*BDIM < src.dim[0]) gridsize.x++;
	gridsize.y = src.dim[indExchange] / BDIM; if(gridsize.y*BDIM < src.dim[indExchange]) gridsize.y++;

	blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

	// Transpose X and Y sizes
	trans.dim[0] = src.dim[indExchange];
	trans.dim[indExchange] = src.dim[0];
	// Flip the partition direction if appropriate
	if(trans.partitionDir == PARTITION_X) {
		trans.partitionDir = newPartDirect;
	} else if(trans.partitionDir == newPartDirect) {
		trans.partitionDir = PARTITION_X;
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

	for(i = 0; i < trans.nGPUs; i++) {
		cudaSetDevice(trans.deviceID[i]);
		CHECK_CUDA_ERROR("cudaSetDevice()");
		calcPartitionExtent(&src, i, sub);

		if(indExchange == 2) {
			cukern_ArrayExchangeZ<<<gridsize, blocksize>>>(src.devicePtr[i], nuClone->devicePtr[i], sub[3], sub[4], sub[5]);
		} else {
			if(is3d) {
				cukern_ArrayExchangeY<<<gridsize, blocksize>>>(src.devicePtr[i], nuClone->devicePtr[i], sub[3],sub[4],sub[5]);
			} else {
				cukern_ArrayTranspose2D<<<gridsize, blocksize>>>(src.devicePtr[i], nuClone->devicePtr[i], sub[3], sub[4]);
			}
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

__global__ void cukern_ArrayExchangeY(double *src, double *dst, int nx, int ny, int nz)
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

__global__ void cukern_ArrayExchangeZ(double*src, double *dst, int nx, int ny, int nz)
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

