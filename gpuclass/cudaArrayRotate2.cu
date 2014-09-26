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

  if((nlhs != 0) || (nrhs != 2)) { mexErrMsgTxt("cudaArrayRotate2 operator is cudaArrayRotate2(array, dir)\n"); }
  CHECK_CUDA_ERROR("entering cudaArrayRotate");

  MGArray src;
  int worked = accessMGArrays(prhs, 0, 0, &src);
  PAR_WARN(src)
  MGArray copy = src;
  MGArray *clone;

  int indExchange = (int)*mxGetPr(prhs[1]);
  int i, sub[6];
  switch(src.dim[2] > 1 ? 3 : 2) { /* on # dimensions */
    case 3:
      if(indExchange == 2) {
        gridsize.x = src.dim[0] / BDIM; if(gridsize.x*BDIM < src.dim[0]) gridsize.x++;
        gridsize.y = src.dim[1] / BDIM; if(gridsize.y*BDIM < src.dim[1]) gridsize.y++;

        blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

	// Transpose X and Y
        copy.dim[0] = src.dim[1];
        copy.dim[1] = src.dim[0];
        // Recalculate the partition sizes
	for(i = 0; i < copy.nGPUs; i++) {
            calcPartitionExtent(&copy, i, sub);
            copy.partNumel[i] = sub[3]*sub[4]*sub[5];
        }
	// Allocate new memory
	clone = allocMGArrays(1, &copy);
        // Overwrite the original tag
        serializeMGArrayToTag(clone, (int64_t *)mxGetData(prhs[0]));

	for(i = 0; i < copy.nGPUs; i++) {
            cudaSetDevice(copy.deviceID[i]);
            CHECK_CUDA_ERROR("cudaSetDevice()");
            cukern_ArrayExchangeY<<<gridsize, blocksize>>>(src.devicePtr[i], clone->devicePtr[i], src.dim[0], src.dim[1], src.dim[2]);
            CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &src, i, "array transposition");
            cudaFree(src.devicePtr[i]);
        }
	free(clone);
        }
      if(indExchange == 3) {
        gridsize.x = src.dim[0] / BDIM; if(gridsize.x*BDIM < src.dim[0]) gridsize.x++;
        gridsize.y = src.dim[2] / BDIM; if(gridsize.y*BDIM < src.dim[2]) gridsize.y++;

        blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

        // Transpose X and Z
        copy.dim[0] = src.dim[2];
        copy.dim[2] = src.dim[0];
        // Recalculate the partition sizes
        for(i = 0; i < copy.nGPUs; i++) {
            calcPartitionExtent(&copy, i, sub);
            copy.partNumel[i] = sub[3]*sub[4]*sub[5];
        }
        // Allocate new memory
        clone = allocMGArrays(1, &copy);
        // Overwrite the original tag
        serializeMGArrayToTag(clone, (int64_t *)mxGetData(prhs[0]));

        for(i = 0; i < copy.nGPUs; i++) {
            cudaSetDevice(copy.deviceID[i]);
            CHECK_CUDA_ERROR("cudaSetDevice()");
            cukern_ArrayExchangeZ<<<gridsize, blocksize>>>(src.devicePtr[i], clone->devicePtr[i], src.dim[0], src.dim[1], src.dim[2]);
            CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &src, i, "array transposition");
            cudaFree(src.devicePtr[i]);
        }

	free(clone);
        }
      break;
    case 2:
      gridsize.x = src.dim[0] / BDIM; if(gridsize.x*BDIM < src.dim[0]) gridsize.x++;
      gridsize.y = src.dim[1] / BDIM; if(gridsize.y*BDIM < src.dim[1]) gridsize.y++;

      blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

      // Transpose X and Y
      copy.dim[0] = src.dim[1];
      copy.dim[1] = src.dim[0];
      // Recalculate the partition sizes
      for(i = 0; i < copy.nGPUs; i++) {
          calcPartitionExtent(&copy, i, sub);
          copy.partNumel[i] = sub[3]*sub[4]*sub[5];
      }
      // Allocate new memory
      clone = allocMGArrays(1, &copy);
      // Overwrite the original tag
      serializeMGArrayToTag(clone, (int64_t *)mxGetData(prhs[0]));

      for(i = 0; i < copy.nGPUs; i++) {
          cudaSetDevice(copy.deviceID[i]);
          CHECK_CUDA_ERROR("cudaSetDevice()");
          cukern_ArrayTranspose2D<<<gridsize, blocksize>>>(src.devicePtr[i], clone->devicePtr[i], src.dim[0], src.dim[1]);
          CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &src, i, "array transposition");
          cudaFree(src.devicePtr[i]);
      }
      free(clone);

      break;      
    }


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

__shared__ double tmp[BDIM][BDIM];

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
__shared__ double tmp[BDIM][BDIM];

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

