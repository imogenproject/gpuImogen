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

/* THIS FUNCTION:
   This function performs array index exchanges.
   Given C's linear-to-n-dimensional mapping, index = x + Nx (y + ny (z + ...) ... ),
   with the understanding that 'dir' means { x = 1, y = 2, z = 3 },
   exchanges the given index with the x direction of the input array
*/

__global__ void cukern_ArrayTranspose2D(double *src, double *dst, int nx, int ny);
__global__ void cukern_ArrayExchangeY(double *src, double *dst, int nx, int ny, int nz);
__global__ void cukern_ArrayExchangeZ(double *src, double *dst, int nx, int ny, int nz);

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

  if((nlhs != 1) || (nrhs != 2)) { mexErrMsgTxt("cudaArrayRotate operator is rotated = cudaArrayRotate(array, dir)\n"); }
  CHECK_CUDA_ERROR("entering cudaArrayRotate");

  ArrayMetadata amd;
  double **srcs = getGPUSourcePointers(prhs, &amd, 0, 0);

  int64_t oldref[5];
  arrayMetadataToTag(&amd, &oldref[0]);

  int indExchange = (int)*mxGetPr(prhs[1]);

  switch(oldref[1]) { /* on # dimensions */
    case 3:
      if(indExchange == 2) {
        gridsize.x = amd.dim[0] / BDIM; if(gridsize.x*BDIM < amd.dim[0]) gridsize.x++;
        gridsize.y = amd.dim[1] / BDIM; if(gridsize.y*BDIM < amd.dim[1]) gridsize.y++;

        blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

        ArrayMetadata newAMD = amd;
        newAMD.dim[1] = amd.dim[0];
        newAMD.dim[0] = amd.dim[1];
        double **destPtr = makeGPUDestinationArrays(&newAMD, plhs, 1);

        cukern_ArrayExchangeY<<<gridsize, blocksize>>>(srcs[0], destPtr[0], amd.dim[0], amd.dim[1], amd.dim[2]);
        }
      if(indExchange == 3) {
        gridsize.x = amd.dim[0] / BDIM; if(gridsize.x*BDIM < amd.dim[0]) gridsize.x++;
        gridsize.y = amd.dim[2] / BDIM; if(gridsize.y*BDIM < amd.dim[2]) gridsize.y++;

        blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

        ArrayMetadata newAMD = amd;
        newAMD.dim[0] = amd.dim[2];
        newAMD.dim[2] = amd.dim[0];
        double **destPtr = makeGPUDestinationArrays(&newAMD, plhs, 1);

        cukern_ArrayExchangeZ<<<gridsize, blocksize>>>(srcs[0], destPtr[0], amd.dim[0], amd.dim[1], amd.dim[2]);
        }
      break;
    case 2:
      gridsize.x = amd.dim[0] / BDIM; if(gridsize.x*BDIM < amd.dim[0]) gridsize.x++;
      gridsize.y = amd.dim[1] / BDIM; if(gridsize.y*BDIM < amd.dim[1]) gridsize.y++;

      blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

      ArrayMetadata newAMD = amd;
      newAMD.dim[0] = amd.dim[1];
      newAMD.dim[1] = amd.dim[0];
      double **destPtr = makeGPUDestinationArrays(&newAMD, plhs, 1);

      cukern_ArrayTranspose2D<<<gridsize, blocksize>>>(srcs[0], destPtr[0], amd.dim[0], amd.dim[1]);

      break;      
    }

cudaError_t epicFail = cudaGetLastError();
if(epicFail != cudaSuccess) CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, oldref[1], "array transposition");

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

