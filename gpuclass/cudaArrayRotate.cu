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
  ArrayMetadata amd;
  double **srcs = getGPUSourcePointers(prhs, &amd, 0, 0);

  int64_t *oldref = (int64_t *)mxGetData(prhs[0]);
  int64_t newref[5];

  int indExchange = (int)*mxGetPr(prhs[1]);

  switch(oldref[1]) { /* on # dimensions */
    case 3:
      if(indExchange == 2) {
        gridsize.x = amd.dim[0] / BDIM; if(gridsize.x*BDIM < amd.dim[0]) gridsize.x++;
        gridsize.y = amd.dim[1] / BDIM; if(gridsize.y*BDIM < amd.dim[1]) gridsize.y++;

        blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

        newref[0] = 0; newref[1] = oldref[1];
        newref[2] = oldref[3]; newref[3] = oldref[2]; newref[4] = oldref[4];

        double **destPtr = makeGPUDestinationArrays(newref, plhs, 1);
        cukern_ArrayExchangeY<<<gridsize, blocksize>>>(srcs[0], destPtr[0], amd.dim[0], amd.dim[1], amd.dim[2]);
        }
      if(indExchange == 3) {
        gridsize.x = amd.dim[0] / BDIM; if(gridsize.x*BDIM < amd.dim[0]) gridsize.x++;
        gridsize.y = amd.dim[2] / BDIM; if(gridsize.y*BDIM < amd.dim[2]) gridsize.y++;

        blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

        newref[0] = 0; newref[1] = oldref[1];
        newref[2] = oldref[4]; newref[3] = oldref[3]; newref[4] = oldref[2];

        double **destPtr = makeGPUDestinationArrays(newref, plhs, 1);
        cukern_ArrayExchangeZ<<<gridsize, blocksize>>>(srcs[0], destPtr[0], amd.dim[0], amd.dim[1], amd.dim[2]);
        }
      break;
    case 2:
      gridsize.x = amd.dim[0] / BDIM; if(gridsize.x*BDIM < amd.dim[0]) gridsize.x++;
      gridsize.y = amd.dim[1] / BDIM; if(gridsize.y*BDIM < amd.dim[1]) gridsize.y++;

      blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

      newref[0] = 0; newref[1] = oldref[1];
      newref[2] = oldref[3]; newref[3] = oldref[2]; newref[4] = oldref[4];

      double **destPtr = makeGPUDestinationArrays(newref, plhs, 1);
      cudaError_t fail = cudaGetLastError();
      if(fail != cudaSuccess) printf("cudaArrayRotate: allocate failed; %s\n", cudaGetErrorString(fail));

      cukern_ArrayTranspose2D<<<gridsize, blocksize>>>(srcs[0], destPtr[0], amd.dim[0], amd.dim[1]);
      fail = cudaGetLastError();
      if(fail != cudaSuccess) printf("cudaArrayRotate: kernel invocation failed; %s\n", cudaGetErrorString(fail));

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

myx = threadIdx.x + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
myy = threadIdx.y + BDIM*blockIdx.x;
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

