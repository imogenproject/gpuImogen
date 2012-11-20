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
#include "cudaEnums.h"

__global__ void cukern_circshift3D(double *in, double *out, int3 dimension, int3 shift);
__global__ void cukern_circshift2D(double *in, double *out, int dimx, int dimy, int shiftx, int shifty);
__global__ void cukern_circshift1D(double *in, double *out, int dimension, int shift);

__global__ void cukern_constshift3D(double *in, double *out, int3 dimension, int3 shift);
__global__ void cukern_constshift2D(double *in, double *out, int dimx, int dimy, int shiftx, int shifty);
__global__ void cukern_constshift1D(double *in, double *out, int dimension, int shift);


#define BLOCKDIMENSION_3D 8
#define BLOCKDIMENSION_2D 8
#define BLOCKDIMENSION_1D 64

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  dim3 blocksize; blocksize.z = 1;
  int numel; dim3 gridsize;

  if( (nlhs != 1) || (nrhs != 3)) { mexErrMsgTxt("circshift operator is shifted = cudaShift([nx ny nz], orig, shift_type)"); }

  cudaCheckError("entering cudaShift");

  double *shiftamt = mxGetPr(prhs[0]);
  ArrayMetadata amd;
  double **srcs = getGPUSourcePointers(prhs, &amd, 1, 1);

  int3 shift;
  shift.x = (int)shiftamt[0];
  shift.y = (int)shiftamt[1];
  shift.z = (int)shiftamt[2];

  int3 arrsize;
  double **destPtr;

  int shiftType = (int)*mxGetPr(prhs[2]);

  switch(amd.ndims) {
    case 3:
    blocksize.x = blocksize.y = BLOCKDIMENSION_3D;
    gridsize.x = amd.dim[0] / BLOCKDIMENSION_3D; if(gridsize.x * BLOCKDIMENSION_3D < amd.dim[0]) gridsize.x++;
    gridsize.y = amd.dim[1] / BLOCKDIMENSION_3D; if(gridsize.y * BLOCKDIMENSION_3D < amd.dim[1]) gridsize.y++;
    gridsize.z = 1;
    arrsize.x = amd.dim[0]; arrsize.y = amd.dim[1]; arrsize.z = amd.dim[2];

    destPtr = makeGPUDestinationArrays((int64_t *)mxGetData(prhs[1]), plhs, 1);
    switch(shiftType) {
        case CUDA_CIRC: cukern_circshift3D<<<gridsize, blocksize>>>(srcs[0], destPtr[0], arrsize, shift); break;
        case CUDA_CONST: cukern_constshift3D<<<gridsize, blocksize>>>(srcs[0], destPtr[0], arrsize, shift); break;
        };
    free(destPtr);
    break;
    case 2:
    blocksize.x = blocksize.y = BLOCKDIMENSION_2D;
    gridsize.x = amd.dim[0] / BLOCKDIMENSION_2D; if(gridsize.x * BLOCKDIMENSION_2D < amd.dim[0]) gridsize.x++;
    gridsize.y = amd.dim[1] / BLOCKDIMENSION_2D; if(gridsize.y * BLOCKDIMENSION_2D < amd.dim[1]) gridsize.y++;
    gridsize.z = 1; blocksize.z = 1;

    destPtr = makeGPUDestinationArrays((int64_t *)mxGetData(prhs[1]), plhs, 1);
    switch(shiftType) {
        case CUDA_CIRC: cukern_circshift2D<<<gridsize, blocksize>>>(srcs[0], destPtr[0], amd.dim[0], amd.dim[1], shift.x, shift.y); break;
        case CUDA_CONST: cukern_circshift2D<<<gridsize, blocksize>>>(srcs[0], destPtr[0], amd.dim[0], amd.dim[1], shift.x, shift.y); break;
        }
    free(destPtr);
    break;
    case 1:
    blocksize.x = BLOCKDIMENSION_1D;
    gridsize.x = amd.dim[0] / BLOCKDIMENSION_1D; if(gridsize.x * BLOCKDIMENSION_1D < amd.dim[0]) gridsize.x++;
    gridsize.y = 1; blocksize.y = 1;
    gridsize.z = 1; blocksize.z = 1;

    destPtr = makeGPUDestinationArrays((int64_t *)mxGetData(prhs[1]), plhs, 1);
    switch(shiftType) {
        case CUDA_CIRC: cukern_circshift1D<<<gridsize, blocksize>>>(srcs[0], destPtr[0], amd.dim[0], shift.x); break;
        case CUDA_CONST: cukern_constshift1D<<<gridsize, blocksize>>>(srcs[0], destPtr[0], amd.dim[0], shift.x); break;
        };
    free(destPtr);
    break;
  }

cudaError_t epicFail = cudaGetLastError();
if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, amd.ndims, "cudaShift");


}

/**************** THREE DIMENSIONAL SHIFT ROUTINES ******************/

__global__ void cukern_circshift3D(double *in, double *out, int3 dimension, int3 shift)
{
__shared__ double lBlock[BLOCKDIMENSION_3D][BLOCKDIMENSION_3D];

int ctrZ;

int idxX = threadIdx.x + BLOCKDIMENSION_3D*blockIdx.x;
int idxY = threadIdx.y + BLOCKDIMENSION_3D*blockIdx.y;
int idxZ = 0;

if((idxX >= dimension.x) || (idxY >= dimension.y)) return;

int idxWrite = idxX + dimension.x * idxY;

idxX = (idxX + shift.x); idxX += (idxX < 0)*dimension.x;
idxY = (idxY + shift.y); idxY += (idxY < 0)*dimension.y;

idxX = idxX % dimension.x;
idxY = idxY % dimension.y;

idxZ = shift.z; idxZ += (idxZ < 0)*dimension.z;
idxZ = idxZ % dimension.z;

int idxRead = idxX + dimension.x * (idxY + dimension.y * idxZ);

for(ctrZ = 0; ctrZ < dimension.z; ctrZ++) {
    lBlock[threadIdx.x][threadIdx.y] = in[idxRead];
    __syncthreads();

    out[idxWrite] = lBlock[threadIdx.x][threadIdx.y];

    idxWrite += dimension.x*dimension.y;
    idxRead  += dimension.x*dimension.y;
    idxRead = idxRead % (dimension.x * dimension.y * dimension.z);    

    __syncthreads();

    }

}


__global__ void cukern_constshift3D(double *in, double *out, int3 dimension, int3 shift)
{
__shared__ double lBlock[BLOCKDIMENSION_3D][BLOCKDIMENSION_3D];

int ctrZ;

int idxX = threadIdx.x + BLOCKDIMENSION_3D*blockIdx.x;
int idxY = threadIdx.y + BLOCKDIMENSION_3D*blockIdx.y;
int idxZ = 0;

if((idxX >= dimension.x) || (idxY >= dimension.y)) return;

int idxWrite = idxX + dimension.x * idxY;

idxX = (idxX + shift.x);
if(idxX < 0) idxX = 0;
if(idxX >= dimension.x) idxX = dimension.x - 1;
idxY = (idxY + shift.y);
if(idxY < 0) idxY = 0;
if(idxY >= dimension.y) idxY = dimension.y - 1;

idxZ = shift.z;
if(idxZ < 0) idxZ = 0;

int idxRead = idxX + dimension.x * (idxY + dimension.y * idxZ);

if(shift.z < 0) { // Nondivergent branch 
    shift.z = -shift.z;

    for(ctrZ = 0; ctrZ < dimension.z; ctrZ++) {
         lBlock[threadIdx.x][threadIdx.y] = in[idxRead];
         __syncthreads();
 
        out[idxWrite] = lBlock[threadIdx.x][threadIdx.y];

        if(ctrZ > shift.z) idxRead += dimension.x*dimension.y; // nondivergent branch
        idxWrite  += dimension.x*dimension.y;
        __syncthreads();
        }
    } else {
    idxRead += dimension.x*dimension.y*shift.z;
    shift.z = dimension.z - shift.z;

    for(ctrZ = 0; ctrZ < dimension.z; ctrZ++) {
         lBlock[threadIdx.x][threadIdx.y] = in[idxRead];
         __syncthreads();

        out[idxWrite] = lBlock[threadIdx.x][threadIdx.y];

        if(ctrZ < shift.z) idxRead += dimension.x*dimension.y; // nondivergent branch
        idxWrite  += dimension.x*dimension.y;
        __syncthreads();
        }
    }



}



/**************** TWO DIMENSIONAL SHIFT ROUTINES *******************/

__global__ void cukern_circshift2D(double *in, double *out, int dimx, int dimy, int shiftx, int shifty)
{
__shared__ double lBlock[BLOCKDIMENSION_2D][BLOCKDIMENSION_2D];

int idxX = threadIdx.x + BLOCKDIMENSION_2D*blockIdx.x;
int idxY = threadIdx.y + BLOCKDIMENSION_2D*blockIdx.y;

if((idxX >= dimx) || (idxY >= dimy)) return;

int idxWrite = idxX + dimx * idxY;

idxX = (idxX + shiftx); idxX += (idxX < 0)*dimx;
idxY = (idxY + shifty); idxY += (idxY < 0)*dimy;

idxX = idxX % dimx;
idxY = idxY % dimy;

int idxRead = idxX + dimx * idxY;

lBlock[threadIdx.x][threadIdx.y] = in[idxRead];
__syncthreads();
out[idxWrite] = lBlock[threadIdx.x][threadIdx.y];

}

__global__ void cukern_constshift2D(double *in, double *out, int dimx, int dimy, int shiftx, int shifty)
{
__shared__ double lBlock[BLOCKDIMENSION_2D][BLOCKDIMENSION_2D];

int idxX = threadIdx.x + BLOCKDIMENSION_2D*blockIdx.x;
int idxY = threadIdx.y + BLOCKDIMENSION_2D*blockIdx.y;

if((idxX >= dimx) || (idxY >= dimy)) return;

int idxWrite = idxX + dimx * idxY;

idxX = (idxX + shiftx);
if(idxX < 0) idxX = 0;
if(idxX >= dimx) idxX = dimx - 1;
idxY = (idxY + shifty);
if(idxY < 0) idxY = 0;
if(idxY >= dimy) idxY = dimy - 1;

int idxRead = idxX + dimx * idxY;

lBlock[threadIdx.x][threadIdx.y] = in[idxRead];
__syncthreads();
out[idxWrite] = lBlock[threadIdx.x][threadIdx.y];

}

/************* ONE DIMENSIONAL SHIFT ROUTINES *******************/

__global__ void cukern_circshift1D(double *in, double *out, int dimension, int shift)
{
int idxX0 = threadIdx.x + BLOCKDIMENSION_1D*blockIdx.x;

if((idxX0 >= dimension)) return;

int idxX = idxX0 + shift; // Implement circular rotation
idxX += (idxX0 < 0)*dimension;
idxX %= dimension;

__shared__ double lblock[BLOCKDIMENSION_1D];

lblock[threadIdx.x] = in[idxX];

__syncthreads();

out[idxX0] = lblock[threadIdx.x];

}


__global__ void cukern_constshift1D(double *in, double *out, int dimension, int shift)
{
int idxX0 = threadIdx.x + BLOCKDIMENSION_1D*blockIdx.x;

if((idxX0 >= dimension)) return;

int idxX = idxX0 + shift;
if(idxX < 0) idxX = 0; // Implement rotation with constant at edges
if(idxX >= dimension) idxX = dimension-1;

__shared__ double lblock[BLOCKDIMENSION_1D];

lblock[threadIdx.x] = in[idxX];

__syncthreads();

out[idxX0] = lblock[threadIdx.x];


}
