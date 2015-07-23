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

/* THIS FUNCTION
   Given F(xi, yi, zi) and direction dir = { X = 1, Y = 2, Z = 3), this function calculates

   F(xi, yi, zi) <- (F(xi, yi, zi) + F(xi + 1*(dir == 1), yi + 1*(dir == 2), zi + 1*(dir == 3) )/2

   using circular boundary conditions on all 3 directions */

__global__ void cukern_ForwardAverageX(double *in, double *out, int nx);
__global__ void cukern_ForwardAverageY(double *in, double *out, int nx, int ny);
__global__ void cukern_ForwardAverageZ(double *in, double *out, int nx, int nz);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if ((nrhs != 2) || (nlhs != 1)) {
        mexErrMsgTxt("Arguments must be result = cudaFwdAverage(array, direction)\n");
        }

    CHECK_CUDA_ERROR("entering cudaFwdAverage");

    MGArray in;
    int worked = MGA_accessMatlabArrays(prhs, 0, 0, &in);
    MGArray *out = MGA_createReturnedArrays(plhs, 1, &in);

    // Establish launch dimensions & a few other parameters
    int direction = (int)*mxGetPr(prhs[1]);

    dim3 arraySize = makeDim3(&in.dim[0]);
    dim3 blocksize, gridsize;
    blocksize.z = 1;
    gridsize.z = 1;

    cudaSetDevice(in.deviceID[0]);
    CHECK_CUDA_ERROR("cudaSetDevice()");

    // Interpolate the grid-aligned velocity
    switch(direction) {
        case 1:
//            blocksize.x = 18; blocksize.y = 8;
//            gridsize.x = arraySize.x / 14; gridsize.x += (14 * gridsize.x < arraySize.x);
//            gridsize.y = arraySize.y / blocksize.y; gridsize.y += (blocksize.y * gridsize.y < arraySize.y);
//            cukern_fwdAverageX<<<gridsize, blocksize>>>(srcs[0], dest[0], arraySize);
            blocksize = makeDim3(128, 1, 1);
            gridsize.x = arraySize.y; gridsize.y = arraySize.z;
            cukern_ForwardAverageX<<<gridsize, blocksize>>>(in.devicePtr[0], out->devicePtr[0], arraySize.x);
            break;
        case 2:
//            blocksize.x = 8; blocksize.y = 18;
//            gridsize.x = arraySize.x / 8; gridsize.x += (8 * gridsize.x < arraySize.x);
//            gridsize.y = arraySize.y / 14; gridsize.y += (14 * gridsize.x < arraySize.y);
//            cukern_fwdAverageY<<<gridsize, blocksize>>>(srcs[0], dest[0], arraySize);
            blocksize = makeDim3(64, 1, 1);
            gridsize.x = arraySize.x / 64; gridsize.x += (64*gridsize.x < arraySize.x);
            gridsize.y = arraySize.z;
            cukern_ForwardAverageY<<<gridsize, blocksize>>>(in.devicePtr[0], out->devicePtr[0], arraySize.x, arraySize.y);
            break;
        case 3:
//            blocksize.x = 18; blocksize.y = 8;
//            gridsize.x = arraySize.z / 14; gridsize.x += (14 * gridsize.x < arraySize.z);
//            gridsize.y = arraySize.x / blocksize.y; gridsize.y += (blocksize.y * gridsize.y < arraySize.x);
//            cukern_fwdAverageZ<<<gridsize, blocksize>>>(srcs[0], dest[0], arraySize);
              blocksize = makeDim3(64, 1, 1);
              gridsize.x = arraySize.x / 64; gridsize.x += (64*gridsize.x < arraySize.x);
              gridsize.y = arraySize.y;
              cukern_ForwardAverageZ<<<gridsize, blocksize>>>(in.devicePtr[0], out->devicePtr[0], arraySize.x, arraySize.z);
            break;
        }

    free(out);

    CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &in, direction, "Forward averaging");

}

/*
Invoke with a grid for which blockdim.x = size(array, Y) and blockdim.y = size(array,Z);
with 128 threads in the X direction.
*/
__global__ void cukern_ForwardAverageX(double *in, double *out, int nx)
{
int yAddr = blockIdx.x;
int zAddr = blockIdx.y;
int ny = gridDim.x;

int addrMax = nx*(yAddr + ny*zAddr + 1); // The address which we must not reach or go beyond is the start of the next line
int readBase = threadIdx.x + nx*(yAddr + ny*zAddr);
int writeBase= readBase;
readBase -= nx*(readBase >= addrMax);

__shared__ double lStore[256];

int locAddr = threadIdx.x;

//lStore[locAddr] = in[globBase + readX]; // load the first memory segment
lStore[locAddr] = in[readBase]; // load the first memory segment

do {
    readBase += 128;
    readBase -= nx*(readBase >= addrMax);
    lStore[(locAddr + 128) % 256] = in[readBase];

    __syncthreads(); // We have now read ahead by a segment. Calculate forward average, comrades!

    if(writeBase < addrMax) { out[writeBase] = .5*(lStore[locAddr] + lStore[(locAddr+1)%256]); } // If we are within range, that is.
    writeBase += 128; // Advance write address
    if(writeBase >= addrMax) return; // If write address is beyond nx, we're finished.
    locAddr ^= 128;

    __syncthreads();

    } while(1);

}

/* Invoke with a blockdim of <64, 1, 1> threads
Invoke with a griddim = <ceil[nx / 64], nz, 1> */
__global__ void cukern_ForwardAverageY(double *in, double *out, int nx, int ny)
{
int xaddr = blockDim.x * blockIdx.x + threadIdx.x; // There are however many X threads
if(xaddr >= nx) return; // truncate this right off

__shared__ double tileA[64];
__shared__ double tileB[64];

double *setA = tileA;
double *setB = tileB;
double *swap;

int readBase = xaddr + nx*ny*blockIdx.y; // set Raddr to x + nx ny z
int writeBase = readBase;
int addrMax = readBase + nx*(ny - 1); // Set this to the max address we want to handle in the loop

setB[threadIdx.x] = in[readBase]; // load set B (e.g. row 0)

while(writeBase < addrMax) { // Exit one BEFORE the max address to handle (since the max is a special case)
    swap = setB; // exchange A/B pointers
    setB = setA;
    setA = swap; // swap so that row 0 is set A
    
//    __syncthreads();

    readBase += nx; // move pointer down one row
    setB[threadIdx.x] = in[readBase]; // load row 1 into set B

//    __syncthreads();

    out[writeBase] = .5*(setB[threadIdx.x] + setA[threadIdx.x]); // average written to output

    writeBase += nx;
    }

readBase = xaddr + nx*ny*blockIdx.y; // reset readbase
setA[threadIdx.x] = in[readBase];
out[writeBase] = .5*(setB[threadIdx.x] + setA[threadIdx.x]); // average written to output

}

/* Invoke with a blockdim of <64, 1, 1> threads
Invoke with a griddim = <ceil[nx / 64], ny, 1> */
__global__ void cukern_ForwardAverageZ(double *in, double *out, int nx, int nz)
{
int xaddr = blockDim.x * blockIdx.x + threadIdx.x; // There are however magridDim.y X threads
if(xaddr >= nx) return; // truncate this right off

__shared__ double tileA[64];
__shared__ double tileB[64];

double *setA = tileA;
double *setB = tileB;
double *swap;

int readBase = xaddr + nx*blockIdx.y; // set Raddr to x + nx gridDim.y z
int writeBase = readBase;
int addrMax = readBase + nx*gridDim.y*(nz - 1); // Set this to the max address we want to handle in the loop

setB[threadIdx.x] = in[readBase]; // load set B (e.g. row 0)

while(writeBase < addrMax) { // Exit one BEFORE the max address to handle (since the max is a special case)
    swap = setB; // exchange A/B pointers
    setB = setA;
    setA = swap; // swap so that row 0 is set A

//    __syncthreads();

    readBase += nx*gridDim.y; // move pointer down one row
    setB[threadIdx.x] = in[readBase]; // load row 1 into set B

//    __syncthreads();

    out[writeBase] = .5*(setB[threadIdx.x] + setA[threadIdx.x]); // average written to output

    writeBase += nx*gridDim.y;
    }

readBase = xaddr + nx*blockIdx.y; // reset readbase
setA[threadIdx.x] = in[readBase];
out[writeBase] = .5*(setB[threadIdx.x] + setA[threadIdx.x]); // average written to output

}

