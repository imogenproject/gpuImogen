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

__global__ void cukern_fwdAverageX(double *a, double *b, int3 dims);
__global__ void cukern_ForwardAverageX(double *in, double *out, int nx);
__global__ void cukern_ForwardAverageY(double *in, double *out, int nx, int ny);
__global__ void cukern_ForwardAverageZ(double *in, double *out, int nx, int nz);

__global__ void cukern_fwdAverageY(double *a, double *b, int3 dims);
__global__ void cukern_fwdAverageZ(double *a, double *b, int3 dims);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if ((nrhs != 2) || (nlhs != 1)) {
        mexErrMsgTxt("Arguments must be result = cudaFwdAverage(array, direction)\n");
        }

    // Get source array info and create destination arrays
    ArrayMetadata amd;
    double **srcs = getGPUSourcePointers(prhs, &amd, 0, 0);
    double **dest = makeGPUDestinationArrays((int64_t *)mxGetData(prhs[0]), plhs, 1);

    // Establish launch dimensions & a few other parameters
    int direction = (int)*mxGetPr(prhs[1]);

    int3 arraySize;
    arraySize.x = amd.dim[0];
    amd.ndims > 1 ? arraySize.y = amd.dim[1] : arraySize.y = 1;
    amd.ndims > 2 ? arraySize.z = amd.dim[2] : arraySize.z = 1;

    dim3 blocksize, gridsize;
    blocksize.z = 1;
    gridsize.z = 1;

    // Interpolate the grid-aligned velocity
    switch(direction) {
        case 1:
//            blocksize.x = 18; blocksize.y = 8;
//            gridsize.x = arraySize.x / 14; gridsize.x += (14 * gridsize.x < arraySize.x);
//            gridsize.y = arraySize.y / blocksize.y; gridsize.y += (blocksize.y * gridsize.y < arraySize.y);
//            cukern_fwdAverageX<<<gridsize, blocksize>>>(srcs[0], dest[0], arraySize);
            blocksize.x = 128; blocksize.y = blocksize.z = 1;
            gridsize.x = arraySize.y; gridsize.y = arraySize.z;
            cukern_ForwardAverageX<<<gridsize, blocksize>>>(srcs[0], dest[0], arraySize.x);
            break;
        case 2:
//            blocksize.x = 8; blocksize.y = 18;
//            gridsize.x = arraySize.x / 8; gridsize.x += (8 * gridsize.x < arraySize.x);
//            gridsize.y = arraySize.y / 14; gridsize.y += (14 * gridsize.x < arraySize.y);
//            cukern_fwdAverageY<<<gridsize, blocksize>>>(srcs[0], dest[0], arraySize);
            blocksize.x = 64; blocksize.y = blocksize.z = 1;
            gridsize.x = arraySize.x / 64; gridsize.x += (64*gridsize.x < arraySize.x);
            gridsize.y = arraySize.z;
            cukern_ForwardAverageY<<<gridsize, blocksize>>>(srcs[0], dest[0], arraySize.x, arraySize.y);
            break;
        case 3:
//            blocksize.x = 18; blocksize.y = 8;
//            gridsize.x = arraySize.z / 14; gridsize.x += (14 * gridsize.x < arraySize.z);
//            gridsize.y = arraySize.x / blocksize.y; gridsize.y += (blocksize.y * gridsize.y < arraySize.x);
//            cukern_fwdAverageZ<<<gridsize, blocksize>>>(srcs[0], dest[0], arraySize);
              blocksize.x = 64; blocksize.y = blocksize.z = 1;
              gridsize.x = arraySize.x / 64; gridsize.x += (64*gridsize.x < arraySize.x);
              gridsize.y = arraySize.y;
              cukern_ForwardAverageZ<<<gridsize, blocksize>>>(srcs[0], dest[0], arraySize.x, arraySize.z);
            break;
        }

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


#undef TILEDIM_X
#undef TILEDIM_Y
#undef DIFFEDGE
#undef FD_DIMENSION
#undef FD_MEMSTEP
#undef OTHER_DIMENSION
#undef OTHER_MEMSTEP
#undef ORTHOG_DIMENSION
#undef ORTHOG_MEMSTEP
/* These define the size of the "tile" each element loads
   which is contrained, basically, by available local memory.
   diffedge determines how wide the buffer zone is for taking
   derivatives. */
#define TILEDIM_X 18
#define TILEDIM_Y 8
#define DIFFEDGE 2
/* These determine how we look at the array. The array is assumed to be 3D
   (though possibly with z extent 1) and stored in C row-major format:
   index = [i j k], size = [Nx Ny Nz], memory step = [1 Nx NxNy]

   Choosing these determines how this operator sees the array: FD_DIM is the
   one we're taking derivatives in, OTHER forms a plane to it, and ORTHOG
   is the final dimension */
#define FD_DIMENSION dims.x
#define FD_MEMSTEP 1
#define OTHER_DIMENSION dims.y
#define OTHER_MEMSTEP dims.x
#define ORTHOG_DIMENSION dims.z
#define ORTHOG_MEMSTEP (dims.x * dims.y)
__global__ void cukern_fwdAverageX(double *in, double *out, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double f[TILEDIM_X * TILEDIM_Y+2];

FINITEDIFFX_PREAMBLE

/* Stick whatever local variables we care to futz with here */

/* We step through the array, one XY plane at a time */
int z;
for(z = 0; z < ORTHOG_DIMENSION; z++) {
    f[tileAddr] = in[globAddr];
    __syncthreads();

    if(ITakeDerivative) {
        out[globAddr] = (f[tileAddr] + f[tileAddr+1])/2;
        }

    __syncthreads();

    /* This determines the "Z" direction */
    globAddr += ORTHOG_MEMSTEP;
    }

}



#undef TILEDIM_X
#undef TILEDIM_Y
#undef DIFFEDGE
#undef FD_DIMENSION
#undef FD_MEMSTEP
#undef OTHER_DIMENSION
#undef OTHER_MEMSTEP
#undef ORTHOG_DIMENSION
#undef ORTHOG_MEMSTEP
/* These define the size of the "tile" each element loads
   which is contrained, basically, by available local memory.
   diffedge determines how wide the buffer zone is for taking
   derivatives. */
#define TILEDIM_X 8
#define TILEDIM_Y 18
#define DIFFEDGE 2
/* These determine how we look at the array. The array is assumed to be 3D
   (though possibly with z extent 1) and stored in C row-major format:
   index = [i j k], size = [Nx Ny Nz], memory step = [1 Nx NxNy]

   Choosing these determines how this operator sees the array: FD_DIM is the
   one we're taking derivatives in, OTHER forms a plane to it, and ORTHOG
   is the final dimension */
#define FD_DIMENSION dims.y
#define FD_MEMSTEP dims.x
#define OTHER_DIMENSION dims.x
#define OTHER_MEMSTEP 1
#define ORTHOG_DIMENSION dims.z
#define ORTHOG_MEMSTEP (dims.x * dims.y)
__global__ void cukern_fwdAverageY(double *in, double *out, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double f[TILEDIM_X * TILEDIM_Y+2];

FINITEDIFFY_PREAMBLE

/* Stick whatever local variables we care to futz with here */

/* We step through the array, one XY plane at a time */
int z;
for(z = 0; z < ORTHOG_DIMENSION; z++) {
    f[tileAddr] = in[globAddr];

    __syncthreads();
    // Keep in mind, ANY operation that refers to other than register variables or flux[tileAddr] MUST have a __syncthreads() after it or there will be sadness.
    if(ITakeDerivative) {
        out[globAddr] = (f[tileAddr] + f[tileAddr+1])/2;
        }

    __syncthreads();
    /* This determines the "Z" direction */
    globAddr += ORTHOG_MEMSTEP;
    }

}

#undef TILEDIM_X
#undef TILEDIM_Y
#undef DIFFEDGE
#undef FD_DIMENSION
#undef FD_MEMSTEP
#undef OTHER_DIMENSION
#undef OTHER_MEMSTEP
#undef ORTHOG_DIMENSION
#undef ORTHOG_MEMSTEP
/* These define the size of the "tile" each element loads
   which is contrained, basically, by available local memory.
   diffedge determines how wide the buffer zone is for taking
   derivatives. */
#define TILEDIM_X 18
#define TILEDIM_Y 8
#define DIFFEDGE 2
/* These determine how we look at the array. The array is assumed to be 3D
   (though possibly with z extent 1) and stored in C row-major format:
   index = [i j k], size = [Nx Ny Nz], memory step = [1 Nx NxNy]

   Choosing these determines how this operator sees the array: FD_DIM is the
   one we're taking derivatives in, OTHER forms a plane to it, and ORTHOG
   is the final dimension */
#define FD_DIMENSION dims.z
#define FD_MEMSTEP (dims.x*dims.y)
#define OTHER_DIMENSION dims.x
#define OTHER_MEMSTEP 1
#define ORTHOG_DIMENSION dims.y
#define ORTHOG_MEMSTEP dims.x
__global__ void cukern_fwdAverageZ(double *in, double *out, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double f[TILEDIM_X * TILEDIM_Y+2];

/* Our assumption implicitly is that differencing occurs in the X direction in the local tile */
int tileAddr = threadIdx.x + TILEDIM_X*threadIdx.y + 1;

int addrX = (threadIdx.x - DIFFEDGE) + blockIdx.x * (TILEDIM_X - 2*DIFFEDGE);
int addrY = threadIdx.y + blockIdx.y * TILEDIM_Y;

addrX += (addrX < 0)*FD_DIMENSION;
/* Nuke the threads hanging out past the end of the X extent of the array */
/* addrX is zero indexed, mind */
if(addrX >= FD_DIMENSION - 1 + DIFFEDGE) return;
if(addrY >= OTHER_DIMENSION) return;

/* Mask out threads who are near the edges to prevent seg violation upon differencing */
bool ITakeDerivative = (threadIdx.x >= DIFFEDGE) && (threadIdx.x < (TILEDIM_X - DIFFEDGE)) && (addrX < FD_DIMENSION);

addrX %= FD_DIMENSION; /* Wraparound (circular boundary conditions) */

/* NOTE: This chooses which direction we "actually" take derivatives in
         along with the conditional add a few lines up */
int globAddr = FD_MEMSTEP * addrX + OTHER_MEMSTEP * addrY;

/* Stick whatever local variables we care to futz with here */

/* We step through the array, one XY plane at a time */
int z;
for(z = 0; z < ORTHOG_DIMENSION; z++) {
    f[tileAddr] = in[globAddr];

    __syncthreads();
    // Keep in mind, ANY operation that refers to other than register variables or flux[tileAddr] MUST have a __syncthreads() after it or there will be sadness.
    if(ITakeDerivative) {
        out[globAddr] = (f[tileAddr] + f[tileAddr+1])/2;
        }

    __syncthreads();
    /* This determines the "Z" direction */
    globAddr += ORTHOG_MEMSTEP;
    }

}

