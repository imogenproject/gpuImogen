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

__global__ void cukern_SimpleVelocity(double *v, double *p, double *m, int numel);

__global__ void cukern_VelocityBkwdAverage_X(double *v, double *p, double *m, int nx);
__global__ void cukern_VelocityBkwdAverage_Y(double *v, double *p, double *m, int nx, int ny);
__global__ void cukern_VelocityBkwdAverage_Z(double *v, double *p, double *m, int nx, int nz);

__global__ void cukern_CentralAverage_X(double *out, double *in, int nx);
__global__ void cukern_CentralAverage_Y(double *out, double *in, int nx, int ny);


__global__ void cukern_magVelInterp_Z(double *velout, double *velin, int3 dims);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // At least 2 arguments expected
    // Input and result
    if ((nrhs!=3) || (nlhs != 1)) mexErrMsgTxt("Wrong number of arguments: need velInterp = cudaMagPrep(mom, mass, [dirvel dirmag])\n");

    // Get source array info and create destination arrays
    ArrayMetadata amd;
    double **srcs = getGPUSourcePointers(prhs, &amd, 0, 1);
    double **dest = makeGPUDestinationArrays((int64_t *)mxGetData(prhs[0]), plhs, 1);

    double *tempVelocity;

    cudaError_t fail = cudaMalloc((void **)&tempVelocity, amd.numel*sizeof(double));
    if(fail != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(fail));
    mexErrMsgTxt("cudaMagPrep: I have an cudaMalloc fail. And a sad.");
    }

    // Establish launch dimensions & a few other parameters
    double *directs = mxGetPr(prhs[2]);
    int velDirection = (int)directs[0];
    int magDirection = (int)directs[1];

    int3 arraySize;
    arraySize.x = amd.dim[0];
    amd.ndims > 1 ? arraySize.y = amd.dim[1] : arraySize.y = 1;
    amd.ndims > 2 ? arraySize.z = amd.dim[2] : arraySize.z = 1;

    dim3 blocksize, gridsize;
    blocksize.z = 1;
    gridsize.z = 1;

    if(amd.dim[velDirection-1] > 1) {

        // Interpolate the grid-aligned velocity
        switch(velDirection) {
           case 1:
                blocksize.x = 128; blocksize.y = blocksize.z = 1;
                gridsize.x = arraySize.y; // / 16; gridsize.x += (16 * gridsize.x < arraySize.x);
                gridsize.y = arraySize.z;// / blocksize.y; gridsize.y += (blocksize.y * gridsize.y < arraySize.y);
                cukern_VelocityBkwdAverage_X<<<gridsize, blocksize>>>(tempVelocity, srcs[0], srcs[1], arraySize.x);
                break;
            case 2:
                blocksize.x = 64; blocksize.y = blocksize.z = 1;
                gridsize.x = arraySize.x/64; gridsize.x += (gridsize.x*64 < arraySize.x);
                gridsize.y = arraySize.z;
                cukern_VelocityBkwdAverage_Y<<<gridsize, blocksize>>>(tempVelocity, srcs[0], srcs[1], arraySize.x, arraySize.y);
                break;
            case 3:
                blocksize.x = 64; blocksize.y = blocksize.z = 1;
                gridsize.x = arraySize.x/64; gridsize.x += (gridsize.x*64 < arraySize.x);
                gridsize.y = arraySize.y;
                cukern_VelocityBkwdAverage_Z<<<gridsize, blocksize>>>(tempVelocity, srcs[0], srcs[1], arraySize.x, arraySize.z);
                break;
            }
        } else {
            blocksize.x = 512; blocksize.y = blocksize.z = 1;
            gridsize.x = amd.numel / 512; gridsize.x += (gridsize.x * 512 < amd.numel); gridsize.y = gridsize.z = 1;
            cukern_SimpleVelocity<<<gridsize, blocksize>>>(tempVelocity, srcs[0], srcs[1], amd.numel);
        }

cudaError_t epicFail = cudaGetLastError();
if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, velDirection, "mag prep velocity avg");

    // Interpolate the velocity to 2nd order
    if(amd.dim[magDirection-1] > 1) {
        switch(magDirection) {
            case 1:
                blocksize.x = 128; blocksize.y = blocksize.z = 1;
                gridsize.x = arraySize.y; // / 16; gridsize.x += (16 * gridsize.x < arraySize.x);
                gridsize.y = arraySize.z;// / blocksize.y; gridsize.y += (blocksize.y * gridsize.y < arraySize.y);
                cukern_CentralAverage_X<<<gridsize, blocksize>>>(dest[0], tempVelocity, arraySize.x);
                break;
            case 2:
                blocksize.x = 64; blocksize.y = blocksize.z = 1;
                gridsize.x = arraySize.x/64; gridsize.x += (gridsize.x*64 < arraySize.x);
                gridsize.y = arraySize.z;
                cukern_CentralAverage_Y<<<gridsize, blocksize>>>(dest[0], tempVelocity, arraySize.x, arraySize.y);
                break;
            case 3:
                blocksize.x = 18; blocksize.y = 8;
                gridsize.x = arraySize.z / 14; gridsize.x += (14 * gridsize.x < arraySize.z);
                gridsize.y = arraySize.x / blocksize.y; gridsize.y += (blocksize.y * gridsize.y < arraySize.x);
                cukern_magVelInterp_Z<<<gridsize, blocksize>>>(dest[0], tempVelocity, arraySize);
                break;
            }
        } else {
        cudaMemcpy(dest[0], tempVelocity, amd.numel, cudaMemcpyDeviceToDevice);
        // FIXME: Detect this condition ahead of time and never bother with this array in the first place
        }

epicFail = cudaGetLastError();
if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, magDirection, "mag prep interpolation");

    cudaFree(tempVelocity); // Because only YOU can prevent memory leaks!
                            // (and this one would be a whopper...)
}

__global__ void cukern_SimpleVelocity(double *v, double *p, double *m, int numel)
{
int addr = threadIdx.x + 512*blockIdx.x;
short int q;

if(addr > numel) return;

v[addr] = p[addr] / m[addr];
}


__global__ void cukern_VelocityBkwdAverage_X(double *v, double *p, double *m, int nx)
{
int xAddr = threadIdx.x;
int yAddr = blockIdx.x;
int zAddr = blockIdx.y;
int ny = gridDim.x;

int addrMax   = nx*(yAddr + ny*zAddr + 1); // The address which we must not reach or go beyond is the start of the next line
int writeBase  = xAddr + nx*(yAddr + ny*zAddr); // the write start position is the left edge
int readBase = writeBase - 1; // the read start position is one to the left of the write start position.

if (threadIdx.x == 0) readBase += nx; // leftmost reads right edge
readBase -= nx*(readBase >= addrMax);

__shared__ double lMom[256];
__shared__ double lRho[256];

int locAddr = threadIdx.x;

//lStore[locAddr] = in[globBase + readX]; // load the first memory segment
lMom[locAddr] = p[readBase]; // load the first memory segment
lRho[locAddr] = m[readBase];

do {
    readBase += 128; // move over one block
    readBase -= nx*(readBase >= addrMax); // loop around if x overflows
    lMom[(locAddr + 128) % 256] = p[readBase];
    lRho[(locAddr + 128) % 256] = m[readBase];

    __syncthreads(); // We have now read ahead by a segment. Calculate forward average, comrades!

    if(writeBase < addrMax) { v[writeBase] = (lMom[locAddr] + lMom[(locAddr+1)%256])/(lRho[locAddr] + lRho[(locAddr+1)%256]); } // If we are within range, that is.
    writeBase += 128; // Advance write address
    if(writeBase >= addrMax) return; // If write address is beyond nx, we're finished.
    locAddr ^= 128;

    __syncthreads();

    } while(1);

}

/* Invoke with a blockdim of <64, 1, 1> threads
Invoke with a griddim = <ceil[nx / 64], nz, 1> */
__global__ void cukern_VelocityBkwdAverage_Y(double *v, double *p, double *m, int nx, int ny)
{
int xaddr = blockDim.x * blockIdx.x + threadIdx.x; // There are however many X threads
if(xaddr >= nx) return; // truncate this right off

__shared__ double tileA[128];
__shared__ double tileB[128];

double *setA = tileA;
double *setB = tileB;
double *swap;

int writeBase = xaddr + nx*ny*blockIdx.y; // set Raddr to x + nx ny z
int addrMax = writeBase + nx*(ny - 1); // Set this to the max address we want to handle in the loop

setB[threadIdx.x]    = p[addrMax]; // load row (y=-1) into set b
setB[threadIdx.x+64] = m[addrMax];

while(writeBase <= addrMax) { // Exit one BEFORE the max address to handle (since the max is a special case)
    swap = setB; // exchange A/B pointers
    setB = setA;
    setA = swap; 

//    __syncthreads();

    setB[threadIdx.x]    = p[writeBase]; // load row (y=0) into set B
    setB[threadIdx.x+64] = m[writeBase];

    v[writeBase] = (setA[threadIdx.x] + setB[threadIdx.x])/(setA[threadIdx.x+64] + setB[threadIdx.x+64]); // average written to output

    __syncthreads();

    writeBase += nx; // increment rw address to y=1
    }

}

/* Invoke with a blockdim of <64, 1, 1> threads
Invoke with a griddim = <ceil[nx / 64], ny, 1> */
__global__ void cukern_VelocityBkwdAverage_Z(double *v, double *p, double *m, int nx, int nz)
{
int xaddr = blockDim.x * blockIdx.x + threadIdx.x; // There are however many X threads
if(xaddr >= nx) return; // truncate this right off

__shared__ double tileA[128];
__shared__ double tileB[128];

double *setA = tileA;
double *setB = tileB;
double *swap;

int writeBase = xaddr + nx*blockIdx.y; // set Raddr to x + nx ny z
int addrMax = writeBase + nx*gridDim.y*(nz-1); // Set this to the max address we want to handle in the loop

setB[threadIdx.x]    = p[addrMax]; // load row (y=-1) into set b
setB[threadIdx.x+64] = m[addrMax];

while(writeBase <= addrMax) { // Exit one BEFORE the max address to handle (since the max is a special case)
    swap = setB; // exchange A/B pointers
    setB = setA;
    setA = swap;

//    __syncthreads();

    setB[threadIdx.x]    = p[writeBase]; // load row (y=0) into set B
    setB[threadIdx.x+64] = m[writeBase];

    v[writeBase] = (setA[threadIdx.x] + setB[threadIdx.x])/(setA[threadIdx.x+64] + setB[threadIdx.x+64]); // average written to output

    __syncthreads();

    writeBase += nx*gridDim.y; // increment rw address to y=1
    }

}

//################ MAGNETIC INTERPOLATION KERNELS


__global__ void cukern_CentralAverage_X(double *out, double *in, int nx)
{
int xAddr = threadIdx.x;
int yAddr = blockIdx.x;
int zAddr = blockIdx.y;
int ny = gridDim.x;

int addrMax   = nx*(yAddr + ny*zAddr + 1); // The address which we must not reach or go beyond is the start of the next line
int writeBase  = xAddr + nx*(yAddr + ny*zAddr); // the write start position is the left edge
int readBase = writeBase - 1; // the read start position is one to the left of the write start position.

if (threadIdx.x == 0) readBase += nx; // leftmost reads right edge
readBase -= nx*(readBase >= addrMax);

__shared__ double funcBuffer[256];

int locAddr = threadIdx.x;

//lStore[locAddr] = in[globBase + readX]; // load the first memory segment
funcBuffer[locAddr] = in[readBase]; // load the first memory segment

do {
    readBase += 128; // move over one block
    readBase -= nx*(readBase >= addrMax); // loop around if x overflows
    funcBuffer[(locAddr + 128) % 256] = in[readBase];

    __syncthreads(); // We have now read ahead by a segment. Calculate forward average, comrades!

    if(writeBase < addrMax) { out[writeBase] = .25*(funcBuffer[locAddr] + funcBuffer[(locAddr+2)%256]+ 2*funcBuffer[(locAddr+1)%256]); } // If we are within range, that is.
    writeBase += 128; // Advance write address
    if(writeBase >= addrMax) return; // If write address is beyond nx, we're finished.
    locAddr ^= 128;

    __syncthreads();

    } while(1);

}


/* Invoke with a blockdim of <64, 1, 1> threads
Invoke with a griddim = <ceil[nx / 64], nz, 1> */
__global__ void cukern_CentralAverage_Y(double *out, double *in, int nx, int ny)
{
int xaddr = blockDim.x * blockIdx.x + threadIdx.x; // There are however many X threads
if(xaddr >= nx) return; // truncate this right off

__shared__ double tileA[64];
__shared__ double tileB[64];
__shared__ double tileC[64];

double *setA = tileA;
double *setB = tileB;
double *setC = tileC;
double *swap;

int writeBase = xaddr + nx*ny*blockIdx.y; // set Raddr to x + nx ny z
int addrMax = writeBase + nx*(ny - 1); // Set this to the max address we want to handle in the loop

setB[threadIdx.x] = in[addrMax]; // load row (y=-1) into set b
setC[threadIdx.x] = in[writeBase];

while(writeBase < addrMax) { // Exit one BEFORE the max address to handle (since the max is a special case)
    swap = setA; // Rotate pointers
    setA = setB;
    setB = setC;
    setC = swap;

//    __syncthreads();
    setC[threadIdx.x]    = in[writeBase + nx]; // load row (y=0) into set B

    out[writeBase] = .25*(setA[threadIdx.x] + setC[threadIdx.x]) + .5*setB[threadIdx.x]; // average written to output

    __syncthreads();

    writeBase += nx; // increment rw address to y=1
    }

// We arrive here when writeBase == addrMax, i.e. we are at the last Y index
setA[threadIdx.x] = in[xaddr + nx*ny*blockIdx.y];

// The weights change because we haven't cycled the pointers
out[writeBase] = .25*(setA[threadIdx.x] + setB[threadIdx.x]) + .5*setC[threadIdx.x];

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
__global__ void cukern_magVelInterp_Z(double *velout, double *velin, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double cellVel[TILEDIM_X * TILEDIM_Y+2];

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
    cellVel[tileAddr] = velin[globAddr];
    __syncthreads();

    // Keep in mind, ANY operation that refers to other than register variables or flux[tileAddr] MUST have a __syncthreads() after it or there will be sadness.
    if(ITakeDerivative) {
        velout[globAddr] = .25*(cellVel[tileAddr-1] + 2.0*cellVel[tileAddr] + cellVel[tileAddr+1]);
        }

    __syncthreads();
    /* This determines the "Z" direction */
    globAddr += ORTHOG_MEMSTEP;
    }

}



