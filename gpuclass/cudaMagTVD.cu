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

__global__ void cukern_magnetTVDstep_uniformX(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims);
__global__ void cukern_magnetTVDstep_uniformY(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims);
__global__ void cukern_magnetTVDstep_uniformZ(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims);

#define BLOCKDIMA 18
#define BLOCKDIMAM2 16
#define BLOCKDIMB 8

#define BLOCKLEN 128
#define BLOCKLENP4 132

__device__ void cukern_FluxLimiter_VanLeer_x(double deriv[2][BLOCKLENP4], double flux[BLOCKLENP4]);
__device__ void cukern_FluxLimiter_VanLeer_yz(double deriv[2][BLOCKDIMB][BLOCKDIMA], double flux[BLOCKDIMB][BLOCKDIMA]);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // At least 2 arguments expected
    // Input and result
    if ((nrhs!=6) || (nlhs != 1)) mexErrMsgTxt("Wrong number of arguments: need [flux] = cudaMagTVD(magW, mag, velgrid, velflow, lambda, dir)\n");

    // Get source array info and create destination arrays
    ArrayMetadata amd;
    double **srcs = getGPUSourcePointers(prhs, &amd, 0, 3);
    double **dest = makeGPUDestinationArrays((int64_t *)mxGetData(prhs[0]), plhs, 1);

    // Establish launch dimensions & a few other parameters
    int fluxDirection = (int)*mxGetPr(prhs[5]);
    double lambda     = *mxGetPr(prhs[4]);

    int3 arraySize;
    arraySize.x = amd.dim[0];
    amd.ndims > 1 ? arraySize.y = amd.dim[1] : arraySize.y = 1;
    amd.ndims > 2 ? arraySize.z = amd.dim[2] : arraySize.z = 1;

    dim3 blocksize, gridsize;
    switch(fluxDirection) {
        case 1: // X direction flux. This is "priveleged" in that the shift and natural memory load directions align
            blocksize.x = 24; blocksize.y = 16; blocksize.z = 1;

            gridsize.x = arraySize.x / 18; gridsize.x += 1*(18*gridsize.x < arraySize.x);
            gridsize.y = arraySize.y / blocksize.y; gridsize.y += 1*(blocksize.y*gridsize.y < arraySize.y);

            cukern_magnetTVDstep_uniformX<<<gridsize , blocksize>>>(srcs[0], srcs[2], srcs[3], srcs[1], dest[0], lambda, arraySize);
//printf("xkern\n");
            break;
        case 2: // Y direction flux: u = y, v = x, w = z
            blocksize.x = 16; blocksize.y = 24; blocksize.z = 1;

            gridsize.x = arraySize.x / 16; gridsize.x += 1*(16*gridsize.x < arraySize.x);
            gridsize.y = arraySize.y / 18; gridsize.y += 1*(18*gridsize.y < arraySize.y);

            cukern_magnetTVDstep_uniformY<<<gridsize , blocksize>>>(srcs[0], srcs[2], srcs[3], srcs[1], dest[0], lambda, arraySize);
//printf("ykern\n");
            break;
        case 3: // Z direction flux: u = z, v = x, w = y;
            blocksize.x = 24; blocksize.y = 16; blocksize.z = 1;

            gridsize.x = arraySize.z / 18; gridsize.x += 1*(18*gridsize.x < arraySize.z);
            gridsize.y = arraySize.x / blocksize.y; gridsize.y += 1*(blocksize.y*gridsize.y < arraySize.x);

            cukern_magnetTVDstep_uniformZ<<<gridsize , blocksize>>>(srcs[0], srcs[2], srcs[3], srcs[1], dest[0], lambda, arraySize);
//printf("zkern\n");
            break;
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
#define TILEDIM_X 24
#define TILEDIM_Y 16
#define DIFFEDGE 3
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
__global__ void cukern_magnetTVDstep_uniformX(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double flux[TILEDIM_X * TILEDIM_Y+2];
__shared__ double derivL[TILEDIM_X * TILEDIM_Y + 2];
__shared__ double derivR[TILEDIM_X * TILEDIM_Y + 2];

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
double derivRatio, locFlux;
int locVF;

/* We step through the array, one XY plane at a time */
int z;
for(z = 0; z < ORTHOG_DIMENSION; z++) {
    locVF = (int)velFlow[globAddr];
    flux[tileAddr] = bW[globAddr]*velGrid[globAddr];

    // Keep in mind, ANY operation that refers to other than register variables or flux[tileAddr] MUST have a __syncthreads() after it or there will be sadness.
    __syncthreads();

    locFlux = flux[tileAddr+locVF]; // This is the one we want to correct to 2nd order

    if(locVF == 1) {
        derivL[tileAddr] = flux[tileAddr] - flux[tileAddr+1];
        derivR[tileAddr] = flux[tileAddr+1] - flux[tileAddr+2];
    } else {
        derivL[tileAddr] = flux[tileAddr] - flux[tileAddr-1];
        derivR[tileAddr] = flux[tileAddr+1] - flux[tileAddr];
    }

    // We're finished with velocityFlow, reuse to store the flux which we're about to limit
    __syncthreads();

    // Use the van Leer limiter
    derivRatio = derivL[tileAddr] * derivR[tileAddr];
    if(derivRatio < 0) derivRatio = 0;

    derivRatio /= (derivL[tileAddr] + derivR[tileAddr]);
    if(isnan(derivRatio)) derivRatio = 0.0;

    flux[tileAddr] = locFlux + derivRatio;

    __syncthreads();

    if(ITakeDerivative) {
        mag[globAddr]     = mag[globAddr] - lambda*(flux[tileAddr] - flux[tileAddr-1]);
        fluxout[globAddr] = flux[tileAddr];
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
#define TILEDIM_X 16
#define TILEDIM_Y 24
#define DIFFEDGE 3
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
__global__ void cukern_magnetTVDstep_uniformY(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double flux[TILEDIM_X * TILEDIM_Y+2];
__shared__ double derivL[TILEDIM_X * TILEDIM_Y + 2];
__shared__ double derivR[TILEDIM_X * TILEDIM_Y + 2];

FINITEDIFFY_PREAMBLE

/* Our assumption implicitly is that differencing occurs in the X direction in the local tile */
//int tileAddr = threadIdx.x + TILEDIM_X*threadIdx.y + 1;

//int addrX = (threadIdx.x - DIFFEDGE) + blockIdx.x * (TILEDIM_X - 2*DIFFEDGE);
//int addrY = threadIdx.y + blockIdx.y * TILEDIM_Y;

//addrX += (addrX < 0)*FD_DIMENSION;

/* Nuke the threads hanging out past the end of the X extent of the array */
/* addrX is zero indexed, mind */
//if(addrX >= FD_DIMENSION - 1 + DIFFEDGE) return;
//if(addrY >= OTHER_DIMENSION) return; 

/* Mask out threads who are near the edges to prevent seg violation upon differencing */
//bool ITakeDerivative = (threadIdx.x >= DIFFEDGE) && (threadIdx.x < (TILEDIM_X - DIFFEDGE)) && (addrX < FD_DIMENSION);

//addrX %= FD_DIMENSION; /* Wraparound (circular boundary conditions) */

/* NOTE: This chooses which direction we "actually" take derivatives in
         along with the conditional add a few lines up */
//int globAddr = FD_MEMSTEP * addrX + OTHER_MEMSTEP * addrY;

/* Stick whatever local variables we care to futz with here */
double derivRatio, locFlux;
int locVF;

/* We step through the array, one XY plane at a time */
int z;
for(z = 0; z < ORTHOG_DIMENSION; z++) {
    locVF = (int)velFlow[globAddr];
    flux[tileAddr] = bW[globAddr]*velGrid[globAddr];

    // Keep in mind, ANY operation that refers to other than register variables or flux[tileAddr] MUST have a __syncthreads() after it or there will be sadness.
    __syncthreads();

    locFlux = flux[tileAddr+locVF]; // This is the one we want to correct to 2nd order

    if(locVF == 1) {
        derivL[tileAddr] = flux[tileAddr] - flux[tileAddr+1];
        derivR[tileAddr] = flux[tileAddr+1] - flux[tileAddr+2];
    } else {
        derivL[tileAddr] = flux[tileAddr] - flux[tileAddr-1];
        derivR[tileAddr] = flux[tileAddr+1] - flux[tileAddr];
    }

    // We're finished with velocityFlow, reuse to store the flux which we're about to limit
    __syncthreads();

    // Use the van Leer limiter
    derivRatio = derivL[tileAddr] * derivR[tileAddr];
    if(derivRatio < 0) derivRatio = 0;

    derivRatio /= (derivL[tileAddr] + derivR[tileAddr]);
    if(isnan(derivRatio)) derivRatio = 0.0;

    flux[tileAddr] = locFlux + derivRatio;

    __syncthreads();

    if(ITakeDerivative) {
        mag[globAddr]     = mag[globAddr] - lambda*(flux[tileAddr] - flux[tileAddr-1]);
        fluxout[globAddr] = flux[tileAddr];
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
#define TILEDIM_X 24
#define TILEDIM_Y 16
#define DIFFEDGE 3
/* These determine how we look at the array. The array is assumed to be 3D
   (though possibly with z extent 1) and stored in C row-major format:
   index = [i j k], size = [Nx Ny Nz], memory step = [1 Nx NxNy]

   Choosing these determines how this operator sees the array: FD_DIM is the
   one we're taking derivatives in, OTHER forms a plane to it, and ORTHOG
   is the final dimension */
#define FD_DIMENSION dims.z
#define FD_MEMSTEP dims.x*dims.y
#define OTHER_DIMENSION dims.x
#define OTHER_MEMSTEP 1
#define ORTHOG_DIMENSION dims.y
#define ORTHOG_MEMSTEP dims.x
__global__ void cukern_magnetTVDstep_uniformZ(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double flux[TILEDIM_X * TILEDIM_Y+2];
__shared__ double derivL[TILEDIM_X * TILEDIM_Y + 2];
__shared__ double derivR[TILEDIM_X * TILEDIM_Y + 2];

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
double derivRatio, locFlux;
int locVF;

/* We step through the array, one XY plane at a time */
int z;
for(z = 0; z < ORTHOG_DIMENSION; z++) {
    locVF = (int)velFlow[globAddr];
    flux[tileAddr] = bW[globAddr]*velGrid[globAddr];

    // Keep in mind, ANY operation that refers to other than register variables or flux[tileAddr] MUST have a __syncthreads() after it or there will be sadness.
    __syncthreads();

    locFlux = flux[tileAddr+locVF]; // This is the one we want to correct to 2nd order

    if(locVF == 1) {
        derivL[tileAddr] = flux[tileAddr] - flux[tileAddr+1];
        derivR[tileAddr] = flux[tileAddr+1] - flux[tileAddr+2];
    } else {
        derivL[tileAddr] = flux[tileAddr] - flux[tileAddr-1];
        derivR[tileAddr] = flux[tileAddr+1] - flux[tileAddr];
    }

    // We're finished with velocityFlow, reuse to store the flux which we're about to limit
    __syncthreads();

    // Use the van Leer limiter
    derivRatio = derivL[tileAddr] * derivR[tileAddr];
    if(derivRatio < 0) derivRatio = 0;

    derivRatio /= (derivL[tileAddr] + derivR[tileAddr]);
    if(isnan(derivRatio)) derivRatio = 0.0;

    flux[tileAddr] = locFlux + derivRatio;

    __syncthreads();

    if(ITakeDerivative) {
        mag[globAddr]     = mag[globAddr] - lambda*(flux[tileAddr] - flux[tileAddr-1]);
        fluxout[globAddr] = flux[tileAddr];
        }

    __syncthreads();

    /* This determines the "Z" direction */
    globAddr += ORTHOG_MEMSTEP;
    }

}


