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

__global__ void cukern_magVelAvg_X(double *v, double *p, double *m, int3 dims);
__global__ void cukern_magVelAvg_Y(double *v, double *p, double *m, int3 dims);
__global__ void cukern_magVelAvg_Z(double *v, double *p, double *m, int3 dims);

__global__ void cukern_magVelInterp_X(double *velout, double *velin, int3 dims);
__global__ void cukern_magVelInterp_Y(double *velout, double *velin, int3 dims);
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
    mexErrMsgTxt("cudaMagPrep: I haz an cudaMalloc fail. And a sad.");
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

    // Interpolate the grid-aligned velocity
    switch(velDirection) {
        case 1:
            blocksize.x = 18; blocksize.y = 8;
            gridsize.x = arraySize.x / 16; gridsize.x += (16 * gridsize.x < arraySize.x);
            gridsize.y = arraySize.y / blocksize.y; gridsize.y += (blocksize.y * gridsize.y < arraySize.y);
            cukern_magVelAvg_X<<<gridsize, blocksize>>>(tempVelocity, srcs[0], srcs[1], arraySize);
            break;
        case 2:
            blocksize.x = 8; blocksize.y = 18;
            gridsize.x = arraySize.x / 8; gridsize.x += (8 * gridsize.x < arraySize.x);
            gridsize.y = arraySize.y / 16; gridsize.y += (16 * gridsize.x < arraySize.y);
            cukern_magVelAvg_Y<<<gridsize, blocksize>>>(tempVelocity, srcs[0], srcs[1], arraySize);
            break;
        case 3:
            blocksize.x = 18; blocksize.y = 8;
            gridsize.x = arraySize.z / 16; gridsize.x += (16 * gridsize.x < arraySize.z);
            gridsize.y = arraySize.x / blocksize.y; gridsize.y += (blocksize.y * gridsize.y < arraySize.x);
            cukern_magVelAvg_Z<<<gridsize, blocksize>>>(tempVelocity, srcs[0], srcs[1], arraySize);
            break;
        }

    // Interpolate the velocity to 2nd order

    switch(magDirection) {
        case 1:
            blocksize.x = 18; blocksize.y = 8;
            gridsize.x = arraySize.x / 14; gridsize.x += (14 * gridsize.x < arraySize.x);
            gridsize.y = arraySize.y / blocksize.y; gridsize.y += (blocksize.y * gridsize.y < arraySize.y);
            cukern_magVelInterp_X<<<gridsize, blocksize>>>(dest[0], tempVelocity, arraySize);
            break;
        case 2:
            blocksize.x = 8; blocksize.y = 18;
            gridsize.x = arraySize.x / 8; gridsize.x += (8 * gridsize.x < arraySize.x);
            gridsize.y = arraySize.y / 14; gridsize.y += (14 * gridsize.y < arraySize.y);
            cukern_magVelInterp_Y<<<gridsize, blocksize>>>(dest[0], tempVelocity, arraySize);
            break;
        case 3:
            blocksize.x = 18; blocksize.y = 8;
            gridsize.x = arraySize.z / 14; gridsize.x += (14 * gridsize.x < arraySize.z);
            gridsize.y = arraySize.x / blocksize.y; gridsize.y += (blocksize.y * gridsize.y < arraySize.x);
            cukern_magVelInterp_Z<<<gridsize, blocksize>>>(dest[0], tempVelocity, arraySize);
            break;
        }

    cudaFree(tempVelocity); // Because only YOU can prevent memory leaks!
                            // (and this one would be a whopper...)

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
#define DIFFEDGE 1
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
__global__ void cukern_magVelAvg_X(double *v, double *p, double *m, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double cellMom[TILEDIM_X * TILEDIM_Y+2];
__shared__ double cellRho[TILEDIM_X * TILEDIM_Y+2];

FINITEDIFFX_PREAMBLE

/* Stick whatever local variables we care to futz with here */

/* We step through the array, one XY plane at a time */
int z;
for(z = 0; z < ORTHOG_DIMENSION; z++) {
    cellMom[tileAddr] = p[globAddr];
    cellRho[tileAddr] = m[globAddr];
    __syncthreads();

    if(ITakeDerivative) {
        v[globAddr] = (cellMom[tileAddr-1] + cellMom[tileAddr])/(cellRho[tileAddr-1]+cellRho[tileAddr]);
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
#define DIFFEDGE 1
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
__global__ void cukern_magVelAvg_Y(double *v, double *p, double *m, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double cellMom[TILEDIM_X * TILEDIM_Y+2];
__shared__ double cellRho[TILEDIM_X * TILEDIM_Y+2];

FINITEDIFFY_PREAMBLE

/* Stick whatever local variables we care to futz with here */

/* We step through the array, one XY plane at a time */
int z;
for(z = 0; z < ORTHOG_DIMENSION; z++) {
    cellMom[tileAddr] = p[globAddr];
    cellRho[tileAddr] = m[globAddr];
    __syncthreads();
    // Keep in mind, ANY operation that refers to other than register variables or flux[tileAddr] MUST have a __syncthreads() after it or there will be sadness.
    if(ITakeDerivative) {
        v[globAddr] = (cellMom[tileAddr-1] + cellMom[tileAddr])/(cellRho[tileAddr-1]+cellRho[tileAddr]);
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
#define DIFFEDGE 1
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
__global__ void cukern_magVelAvg_Z(double *v, double *p, double *m, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double cellMom[TILEDIM_X * TILEDIM_Y+2];
__shared__ double cellRho[TILEDIM_X * TILEDIM_Y+2];

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
    cellMom[tileAddr] = p[globAddr];
    cellRho[tileAddr] = m[globAddr];
    __syncthreads();
    // Keep in mind, ANY operation that refers to other than register variables or flux[tileAddr] MUST have a __syncthreads() after it or there will be sadness.
    if(ITakeDerivative) {
        v[globAddr] = (cellMom[tileAddr-1] + cellMom[tileAddr])/(cellRho[tileAddr-1]+cellRho[tileAddr]);
        }

    __syncthreads();
    /* This determines the "Z" direction */
    globAddr += ORTHOG_MEMSTEP;
    }

}

//################ MAGNETIC INTERPOLATION KERNELS

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
__global__ void cukern_magVelInterp_X(double *velout, double *velin, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double cellVel[TILEDIM_X * TILEDIM_Y+2];

FINITEDIFFX_PREAMBLE

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
__global__ void cukern_magVelInterp_Y(double *velout, double *velin, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double cellVel[TILEDIM_X * TILEDIM_Y+2];

FINITEDIFFY_PREAMBLE

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



