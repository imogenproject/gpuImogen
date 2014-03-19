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
__global__ void cukern_magnetTVDstep_uniformX(double *bW, double *velGrid, double *Cf, double *mag, double *fluxout, double lambda, int3 dims);
//__global__ void cukern_magnetTVDstep_uniformX(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims);
__global__ void cukern_magnetTVDstep_uniformY(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims);
__global__ void cukern_magnetTVDstep_uniformZ(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims);

#define BLOCKDIMA 18
#define BLOCKDIMAM2 16
#define BLOCKDIMB 8

#define BLOCKLEN 128
#define BLOCKLENP4 132

#define LIMITERFUNC fluxLimiter_VanLeer

__device__ void cukern_FluxLimiter_VanLeer_x(double deriv[2][BLOCKLENP4], double flux[BLOCKLENP4]);
__device__ void cukern_FluxLimiter_VanLeer_yz(double deriv[2][BLOCKDIMB][BLOCKDIMA], double flux[BLOCKDIMB][BLOCKDIMA]);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // At least 2 arguments expected
    // Input and result
    if ((nrhs!=6) || (nlhs != 1)) mexErrMsgTxt("Wrong number of arguments: need [flux] = cudaMagTVD(magW, mag, velgrid, C_freeze, lambda, dir)\n");

    cudaCheckError("entering cudaMagTVD");

    // Get source array info and create destination arrays
    ArrayMetadata amd;
    double **srcs = getGPUSourcePointers(prhs, &amd, 0, 3);

    // Get freezing speed array
    ArrayMetadata cf_amd;
    double **cfreeze = getGPUSourcePointers(prhs, &cf_amd, 3,3);

    double **dest = makeGPUDestinationArrays(&amd, plhs, 1);

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
            cukern_magnetTVDstep_uniformX<<<gridsize , blocksize>>>(srcs[0], srcs[2], cfreeze[0], srcs[1], dest[0], lambda, arraySize);
//            cukern_magnetTVDstep_uniformX<<<gridsize , blocksize>>>(srcs[0], srcs[2], srcs[3], srcs[1], dest[0], lambda, arraySize);
            break;
        case 2: // Y direction flux: u = y, v = x, w = z
            blocksize.x = 16; blocksize.y = 24; blocksize.z = 1;

            gridsize.x = arraySize.x / 16; gridsize.x += 1*(16*gridsize.x < arraySize.x);
            gridsize.y = arraySize.y / 18; gridsize.y += 1*(18*gridsize.y < arraySize.y);

            cukern_magnetTVDstep_uniformY<<<gridsize , blocksize>>>(srcs[0], srcs[2], srcs[3], srcs[1], dest[0], lambda, arraySize);
            break;
        case 3: // Z direction flux: u = z, v = x, w = y;
            blocksize.x = 24; blocksize.y = 16; blocksize.z = 1;

            gridsize.x = arraySize.z / 18; gridsize.x += 1*(18*gridsize.x < arraySize.z);
            gridsize.y = arraySize.x / blocksize.y; gridsize.y += 1*(blocksize.y*gridsize.y < arraySize.x);

            cukern_magnetTVDstep_uniformZ<<<gridsize , blocksize>>>(srcs[0], srcs[2], srcs[3], srcs[1], dest[0], lambda, arraySize);
            break;
    }

cudaError_t epicFail = cudaGetLastError();
if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, fluxDirection, "magnetic TVD step");

}


/* Warp-synchronous reduction of data */
/* Compute 2+ devices are only half-warp synchronous */
__device__ void warpReduction(volatile double *q, int tid)
{
q[tid] = (q[tid] > q[tid+16]) ? q[tid] : q[tid+16];
q[tid] = (q[tid] > q[tid+8]) ? q[tid] : q[tid+8];
q[tid] = (q[tid] > q[tid+4]) ? q[tid] : q[tid+4];
q[tid] = (q[tid] > q[tid+2]) ? q[tid] : q[tid+2];
q[tid] = (q[tid] > q[tid+1]) ? q[tid] : q[tid+1];
}
/* Expects to be launched with block size [A 1 1] and grid size [Ny Nz]
 * Calculate max(v, x) */
__global__ void cuda_advectFreezeSpeedX(double *v, double *cf, int3 dims)
{

v += dims.x*(blockIdx.x + dims.y*blockIdx.y);

int tid = threadIdx.x;
__shared__ double vmax[CF_BLKX];
double umax = 0.0;
double t;
vmax[tid] = 0.0;
/* Calculate max per stride of blockDim.x */
int x = threadIdx.x;
for(x = tid; x < dims.x; x += CF_BLKX) {
  t = fabs(v[x]);
  umax = (t > umax) ? t : umax;
  }
vmax[tid] = umax;

if(tid > CF_BLKX/2) return;
/* Begin reduction */
__syncthreads();

for(x = blockDim.x/2; x > 16; x/=2) {
  if(tid < x) vmax[tid] = (vmax[tid] > vmax[tid+x]) ? vmax[tid] : vmax[tid+x];
  __syncthreads();
  }

if(tid < 16) warpReduction(&vmax[0], tid);

if(tid == 0) cf[blockIdx.x + dims.y*blockIdx.y] = vmax[0];
}

/* Must be launched 16x16 with grid size [ceil(Nx/16) nz] */
__global__ void cuda_advectFreezeSpeedY(double *v, double *cf, int3 dims)
{
__shared__ double tile[256];
double q;

int tix = threadIdx.x;
int tiy = threadIdx.y;

if( (threadIdx.x + 16*blockIdx.x) >= dims.x) return;

// Do all the pointer arithmetic once,
// Position ourselves at the start
// Translate v by our global x index, the thread y index, and the block z index.
int offset = (threadIdx.x + 16*blockIdx.x) + (dims.x*tiy) + (dims.x*dims.y*blockIdx.y);
v += offset;

int tileIdx = tix + 16*tiy;

tile[tileIdx] = 0.0; // preload zero for her pleasure

int y;
for(y = tiy; y < dims.y; y += 16) {
  // Load one tile
  q = fabs(*v);
  if(q > tile[tileIdx]) tile[tileIdx] = q;
  v += 16*dims.x;
  }

  __syncthreads();

  if(threadIdx.y < 8) {
    if(tile[tileIdx+128] > tile[tileIdx]) tile[tileIdx] = tile[tileIdx+128];
  }
  __syncthreads();
  if(threadIdx.y < 4) {
    if(tile[tileIdx+64] > tile[tileIdx]) tile[tileIdx] = tile[tileIdx+64];
  }
  __syncthreads();
  if(threadIdx.y < 2) {
    if(tile[tileIdx+32] > tile[tileIdx]) tile[tileIdx] = tile[tileIdx+32];
  }
  __syncthreads();
  if(threadIdx.y == 0) {
    cf[threadIdx.x + 16*blockIdx.x + dims.x*blockIdx.y] = (tile[tileIdx+16] > tile[tileIdx]) ? tile[tileIdx+16] : tile[tileIdx];
  }
  
}

/* Finds max in Z direction */
/* Invoke with 16x16 threads */
__global__ void cuda_advectFreezeSpeedZ(double *v, double *cf, int3 dims)
{

int stride = (threadIdx.x + 16*blockIdx.x) + dims.x*(threadIdx.y + 16*blockIdx.y);
v += stride;
cf += stride;

stride = dims.x * dims.y;

if(threadIdx.x + 16*blockIdx.x >= dims.x) return;
if(threadIdx.y + 16*blockIdx.y >= dims.y) return;

double qmax = 0.0;
double q;
int z;
for(z = 0; z < dims.z; z++) {
  q = fabs(*v);
  if(q > qmax) qmax = q;
  v += stride;
  }

*cf = qmax;

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
__global__ void cukern_magnetTVDstep_uniformX(double *bW, double *velGrid, double *Cf, double *mag, double *fluxout, double lambda, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double fluxR[TILEDIM_X * TILEDIM_Y + 2];
__shared__ double fluxL[TILEDIM_X * TILEDIM_Y + 2];
__shared__ double derivR[TILEDIM_X * TILEDIM_Y + 2];
__shared__ double derivL[TILEDIM_X * TILEDIM_Y + 2];

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
    fluxR[tileAddr] = (C_f + velGrid[globAddr])*bW[globAddr]/2.0;
    fluxL[tileAddr] = (C_f - velGrid[globAddr])*bW[globAddr]/2.0;

    // Keep in mind, ANY operation that refers to other than register variables or flux[tileAddr] MUST have a __syncthreads() after it or there will be sadness.
    __syncthreads();

    /* Leftgoing advection */
    derivL[tileAddr] = (fluxL[tileAddr-1] - fluxL[tileAddr])/2.0;
    /* Rightgoing advection */
    derivR[tileAddr] = (fluxR[tileAddr] - fluxR[tileAddr-1])/2.0; /* bkw deriv */

    // We're finished with velocityFlow, reuse to store the flux which we're about to limit
    __syncthreads();

    fluxR[tileAddr] += LIMITERFUNC(derivR[tileAddr+1],derivR[tileAddr]);
    fluxL[tileAddr] += LIMITERFUNC(derivL[tileAddr],derivL[tileAddr+1]);

    __syncthreads();

    if(ITakeDerivative) {
        mag[globAddr]     = mag[globAddr] - lambda*(fluxR[tileAddr] - fluxR[tileAddr-1] - fluxL[tileAddr+1] + fluxL[tileAddr]);
        fluxout[globAddr] = fluxR[tileAddr-1] - fluxL[tileAddr];
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
double locFlux;
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

    flux[tileAddr] = locFlux + .5*LIMITERFUNC(derivL[tileAddr],derivR[tileAddr]);

    __syncthreads();

    if(ITakeDerivative) {
        mag[globAddr]     = mag[globAddr] - lambda*(flux[tileAddr] - flux[tileAddr-1]);
        fluxout[globAddr] = flux[tileAddr-1];
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
double locFlux;
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

    flux[tileAddr] = locFlux + .5*LIMITERFUNC(derivL[tileAddr], derivR[tileAddr]);

    __syncthreads();

    if(ITakeDerivative) {
        mag[globAddr]     = mag[globAddr] - lambda*(flux[tileAddr] - flux[tileAddr-1]);
        fluxout[globAddr] = flux[tileAddr-1];
        }

    __syncthreads();

    /* This determines the "Z" direction */
    globAddr += ORTHOG_MEMSTEP;
    }

}


