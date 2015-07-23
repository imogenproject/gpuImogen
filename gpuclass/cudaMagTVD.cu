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

   Calculates the advection of magnetic field 'Bw' by velocity 'velGrid' and stores the result in 'mag',

   This update is calculated using the second-order TVD method for linear advection, and also outputs
   the flux used in the calculation to 'fluxout'. In order to preserve the divergence of the magnetic
   field to machine precision, this flux is used to update the mirror component of the magnetic field
   (i.e. if this function updates bx due to vy, the flux is used to update by).

   */

__global__ void cukern_advectFreezeSpeedX(double *v, double *cf, int3 dims);
__global__ void cukern_advectFreezeSpeedY(double *v, double *cf, int3 dims);
__global__ void cukern_advectFreezeSpeedZ(double *v, double *cf, int3 dims);

__global__ void cukern_magnetTVDstep_uniformX(double *bW, double *velGrid, double *Cf, double *mag, double *fluxout, double lambda, int3 dims);
//__global__ void cukern_magnetTVDstep_uniformX(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims);
__global__ void cukern_magnetTVDstep_uniformY(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims);
__global__ void cukern_magnetTVDstep_uniformZ(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims);

#define BLOCKDIMA 18
#define BLOCKDIMAM2 16
#define BLOCKDIMB 8

#define BLOCKLEN 128
#define BLOCKLENP4 132

#define FREEZE_BLKDIM 256

#define LIMITERFUNC fluxLimiter_VanLeer

__device__ void cukern_FluxLimiter_VanLeer_x(double deriv[2][BLOCKLENP4], double flux[BLOCKLENP4]);
__device__ void cukern_FluxLimiter_VanLeer_yz(double deriv[2][BLOCKDIMB][BLOCKDIMA], double flux[BLOCKDIMB][BLOCKDIMA]);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // At least 2 arguments expected
    // Input and result
    if ((nrhs != 5) || (nlhs != 1)) mexErrMsgTxt("Wrong number of arguments: need [flux] = cudaMagTVD(magW, mag, velgrid, lambda, dir)\n");

    CHECK_CUDA_ERROR("entering cudaMagTVD");

    // Get source array info and create destination arrays
    MGArray src[3];
    int worked = MGA_accessMatlabArrays(prhs, 0, 2, src);
    MGArray *dst = MGA_createReturnedArrays(plhs, 1, src);

    // Establish launch dimensions & a few other parameters
    double lambda     = *mxGetPr(prhs[3]);
    int fluxDirection = (int)*mxGetPr(prhs[4]);

    int3 arraySize;
    arraySize.x = src->dim[0];
    arraySize.y = src->dim[1];
    arraySize.z = src->dim[2];

    dim3 blocksize, gridsize;
    gridsize.z = 1; blocksize.z = 1;
    double *cf;

    switch(fluxDirection) {
        case 1: // X direction
            blocksize.x = 256; blocksize.y = 1; 
            gridsize.x = arraySize.y; gridsize.y = arraySize.z;

            cudaMalloc(&cf, arraySize.y*arraySize.z*sizeof(double));
            CHECK_CUDA_ERROR("magnetic cfreeze allocate");
  
            cukern_advectFreezeSpeedX<<<gridsize, blocksize>>>(src[2].devicePtr[0], cf, arraySize);
            CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, src, fluxDirection, "magnetic calculate cfreeze X dir");

            blocksize.x = 24; blocksize.y = 16;
            gridsize.x = (int)ceil(arraySize.x / 18.0);
            gridsize.y = (int)ceil((double)arraySize.y / (double)blocksize.y);

            cukern_magnetTVDstep_uniformX<<<gridsize , blocksize>>>(src[0].devicePtr[0], src[2].devicePtr[0], cf, src[1].devicePtr[0], dst->devicePtr[0], lambda, arraySize);
            CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, src, fluxDirection, "magnetic TVD X step");

            cudaFree(cf);
            break;
        case 2: // Y direction flux: u = y, v = x, w = z
            blocksize.x = blocksize.y = 16;
            gridsize.x = (int)ceil(arraySize.x/16.0); gridsize.y = arraySize.z;

            cudaMalloc(&cf, arraySize.x*arraySize.z*sizeof(double));
            CHECK_CUDA_ERROR("magnetic cfreeze allocate.");

            cukern_advectFreezeSpeedY<<<gridsize, blocksize>>>(src[2].devicePtr[0], cf, arraySize);
            CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, src, fluxDirection, "magnetic calculate cfreeze X dir");

            blocksize.x = 16; blocksize.y = 24;
            gridsize.x = (int)ceil(arraySize.x / 16.0);
            gridsize.y = (int)ceil(arraySize.y / 18.0);

            cukern_magnetTVDstep_uniformY<<<gridsize , blocksize>>>(src[0].devicePtr[0], src[2].devicePtr[0], cf, src[1].devicePtr[0], dst->devicePtr[0], lambda, arraySize);
            CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, src, fluxDirection, "magnetic TVD Y step");

            cudaFree(cf);
            break;
        case 3: // Z direction flux: u = z, v = x, w = y;
            blocksize.x = blocksize.y = 16;
            gridsize.x = (int)ceil(arraySize.x/16.0);
            gridsize.y = (int)ceil(arraySize.y/16.0);

            cudaMalloc(&cf, arraySize.x*arraySize.y*sizeof(double));
            CHECK_CUDA_ERROR("magnetic cfreeze allocate");

            cukern_advectFreezeSpeedZ<<<gridsize, blocksize>>>(src[2].devicePtr[0], cf, arraySize);
            CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, src, fluxDirection, "magnetic calculate cfreeze X dir");

            blocksize.x = 24; blocksize.y = 16;
            gridsize.x = (int)ceil(arraySize.z / 18.0);
            gridsize.y = (int)ceil((double)arraySize.x / (double)blocksize.y);

            cukern_magnetTVDstep_uniformZ<<<gridsize , blocksize>>>(src[0].devicePtr[0], src[2].devicePtr[0], cf, src[1].devicePtr[0], dst->devicePtr[0], lambda, arraySize);
            CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, src, fluxDirection, "magnetic TVD Y step");

            cudaFree(cf);
            break;
        }

    free(dst);

}


/* Warp-synchronous reduction of data */
/* Compute 2+ devices are only half-warp synchronous so use 16 threads */
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
__global__ void cukern_advectFreezeSpeedX(double *v, double *cf, int3 dims)
{

v += dims.x*(blockIdx.x + dims.y*blockIdx.y);

int tid = threadIdx.x;
__shared__ double vmax[FREEZE_BLKDIM];
double umax = 0.0;
double t;
vmax[tid] = 0.0;
/* Calculate max per stride of blockDim.x */
int x = threadIdx.x;
for(x = tid; x < dims.x; x += FREEZE_BLKDIM) {
  t = fabs(v[x]);
  umax = (t > umax) ? t : umax;
  }
vmax[tid] = umax;

if(tid > FREEZE_BLKDIM/2) return;
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
__global__ void cukern_advectFreezeSpeedY(double *v, double *cf, int3 dims)
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
__global__ void cukern_advectFreezeSpeedZ(double *v, double *cf, int3 dims)
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
__shared__ double flux [TILEDIM_X * TILEDIM_Y + 2];
__shared__ double deriv[TILEDIM_X * TILEDIM_Y + 2];
__shared__ double fluxB[TILEDIM_X * TILEDIM_Y + 2];

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
double v0;

/* We step through the array, one XY plane at a time */
int z;
for(z = 0; z < ORTHOG_DIMENSION; z++) {
    v0 = velGrid[globAddr];
    flux[tileAddr] = v0 * bW[globAddr];
    __syncthreads();
    deriv[tileAddr] = (flux[tileAddr] - flux[tileAddr-1])/2.0;
    __syncthreads();

    if(v0 > 0) { 
        fluxB[tileAddr] = flux[tileAddr] + LIMITERFUNC(deriv[tileAddr],deriv[tileAddr+1]);
    } else {
        fluxB[tileAddr] = flux[tileAddr+1]+LIMITERFUNC(deriv[tileAddr+1],deriv[tileAddr+2]); 
    }
    __syncthreads();

    if(ITakeDerivative) {
        mag[globAddr]     = mag[globAddr] - lambda*(fluxB[tileAddr] - fluxB[tileAddr-1]);
        fluxout[globAddr] = fluxB[tileAddr-1];
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
__global__ void cukern_magnetTVDstep_uniformY(double *bW, double *velGrid, double *Cf, double *mag, double *fluxout, double lambda, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double flux [TILEDIM_X * TILEDIM_Y + 2];
__shared__ double deriv[TILEDIM_X * TILEDIM_Y + 2];
__shared__ double fluxB[TILEDIM_X * TILEDIM_Y + 2];

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
double v0;

/* We step through the array, one XY plane at a time */
int z;
for(z = 0; z < ORTHOG_DIMENSION; z++) {
    v0 = velGrid[globAddr];
    flux[tileAddr] = v0 * bW[globAddr];
    __syncthreads();
    deriv[tileAddr] = (flux[tileAddr] - flux[tileAddr-1])/2.0;
    __syncthreads();
    flux[tileAddr] += LIMITERFUNC(deriv[tileAddr],deriv[tileAddr+1]);
    __syncthreads();

    if(v0 > 0) {
        fluxB[tileAddr] = flux[tileAddr] + LIMITERFUNC(deriv[tileAddr],deriv[tileAddr+1]);
    } else {
        fluxB[tileAddr] = flux[tileAddr+1]+LIMITERFUNC(deriv[tileAddr+1],deriv[tileAddr+2]);
    }
    __syncthreads();

    if(ITakeDerivative) {
        mag[globAddr]     = mag[globAddr] - lambda*(fluxB[tileAddr] - fluxB[tileAddr-1]);
        fluxout[globAddr] = fluxB[tileAddr-1];
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
__global__ void cukern_magnetTVDstep_uniformZ(double *bW, double *velGrid, double *Cf, double *mag, double *fluxout, double lambda, int3 dims)
{
/* Declare any arrays to be used for storage/differentiation similarly. */
__shared__ double flux [TILEDIM_X * TILEDIM_Y + 2];
__shared__ double deriv[TILEDIM_X * TILEDIM_Y + 2];
__shared__ double fluxB[TILEDIM_X * TILEDIM_Y + 2];

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
double v0;

/* We step through the array, one XY plane at a time */
int z;
for(z = 0; z < ORTHOG_DIMENSION; z++) {
    v0 = velGrid[globAddr];
    flux[tileAddr] = v0 * bW[globAddr];
    __syncthreads();
    deriv[tileAddr] = (flux[tileAddr] - flux[tileAddr-1])/2.0;
    __syncthreads();
    flux[tileAddr] += LIMITERFUNC(deriv[tileAddr],deriv[tileAddr+1]);
    __syncthreads();

    if(v0 > 0) {
        fluxB[tileAddr] = flux[tileAddr] + LIMITERFUNC(deriv[tileAddr],deriv[tileAddr+1]);
    } else {
        fluxB[tileAddr] = flux[tileAddr+1]+LIMITERFUNC(deriv[tileAddr+1],deriv[tileAddr+2]);
    }
    __syncthreads();

    if(ITakeDerivative) {
        mag[globAddr]     = mag[globAddr] - lambda*(fluxB[tileAddr] - fluxB[tileAddr-1]);
        fluxout[globAddr] = fluxB[tileAddr-1];
        }

    __syncthreads();

    /* This determines the "Z" direction */
    globAddr += ORTHOG_MEMSTEP;
    }

}


