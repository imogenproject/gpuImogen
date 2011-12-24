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

__global__ void cukern_magnetTVDstep_uniformX(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int nx);
__global__ void cukern_magnetTVDstep_uniformY(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims);
__global__ void cukern_magnetTVDstep_uniformZ(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims);

#define BLOCKDIMA 18
#define BLOCKDIMAM2 16
#define BLOCKDIMB 8

#define BLOCKLEN 128
#define BLOCKLENP4 132

__device__ void cukern_FluxLimiter_VanLeerx(double deriv[2][BLOCKLENP4], double flux[BLOCKLENP4]);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // At least 2 arguments expected
    // Input and result
    if ((nrhs!=5) || (nlhs != 2)) mexErrMsgTxt("Wrong number of arguments: need [B, flux] = cudaMagTVD(magW, velgrid, velflow, lambda, dir)\n");

    // Get source array info and create destination arrays
    ArrayMetadata amd;
    double **srcs = getGPUSourcePointers(prhs, &amd, 0, 2);
    double **dest = makeGPUDestinationArrays((int64_t *)mxGetData(prhs[0]), plhs, 2);

    // Establish launch dimensions & a few other parameters
    int fluxDirection = (int)*mxGetPr(prhs[4]);
    double lambda     = *mxGetPr(prhs[3]);

    int3 arraySize;
    arraySize.x = amd.dim[0];
    amd.ndims > 1 ? arraySize.y = amd.dim[1] : arraySize.y = 1;
    amd.ndims > 2 ? arraySize.z = amd.dim[2] : arraySize.z = 1;

    dim3 blocksize, gridsize;
    switch(fluxDirection) {
        case 1: // X direction flux. This is "priveleged" in that the shift and natural memory load directions align
            blocksize.x = BLOCKLEN+4; blocksize.y = blocksize.z = 1;
            gridsize.x = arraySize.y;
            gridsize.y = arraySize.z;
            cukern_magnetTVDstep_uniformX<<<gridsize , blocksize>>>(srcs[0], srcs[1], srcs[2], dest[0], dest[1], lambda, arraySize.x);
            break;
        case 2: // Y direction flux: u = y, v = x, w = z
            blocksize.x = BLOCKDIMB; blocksize.y = BLOCKDIMAM2;

            gridsize.x = arraySize.x / blocksize.x; gridsize.x += 1*(blocksize.x*gridsize.x < arraySize.x);
            gridsize.y = arraySize.y / blocksize.y; gridsize.y += 1*(blocksize.y*gridsize.y < arraySize.y);

            blocksize.y = BLOCKDIMA;

            cukern_magnetTVDstep_uniformY<<<gridsize , blocksize>>>(srcs[0], srcs[1], srcs[2], dest[0], dest[1], lambda, arraySize);
            break;
        case 3: // Z direction flux: u = z, v = x, w = y;
            blocksize.x = BLOCKDIMB; blocksize.y = BLOCKDIMAM2;

            gridsize.x = arraySize.x / blocksize.x; gridsize.x += 1*(blocksize.x*gridsize.x < arraySize.x);
            gridsize.y = arraySize.z / blocksize.y; gridsize.y += 1*(blocksize.y*gridsize.y < arraySize.z);

            blocksize.y = BLOCKDIMA;

            cukern_magnetTVDstep_uniformZ<<<gridsize , blocksize>>>(srcs[0], srcs[1], srcs[2], dest[0], dest[1], lambda, arraySize);

            break;
    }

}

__global__ void cukern_magnetTVDstep_uniformX(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int nx)
{
double locVelFlow;
__shared__ double flux[BLOCKLENP4];
__shared__ double derivLR[2][BLOCKLEN+4];

/* Step 0 - obligatory annoying setup stuff (ASS) */
int I0 = nx * (blockIdx.x + gridDim.x * blockIdx.y);
int Xindex = (threadIdx.x-2);
int Xtrack = Xindex;
Xindex += nx * (threadIdx.x < 2);

int x;
bool doIflux = (threadIdx.x > 1) && (threadIdx.x < BLOCKLEN+2);

while(Xtrack < nx + 2) {
    x = I0 + (Xindex % nx) ;

    // First step: calculate local flux
    flux[threadIdx.x] = velGrid[x]*bW[x];
    if(doIflux && (Xindex < nx)) fluxout[x] = flux[threadIdx.x];
    locVelFlow = velFlow[x];
    __syncthreads();
    
    // Second step - calculate derivatives and apply limiter
    // right derivative
    if(locVelFlow == 1) { derivLR[1][threadIdx.x] = flux[(threadIdx.x+1)%BLOCKLENP4] - flux[(threadIdx.x+2)%BLOCKLENP4]; }
        else            { derivLR[1][threadIdx.x] = flux[(threadIdx.x+1)%BLOCKLENP4] - flux[threadIdx.x]; }
    // left derivative
    if(locVelFlow == 1) { derivLR[1][threadIdx.x] = flux[threadIdx.x] - flux[(threadIdx.x+1)%BLOCKLENP4]; }
        else            { derivLR[1][threadIdx.x] = flux[threadIdx.x] - flux[threadIdx.x-1]; }

    // Third step - Apply flux limiter function
    __syncthreads();
    cukern_FluxLimiter_VanLeerx(derivLR, flux);
    __syncthreads();

    // Fourth step - write to output array
    if( doIflux && (Xindex < nx) ) {
        mag[x] = mag[x] - lambda * ( flux[threadIdx.x] - flux[threadIdx.x - 1] ); 
    }

    Xindex += BLOCKLEN;
    Xtrack += BLOCKLEN;
    __syncthreads();
    }
}

__global__ void cukern_magnetTVDstep_uniformY(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims)
{
double v, b, locVelFlow;

__shared__ double tile[BLOCKDIMB][BLOCKDIMA];
__shared__ double flux[BLOCKDIMB][BLOCKDIMA];

// Dimensions into the array
int myx = blockIdx.x*BLOCKDIMB + threadIdx.x;
int myy = blockIdx.y*BLOCKDIMAM2 + threadIdx.y - 1;

if((myx >= dims.x) || (myy > dims.y)) return; // we keep an extra Y thread for the finite diff.

bool IWrite = (threadIdx.y > 0) && (threadIdx.y <= BLOCKDIMAM2) && (myy < dims.y) && (myy >= 0);
// Exclude threads at the boundary of the fluxing direction from writing back

if(myy < 0) myy += dims.y; // wrap left edge back to right edge
myy = myy % dims.y; // wrap right edge back to left

int x = myx + dims.x*myy;
int z;

for(z = 0; z < dims.z; z++) {
    v = velGrid[x];
    b = mag[x];

    // first calculate velocityFlow
    tile[threadIdx.x][threadIdx.y] = v;
    flux[threadIdx.x][threadIdx.y] = b*v;
    __syncthreads();

    locVelFlow = (tile[threadIdx.x][threadIdx.y] + tile[threadIdx.x][(threadIdx.y+1) % BLOCKDIMA]);
    if(locVelFlow < 0.0) { locVelFlow = 1.0; } else { locVelFlow = 0.0; }

    __syncthreads();

    // Second step - calculate flux
    if(locVelFlow == 1) { tile[threadIdx.x][threadIdx.y] = flux[threadIdx.x][(threadIdx.y + 1)%BLOCKDIMA]; } else 
                        { tile[threadIdx.x][threadIdx.y] = flux[threadIdx.x][threadIdx.y]; }
   
    __syncthreads();

    // Third step - Perform flux and write to output array
    if( IWrite ) {
            bW[x] = b - lambda * ( tile[threadIdx.x][threadIdx.y] - tile[threadIdx.x][threadIdx.y-1]);
            velFlow[x] = locVelFlow;
        }

    x += dims.x*dims.y;
    __syncthreads(); 
    }

}

__global__ void cukern_magnetTVDstep_uniformZ(double *bW, double *velGrid, double *velFlow, double *mag, double *fluxout, double lambda, int3 dims)
{
double v, b, locVelFlow;

__shared__ double tile[BLOCKDIMB][BLOCKDIMA];
__shared__ double flux[BLOCKDIMB][BLOCKDIMA];

int myx = blockIdx.x*BLOCKDIMB + threadIdx.x;
int myz = blockIdx.y*BLOCKDIMAM2 + threadIdx.y - 1;

if((myx >= dims.x) || (myz > dims.z)) return; // we keep an extra Y thread for the finite diff.

bool IWrite = (threadIdx.y > 0) && (threadIdx.y <= BLOCKDIMAM2) && (myz < dims.y) && (myz >= 0);
// Exclude threads at the boundary of the fluxing direction from writing back

if(myz < 0) myz += dims.z; // wrap left edge back to right edge
myz = myz % dims.z; // wrap right edge back to left

int x = myx + dims.x*dims.y*myz;
int y;

for(y = 0; y < dims.y; y++) {
    v = velGrid[x];
    b = mag[x];

    // first calculate velocityFlow
    tile[threadIdx.x][threadIdx.y] = v;
    flux[threadIdx.x][threadIdx.y] = b*v;
    __syncthreads();

    locVelFlow = (tile[threadIdx.x][threadIdx.y] + tile[threadIdx.x][(threadIdx.y+1) % BLOCKDIMA]);
    if(locVelFlow < 0.0) { locVelFlow = 1.0; } else { locVelFlow = 0.0; }

    __syncthreads();

    // Second step - calculate flux
    if(locVelFlow == 1) { tile[threadIdx.x][threadIdx.y] = flux[threadIdx.x][(threadIdx.y + 1)%BLOCKDIMA]; } else 
                        { tile[threadIdx.x][threadIdx.y] = flux[threadIdx.x][threadIdx.y]; }
   
    __syncthreads();

    // Third step - Perform flux and write to output array
    if( IWrite ) {
            bW[x] = b - lambda * ( tile[threadIdx.x][threadIdx.y] - tile[threadIdx.x][threadIdx.y-1]);
            velFlow[x] = locVelFlow;
        }

    x += dims.x;
    __syncthreads(); 
    }

}

// NOTE: This function divides the fluxes by two, as this is NOT done in the left/right decoupling phase
// in order to avoid any more multiplies by 1/2 than necessary.
__device__ void cukern_FluxLimiter_VanLeerx(double deriv[2][BLOCKLENP4], double flux[BLOCKLENP4])
{

double r;

r = deriv[0][threadIdx.x] * deriv[1][threadIdx.x];
if(r < 0.0) r = 0.0;

r = r / ( deriv[0][threadIdx.x] + deriv[1][threadIdx.x]);
if (isnan(r)) { r = 0.0; }

flux[threadIdx.x] += r;

}
