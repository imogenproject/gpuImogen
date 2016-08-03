#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif
#include "mex.h"

#include "mpi.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas.h"

#include "cudaCommon.h"
#include "directionalMaxFinder.h"


/* THIS FUNCTION:
   directionalMaxFinder has three different behaviors depending on how it is called.
   m = directionalMaxFinder(array) will calculate the global maximum of array
   c = directionalMaxFinder(a1, a2, direct) will find the max of |a1(r)+a2(r)| in the
      'direct' direction (1=X, 2=Y, 3=Z)
   c = directionalMaxFinder(rho, c_s, px, py, pz) will specifically calculate the x direction
       CFL limiting speed, max(|px/rho| + c_s)
    */

__global__ void cukern_DirectionalMax(double *d1, double *d2, double *out, int direct, int nx, int ny, int nz);
__global__ void cukern_GlobalMax(double *din, int n, double *dout);
__global__ void cukern_GlobalMax_forCFL(double *rho, double *cs, double *px, double *py, double *pz, int n, double *dout, int *dirOut);

#define BLOCKDIM 8
#define GLOBAL_BLOCKDIM 128

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // At least 2 arguments expected
  // Input and result
  if((nlhs == 0) || (nlhs > 2))
     mexErrMsgTxt("Either 1 return argument for simple & directional max or 2 for CFL max");

  if((nlhs == 2) && (nrhs != 5))
     mexErrMsgTxt("For CFL max require [max dir] = directionalMaxFinder(rho, soundspeed, px, py, pz)");

  if((nlhs == 1) && ((nrhs != 3) && (nrhs != 1)))
     mexErrMsgTxt("Either 1 or 3 arguments for one rturn argument");

  CHECK_CUDA_ERROR("entering directionalMaxFinder");

  int i; 
  int sub[6];

switch(nrhs) {
  case 3: {
  /* m = directionalMaxFinder(v, c, dir) 
   * computes MAX[ (|v|+c)_{ijk} ] in the dir = 1/2/3/ ~ i/j/k direction
   */ 
  MGArray in[2];
  int worked   = MGA_accessMatlabArrays(prhs, 0, 1, in);
  MGArray *out = MGA_createReturnedArrays(plhs, 1, in);

  dim3 blocksize, gridsize, dims;

  int maxDirection = (int)*mxGetPr(prhs[2]);

  for(i = 0; i < in->nGPUs; i++) {
    calcPartitionExtent(in, i, sub);
    dims = makeDim3(&sub[3]);
    blocksize = makeDim3(BLOCKDIM, BLOCKDIM, 1);

    switch(maxDirection) {
      case 1:
        gridsize.x = dims.y / BLOCKDIM; if (gridsize.x * BLOCKDIM < dims.y) gridsize.x++;
        gridsize.y = dims.z / BLOCKDIM; if (gridsize.y * BLOCKDIM < dims.z) gridsize.y++;
        break;
      case 2:
        gridsize.x = dims.x / BLOCKDIM; if (gridsize.x * BLOCKDIM < dims.x) gridsize.x++;
        gridsize.y = dims.z / BLOCKDIM; if (gridsize.y * BLOCKDIM < dims.z) gridsize.y++;
        break;
      case 3:
        gridsize.x = dims.x / BLOCKDIM; if (gridsize.x * BLOCKDIM < dims.x) gridsize.x++;
        gridsize.y = dims.y / BLOCKDIM; if (gridsize.y * BLOCKDIM < dims.y) gridsize.y++;
        break;
      default: mexErrMsgTxt("Direction passed to directionalMaxFinder is not in { 1,2,3 }");
    }
  
    cudaSetDevice(in->deviceID[i]);
    CHECK_CUDA_ERROR("setCudaDevice()");
    cukern_DirectionalMax<<<gridsize, blocksize>>>(in[0].devicePtr[i], in[1].devicePtr[i], out->devicePtr[i], maxDirection, dims.x, dims.y, dims.z);
    CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, in, i, "directionalMaxFinder(a,b,direct)");
  }

// FIXME the above function executes in the worst possible way
// FIXME it stores the local max in N locations in a non-flattened array
// FIXME which will effectively stymie MGA_globalPancakeReduce
//  MGArray *finalResult;
//    MGA_globalPancakeReduce(out, finalResult, maxDirection, 0, 1);

  free(out);

  } break;
  case 1: { // NOTE: This function has been 80% uncrapped by the new MGA_*ReduceScalar function
    MGArray a;
    MGA_accessMatlabArrays(prhs, 0, 0, &a);

    // FIXME: This lacks the proper topology to pass to the global reducer so we "fake" it here
    double maxval;
    int returnCode = MGA_globalReduceScalar(&a, &maxval, MGA_OP_MAX, NULL);

    mwSize dims[2];
    dims[0] = 1;
    dims[1] = 1;
    plhs[0] = mxCreateNumericArray (2, dims, mxDOUBLE_CLASS, mxREAL);

    // Now apply result to all nodes
    double *globalMax = mxGetPr(plhs[0]);

    MPI_Allreduce((void *)&maxval, (void *)globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  } break;
  case 5: {
    // Get input arrays: [rho, c_s, px, py, pz]
    MGArray fluid[5];
    int worked = MGA_accessMatlabArrays(prhs, 0, 4, &fluid[0]);

    dim3 blocksize, gridsize;
    blocksize.x = GLOBAL_BLOCKDIM; blocksize.y = blocksize.z = 1;

    // Launches enough blocks to fully occupy the GPU
    gridsize.x = 32;
    gridsize.y = gridsize.z =1;

    // Allocate enough pinned memory to hold results
    double *blkA[fluid->nGPUs];
    int *blkB[fluid->nGPUs];
    int hblockElements = gridsize.x;

    int i;
    for(i = 0; i < fluid->nGPUs; i++) {
        cudaSetDevice(fluid->deviceID[i]);
        CHECK_CUDA_ERROR("cudaSetDevice()");
        cudaMallocHost((void **)&blkA[i], hblockElements * sizeof(double));
        CHECK_CUDA_ERROR("CFL malloc doubles");
        cudaMallocHost((void **)&blkB[i], hblockElements * sizeof(int));
        CHECK_CUDA_ERROR("CFL malloc ints");

        cukern_GlobalMax_forCFL<<<gridsize, blocksize>>>(
		fluid[0].devicePtr[i],
		fluid[1].devicePtr[i],
		fluid[2].devicePtr[i],
		fluid[3].devicePtr[i],
		fluid[4].devicePtr[i],
		fluid[0].partNumel[i], blkA[i], blkB[i]);
        CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &fluid[0], i, "CFL max finder");
    }

    mwSize dims[2];
    dims[0] = 1;
    dims[1] = 1;
    plhs[0] = mxCreateNumericArray (2, dims, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray (2, dims, mxDOUBLE_CLASS, mxREAL);

    double *maxout = mxGetPr(plhs[0]);
    double *dirout = mxGetPr(plhs[1]);
    int devCount;

    for(devCount = 0; devCount < fluid->nGPUs; devCount++) {
        cudaSetDevice(fluid->deviceID[devCount]);
        CHECK_CUDA_ERROR("cudaSetDevice()");
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR("cudaDeviceSynchronize()");
        if(devCount == 0) { maxout[0] = blkA[0][0]; dirout[0] = blkB[devCount][0]; } // Special first case: initialize nodeMax

        for(i = 0; i < gridsize.x; i++)
    	    if(blkA[devCount][i] > maxout[0]) { maxout[0] = blkA[devCount][i]; dirout[0] = blkB[devCount][i]; }

        cudaFreeHost(blkA[devCount]);
        CHECK_CUDA_ERROR("cudaFreeHost");
	cudaFreeHost(blkB[devCount]);
	CHECK_CUDA_ERROR("cudaFreeHost");
    }

// FIXME This needs the mpi_reduce with complex structures thingie done to it
//    MPI_Allreduce((void *)&nodeMax, (void *)globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  } break;
}

}

__global__ void cukern_DirectionalMax(double *d1, double *d2, double *out, int direct, int nx, int ny, int nz)
{

int myU = threadIdx.x + blockDim.x*blockIdx.x;
int myV = threadIdx.y + blockDim.y*blockIdx.y;

double maxSoFar = -1e37;
int addrMax, myBaseaddr;

switch(direct) {
  case 1: { // Seek maxima in the X direction. U=y, V=z
    if ((myU >= ny) || (myV >= nz)) return;

    myBaseaddr = nx*(myU + ny*myV);
    addrMax = myBaseaddr + nx;

    for(; myBaseaddr < addrMax ; myBaseaddr++) {
      if ( abs(d1[myBaseaddr]) + d2[myBaseaddr] > maxSoFar) maxSoFar = abs(d1[myBaseaddr]) + d2[myBaseaddr];
      }

    myBaseaddr = nx*(myU + ny*myV);
    for(; myBaseaddr < addrMax ; myBaseaddr++) { out[myBaseaddr] = maxSoFar; }

  } break;
  case 2: { // Seek maxima in the Y direction. U=x, V=z
    if ((myU >= nx) || (myV >= nz)) return;

    myBaseaddr = myU + nx*ny*myV;
    addrMax = myBaseaddr + ny*nx;

    for(; myBaseaddr < addrMax ; myBaseaddr += nx) {
      if ( abs(d1[myBaseaddr]) + d2[myBaseaddr] > maxSoFar) maxSoFar = abs(d1[myBaseaddr]) + d2[myBaseaddr];
      }

    myBaseaddr = myU + nx*ny*myV;
    for(; myBaseaddr < addrMax ; myBaseaddr += nx) { out[myBaseaddr] = maxSoFar; }
  } break;
  case 3: { // Seek maxima in the Z direction; U=x, V=y
  if ((myU >= nx) || (myV >= ny)) return;

    myBaseaddr = myU + nx*myV;
    addrMax = myBaseaddr + nx*ny*nz;

    for(; myBaseaddr < addrMax ; myBaseaddr += nx*ny) {
      if ( abs(d1[myBaseaddr]) + d2[myBaseaddr] > maxSoFar) maxSoFar = abs(d1[myBaseaddr]) + d2[myBaseaddr];
      }

    myBaseaddr = myU + nx*myV;
    for(; myBaseaddr < addrMax ; myBaseaddr += nx) { out[myBaseaddr] = maxSoFar; }

  } break;
}

}

__global__ void cukern_GlobalMax(double *phi, int n, double *dout)
{
unsigned int tix = threadIdx.x;
int x = blockIdx.x * blockDim.x + tix;

__shared__ double W[256];

double Wmax = -1e37;
W[tix] = -1e37;
if(tix == 0) dout[blockIdx.x] = Wmax; // As a safety measure incase we return below

if(x >= n) return; // If we're fed a very small array, this will be easy

// Threads step through memory with a stride of (total # of threads), finphig the max in this space
while(x < n) {
  if(phi[x] > Wmax) Wmax = phi[x];
  x += blockDim.x * gridDim.x;
  }
W[tix] = Wmax;

x = 128;
while(x > 16) {
	if(tix >= x) return;
	__syncthreads();
	if(W[tix+x] > W[tix]) W[tix] = W[tix+x];
        x=x/2;
}

__syncthreads();

// We have one halfwarp (16 threads) remaining, proceed synchronously
if(W[tix+16] > W[tix]) W[tix] = W[tix+16]; if(tix >= 8) return;
if(W[tix+8] > W[tix]) W[tix] = W[tix+8]; if(tix >= 4) return;
if(W[tix+4] > W[tix]) W[tix] = W[tix+4]; if(tix >= 2) return;
if(W[tix+2] > W[tix]) W[tix] = W[tix+2]; if(tix) return;

dout[blockIdx.x] = (W[1] > W[0]) ? W[1] : W[0];

}

__global__ void cukern_GlobalMax_forCFL(double *rho, double *cs, double *px, double *py, double *pz, int n, double *out, int *dirOut)
{
unsigned int tix = threadIdx.x;
int x = blockIdx.x * blockDim.x + tix; // address
int blockhop = blockDim.x * gridDim.x;         // stepsize

// Do not use struct because 12-byte struct = bad memory pattern
__shared__ int    maxdir[GLOBAL_BLOCKDIM];
__shared__ double freeze[GLOBAL_BLOCKDIM];

double u, v;
int q;

freeze[tix] = 0.0;

if(x >= n) return; // This is unlikely but we may get a stupid-small resolution

// load first set and set maxdir
maxdir[tix] = 1;

u = abs(px[x]);
v = abs(py[x]);
if(v > u) { u = v; maxdir[tix] = 2; }
v = abs(pz[x]);
if(v > u) { u = v; maxdir[tix] = 3; }

freeze[tix] = u / rho[x] + cs[x];

x += blockhop; // skip the first block since we've already done it.

// load next set and compare until reaching end of array
while(x < n) {
  // Perform the max operation for this cell
  u = abs(px[x]);
  v = abs(py[x]);
  q = 1;
  if(v > u) { u = v; q = 2; }
  v = abs(pz[x]);
  if(v > u) { u = v; q = 3; }

  u = u / rho[x] + cs[x];
  // And compare-write to the shared array
  if(u > freeze[tix]) { freeze[tix] = u; maxdir[tix] = q; }

  x += blockhop;
  }

x = GLOBAL_BLOCKDIM / 2;
while(x > 16) {
        if(tix >= x) return;
        __syncthreads();
        if(freeze[tix+x] > freeze[tix]) { freeze[tix] = freeze[tix+x]; maxdir[tix] = maxdir[tix+x]; }
        x=x/2;
}

__syncthreads();

// We have one halfwarp (16 threads) remaining, proceed synchronously
if(freeze[tix+16] > freeze[tix]) { freeze[tix] = freeze[tix+16]; maxdir[tix] = maxdir[tix+16]; } if(tix >= 8) return;
if(freeze[tix+8] > freeze[tix])  { freeze[tix] = freeze[tix+8 ]; maxdir[tix] = maxdir[tix+8];  } if(tix >= 4) return;
if(freeze[tix+4] > freeze[tix])  { freeze[tix] = freeze[tix+4 ]; maxdir[tix] = maxdir[tix+4];  } if(tix >= 2) return;
if(freeze[tix+2] > freeze[tix])  { freeze[tix] = freeze[tix+2 ]; maxdir[tix] = maxdir[tix+2];  } if(tix) return;

out[blockIdx.x] = (freeze[1] > freeze[0]) ? freeze[1] : freeze[0];
dirOut[blockIdx.x] = (freeze[1] > freeze[0]) ? maxdir[1] : maxdir[0];

}

