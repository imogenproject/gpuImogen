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

  cudaCheckError("entering directionalMaxFinder");

ArrayMetadata amd;

switch(nrhs) {
  case 3: {
  double **srcs = getGPUSourcePointers(prhs, &amd, 0, 1);

  double **dest = makeGPUDestinationArrays(&amd, plhs, 1);

  dim3 blocksize, gridsize, dims;

  dims.x = amd.dim[0];
  dims.y = amd.dim[1];
  dims.z = amd.dim[2];

  blocksize.x = blocksize.y = BLOCKDIM; blocksize.z =1;

  switch((int)*mxGetPr(prhs[2])) {
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
  
  //printf("%i %i %i %i %i %i\n", gridsize.x, gridsize.y, gridsize.z, blocksize.x, blocksize.y, blocksize.z);
  cukern_DirectionalMax<<<gridsize, blocksize>>>(srcs[0], srcs[1], dest[0], (int)*mxGetPr(prhs[2]), dims.x, dims.y, dims.z);


  cudaError_t epicFail = cudaGetLastError();
  if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, nrhs, "directional max finder");


  } break;
  case 1: {
    double **arr  = getGPUSourcePointers(prhs, &amd, 0, 0);

    dim3 blocksize, gridsize;
    blocksize.x = 256; blocksize.y = blocksize.z = 1;

    gridsize.x = 64;
    gridsize.y = gridsize.z =1;

    double *blkA;
    cudaMalloc(&blkA, gridsize.x * sizeof(double));

    cukern_GlobalMax<<<gridsize, blocksize>>>(arr[0], amd.numel, blkA);

    double maxes[gridsize.x];
    cudaMemcpy(&maxes[0], blkA, sizeof(double)*gridsize.x, cudaMemcpyDeviceToHost);
    cudaFree(blkA);

    mwSize dims[2];
    dims[0] = 1;
    dims[1] = 1;
    plhs[0] = mxCreateNumericArray (2, dims, mxDOUBLE_CLASS, mxREAL);

    double *d = mxGetPr(plhs[0]);
    d[0] = maxes[0];

    int i; for(i = 1; i < gridsize.x; i++) { if(maxes[i] > d[0]) d[0] = maxes[i];  }

    cudaError_t epicFail = cudaGetLastError();
    if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, nrhs, "cuda global max kern");

  } break;
  case 5: {
    // Get input arrays: [rho, c_s, px, py, pz]
    double **arraysIn = getGPUSourcePointers(prhs, &amd, 0, 4); 

    dim3 blocksize, gridsize;
    blocksize.x = GLOBAL_BLOCKDIM; blocksize.y = blocksize.z = 1;

    // Launches enough blocks to fully occupy the GPU
    gridsize.x = 64;
    gridsize.y = gridsize.z =1;

    // Allocate space for each block to print its max into
    double *blkA; int *blkB;
    cudaMalloc(&blkA, gridsize.x * sizeof(double));
    cudaMalloc(&blkB, gridsize.x * sizeof(int));
    // Searches (blockdim*griddim) at a time until getting to the end.
    cudaError_t epicFail = cudaGetLastError();
    if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, nrhs, "CFL malloc");


    cukern_GlobalMax_forCFL<<<gridsize, blocksize>>>(arraysIn[0], arraysIn[1], arraysIn[2], arraysIn[3], arraysIn[4], amd.numel, blkA, blkB);

    epicFail = cudaGetLastError();
    if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, nrhs, "CFL max finder");


    // Copy the result back to the CPU for final analysis of the 64 potential maxima
    double maxes[gridsize.x]; int maxIndices[gridsize.x];
    cudaMemcpy(&maxes[0], blkA, sizeof(double)*gridsize.x, cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxIndices[0], blkB, sizeof(int)*gridsize.x, cudaMemcpyDeviceToHost);

    epicFail = cudaGetLastError();
    if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, nrhs, "CFL max finder memcopy");

    cudaFree(blkA);
    cudaFree(blkB);

    mwSize dims[2];
    dims[0] = 1;
    dims[1] = 1;
    plhs[0] = mxCreateNumericArray (2, dims, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray (2, dims, mxDOUBLE_CLASS, mxREAL);

    double *maxout = mxGetPr(plhs[0]);
    double *dirout = mxGetPr(plhs[1]);
    maxout[0] = maxes[0];
    dirout[0] = (double)maxIndices[0];
    
    int i; for(i = 1; i < gridsize.x; i++) { if(maxes[i] > maxout[0]) { maxout[0] = maxes[i]; dirout[0] = maxIndices[0]; }  }

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
  case 3: { // Seeek maxima in the Z direction; U=x, V=y
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

__global__ void cukern_GlobalMax(double *din, int n, double *dout)
{

int x = blockIdx.x * blockDim.x + threadIdx.x;
__shared__ double locBloc[256];

double CsMax = -1e37;
locBloc[threadIdx.x] = -1e37;
if(threadIdx.x == 0) dout[blockIdx.x] = locBloc[0]; // As a safety measure incase we return below

if(x >= n) return; // If we're fed a very small array, this will be easy

// Threads step through memory with a stride of (total # of threads), finding the max in this space
while(x < n) {
  if(din[x] > CsMax) CsMax = din[x];
  x += blockDim.x * gridDim.x;
  }
locBloc[threadIdx.x] = CsMax;

// Synchronize, then logarithmically fan in to identify each block's maximum
__syncthreads();

x = 2;
while(x < 256) {
  if(threadIdx.x % x != 0) break;

  if(locBloc[threadIdx.x + x/2] > locBloc[threadIdx.x]) locBloc[threadIdx.x] = locBloc[threadIdx.x + x/2];

  x *= 2;
  }

__syncthreads();

// Make sure the max is written and visible; each block writes one value. We test these 30 or so in CPU.
if(threadIdx.x == 0) dout[blockIdx.x] = locBloc[0];

}

__global__ void cukern_GlobalMax_forCFL(double *rho, double *cs, double *px, double *py, double *pz, int n, double *out, int *dirOut)
{
int x = blockIdx.x * blockDim.x + threadIdx.x; // address
int blockhop = blockDim.x * gridDim.x;         // stepsize

__shared__ int    maxdir[GLOBAL_BLOCKDIM];
__shared__ double setA[GLOBAL_BLOCKDIM];

double u, v;
int q;

setA[threadIdx.x] = 0.0;

if(x >= n) return; // This is unlikely but we may get a stupid-small resolution

// load first set and set maxdir
maxdir[threadIdx.x] = 1;
u = abs(px[x]);
v = abs(py[x]);
if(v > u) { u = v; maxdir[threadIdx.x] = 2; }
v = abs(pz[x]);
if(v > u) { u = v; maxdir[threadIdx.x] = 3; }

setA[threadIdx.x] = u / rho[x] + cs[x];

x += blockhop; // skip the first block since we've already done it.

// load next set and compare until reaching end of array
while(x < n) {
  //__syncthreads(); // prevent the memory accesses from breaking too far apart

  // Perform the max operation for this cell
  u = abs(px[x]);
  v = abs(py[x]);
  q = 1;
  if(v > u) { u = v; q = 2; }
  v = abs(pz[x]);
  if(v > u) { u = v; q = 3; }

  u = u / rho[x] + cs[x];
  // And compare-write to the shared array
  if(u > setA[threadIdx.x]) { setA[threadIdx.x] = u; maxdir[threadIdx.x] = q; }

  x += blockhop;
  }

__syncthreads();
// logarithmic foldin to determine max for block
if(threadIdx.x == 0) {
  int a;
  u = setA[0];
  q = maxdir[0];
  
  // do this the stupid way for now
  for(a = 1; a < GLOBAL_BLOCKDIM; a++) {
    if(setA[a] > u) { u = setA[a]; q = maxdir[a]; }
    }
  }


// write final values to out[blockIdx.x] and dirOut[blockidx.x]
if(threadIdx.x == 0) {
  out[blockIdx.x] = u;
  dirOut[blockIdx.x] = q;
  }

}


/*
// This is specifically for finding globalmax( max(abs(p_i))/rho + c_s) for the CFL constraint
__global__ void cukern_GlobalMax_forCFL(double *rho, double *cs, double *px, double *py, double *pz, int n, double *dout, int *dirOut)
{

int x = blockIdx.x * blockDim.x + threadIdx.x;
// In the end threads must share their maxima and fold them in logarithmically
__shared__ double locBloc[GLOBAL_BLOCKDIM];
__shared__ int locDir[GLOBAL_BLOCKDIM];

// Threadwise: The largest sound speed and it's directional index yet seen; Local comparision direction.
// temporary float values used to evaluate each cell
double CsMax = -1e37; int IndMax, locImax;
double tmpA, tmpB;

// Set all maxima to ~-infinity and index to invalid.
locBloc[threadIdx.x] = -1e37;
locDir[threadIdx.x] = 0;

return; 

// Have thread 0 write such to the globally shared values (overwrite garbage before we possibly get killed next line)
if(threadIdx.x == 0) { dout[blockIdx.x] = -1e37; dirOut[blockIdx.x] = 0; }

if(x >= n) return; // If we get a very low resolution, save time & space on wasted threads

// Jumping through memory,
while(x < n) {
  // Find the maximum |momentum| first; Convert it to velocity and add to soundspeed, then compare with this thread's previous max.
  tmpA = abs(px[x]);
  tmpB = abs(py[x]);

  if(tmpB > tmpA) { tmpA = tmpB; locImax = 2; } else { locImax = 1; }
  tmpB = abs(pz[x]);
  if(tmpB > tmpA) { tmpA = tmpB; locImax = 3; }  

  tmpA = tmpA / rho[x] + cs[x];

  if(tmpA > CsMax) { CsMax = tmpA; IndMax = locImax; }

  // Jump to next address to compare
  x += blockDim.x * gridDim.x;
  }

// Between them threads have surveyed entire array
// Flush threadwise maxima to shared memory
locBloc[threadIdx.x] = CsMax;
locDir[threadIdx.x] = IndMax;

// Now we need the max of the stored shared array to write back to the global array
__syncthreads();

if (threadIdx.x % 8 > 0) return; // keep one in 8 threads

// Each searches the max of the nearest 8 points
for(x = 1; x < 8; x++) {
  if(locBloc[threadIdx.x+x] > locBloc[threadIdx.x]) locBloc[threadIdx.x] = locBloc[threadIdx.x+x];
  }

// The last thread takes the max of these maxes
if(threadIdx.x > 0) return;
for(x = 8; x < GLOBAL_BLOCKDIM; x+= 8) {
  if(locBloc[threadIdx.x+x] > locBloc[0]) locBloc[0] = locBloc[threadIdx.x+x];
  }

// NOTE: This is the dead-stupid backup if all else fails.
//if(threadIdx.x > 0) return;
//for(x = 1; x < GLOBAL_BLOCKDIM; x++)  if(locBloc[x] > locBloc[0]) locBloc[0] = locBloc[x];

dout[blockIdx.x] = locBloc[0];
dirOut[blockIdx.x] = locDir[0];

}

*/

