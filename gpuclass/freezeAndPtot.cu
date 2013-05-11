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

__global__ void cukern_FreezeSpeed_mhd(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *freeze, double *ptot, int nx);
__global__ void cukern_FreezeSpeed_hydro(double *rho, double *E, double *px, double *py, double *pz, double *freeze, double *ptot, int nx);

#define BLOCKDIM 64
#define MAXPOW   5

__device__ __constant__ double gammafunc[5];

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  // At least 2 arguments expected
  // Input and result
  if ( (nrhs!=11) && (nrhs!=2))
     mexErrMsgTxt("Wrong number of arguments. Call using [ptot freeze] = FreezeAndPtot(mass, ener, momx, momy, momz, bz, by, bz, gamma, direct=1, csmin)");

  cudaCheckError("entering freezeAndPtot");

  int ispurehydro = (int)*mxGetPr(prhs[9]);

  int nArrays;
  if(ispurehydro) { nArrays = 5; } else { nArrays = 8; }

  ArrayMetadata amd;
  double **args = getGPUSourcePointers(prhs, &amd, 0, nArrays-1);

  dim3 arraySize;
  arraySize.x = amd.dim[0];
  arraySize.y = amd.dim[1];
  arraySize.z = amd.dim[2];
  dim3 blocksize, gridsize;

  blocksize.x = BLOCKDIM; blocksize.y = blocksize.z = 1;
  gridsize.x = arraySize.y;
  gridsize.y = arraySize.z;

  double **ptot = makeGPUDestinationArrays((int64_t *)mxGetData(prhs[0]), plhs, 1); // ptotal array

  int64_t *oldref = (int64_t *)mxGetData(prhs[0]); 
  int64_t fref[5];
  fref[0] = 0;
  fref[1] = oldref[1] - 1;
  fref[2] = arraySize.y;
  fref[3] = arraySize.z;
  fref[4] = 1;

  double **freezea = makeGPUDestinationArrays(&fref[0], &plhs[1], 1); // freeze array

  double hostgf[5];
    hostgf[0] = *mxGetPr(prhs[8]);
    hostgf[1] = hostgf[0] - 1.0;
    hostgf[2] = hostgf[0]*hostgf[1];
    hostgf[3] = 2.0 - hostgf[0];
    hostgf[4] = (*mxGetPr(prhs[10]))*(*mxGetPr(prhs[10])); // min c_s squared ;
  
  cudaMemcpyToSymbol(gammafunc, &hostgf[0],     5*sizeof(double), 0, cudaMemcpyHostToDevice);


  if(ispurehydro) {
    cukern_FreezeSpeed_hydro<<<gridsize, blocksize>>>(args[0], args[1], args[2], args[3], args[4]     , freezea[0], ptot[0], arraySize.x);
//                                                   (*rho,    *E,      *px,     *py,     *pz,      gam,              *freeze,  *ptot,  nx)
  } else {
    cukern_FreezeSpeed_mhd<<<gridsize, blocksize>>>(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]     , freezea[0], ptot[0], arraySize.x);
//                                                 (*rho,    *E,      *px,     *py,     *pz,     *bx,     *by,     *bz,     gam,              *freeze,  *ptot,  nx)
  }
  free(ptot);
  free(args);
  free(freezea);

cudaError_t epicFail = cudaGetLastError();
if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, ispurehydro, "Getting freezing speed");

}

#define gam gammafunc[0]
#define gm1 gammafunc[1]
#define gg1 gammafunc[2]
#define twomg gammafunc[3]
#define cs0sq gammafunc[4]

__global__ void cukern_FreezeSpeed_mhd(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *freeze, double *ptot, int nx)
{
/* gridDim = [ny nz], nx = nx */
int x = threadIdx.x + nx*(blockIdx.x + gridDim.x*blockIdx.y);
nx += nx*(blockIdx.x + gridDim.x*blockIdx.y);
//int addrMax = nx + nx*(blockIdx.x + gridDim.x*blockIdx.y);

double pressvar;
double T, bsqhf;
double rhoinv;
//double gg1 = gam*(gam-1.0);
//double gm1 = gam - 1.0;
//double twomg = 2.0 - gam;

__shared__ double locBloc[BLOCKDIM];

//CsMax = 0.0;
locBloc[threadIdx.x] = 0.0;

if(x >= nx) return; // If we get a very low resolution

while(x < nx) {
  rhoinv = 1.0/rho[x];
  T = .5*rhoinv*(px[x]*px[x] + py[x]*py[x] + pz[x]*pz[x]);
  bsqhf = .5*(bx[x]*bx[x] + by[x]*by[x] + bz[x]*bz[x]);

// THIS IS THE NEW CODE
//  Cs = gm1*(E[x] - T) + twomg*bsqhf;
  // Calculate pressvar as the internal plus magnetic energy density
  pressvar = E[x] - T - bsqhf;
  if(gam*pressvar*rhoinv < cs0sq) {
    E[x] = T + bsqhf + cs0sq/(gam*rhoinv);
    pressvar = bsqhf + cs0sq/(gam*rhoinv);
    } else { pressvar += bsqhf; }

  // P = gm1(epsilon + bsq/2) + (2 - (gamma-1))*(bsq/2)
  ptot[x] = gm1*pressvar + twomg*bsqhf;

  // We calculate the freezing speed in the X direction: max of |v|+c_fast
  // We double the Alfven contribution
  pressvar = (gg1*pressvar + (8.0 - gg1)*bsqhf)*rhoinv;
  pressvar = sqrt(pressvar) + abs(px[x]*rhoinv);

// THIS IS THE ORIGINAL CODE
  // we calculate P* = Pgas + Pmag
/*
  Cs = gm1*(E[x] - T) + twomg*bsqhf;
  if(gam*PRESSURE*rhoinv < cs0sq) PRESSURE = cs0sq/(gam*rhoinv);
//  if(Cs > 0.0) { ptot[x] = Cs; } else { ptot[x] = 0.0; } // Enforce positive semi-definiteness
  ptot[x] = PRESSURE;

  // We calculate the freezing speed in the X direction: max of |v|+c_fast
  Cs = gg1*(E[x] - T) + (2.0 - gg1)*bsqhf*rhoinv;
  if(Cs < 0.0) Cs = 0.0;
  Cs = sqrt(Cs) + abs(px[x]*rhoinv); */


  if(pressvar > locBloc[threadIdx.x]) locBloc[threadIdx.x] = pressvar;

  x += BLOCKDIM;
  }

__syncthreads();

if(threadIdx.x >= 32) return; // We have only one block left: haha, no more __syncthreads() needed
if(locBloc[threadIdx.x+32] > locBloc[threadIdx.x]) { locBloc[threadIdx.x] = locBloc[threadIdx.x+32]; }

if(threadIdx.x >= 16) return;
if(locBloc[threadIdx.x+16] > locBloc[threadIdx.x]) { locBloc[threadIdx.x] = locBloc[threadIdx.x+16]; }

if(threadIdx.x >= 8) return;
if(locBloc[threadIdx.x+8] > locBloc[threadIdx.x]) {  locBloc[threadIdx.x] = locBloc[threadIdx.x+8];  }

if(threadIdx.x >= 4) return;
if(locBloc[threadIdx.x+4] > locBloc[threadIdx.x]) {  locBloc[threadIdx.x] = locBloc[threadIdx.x+4];  }

if(threadIdx.x >= 2) return;
if(locBloc[threadIdx.x+2] > locBloc[threadIdx.x]) {  locBloc[threadIdx.x] = locBloc[threadIdx.x+2];  }

if(threadIdx.x == 0) {
  if(locBloc[1] > locBloc[0]) {  locBloc[0] = locBloc[1];  }

  freeze[blockIdx.x + gridDim.x*blockIdx.y] = locBloc[0];
  }

/*if (threadIdx.x % 8 > 0) return; // keep one in 8 threads

// Each searches the max of the nearest 8 points
for(x = 1; x < 8; x++) {
  if(locBloc[threadIdx.x+x] > locBloc[threadIdx.x]) locBloc[threadIdx.x] = locBloc[threadIdx.x+x];
  }

__syncthreads();

// The last thread takes the max of these maxes
if(threadIdx.x > 0) return;
for(x = 8; x < BLOCKDIM; x+= 8) {
  if(locBloc[threadIdx.x+x] > locBloc[0]) locBloc[0] = locBloc[threadIdx.x+x];
  }

freeze[blockIdx.x + gridDim.x*blockIdx.y] = locBloc[0]; */
}

#define PRESSURE Cs
// cs0sq = gamma rho^(gamma-1))
__global__ void cukern_FreezeSpeed_hydro(double *rho, double *E, double *px, double *py, double *pz, double *freeze, double *ptot, int nx)
{
int x = threadIdx.x + nx*(blockIdx.x + gridDim.x*blockIdx.y);
int addrMax = nx + nx*(blockIdx.x + gridDim.x*blockIdx.y);

double Cs, CsMax;
double psqhf, rhoinv;
//double gg1 = gam*(gam-1.0);
//double gm1 = gam - 1.0;

__shared__ double locBloc[BLOCKDIM];

CsMax = 0.0;
locBloc[threadIdx.x] = 0.0;

if(x >= addrMax) return; // If we get a very low resolution

while(x < addrMax) {
  rhoinv   = 1.0/rho[x];
  psqhf    = .5*(px[x]*px[x]+py[x]*py[x]+pz[x]*pz[x]);

  PRESSURE = gm1*(E[x] - psqhf*rhoinv);
  if(gam*PRESSURE*rhoinv < cs0sq) {
    PRESSURE = cs0sq/(gam*rhoinv);
    E[x] = psqhf*rhoinv + PRESSURE/gm1;
    } /* Constrain temperature to a minimum value */
  ptot[x] = PRESSURE;

  Cs      = sqrt(gam * PRESSURE *rhoinv) + abs(px[x]*rhoinv);
  if(Cs > CsMax) CsMax = Cs;

  x += BLOCKDIM;
  }

locBloc[threadIdx.x] = CsMax;

__syncthreads();

if (threadIdx.x % 8 > 0) return; // keep threads  [0 8 16 ...]

// Each searches the max of the nearest 8 points
for(x = 1; x < 8; x++) {
  if(locBloc[threadIdx.x+x] > locBloc[threadIdx.x]) locBloc[threadIdx.x] = locBloc[threadIdx.x+x];
  __syncthreads();
  }

// The last thread takes the max of these maxes
if(threadIdx.x > 0) return;
for(x = 8; x < BLOCKDIM; x+= 8) {
  if(locBloc[x] > locBloc[0]) locBloc[0] = locBloc[x];
  }

// NOTE: This is the dead-stupid backup if all else fails.
//if(threadIdx.x > 0) return;
//for(x = 1; x < GLOBAL_BLOCKDIM; x++)  if(locBloc[x] > locBloc[0]) locBloc[0] = locBloc[x];

freeze[blockIdx.x + gridDim.x*blockIdx.y] = locBloc[0];


}
