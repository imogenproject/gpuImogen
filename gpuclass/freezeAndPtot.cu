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

/* THIS FUNCTION:
   Calculates the maximum in the x direction of the freezing speed c_f, defined
   as the fastest characteristic velocity in the x direction.

   In the hydrodynamic case this is the adiabatic sounds speed, in the MHD case
   this is the fast magnetosonic speed.

   */

__global__ void cukern_FreezeSpeed_mhd(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *freeze, double *ptot, int nx);
__global__ void cukern_FreezeSpeed_hydro(double *rho, double *E, double *px, double *py, double *pz, double *freeze, double *ptot, int nx);

#define BLOCKDIM 64
#define MAXPOW   5

__device__ __constant__ double gammafunc[6];

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  // At least 2 arguments expected
  // Input and result
  if ( (nrhs!=11) && (nrhs!=2))
     mexErrMsgTxt("Wrong number of arguments. Call using [ptot freeze] = FreezeAndPtot(mass, ener, momx, momy, momz, bz, by, bz, gamma, direct=1, csmin)");

  CHECK_CUDA_ERROR("entering freezeAndPtot");

  int ispurehydro = (int)*mxGetPr(prhs[9]);

  int nArrays;
  if(ispurehydro) { nArrays = 5; } else { nArrays = 8; }

  MGArray fluid[8];
  accessMGArrays(prhs, 0, nArrays-1, fluid);

  dim3 arraySize;
  arraySize.x = fluid->dim[0];
  arraySize.y = fluid->dim[1];
  arraySize.z = fluid->dim[2];
  dim3 blocksize, gridsize;

  blocksize.x = BLOCKDIM; blocksize.y = blocksize.z = 1;
  gridsize.x = arraySize.y;
  gridsize.y = arraySize.z;

  MGArray clone;
  MGArray *POut;
  MGArray *cfOut;

  clone = fluid[0];
  clone.dim[0] = arraySize.y;
  clone.dim[1] = arraySize.z;
  clone.dim[2] = 1;
  POut = createMGArrays(plhs, 1, fluid); 
  cfOut= createMGArrays(plhs+1, 1, &clone);

  double hostgf[6];
  double gam = *mxGetPr(prhs[8]);
    hostgf[0] = gam;
    hostgf[1] = gam - 1.0;
    hostgf[2] = gam*(gam-1.0);
    hostgf[3] = (1.0 - .5*gam);
    hostgf[4] = (*mxGetPr(prhs[10]))*(*mxGetPr(prhs[10])); // min c_s squared ;
    hostgf[5] = (ALFVEN_CSQ_FACTOR - .5*gam*(gam-1.0));
  
  cudaMemcpyToSymbol(gammafunc, &hostgf[0],     6*sizeof(double), 0, cudaMemcpyHostToDevice);
  CHECK_CUDA_ERROR("cfreeze symbol upload");

  int i;
  int sub[6];

  if(ispurehydro) {
    for(i = 0; i < fluid->nGPUs; i++) {
      cudaSetDevice(fluid->deviceID[i]);
      CHECK_CUDA_ERROR("cudaSetDevice()");
      calcPartitionExtent(fluid, i, sub);
      cukern_FreezeSpeed_hydro<<<gridsize, blocksize>>>(
		fluid[0].devicePtr[i],
		fluid[1].devicePtr[i],
		fluid[2].devicePtr[i],
		fluid[3].devicePtr[i],
		fluid[4].devicePtr[i],
		cfOut->devicePtr[i], POut->devicePtr[i], arraySize.x);
      CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "Freeze speed hydro");
      
      }

//  (*rho,    *E,      *px,     *py,     *pz,      gam,              *freeze,  *ptot,  nx)
  } else {
    for(i = 0; i < fluid->nGPUs; i++) {
      cudaSetDevice(fluid->deviceID[i]);
      calcPartitionExtent(fluid, i, sub);
      cukern_FreezeSpeed_mhd<<<gridsize, blocksize>>>(
                fluid[0].devicePtr[i],
                fluid[1].devicePtr[i],
                fluid[2].devicePtr[i],
                fluid[3].devicePtr[i],
                fluid[4].devicePtr[i],
                fluid[5].devicePtr[i],
                fluid[6].devicePtr[i],
                fluid[7].devicePtr[i],
                cfOut->devicePtr[i], POut->devicePtr[i], arraySize.x);
  CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "freeze speed MHD");
//     (*rho, *E,   *px,  *py,  *pz,  *bx,  *by,  *bz,  gam,     *freeze,  *ptot,  nx)
    }
  }

  free(POut);
  free(cfOut);

}

#define gam gammafunc[0]
#define gm1 gammafunc[1]
#define gg1 gammafunc[2]
#define MHD_PRESS_B gammafunc[3]
#define cs0sq gammafunc[4]
#define MHD_CS_B gammafunc[5]

__global__ void cukern_FreezeSpeed_mhd(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *freeze, double *ptot, int nx)
{
/* gridDim = [ny nz], nx = nx */
int x = threadIdx.x + nx*(blockIdx.x + gridDim.x*blockIdx.y);
nx += nx*(blockIdx.x + gridDim.x*blockIdx.y);
//int addrMax = nx + nx*(blockIdx.x + gridDim.x*blockIdx.y);

double pressvar;
double T, bsquared;
double rhoinv;
__shared__ double locBloc[BLOCKDIM];

//CsMax = 0.0;
locBloc[threadIdx.x] = 0.0;

if(x >= nx) return; // If we get a very low resolution

while(x < nx) {
  rhoinv = 1.0/rho[x];
  T = .5*rhoinv*(px[x]*px[x] + py[x]*py[x] + pz[x]*pz[x]);
  bsquared = bx[x]*bx[x] + by[x]*by[x] + bz[x]*bz[x];

  // Calculate internal + magnetic energy
  pressvar = E[x] - T;

  // Assert minimum thermal soundspeed / temperature
/*  if(gam*pressvar*rhoinv < cs0sq) {
    E[x] = T + bsqhf + cs0sq/(gam*rhoinv);
    pressvar = cs0sq/(gam*rhoinv);
    } */

  // Calculate gas + magnetic pressure
  ptot[x] = gm1*pressvar + MHD_PRESS_B*bsquared;

  // We calculate the freezing speed in the X direction: max of |v|+c_fast
  // MHD_CS_B includes an "alfven factor" to stabilize the code in low-beta situations
  pressvar = (gg1*pressvar + MHD_CS_B*bsquared)*rhoinv;
  pressvar = sqrt(abs(pressvar)) + abs(px[x]*rhoinv);

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
