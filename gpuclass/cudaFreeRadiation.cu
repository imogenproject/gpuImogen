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

__global__ void cukern_FreeHydroRadiationRate(double *rho, double *px, double *py, double *pz, double *E, double *radrate, int numel);
__global__ void cukern_FreeMHDRadiationRate(double *rho, double *px, double *py, double *pz, double *E, double *bx, double *by, double *bz, double *radrate, int numel);

__global__ void cukern_FreeHydroRadiation(double *rho, double *px, double *py, double *pz, double *E, int numel);
__global__ void cukern_FreeMHDRadiation(double *rho, double *px, double *py, double *pz, double *E, double *bx, double *by, double *bz, int numel);

__constant__ __device__ double radparam[5];
#define GAMMA_M1 radparam[0]
#define STRENGTH radparam[1]
#define EXPONENT radparam[2]
#define TWO_MEXPONENT radparam[3]
#define PFLOOR radparam[4]


#define BLOCKDIM 256
#define GRIDDIM 64

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if ((nrhs != 9) || (nlhs > 1))
     mexErrMsgTxt("Wrong number of arguments. Expected forms: rate = cudaFreeRadiation(rho, px, py, pz, E, bx, by, bz, [gamma theta 0 isPureHydro]) or cudaFreeRadiation(rho, px, py, pz, E, bx, by , bz, [gamma theta beta*dt isPureHydro]\n");

  double gam       = (mxGetPr(prhs[8]))[0];
  double exponent  = (mxGetPr(prhs[8]))[1];
  double strength  = (mxGetPr(prhs[8]))[2];
  int isHydro   = ((int)(mxGetPr(prhs[8]))[3]) != 0;

  ArrayMetadata amd;
  double **arrays = getGPUSourcePointers(prhs, &amd, 0, 4);
  cudaCheckError("Entering cudaFreeRadiation");

  double **dest = NULL;
  if(nlhs == 1) {
    dest = makeGPUDestinationArrays(&amd, plhs, 1);
    }

  double hostRP[5];
  hostRP[0] = gam-1.0;
  hostRP[1] = strength;
  hostRP[2] = exponent;
  hostRP[3] = 2.0 - exponent;
  hostRP[4] = 1.0; // pressure floor = 1.0
  cudaMemcpyToSymbol(radparam, hostRP, 5*sizeof(double), 0, cudaMemcpyHostToDevice);

  switch(isHydro + 2*nlhs) {
    case 0: {
      double **B = getGPUSourcePointers(prhs, &amd, 5, 7);
      cukern_FreeMHDRadiation<<<GRIDDIM, BLOCKDIM>>>(arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], B[0], B[1], B[2], amd.numel);
      break; }
    case 1: {
      cukern_FreeHydroRadiation<<<GRIDDIM, BLOCKDIM>>>(arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], amd.numel);
      break; }
    case 2: {
      double **B = getGPUSourcePointers(prhs, &amd, 5, 7);
      cukern_FreeMHDRadiationRate<<<GRIDDIM, BLOCKDIM>>>(arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], B[0], B[1], B[2], dest[0], amd.numel);
      break; }
    case 3: {
      cukern_FreeHydroRadiationRate<<<GRIDDIM, BLOCKDIM>>>(arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], dest[0], amd.numel);
      break; }
    }

cudaError_t epicFail = cudaGetLastError();
if(epicFail != cudaSuccess) cudaLaunchError(epicFail, BLOCKDIM, GRIDDIM, &amd, 666, "cudaFreeGasRadiation");

}

__global__ void cukern_FreeHydroRadiation(double *rho, double *px, double *py, double *pz, double *E, int numel)
{
int x = threadIdx.x + BLOCKDIM*blockIdx.x;

double P; double dE;

while(x < numel) {
  P = GAMMA_M1*(E[x] - (px[x]*px[x]+py[x]*py[x]+pz[x]*pz[x])/(2*rho[x])); // gas pressure
  dE = STRENGTH*pow(rho[x], TWO_MEXPONENT)*pow(P, EXPONENT);
  if(P - (GAMMA_M1*dE) < PFLOOR) { E[x] -= (P-PFLOOR)/GAMMA_M1; } else { E[x] -= dE; }

  x += BLOCKDIM*GRIDDIM;
  }

}

__global__ void cukern_FreeMHDRadiation(double *rho, double *px, double *py, double *pz, double *E, double *bx, double *by, double *bz, int numel)
{
int x = threadIdx.x + BLOCKDIM*blockIdx.x;

double P;
double dE;
while(x < numel) {
  P = GAMMA_M1*(E[x] - (  (px[x]*px[x]+py[x]*py[x]+pz[x]*pz[x])/rho[x] +\
                           (bx[x]*bx[x]+by[x]*by[x]+bz[x]*bz[x]))/2.0); // gas pressure
  dE = STRENGTH*pow(rho[x], TWO_MEXPONENT)*pow(P, EXPONENT);
  if(P - (GAMMA_M1 * dE) < PFLOOR) { E[x] -= (P-PFLOOR)/GAMMA_M1; } else { E[x] -= dE; }

  x += BLOCKDIM*GRIDDIM;
  }

}

__global__ void cukern_FreeHydroRadiationRate(double *rho, double *px, double *py, double *pz, double *E, double *radrate, int numel)
{
int x = threadIdx.x + BLOCKDIM*blockIdx.x;

double P;
while(x < numel) {
  P = GAMMA_M1*(E[x] - (px[x]*px[x]+py[x]*py[x]+pz[x]*pz[x])/(2*rho[x])); // gas pressure
  radrate[x] = pow(rho[x], TWO_MEXPONENT)*pow(P, EXPONENT);

  x += BLOCKDIM*GRIDDIM;
  }

}

__global__ void cukern_FreeMHDRadiationRate(double *rho, double *px, double *py, double *pz, double *E, double *bx, double *by, double *bz, double *radrate, int numel)
{
int x = threadIdx.x + BLOCKDIM*blockIdx.x;

double P;
while(x < numel) {
  P = GAMMA_M1*(E[x] - (  (px[x]*px[x]+py[x]*py[x]+pz[x]*pz[x])/rho[x] +\
                           (bx[x]*bx[x]+by[x]+by[x]+bz[x]+bz[x]))/2.0); // gas pressure
  radrate[x] = pow(rho[x], TWO_MEXPONENT)*pow(P, EXPONENT);

  x += BLOCKDIM*GRIDDIM;
  }

}


//            gasPressure = pressure(ENUM.PRESSURE_GAS, run, mass, mom, ener, mag);
//            result      = obj.strength*mass.array.^(2 - obj.exponent) .* gasPressure.^obj.exponent;

