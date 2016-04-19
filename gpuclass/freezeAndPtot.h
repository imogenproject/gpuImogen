#ifndef CUDA_FREEZEPTOTH_
#define CUDA_FREEZEPTOTH_

__global__ void cukern_FreezeSpeed_mhd(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *freeze, double *ptot, int nx);
__global__ void cukern_FreezeSpeed_hydro(double *rho, double *E, double *px, double *py, double *pz, double *freeze, double *ptot, int nx);

__device__ __constant__ double gammafunc[6];

#endif
