#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif
#include "mex.h"

// CUDA^M
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas.h"
#include "cudaCommon.h"

/* THIS FUNCTION 

This is the Cuda Fluid TVD function; It takes a single forward-time step, CFD or MHD, of the
conserved-transport part of the fluid equations using a total variation diminishing scheme to
perform a non-oscillatory update.

Requires predicted half-step values from a 1st order upwind scheme.

*/

#define BLOCKLEN 60
#define BLOCKLENP2 62
#define BLOCKLENP4 64

/*__global__ void cukern_TVDStep_mhd_uniform(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *P, double *Cfreeze, double *rhoW, double *enerW, double *pxW, double *pyW, double *pzW, double lambda, int nx);*/
__global__ void cukern_TVDStep_mhd_uniform(double *P, double *Cfreeze, double halflambda, int nx);
/*__global__ void cukern_TVDStep_hydro_uniform(double *rho, double *E, double *px, double *py, double *pz, double *P, double *Cfreeze, double *rhoW, double *enerW, double *pxW, double *pyW, double *pzW, double lambdahf, int nx);*/
__global__ void cukern_TVDStep_hydro_uniform(double *P, double *Cfreeze, double halfLambda, int nx);

__device__ void cukern_FluxLimiter_VanLeer(double deriv[2][BLOCKLENP4], double flux[2][BLOCKLENP4], int who);
__device__ __inline__ double fluxLimiter_Vanleer(double derivL, double derivR);

__constant__ __device__ double *inputPointers[8];
__constant__ __device__ double *outputPointers[5];

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // At least 2 arguments expected
  // Input and result
  if ((nrhs!=17) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: call cudaTVDStep(rho, E, px, py, pz, bx, by, bz, P, rho_out, E_out, px_out, py_out, pz_out, C_freeze, lambda, purehydro?)\n");

  // Get source array info and create destination arrays
  ArrayMetadata amd;

  // Get the source gpu arrays as enumerated in the error message
  double **srcs   = getGPUSourcePointers(prhs, &amd, 0, 13);

  // Get the GPU freeze speed. Differs in that it is not the same size
  ArrayMetadata fmd;
  double **gpu_cf = getGPUSourcePointers(prhs, &fmd, 14, 14); 

  // Get flux factor (dt / d) amd determine if we are doing the hydro case or the MHD case
  double lambda   = *mxGetPr(prhs[15]);
  int isPureHydro = (int)*mxGetPr(prhs[16]);

  // Do the usual rigamarole determining array sizes and GPU launch dimensions
  dim3 arraySize;
  arraySize.x = amd.dim[0];
  arraySize.y = amd.dim[1];
  arraySize.z = amd.dim[2];

  dim3 blocksize, gridsize;

  blocksize.x = BLOCKLEN+4;
  blocksize.y = blocksize.z = 1;

  gridsize.x = arraySize.y;
  gridsize.y = arraySize.z;

  // Invoke the kernel
  if(arraySize.x > 1) {
    if(isPureHydro) {
    //cukern_TVDStep_hydro_uniform                         (*rho,    *E,     *px,      *py,     *pz,     *P,      *Cfreeze, *rhoW,  *enerW,     *pxW,     *pyW,     *pzW,     lambda, nx);
      cudaMemcpyToSymbol(inputPointers,  srcs,     5*sizeof(double *), 0, cudaMemcpyHostToDevice);
      cudaMemcpyToSymbol(outputPointers, &srcs[9], 5*sizeof(double *), 0, cudaMemcpyHostToDevice);
      cukern_TVDStep_hydro_uniform<<<gridsize, blocksize>>>(srcs[8], *gpu_cf, .5*lambda, arraySize.x);
    } else {
    //cukern_TVDStep_mhd_uniform                         (*rho,    *E,      *px,     *py,     *pz,     *bx,     *by,     *bz,     *P,           *Cfreeze, *rhoW,  *enerW,   *pxW,     *pyW,     *pzW, double lambda, int nx);
      cudaMemcpyToSymbol(inputPointers,  srcs,     8*sizeof(double *), 0, cudaMemcpyHostToDevice);
      cudaMemcpyToSymbol(outputPointers, &srcs[9], 5*sizeof(double *), 0, cudaMemcpyHostToDevice);      
      cukern_TVDStep_mhd_uniform  <<<gridsize, blocksize>>>(srcs[8], *gpu_cf, .5*lambda, arraySize.x);
    }
  }

cudaError_t epicFail = cudaGetLastError();
if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, isPureHydro, "fluid TVD");

}

/* blockidx.{xy} is our index in {yz}, and gridDim.{xy} gives the {yz} size */
/* Expect invocation with n+4 threads */
__global__ void cukern_TVDStep_mhd_uniform(double *P, double *Cfreeze, double halfLambda, int nx)
{
double c_f, velocity;
double q_i[5];
double b_i[3];
double w_i;
__shared__ double fluxLR[2][BLOCKLENP4];
__shared__ double derivLR[2][BLOCKLENP4];

/* Step 0 - obligatory annoying setup stuff (ASS) */
int I0 = nx*(blockIdx.x + gridDim.x * blockIdx.y);
int Xindex = (threadIdx.x-2);
int Xtrack = Xindex;
Xindex += nx*(threadIdx.x < 2);

int x; /* = Xindex % nx; */
int i;
bool doIflux = (threadIdx.x > 1) && (threadIdx.x < BLOCKLEN+2);

/* Step 1 - calculate W values */
c_f = Cfreeze[blockIdx.x + gridDim.x * blockIdx.y];

while(Xtrack < nx+2) {
    x = I0 + (Xindex % nx);

    q_i[0] = inputPointers[0][x];
    q_i[1] = inputPointers[1][x];       /* So we avoid multiple loops */
    q_i[2] = inputPointers[2][x];      /* over them inside the flux loop */
    q_i[3] = inputPointers[3][x];
    q_i[4] = inputPointers[4][x];
    b_i[0] = inputPointers[5][x];
    b_i[1] = inputPointers[6][x];
    b_i[2] = inputPointers[7][x];
    velocity = q_i[2]/q_i[0];

    /* rho, E, px, py, pz going down */
    /* Iterate over variables to flux */
    for(i = 0; i < 5; i++) {
        /* Step 1 - Calculate raw fluxes */
        switch(i) {
            case 0: w_i = q_i[2]; break;
            case 1: w_i = (velocity * (q_i[1] + P[x]) - b_i[0]*(q_i[2]*b_i[0]+q_i[3]*b_i[1]+q_i[4]*b_i[2])/q_i[0] ); break;
            case 2: w_i = (velocity*q_i[2] + P[x] - b_i[0]*b_i[0]); break;
            case 3: w_i = (velocity*q_i[3]        - b_i[0]*b_i[1]); break;
            case 4: w_i = (velocity*q_i[4]        - b_i[0]*b_i[2]); break;
            }

        /* Step 2 - Decouple to L/R flux */
        fluxLR[0][threadIdx.x] = (q_i[i]*c_f - w_i); /* Left  going flux */
        fluxLR[1][threadIdx.x] = (q_i[i]*c_f + w_i); /* Right going flux */
        __syncthreads();

        /* Step 3 - Differentiate fluxes & call limiter */
            /* left flux */
        derivLR[0][threadIdx.x] = fluxLR[0][(threadIdx.x-1)%BLOCKLENP4] - fluxLR[0][threadIdx.x]; /* left derivative */
        derivLR[1][threadIdx.x] = fluxLR[0][threadIdx.x] - fluxLR[0][(threadIdx.x+1)%BLOCKLENP4]; /* right derivative */
        __syncthreads();
        fluxLR[0][threadIdx.x] += fluxLimiter_Vanleer(derivLR[0][threadIdx.x], derivLR[1][threadIdx.x]);
//        cukern_FluxLimiter_VanLeer(derivLR, fluxLR, 0);
        __syncthreads();

            /* Right flux */
        derivLR[0][threadIdx.x] = fluxLR[1][threadIdx.x] - fluxLR[1][(threadIdx.x-1)%BLOCKLENP4]; /* left derivative */
        derivLR[1][threadIdx.x] = fluxLR[1][(threadIdx.x+1)%BLOCKLENP4] - fluxLR[1][threadIdx.x]; /* right derivative */
        __syncthreads();
        fluxLR[1][threadIdx.x] += fluxLimiter_Vanleer(derivLR[0][threadIdx.x], derivLR[1][threadIdx.x]);
  //      cukern_FluxLimiter_VanLeer(derivLR, fluxLR, 1); 
        __syncthreads();

        /* Step 4 - Perform flux and write to output array */
       if( doIflux && (Xindex < nx) ) {
            outputPointers[i][x] -= halfLambda * ( fluxLR[0][threadIdx.x] - fluxLR[0][threadIdx.x+1] + \
                                                   fluxLR[1][threadIdx.x] - fluxLR[1][threadIdx.x-1]  ); 
          }

        __syncthreads();
        }

    Xindex += BLOCKLEN;
    Xtrack += BLOCKLEN;
    }

}

__global__ void cukern_TVDStep_hydro_uniform(double *P, double *Cfreeze, double halfLambda, int nx)
{
double C_f, velocity;
double q_i[5];
double w_i;
__shared__ double fluxLR[2][BLOCKLENP4];
__shared__ double derivLR[2][BLOCKLENP4];

/* Step 0 - obligatory annoying setup stuff (ASS) */
int I0 = nx*(blockIdx.x + gridDim.x * blockIdx.y);
int Xindex = (threadIdx.x-2);
int Xtrack = Xindex;
Xindex += nx*(threadIdx.x < 2);

int x; /* = Xindex % nx; */
int i;
bool doIflux = (threadIdx.x > 1) && (threadIdx.x < BLOCKLEN+2);

/* Step 1 - calculate W values */
C_f = Cfreeze[blockIdx.x + gridDim.x * blockIdx.y];

while(Xtrack < nx+2) {
    x = I0 + (Xindex % nx);

    q_i[0] = inputPointers[0][x]; /* Preload these out here */
    q_i[1] = inputPointers[1][x]; /* So we avoid multiple loops */
    q_i[2] = inputPointers[2][x]; /* over them inside the flux loop */
    q_i[3] = inputPointers[3][x];
    q_i[4] = inputPointers[4][x];
    velocity = q_i[2] / q_i[0];

    /* rho, E, px, py, pz going down */
    /* Iterate over variables to flux */
    for(i = 0; i < 5; i++) {
        /* Step 1 - Calculate raw fluxes */
        switch(i) {
            case 0: w_i = q_i[2]; break;
            case 1: w_i = (velocity * (q_i[1] + P[x]) ) ; break;
            case 2: w_i = (velocity * q_i[2] + P[x]); break;
            case 3: w_i = (velocity * q_i[3]); break;
            case 4: w_i = (velocity * q_i[4]); break;
            }

        /* Step 2 - Decouple to L/R flux */
/* NOTE there is a missing .5 here, accounted for in the h(al)f of lambdahf */
        fluxLR[0][threadIdx.x] = (C_f*q_i[i] - w_i); /* Left  going flux */
        fluxLR[1][threadIdx.x] = (C_f*q_i[i] + w_i); /* Right going flux */
        __syncthreads();

        /* Step 3 - Differentiate fluxes & call limiter */
            /* left flux */
        derivLR[0][threadIdx.x] = fluxLR[0][(threadIdx.x-1)%BLOCKLENP4] - fluxLR[0][threadIdx.x]; /* left derivative */
        derivLR[1][threadIdx.x] = fluxLR[0][threadIdx.x] - fluxLR[0][(threadIdx.x+1)%BLOCKLENP4]; /* right derivative */
        __syncthreads();
        fluxLR[0][threadIdx.x] += fluxLimiter_Vanleer(derivLR[0][threadIdx.x], derivLR[1][threadIdx.x]);
/*        cukern_FluxLimiter_VanLeer(derivLR, fluxLR, 0);*/
        __syncthreads();

            /* Right flux */
        derivLR[0][threadIdx.x] = fluxLR[1][threadIdx.x] - fluxLR[1][(threadIdx.x-1)%BLOCKLENP4]; /* left derivative */
        derivLR[1][threadIdx.x] = fluxLR[1][(threadIdx.x+1)%BLOCKLENP4] - fluxLR[1][threadIdx.x]; /* right derivative */
        __syncthreads();
        fluxLR[1][threadIdx.x] += fluxLimiter_Vanleer(derivLR[0][threadIdx.x], derivLR[1][threadIdx.x]);
/*        cukern_FluxLimiter_VanLeer(derivLR, fluxLR, 1);*/
        __syncthreads();

        /* Step 4 - Perform flux and write to output array */
       if( doIflux && (Xindex < nx) ) {
            outputPointers[i][x] -= halfLambda * ( fluxLR[0][threadIdx.x] - fluxLR[0][threadIdx.x+1] + \
                                                   fluxLR[1][threadIdx.x] - fluxLR[1][threadIdx.x-1]  );
            }

        __syncthreads();
        }

    Xindex += BLOCKLEN;
    Xtrack += BLOCKLEN;
    }

}


__device__ void cukern_FluxLimiter_VanLeer(double deriv[2][BLOCKLENP4], double flux[2][BLOCKLENP4], int who)
{

double r;

r = deriv[0][threadIdx.x] * deriv[1][threadIdx.x];
if(r < 0.0) r = 0.0;

r = r / ( deriv[0][threadIdx.x] + deriv[1][threadIdx.x]);
if (isnan(r)) { r = 0.0; }

flux[who][threadIdx.x] += r;

}

__device__ double fluxLimiter_Vanleer(double derivL, double derivR)
{
double r;

r = derivL * derivR;
if(r < 0.0) r = 0.0;

r = r / ( derivL + derivR);
if (isnan(r)) { r = 0.0; }

return r;
}


