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

#include "cudaCommon.h" // This defines the getGPUSourcePointers and makeGPUDestinationArrays utility functions

/* THIS FUNCTION

This function calculates a single half-step of the conserved transport part of the fluid equations
(CFD or MHD) which is used as the predictor input to the matching TVD function.

*/

__global__ void cukern_Wstep_mhd_uniform  (double *P, double *Cfreeze, double lambdaqtr, int nx);
__global__ void cukern_Wstep_hydro_uniform(double *P, double *Cfreeze, double lambdaqtr, int nx);

#define BLOCKLEN 60
#define BLOCKLENP2 62
#define BLOCKLENP4 64

__constant__ __device__ double *inputPointers[8];
__constant__ __device__ double *outputPointers[5];
__constant__ __device__ double fluidQtys[5];
#define FLUID_GAMMA   fluidQtys[0]
#define FLUID_GM1     fluidQtys[1]
#define FLUID_GG1     fluidQtys[2]
#define FLUID_MINMASS fluidQtys[3]
#define FLUID_MINEINT fluidQtys[4]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Input and result
  if ((nrhs!=13) || (nlhs != 5)) mexErrMsgTxt("Wrong number of arguments: need [5] = cudaWflux(rho, E, px, py, pz, bx, by, bz, Ptot, c_f, lambda, purehydro?, fluid gamma)\n");

  cudaCheckError("entering cudaFluidW");

  ArrayMetadata amd;
  double **srcs = getGPUSourcePointers(prhs, &amd, 0, 9);
  double **dest = makeGPUDestinationArrays((int64_t *)mxGetData(prhs[0]),  plhs, 5);

  // Establish launch dimensions & a few other parameters
  int fluxDirection = 1;
  double lambda     = *mxGetPr(prhs[10]);

  dim3 arraySize;
  arraySize.x = amd.dim[0];
  arraySize.y = amd.dim[1];
  arraySize.z = amd.dim[2];

  dim3 blocksize, gridsize;

  int nu;
  // This bit is actually redundant now since arrays are always rotated so the fluid step is finite-differenced in the X direction
  blocksize.x = BLOCKLEN+4; blocksize.y = blocksize.z = 1;
  switch(fluxDirection) {
    case 1: // X direction flux: u = x, v = y, w = z;
      gridsize.x = arraySize.y;
      gridsize.y = arraySize.z;
      nu = gridsize.x;
      break;
    case 2: // Y direction flux: u = y, v = x, w = z
      gridsize.x = arraySize.x;
      gridsize.y = arraySize.z;
      nu = gridsize.y;
      break;
    case 3: // Z direction flux: u = z, v = x, w = y;
      gridsize.x = arraySize.x;
      gridsize.y = arraySize.y;
      nu = gridsize.z;
      break;
    }
  double *thermo = mxGetPr(prhs[12]);
  double gamma = thermo[0];
  double rhomin= thermo[1];
  double gamHost[5];
  gamHost[0] = gamma;
  gamHost[1] = gamma-1.0;
  gamHost[2] = gamma*(gamma-1.0);
  gamHost[3] = rhomin;
// assert     cs > cs_min
//     g P / rho > g rho_min^(g-1)
// (g-1) e / rho > rho_min^(g-1)
//             e > rho rho_min^(g-1)/(g-1)
  gamHost[4] = powl(rhomin, gamma-1.0)/(gamma-1.0);
// Even for gamma=5/3, soundspeed is very weakly dependent on density (cube root)

  cudaMemcpyToSymbol(fluidQtys, &gamHost[0], 5*sizeof(double), 0, cudaMemcpyHostToDevice);

// It appears this is only used in the null step. It was used in a previous W step but that kernel was irreperably broken.

// If the dimension has finite extent, performs actual step; If not, blits input arrays to output arrays
// NOTE: this situation should not occur, since the flux routine itself skips singleton dimensions for 1- and 2-d sims.

int hydroOnly;
hydroOnly = (int)*mxGetPr(prhs[11]);
  
if(hydroOnly == 1) {
  cudaMemcpyToSymbol(inputPointers,  srcs, 5*sizeof(double *), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(outputPointers, dest, 5*sizeof(double *), 0, cudaMemcpyHostToDevice);
  cukern_Wstep_hydro_uniform<<<gridsize, blocksize>>>(srcs[8], srcs[9], .25*lambda, arraySize.x);
  } else {
  cudaMemcpyToSymbol(inputPointers,  srcs, 8*sizeof(double *), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(outputPointers, dest, 5*sizeof(double *), 0, cudaMemcpyHostToDevice);
  cukern_Wstep_mhd_uniform<<<gridsize, blocksize>>>(srcs[8], srcs[9], lambda/4.0, arraySize.x);
}

cudaError_t epicFail = cudaGetLastError();
if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, hydroOnly, "fluid W step");

}

__global__ void cukern_Wstep_mhd_uniform(double *P, double *Cfreeze, double lambdaqtr, int nx)
{
double C_f, velocity;
double q_i[5];
double b_i[3];
double w_i;
__shared__ double fluxLR[2][BLOCKLENP4];

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

    q_i[0] = inputPointers[0][x];
    q_i[1] = inputPointers[1][x];       /* So we avoid multiple loops */
    q_i[2] = inputPointers[2][x];      /* over them inside the flux loop */
    q_i[3] = inputPointers[3][x];
    q_i[4] = inputPointers[4][x];
    b_i[0] = inputPointers[5][x];
    b_i[1] = inputPointers[6][x];
    b_i[2] = inputPointers[7][x];
    velocity = q_i[2] / q_i[0];

    /* rho, E, px, py, pz going down */
    /* Iterate over variables to flux */
    for(i = 0; i < 5; i++) {
        switch(i) {
            case 0: w_i = q_i[2]; break;
            case 1: w_i = (velocity * (q_i[1] + P[x]) - b_i[0]*(q_i[2]*b_i[0]+q_i[3]*b_i[1]+q_i[4]*b_i[2])/q_i[0] ) ; break;
            case 2: w_i = (velocity * q_i[2] + P[x] - b_i[0]*b_i[0]); break;
            case 3: w_i = (velocity * q_i[3]        - b_i[0]*b_i[1]); break;
            case 4: w_i = (velocity * q_i[4]        - b_i[0]*b_i[2]); break;
            }

        /* Step 2 - decouple to L/R flux */
        fluxLR[0][threadIdx.x] = (C_f*q_i[i] - w_i); /* Left  going flux */
        fluxLR[1][threadIdx.x] = (C_f*q_i[i] + w_i); /* Right going flux */
        // NOTE: a 0.5 is eliminated here. THis requires lambda to be rescaled by .5 in launch.
        /* Step 4 - Perform flux and write to output array */
        __syncthreads();

       if( doIflux && (Xindex < nx) ) {
            // NOTE: a .5 is missing here also, so lambda must ultimately be divided by 4.
            outputPointers[i][x] = q_i[i] - lambdaqtr * ( fluxLR[0][threadIdx.x] - fluxLR[0][threadIdx.x+1] + \
                                                          fluxLR[1][threadIdx.x] - fluxLR[1][threadIdx.x-1]  ); 

            }

        __syncthreads();
        }

    Xindex += BLOCKLEN;
    Xtrack += BLOCKLEN;
    __syncthreads();
    }

}

#define FLUXLa_OFFSET 0
#define FLUXLb_OFFSET (BLOCKLENP4)
#define FLUXRa_OFFSET (2*(BLOCKLENP4))
#define FLUXRb_OFFSET (3*(BLOCKLEN+4))
__global__ void cukern_Wstep_hydro_uniform(double *P, double *Cfreeze, double lambdaqtr, int nx)
{
double C_f, velocity;
double q_i[3];
double w_i;
double velocity_half;
__shared__ double fluxArray[4*(BLOCKLENP4)];
__shared__ double freezeSpeed[BLOCKLENP4];
/*
__shared__ double fluxL_a[BLOCKLENP4];
__shared__ double fluxL_b[BLOCKLENP4];
__shared__ double fluxR_a[BLOCKLENP4];
__shared__ double fluxR_b[BLOCKLENP4];*/

freezeSpeed[threadIdx.x] = 0;

/* Step 0 - obligatory annoying setup stuff (ASS) */
int I0 = nx*(blockIdx.x + gridDim.x * blockIdx.y);
int Xindex = (threadIdx.x-2);
int Xtrack = Xindex;
Xindex += nx*(threadIdx.x < 2);

int x; /* = Xindex % nx; */
bool doIflux = (threadIdx.x > 1) && (threadIdx.x < BLOCKLEN+2);

/* Step 1 - calculate W values */
C_f = Cfreeze[blockIdx.x + gridDim.x * blockIdx.y];
double locPsq;
double locE;

while(Xtrack < nx+2) {
    x = I0 + (Xindex % nx);

/*rho    q_i[0] = inputPointers[0][x];  Preload these out here 
E    q_i[1] = inputPointers[1][x];  So we avoid multiple loops 
px    q_i[2] = inputPointers[2][x];  over them inside the flux loop 
 py   q_i[3] = inputPointers[3][x];
  pz  q_i[4] = inputPointers[4][x];*/
    q_i[0] = inputPointers[0][x];
    q_i[1] = inputPointers[2][x];
    q_i[2] = inputPointers[1][x];
    locPsq   = P[x];

    velocity = q_i[1] / q_i[0];

    #define FLUXA_DECOUPLE(i) fluxArray[FLUXLa_OFFSET+threadIdx.x] = q_i[i]*C_f - w_i; fluxArray[FLUXRa_OFFSET+threadIdx.x] = q_i[i]*C_f + w_i;
    #define FLUXB_DECOUPLE(i) fluxArray[FLUXLb_OFFSET+threadIdx.x] = q_i[i]*C_f - w_i; fluxArray[FLUXRb_OFFSET+threadIdx.x] = q_i[i]*C_f + w_i;
/*    #define FLUXA_DECOUPLE(i) fluxL_a[threadIdx.x] = q_i[i]*C_f - w_i; fluxR_a[threadIdx.x] = q_i[i]*C_f + w_i;
    #define FLUXB_DECOUPLE(i) fluxL_b[threadIdx.x] = q_i[i]*C_f - w_i; fluxR_b[threadIdx.x] = q_i[i]*C_f + w_i;*/

    #define FLUXA_DELTA lambdaqtr*(fluxArray[FLUXLa_OFFSET+threadIdx.x] - fluxArray[FLUXLa_OFFSET+threadIdx.x+1] + fluxArray[FLUXRa_OFFSET+threadIdx.x] - fluxArray[FLUXRa_OFFSET+threadIdx.x-1])
    #define FLUXB_DELTA lambdaqtr*(fluxArray[FLUXLb_OFFSET+threadIdx.x] - fluxArray[FLUXLb_OFFSET+threadIdx.x+1] + fluxArray[FLUXRb_OFFSET+threadIdx.x] - fluxArray[FLUXRb_OFFSET+threadIdx.x-1])
/*    #define FLUXA_DELTA lambdaqtr*(fluxL_a[threadIdx.x] - fluxL_a[threadIdx.x+1] + fluxR_a[threadIdx.x] - fluxR_a[threadIdx.x-1])
    #define FLUXB_DELTA lambdaqtr*(fluxL_b[threadIdx.x] - fluxL_b[threadIdx.x+1] + fluxR_b[threadIdx.x] - fluxR_b[threadIdx.x-1])*/

    w_i = velocity*(q_i[2]+locPsq); /* E flux = v*(E+P) */
    FLUXA_DECOUPLE(2)
    w_i = (velocity*q_i[1] + locPsq); /* px flux = v*px + P */
    FLUXB_DECOUPLE(1)
    __syncthreads();
    if(doIflux && (Xindex < nx)) {
        locE = q_i[2] - FLUXA_DELTA; /* Calculate Ehalf */
        velocity_half = locPsq = q_i[1] - FLUXB_DELTA; /* Calculate Pxhalf */
        outputPointers[2][x] = locPsq; /* store pxhalf */
        }
    __syncthreads();

    locPsq *= locPsq; /* store p^2 in locPsq */

    q_i[0] = inputPointers[3][x];
    q_i[2] = inputPointers[4][x];
    w_i = velocity*q_i[0]; /* py flux = v*py */
    FLUXA_DECOUPLE(0)
    w_i = velocity*q_i[2]; /* pz flux = v pz */
    FLUXB_DECOUPLE(2)
    __syncthreads();
    if(doIflux && (Xindex < nx)) {
        q_i[0] -= FLUXA_DELTA;
        locPsq += q_i[0]*q_i[0];
        outputPointers[3][x] = q_i[0];
        q_i[2] -= FLUXB_DELTA;
        locPsq += q_i[2]*q_i[2]; /* Finished accumulating p^2 */
        outputPointers[4][x] = q_i[2];
        }
    __syncthreads();

    q_i[0] = inputPointers[0][x];
    w_i = q_i[1]; /* rho flux = px */
    FLUXA_DECOUPLE(0)
    __syncthreads();
    if(doIflux && (Xindex < nx)) {
        q_i[0] -= FLUXA_DELTA; /* Calculate rho_half */
//      outputPointers[0][x] = q_i[0];
        q_i[0] = (q_i[0] < FLUID_MINMASS) ? FLUID_MINMASS : q_i[0]; /* Enforce minimum mass density */
        outputPointers[0][x] = q_i[0];

        velocity_half /= q_i[0]; /* calculate velocity at the halfstep */


//      outputPointers[1][x] = locE; /* store total energy: We need to correct this for negativity shortly */
        locPsq = (locE - .5*(locPsq/q_i[0])); /* Calculate epsilon = E - T */
//      P[x] = FLUID_GM1*locPsq; /* Calculate P = (gamma-1) epsilon */

// For now we have to store the above before fixing them so the original freezeAndPtot runs unperturbed
// but assert the corrected P, C_f values below to see what we propose to do.
// it should match the freezeAndPtot very accurately.

// assert   cs^2 > cs^2(rho minimum)
//     g P / rho > g rho_min^(g-1) under polytropic EOS
//g(g-1) e / rho > g rho_min^(g-1)
//             e > rho rho_min^(g-1)/(g-1) = rho FLUID_MINEINT
        if(locPsq < q_i[0]*FLUID_MINEINT) {
          locE = locE - locPsq + q_i[0]*FLUID_MINEINT; // Assert minimum E = T + epsilon_min
          locPsq = q_i[0]*FLUID_MINEINT; // store minimum epsilon.
          } /* Assert minimum temperature */

        P[x] = FLUID_GM1*locPsq; /* Calculate P = (gamma-1) epsilon */
        outputPointers[1][x] = locE; /* store total energy: We need to correct this for negativity shortly */

        /* calculate local freezing speed */
        locPsq = abs(velocity_half) + sqrt(FLUID_GG1*locPsq/q_i[0]);
        if(locPsq > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = locPsq;
        }

    Xindex += BLOCKLEN;
    Xtrack += BLOCKLEN;
    __syncthreads();
    }

/* We have a block of 64 threads. Fold this shit in */

if(threadIdx.x > 32) return;

if(freezeSpeed[threadIdx.x+32] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+32];
__syncthreads();
if(threadIdx.x > 16) return;

if(freezeSpeed[threadIdx.x+16] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+16];
__syncthreads();
if(threadIdx.x > 8) return;

if(freezeSpeed[threadIdx.x+8] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+8];
__syncthreads();
if(threadIdx.x > 4) return;

if(freezeSpeed[threadIdx.x+4] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+4];
__syncthreads();
if(threadIdx.x > 2) return;

if(freezeSpeed[threadIdx.x+2] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+2];
__syncthreads();
if(threadIdx.x > 1) return;
/*if(threadIdx.x > 0) return;
for(x = 0; x < BLOCKLENP4; x++) { if(freezeSpeed[x] > freezeSpeed[0]) freezeSpeed[0] = freezeSpeed[x]; }
Cfreeze[blockIdx.x + gridDim.x * blockIdx.y] = freezeSpeed[0];*/

Cfreeze[blockIdx.x + gridDim.x * blockIdx.y] = (freezeSpeed[1] > freezeSpeed[0]) ? freezeSpeed[1] : freezeSpeed[0];

}

