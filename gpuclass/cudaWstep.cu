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

typedef struct{ double *q[5]; } fluidQ;

__global__ void cukern_Wstep_mhd_uniform(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *P, double *Cfreeze, double *rhoW, double *enerW, double *pxW, double *pyW, double *pzW, double lambdaqtr, int nx);
__global__ void cukern_Wstep_hydro_uniform(double *rho, double *E, double *px, double *py, double *pz, double *P, double *Cfreeze, fluidQ f, double lambdaqtr, int nx);
__global__ void nullStep(fluidVarPtrs fluid, int numel);

#define BLOCKLEN 60
#define BLOCKLENP2 62
#define BLOCKLENP4 64

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Input and result
  if ((nrhs!=12) || (nlhs != 5)) mexErrMsgTxt("Wrong number of arguments: need [5] = cudaWflux(rho, E, px, py, pz, bx, by, bz, Ptot, c_f, lambda, purehydro?)\n");

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

// It appears this is only used in the null step. It was used in a previous W step but that kernel was irreperably broken.
fluidVarPtrs fluid;
int i;
for(i = 0; i < 5; i++) { fluid.fluidIn[i] = srcs[i]; fluid.fluidOut[i] = dest[i]; }
fluid.B[0] = srcs[5];
fluid.B[1] = srcs[6];
fluid.B[2] = srcs[7];

fluid.Ptotal = srcs[8];
fluid.cFreeze = srcs[9];

// If the dimension has finite extent, performs actual step; If not, blits input arrays to output arrays
// NOTE: this situation should not occur, since the flux routine itself skips singleton dimensions for 1- and 2-d sims.
int hydroOnly;

if(nu > 1) {
  hydroOnly = (int)*mxGetPr(prhs[11]);
  
  if(hydroOnly == 1) {
    fluidQ f;
    for(i = 0; i < 5; i++) { f.q[i] = dest[i]; }

    cukern_Wstep_hydro_uniform<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[8], srcs[9], f, .25*lambda, arraySize.x);

    } else {
    cukern_Wstep_mhd_uniform<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], srcs[6], srcs[7], srcs[8], srcs[9], dest[0], dest[1], dest[2], dest[3], dest[4], lambda/4.0, arraySize.x);
    }
  } else {
  nullStep<<<32, 128>>>(fluid, amd.numel);
  }

cudaError_t epicFail = cudaGetLastError();
if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, hydroOnly, "fluid W step");

}

__global__ void cukern_Wstep_mhd_uniform(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *P, double *Cfreeze, double *rhoW, double *enerW, double *pxW, double *pyW, double *pzW, double lambdaqtr, int nx)
{
double Cinv, rhoinv;
double q_i[5];
double b_i[3];
double w_i;
__shared__ double fluxLR[2][BLOCKLENP4];
double *fluxdest;

/* Step 0 - obligatory annoying setup stuff (ASS) */
int I0 = nx*(blockIdx.x + gridDim.x * blockIdx.y);
int Xindex = (threadIdx.x-2);
int Xtrack = Xindex;
Xindex += nx*(threadIdx.x < 2);

int x; /* = Xindex % nx; */
int i;
bool doIflux = (threadIdx.x > 1) && (threadIdx.x < BLOCKLEN+2);

/* Step 1 - calculate W values */
Cinv = 1.0/Cfreeze[blockIdx.x + gridDim.x * blockIdx.y];

while(Xtrack < nx+2) {
    x = I0 + (Xindex % nx);

    rhoinv = 1.0/rho[x]; /* Preload all these out here */
    q_i[0] = rho[x];
    q_i[1] = E[x];       /* So we avoid multiple loops */
    q_i[2] = px[x];      /* over them inside the flux loop */
    q_i[3] = py[x];
    q_i[4] = pz[x];
    b_i[0] = bx[x];
    b_i[1] = by[x];
    b_i[2] = bz[x];

    /* rho, E, px, py, pz going down */
    /* Iterate over variables to flux */
    for(i = 0; i < 5; i++) {
        switch(i) {
            case 0: w_i = q_i[2] * Cinv; break;
            case 1: w_i = (q_i[2] * (q_i[1] + P[x]) - b_i[0]*(q_i[2]*b_i[0]+q_i[3]*b_i[1]+q_i[4]*b_i[2]) ) * (rhoinv*Cinv); break;
            case 2: w_i = (q_i[2]*q_i[2]*rhoinv + P[x] - b_i[0]*b_i[0])*Cinv; break;
            case 3: w_i = (q_i[2]*q_i[3]*rhoinv        - b_i[0]*b_i[1])*Cinv; break;
            case 4: w_i = (q_i[2]*q_i[4]*rhoinv        - b_i[0]*b_i[2])*Cinv; break;
            }

        /* Step 2 - decouple to L/R flux */
        fluxLR[0][threadIdx.x] = (q_i[i] - w_i); /* Left  going flux */
        fluxLR[1][threadIdx.x] = (q_i[i] + w_i); /* Right going flux */
        // NOTE: a 0.5 is eliminated here. THis requires lambda to be rescaled by .5 in launch.
        __syncthreads();

        /* Step 4 - Perform flux and write to output array */
        __syncthreads();
       if( doIflux && (Xindex < nx) ) {
            switch(i) {
                case 0: fluxdest = rhoW; break;
                case 1: fluxdest = enerW; break;
                case 2: fluxdest = pxW; break;
                case 3: fluxdest = pyW; break;
                case 4: fluxdest = pzW; break;
                }

            // NOTE: a .5 is missing here also, so lambda must ultimately be divided by 4.
            fluxdest[x] = q_i[i] - lambdaqtr * ( fluxLR[0][threadIdx.x] - fluxLR[0][threadIdx.x+1] + \
                                      fluxLR[1][threadIdx.x] - fluxLR[1][threadIdx.x-1]  ) / Cinv; 

            }

        __syncthreads();
        }

    Xindex += BLOCKLEN;
    Xtrack += BLOCKLEN;
    __syncthreads();
    }

}


__global__ void cukern_Wstep_hydro_uniform(double *rho, double *E, double *px, double *py, double *pz, double *P, double *Cfreeze, fluidQ f, double lambdaqtr, int nx)
{
double C_f, velocity;
double q_i[5];
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

    q_i[0] = rho[x];     /* Prelaod these out here */
    q_i[1] = E[x];       /* So we avoid multiple loops */
    q_i[2] = px[x];      /* over them inside the flux loop */
    q_i[3] = py[x];
    q_i[4] = pz[x];
    velocity = q_i[2] / q_i[0];

    /* rho, E, px, py, pz going down */
    /* Iterate over variables to flux */
    for(i = 0; i < 5; i++) {
        switch(i) {
            case 0: w_i = q_i[2] ; break;
            case 1: w_i = (velocity * (q_i[1] + P[x])); break;
            case 2: w_i = (velocity*q_i[2] + P[x]); break;
            case 3: w_i = (velocity*q_i[3]       ); break;
            case 4: w_i = (velocity*q_i[4]       ); break;
            }

        /* Step 2 - decouple to L/R flux */
/* NOTE: factor of .5 moveed from here to output assignment (onto the second .5) */
        fluxLR[0][threadIdx.x] = (q_i[i]*C_f - w_i); /* Left  going flux */
        fluxLR[1][threadIdx.x] = (q_i[i]*C_f + w_i); /* Right going flux */
        __syncthreads();

        /* Step 4 - Perform flux and write to output array */
/*        __syncthreads(); */
	if( doIflux && (Xindex < nx) ) {
            f.q[i][x]= q_i[i] - lambdaqtr * ( fluxLR[0][threadIdx.x] - fluxLR[0][threadIdx.x+1] + \
                                      fluxLR[1][threadIdx.x] - fluxLR[1][threadIdx.x-1]  ); 

            }

        __syncthreads();
        }

    Xindex += BLOCKLEN;
    Xtrack += BLOCKLEN;
    __syncthreads();
    }

}

// Function simply blits fluid input variables to output, since with only 1 plane there's no derivative possible.
__global__ void nullStep(fluidVarPtrs fluid, int numel)
{
int idx0 = threadIdx.x + blockIdx.x*blockDim.x;
int didx = blockDim.x * gridDim.x;
int i;

while(idx0 < numel) {
  for(i = 0; i < 5; i++) { fluid.fluidOut[i][idx0] = fluid.fluidIn[i][idx0]; }

  idx0 += didx;
  }

}

