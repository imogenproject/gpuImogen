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

__global__ void cukern_Wstep_mhd_uniform  (double *P, double *Cfreeze, double lambdaqtr, int nx);
__global__ void cukern_Wstep_hydro_uniform(double *P, double *Cfreeze, double lambdaqtr, int nx);

#define BLOCKLEN 60
#define BLOCKLENP2 62
#define BLOCKLENP4 64

__constant__ __device__ double *inputPointers[8];
__constant__ __device__ double *outputPointers[5];

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


__global__ void cukern_Wstep_hydro_uniform(double *P, double *Cfreeze, double lambdaqtr, int nx)
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

    q_i[0] = inputPointers[0][x];     /* Prelaod these out here */
    q_i[1] = inputPointers[1][x];       /* So we avoid multiple loops */
    q_i[2] = inputPointers[2][x];      /* over them inside the flux loop */
    q_i[3] = inputPointers[3][x];
    q_i[4] = inputPointers[4][x];
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
            outputPointers[i][x]= q_i[i] - lambdaqtr * ( fluxLR[0][threadIdx.x] - fluxLR[0][threadIdx.x+1] + \
                                                         fluxLR[1][threadIdx.x] - fluxLR[1][threadIdx.x-1]  ); 
            }

        __syncthreads();
        }

    Xindex += BLOCKLEN;
    Xtrack += BLOCKLEN;
    __syncthreads();
    }

}

