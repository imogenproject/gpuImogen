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
perform a non-oscillatory update. (Trac, Pen 2003; Xin & Jin, 1995)

The 1D segment of the fluid equations solved is 
     | rho |         | (vx rho == px)           |
     | px  |         | vx px + P - bx^2         |
d/dt | py  | = -d/dx | vx py     - bx by        |
     | pz  |         | vx pz     - bx bz        |
     | E   |         | vx (E+P)  - bx (B dot v) |

with auxiliary equations
  vx = px / rho
  P  = (gamma-1)e + .5*B^2 = thermal pressure + magnetic pressure
  e  = E - .5*(p^2)/rho - .5*(B^2)

In general thermal pressure is an arbitrary positive function of e, however the ideal gas
law is built into Imogen in multiple locations and significant re-checking would be needed
if it were to be generalized.

The hydrodynamic functions solve the same equations under the assumption B = <0,0,0> which
simplifies and considerably speeds up the process

Normally second time order accuracy is acheived using the standard half-step/full-step
Runge-Kutta method, with the half-step supplied by a first-order upwind method. It is possible
to acheive higher accuracy (particularly at CFL > 0.5) by using the TVD function for both
the half- and full-step, but the increase in computation is lardefge

*/

#define BLOCKLEN 92
#define BLOCKLENP2 94
#define BLOCKLENP4 96

__global__ void cukern_TVDStep_mhd_uniform(double *P, double *Cfreeze, double halflambda, int nx);
__global__ void cukern_TVDStep_hydro_uniform(double *P, double *Cfreeze, double halfLambda, int nx);

//#define LIMITERFUNC fluxLimiter_Osher
#define LIMITERFUNC fluxLimiter_minmod

__constant__ __device__ double *inputPointers[8];
__constant__ __device__ double *outputPointers[5];
__constant__ __device__ double fluidParams[2];

#define RHOMIN fluidParams[0]
#define MIN_ETHERM fluidParams[1]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // At least 2 arguments expected
  // Input and result
  if ((nrhs!=18) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: call cudaTVDStep(rho, E, px, py, pz, bx, by, bz, P, rho_out, E_out, px_out, py_out, pz_out, C_freeze, lambda, purehydro?)\n");

  CHECK_CUDA_ERROR("entering FluidTVD");

  // Get source array info and create destination arrays
  ArrayMetadata amd;

  // Get the source gpu arrays as enumerated in the error message
  double **srcs   = getGPUSourcePointers(prhs, &amd, 0, 13);

  // Get the GPU freeze speed. Differs in that it is not the same size
  ArrayMetadata fmd;
  double **gpu_cf = getGPUSourcePointers(prhs, &fmd, 14, 14); 

  // Get flux factor (dt / dx) amd determine if we are doing the hydro case or the MHD case
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

  double *mins = mxGetPr(prhs[17]);
  double rhomin = mins[0];
  double gamma = mins[1];
  double gamHost[2];

  gamHost[0] = rhomin;
// assert     cs > cs_min
//     g P / rho > g rho_min^(g-1)
// (g-1) e / rho > rho_min^(g-1)
//             e > rho rho_min^(g-1)/(g-1)
  gamHost[1] = powl(rhomin, gamma-1.0)/(gamma-1.0);
  cudaMemcpyToSymbol(fluidParams, &gamHost[0], 2*sizeof(double), 0, cudaMemcpyHostToDevice);

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

CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, isPureHydro, "fluid TVD");

}

#define LEFTMOST_FLAG 1
#define RIGHTMOST_FLAG 2
#define ENDINGRHS_FLAG 4
#define IAM_MAIN_BLOCK 8

#define IAMLEFTMOST (whoflags & LEFTMOST_FLAG)
#define IAMRIGHTMOST (whoflags & RIGHTMOST_FLAG)
#define IAMENDRHS   (whoflags & ENDINGRHS_FLAG)
#define IAMMAIN     (whoflags & IAM_MAIN_BLOCK)

/* blockidx.{xy} is our index in {yz}, and gridDim.{xy} gives the {yz} size */
/* Expect invocation with n+4 threads */
__global__ void cukern_TVDStep_mhd_uniform(double *P, double *Cfreeze, double halfLambda, int nx)
{
// Declare a bunch of crap, much more than needed.
// In NVCC -O2 and symbolic algebra transforms we trust
double c_f, velocity;
double q_i[5];
double prop_i[5]; // proposed q_i values
double b_i[3];
double w_i;
__shared__ double fluxLR[2][BLOCKLENP4];
__shared__ double fluxDerivA[BLOCKLENP4+1];
__shared__ double fluxDerivB[BLOCKLENP4+1];

// Precompute some information about "special" threads.
int whoflags = 0;
if(threadIdx.x < 2)           whoflags += LEFTMOST_FLAG; // Mark which threads form the left most of the block,
if(threadIdx.x >= BLOCKLENP2) whoflags += RIGHTMOST_FLAG; // The rightmost, and the two which will form the final RHS
if((threadIdx.x == ( (nx % BLOCKLEN) + 2)) || (threadIdx.x == ( (nx % BLOCKLEN) + 3)) ) whoflags += ENDINGRHS_FLAG;
if((threadIdx.x > 1) && (threadIdx.x < BLOCKLENP2)) whoflags += IAM_MAIN_BLOCK;

// Calculate the usual stupid indexing tricks
int I0 = nx*(blockIdx.x + gridDim.x * blockIdx.y);
int Xindex = (threadIdx.x-2);
int Xtrack = Xindex;
Xindex += nx*(threadIdx.x < 2);
int x;
int i;

unsigned int threadIndexL = (threadIdx.x-1)%BLOCKLENP4;

// Load the freezing speed once
c_f = Cfreeze[blockIdx.x + gridDim.x * blockIdx.y];

while(Xtrack < nx+2) {
    x = I0 + (Xindex % nx);

    q_i[0] = inputPointers[0][x]; // rho
    q_i[1] = inputPointers[1][x]; // Etot      /* So we avoid multiple loops */
    q_i[2] = inputPointers[2][x]; // Px     /* over them inside the flux loop */
    q_i[3] = inputPointers[3][x]; // Py
    q_i[4] = inputPointers[4][x]; // Pz
    b_i[0] = inputPointers[5][x]; // Bx
    b_i[1] = inputPointers[6][x]; // By
    b_i[2] = inputPointers[7][x]; // Bz

    velocity = q_i[2]/q_i[0];

    __syncthreads();

    /* rho, E, px, py, pz going down */
    /* Iterate over variables to flux */
    for(i = 0; i < 5; i++) {
        /* Calculate raw fluxes */
        switch(i) {
            case 0: w_i = q_i[2]; break;
            case 1: w_i = (velocity * (q_i[1] + P[x]) - b_i[0]*(q_i[2]*b_i[0]+q_i[3]*b_i[1]+q_i[4]*b_i[2])/q_i[0] ); break;
            case 2: w_i = (velocity*q_i[2] + P[x] - b_i[0]*b_i[0]); break;
            case 3: w_i = (velocity*q_i[3]        - b_i[0]*b_i[1]); break;
            case 4: w_i = (velocity*q_i[4]        - b_i[0]*b_i[2]); break;
            }

        /* Decouple to L/R flux. */
        fluxLR[0][threadIdx.x] = (q_i[i]*c_f - w_i); /* Left  going flux */
        fluxLR[1][threadIdx.x] = (q_i[i]*c_f + w_i); /* Right going flux */
        __syncthreads();

        /* Derivative of left flux, then right flux */
        fluxDerivA[threadIdx.x] = (fluxLR[0][threadIdx.x] - fluxLR[0][threadIndexL])/2.0; 
        fluxDerivB[threadIdx.x] = (fluxLR[1][threadIdx.x] - fluxLR[1][threadIndexL])/2.0;
        __syncthreads();

        /* Apply limiter function to 2nd order corrections */
        fluxLR[0][threadIdx.x] -= LIMITERFUNC(fluxDerivA[threadIdx.x], fluxDerivA[threadIdx.x+1]); // A=bkwd(x), B=fwd(x)
        fluxLR[1][threadIdx.x] += LIMITERFUNC(fluxDerivB[threadIdx.x+1], fluxDerivB[threadIdx.x]); // A=fwd(x), B=bkwd(x)
        __syncthreads();

        /* Perform flux and propose output value */
       if( IAMMAIN && (Xindex < nx) ) {
            prop_i[i] = outputPointers[i][x] - halfLambda * ( -fluxLR[0][threadIdx.x+1] + fluxLR[0][threadIdx.x] + \
                                                              fluxLR[1][threadIdx.x] - fluxLR[1][threadIndexL]  ); 
          }

        __syncthreads();
        }

    if( IAMMAIN && (Xindex < nx) ) {
      prop_i[0] = (prop_i[0] < RHOMIN) ? RHOMIN : prop_i[0]; // enforce min density

      w_i = .5*(prop_i[2]*prop_i[2] + prop_i[3]*prop_i[3] + prop_i[4]*prop_i[4])/prop_i[0] + .5*(b_i[0]*b_i[0] + b_i[1]*b_i[1] + b_i[2]*b_i[2]);

      if((prop_i[1] - w_i) < prop_i[0]*MIN_ETHERM) {
        prop_i[1] = prop_i[0]*MIN_ETHERM + w_i;
        } 

      outputPointers[0][x] = prop_i[0];
      outputPointers[1][x] = prop_i[1];
      outputPointers[2][x] = prop_i[2];
      outputPointers[3][x] = prop_i[3];
      outputPointers[4][x] = prop_i[4];
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
__shared__ double fluxDerivA[BLOCKLENP4+1];
__shared__ double fluxDerivB[BLOCKLENP4+1];

/* Step 0 - obligatory annoying setup stuff (ASS) */
int I0 = nx*(blockIdx.x + gridDim.x * blockIdx.y);
int Xindex = (threadIdx.x-2);
int Xtrack = Xindex;
Xindex += nx*(threadIdx.x < 2);

int x; /* = Xindex % nx; */
int i;
bool doIflux = (threadIdx.x > 1) && (threadIdx.x < BLOCKLENP2);
double prop_i[5];

unsigned int threadIndexL = (threadIdx.x-1+BLOCKLENP4)%BLOCKLENP4;

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
        /* Calculate raw fluxes */
        switch(i) {
            case 0: w_i = q_i[2]; break;
            case 1: w_i = (velocity * (q_i[1] + P[x]) ) ; break;
            case 2: w_i = (velocity * q_i[2] + P[x]); break;
            case 3: w_i = (velocity * q_i[3]); break;
            case 4: w_i = (velocity * q_i[4]); break;
            }

        /* Decouple to L/R flux */
        fluxLR[0][threadIdx.x] = (C_f*q_i[i] - w_i); /* Left  going flux */
        fluxLR[1][threadIdx.x] = (C_f*q_i[i] + w_i); /* Right going flux */
        __syncthreads();

        /* Calculate proposed flux corrections */
        fluxDerivA[threadIdx.x] = (fluxLR[0][threadIndexL] - fluxLR[0][threadIdx.x]) / 2.0; /* Deriv of leftgoing flux */
        fluxDerivB[threadIdx.x] = (fluxLR[1][threadIdx.x] - fluxLR[1][threadIndexL]) / 2.0; /* Deriv of rightgoing flux */
        __syncthreads();
        
        /* Impose TVD limiter */
        fluxLR[0][threadIdx.x] += LIMITERFUNC(fluxDerivA[threadIdx.x], fluxDerivA[threadIdx.x+1]);
        fluxLR[1][threadIdx.x] += LIMITERFUNC(fluxDerivB[threadIdx.x+1], fluxDerivB[threadIdx.x]); // A=fwd(x), B=bkwd(x)
        __syncthreads();

        /* Perform flux and write to output array */
       if( doIflux && (Xindex < nx) ) {
            prop_i[i] = outputPointers[i][x] - halfLambda * ( fluxLR[1][threadIdx.x] - fluxLR[1][threadIndexL] + 
                                                              -fluxLR[0][threadIdx.x+1] + fluxLR[0][threadIdx.x]);//
            }

        __syncthreads();
        }

    if( doIflux && (Xindex < nx) ) {
        prop_i[0] = (prop_i[0] < RHOMIN) ? RHOMIN : prop_i[0];
        w_i = .5*(prop_i[2]*prop_i[2] + prop_i[3]*prop_i[3] + prop_i[4]*prop_i[4])/prop_i[0];

        if((prop_i[1] - w_i) < prop_i[0]*MIN_ETHERM) {
            prop_i[1] = w_i + prop_i[0]*MIN_ETHERM;
            }

        outputPointers[0][x] = prop_i[0];
        outputPointers[1][x] = prop_i[1];
        outputPointers[2][x] = prop_i[2];
        outputPointers[3][x] = prop_i[3];
        outputPointers[4][x] = prop_i[4];
        }

    __syncthreads();

    Xindex += BLOCKLEN;
    Xtrack += BLOCKLEN;
    }

}

