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

#define BLOCKDIMX 18
#define BLOCKDIMY 18

__global__ void  cukern_applyScalarPotential(double *rho, double *E, double *px, double *py, double *pz, double *phi, int3 arraysize);
/*mass.gputag, mom(1).gputag, mom(2).gputag, mom(3).gputag, ener.gputag, run.potentialField.gputag, 2*run.time.dTime);*/

__constant__ __device__ double devLambda[7];

#define LAMX devLambda[0]
#define LAMY devLambda[1]
#define LAMZ devLambda[2]

// Define: F = -beta * rho * grad(phi)
// rho_g = density for full effect of gravity 
// rho_c = minimum density to feel gravity at all
// beta = { rho_g < rho         : 1                                 }
//        { rho_c < rho < rho_g : [(rho-rho_c)/(rho_rho_g-rho_c)]^2 }
//        {         rho < rho_c : 0                                 }

// This provides a continuous (though not differentiable at rho = rho_g) way to surpress gravitation of the background fluid
// The original process of cutting gravity off below a critical density a few times the minimum
// density is believed to cause "blowups" at the inner edge of circular flow profiles due to being
// discontinuous. If even smoothness is insufficient and smooth differentiability is required,
// a more-times-continuous profile can be constructed, but let's not go there unless forced.

// Density below which we force gravity effects to zero
#define RHOMIN devLambda[3]
#define RHOGRAV devLambda[4]
// 1 / (rho_g - rho_c)
#define G1 devLambda[5]
// rho_c / (rho_g - rho_c)
#define G2 devLambda[6]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // At least 2 arguments expected
    // Input and result
    if ((nrhs!=10) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaApplyScalarPotential(rho, E, px, py, pz, phi, dt, d3x, rhomin, rho_fullg)\n");

    // Get source array info and create destination arrays
    ArrayMetadata amd;
    double **srcs = getGPUSourcePointers(prhs, &amd, 0, 5);

    dim3 gridsize, blocksize;
    int3 arraysize; arraysize.x = amd.dim[0]; arraysize.y = amd.dim[1]; arraysize.z = amd.dim[2];

    blocksize.x = BLOCKDIMX; blocksize.y = BLOCKDIMY; blocksize.z = 1;
    gridsize.x = arraysize.x / (blocksize.x - 2); gridsize.x += ((blocksize.x-2) * gridsize.x < amd.dim[0]);
    gridsize.y = arraysize.y / (blocksize.y - 2); gridsize.y += ((blocksize.y-2) * gridsize.y < amd.dim[1]);
    gridsize.z = 1;

    double dt = *mxGetPr(prhs[6]);
    double *dx = mxGetPr(prhs[7]);    
    double lambda[7];
    lambda[0] = dt/(2.0*dx[0]);
    lambda[1] = dt/(2.0*dx[1]);
    lambda[2] = dt/(2.0*dx[2]);
    lambda[3] = *mxGetPr(prhs[8]); /* minimum rho, rho_c */
    lambda[4] = *mxGetPr(prhs[9]); /* rho_g */

    lambda[5] = 1.0/(lambda[4] - lambda[3]); /* 1/(rho_g - rho_c) */
    lambda[6] = lambda[3]*lambda[5];

    cudaMemcpyToSymbol(devLambda, lambda, 7*sizeof(double), 0, cudaMemcpyHostToDevice);
    cukern_applyScalarPotential<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], arraysize);

    cudaError_t epicFail = cudaGetLastError();
    if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, -1, "applyScalarPotential");

}

/*
 * dP = -rho grad(phi) dt
 * dE = -rho v \cdot grad(phi) dt
 */
__global__ void  cukern_applyScalarPotential(double *rho, double *E, double *px, double *py, double *pz, double *phi, int3 arraysize)
{

int myLocAddr = threadIdx.x + BLOCKDIMX*threadIdx.y;

int myX = threadIdx.x + (BLOCKDIMX-2)*blockIdx.x - 1;
int myY = threadIdx.y + (BLOCKDIMY-2)*blockIdx.y - 1;

if((myX > arraysize.x) || (myY > arraysize.y)) return;

bool IWrite = (threadIdx.x > 0) && (threadIdx.x < (BLOCKDIMX-1)) && (threadIdx.y > 0) && (threadIdx.y < (BLOCKDIMY-1));
IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.y);

myX = (myX + arraysize.x) % arraysize.x;
myY = (myY + arraysize.y) % arraysize.y;

int globAddr = myX + arraysize.x*myY;

double deltaphi; // Store derivative of phi in one direction
double rhomin = devLambda[3];

__shared__ double phiA[BLOCKDIMX*BLOCKDIMY];
__shared__ double phiB[BLOCKDIMX*BLOCKDIMY];
__shared__ double phiC[BLOCKDIMX*BLOCKDIMY];

__shared__ double locrho[BLOCKDIMX*BLOCKDIMY];
__shared__ double ener[BLOCKDIMX*BLOCKDIMY];

double *U; double *V; double *W;
double *temp;

U = phiA; V = phiB; W = phiC;

// Preload lower and middle planes
U[myLocAddr] = phi[globAddr + arraysize.x*arraysize.y*(arraysize.z-1)];
V[myLocAddr] = phi[globAddr];

__syncthreads();

int z;
int deltaz = arraysize.x*arraysize.y;
for(z = 0; z < arraysize.z; z++) {
  if(z >= arraysize.z - 1) deltaz = - arraysize.x*arraysize.y*(arraysize.z-1);

  ener[myLocAddr]   = 0;
  locrho[myLocAddr] = rho[globAddr]; // rho(z) -> rho
  W[myLocAddr]      = px[globAddr]; // load px(z) -> phiC
  __syncthreads();

  if(IWrite && (locrho[myLocAddr] > rhomin)) {
    deltaphi         = devLambda[0]*(V[myLocAddr+1]-V[myLocAddr-1]);
    if(locrho[myLocAddr] < RHOGRAV) { deltaphi *= (locrho[myLocAddr]*G1 - G2); } // reduce G for low density
    ener[myLocAddr] -= deltaphi*W[myLocAddr]; // ener -= dt * px * dphi/dx
    px[globAddr]     = W[myLocAddr] - deltaphi*locrho[myLocAddr]; // store px <- px - dt * rho dphi/dx;
  }

  W[myLocAddr] = py[globAddr]; // load py(z) -> phiC
  __syncthreads();
  if(IWrite && (locrho[myLocAddr] > rhomin)) {
    deltaphi         = devLambda[1]*(V[myLocAddr+BLOCKDIMX]-V[myLocAddr-BLOCKDIMX]);
   if(locrho[myLocAddr] < RHOGRAV) { deltaphi *= (locrho[myLocAddr]*G1 - G2); } // reduce G for low density
    ener[myLocAddr] -= deltaphi*W[myLocAddr]; // ener -= dt * py * dphi/dy
    py[globAddr]     = W[myLocAddr] - deltaphi*locrho[myLocAddr]; // store py <- py - rho dphi/dy;
  }

  W[myLocAddr]       = phi[globAddr + deltaz]; // load phi(z+1) -> phiC
  __syncthreads();
  deltaphi           = devLambda[2]*(W[myLocAddr] - U[myLocAddr]);
  if(locrho[myLocAddr] < RHOGRAV) { deltaphi *= (locrho[myLocAddr]*G1 - G2); } // reduce G for low density
  __syncthreads();

  U[myLocAddr]       = pz[globAddr]; // load pz(z) -> phiA
  __syncthreads();
  if(IWrite && (locrho[myLocAddr] > rhomin)) {
    E[globAddr]     += ener[myLocAddr] - deltaphi*U[myLocAddr]; // Store E[x] <- ener - dt *pz * dphi/dz
    pz[globAddr]     = U[myLocAddr] - deltaphi*locrho[myLocAddr]; // store pz <- pz - rho dphi/dz;
  }

  temp = U; U = V; V = W; W = temp; // cyclically shift them back
  globAddr += arraysize.x * arraysize.y;

}


}

