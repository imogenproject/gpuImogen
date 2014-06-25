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

   When not passed a magnetic field, calculates the local adiabatic sound speed of the fluid at 
   all points,

   c_s^2 = gamma*P/rho

   where c_s is the adiabatic sound speed, gamma is the adiabatic index (1 <= gamma <= 5/3), P
   is the thermal pressure (gamma-1)*(Etotal - rho v^2/2) and rho is the matter density.

   When passed a magnetic field, calculates the maximal fast Alfven velocity,
   C_fast^2 = C_s^2 + C_a^2,

   where C_s is the thermal sound speed above (Except subtracting magnetic energy density from
   the total energy as well) and C_a is the Alfven speed,

   C_a^2 = (B^2)/rho.
*/

__global__ void cukern_Soundspeed_mhd(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *dout, int n);
__global__ void cukern_Soundspeed_hd(double *rho, double *E, double *px, double *py, double *pz, double *dout, int n);

#define BLOCKDIM 256

__constant__ double pressParams[6];
#define MHD_CS_B pressParams[0]
#define GG1 pressParams[1]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Determine appropriate number of arguments for RHS
  if( (nlhs != 1) || ( (nrhs != 9) && (nrhs != 6) ))
    mexErrMsgTxt("calling form for cudaSoundspeed is c_s = cudaSoundspeed(mass, ener, momx, momy, momz, [bx, by, bz,] gamma);");

  CHECK_CUDA_ERROR("entering cudaSoundspeed");

  dim3 blocksize; blocksize.x = BLOCKDIM; blocksize.y = blocksize.z = 1;
  ArrayMetadata amd;
  dim3 gridsize;

  // Select the appropriate kernel to invoke
    int pureHydro = (nrhs == 6);

    double gam; double **srcs;

    if(pureHydro == 1) {
      gam = *mxGetPr(prhs[5]);
      srcs = getGPUSourcePointers(prhs, &amd, 0, 4);
      } else {
      gam = *mxGetPr(prhs[8]);
      srcs = getGPUSourcePointers(prhs, &amd, 0, 7);
      }

    gridsize.x = BLOCKDIM;
    gridsize.y = gridsize.z =1;
    double **destPtr = makeGPUDestinationArrays(&amd, plhs, 1);
    double gg1 = gam*(gam-1);

    double hostP[6];
    hostP[0] = ALFVEN_CSQ_FACTOR - .5*gg1;
    hostP[1] = gg1;
    
    cudaMemcpyToSymbol(pressParams, &hostP[0], 6*sizeof(double), 0, cudaMemcpyHostToDevice);

    if(pureHydro == 1) {
      cukern_Soundspeed_hd<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], destPtr[0], amd.numel);
      } else {
      cukern_Soundspeed_mhd<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], srcs[6], srcs[7], destPtr[0], amd.numel);
      }

    CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, nrhs, "cuda sound speed");

    free(destPtr);
}

// THIS KERNEL CALCULATES SOUNDSPEED IN THE MHD CASE, TAKEN AS THE FAST MA SPEED
// We increase the Alfven contribution to stabilize the code
__global__ void cukern_Soundspeed_mhd(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *dout, int n)
{

int x = threadIdx.x + blockIdx.x * BLOCKDIM;
int dx = blockDim.x * gridDim.x;
double csq, T, Bsq;
double invrho;

while(x < n) {
    invrho = 1.0 / rho[x];
    T = .5*(px[x]*px[x] + py[x]*py[x] + pz[x]*pz[x])*invrho;
    Bsq = bx[x]*bx[x] + by[x]*by[x] + bz[x]*bz[x];

    // MHD_CS_B is (alfven constant A) - .5(gamma)(gamma-1), where A is physically 1
    // but may be increased beyond 1 to stabilize simulations where low-beta conditions occur
    csq = (GG1*(E[x] - T) + MHD_CS_B * Bsq ) * invrho ;
    if(csq < 0.0) csq = 0.0;
    dout[x] = sqrt(csq);
    x += dx;
    }

}

// THIS KERNEL CALCULATES SOUNDSPEED IN THE HYDRODYNAMIC CASE
__global__ void cukern_Soundspeed_hd(double *rho, double *E, double *px, double *py, double *pz, double *dout, int n)
{
int x = threadIdx.x + blockIdx.x * BLOCKDIM;
int dx = blockDim.x * gridDim.x;
double csq, rhoinv;

while(x < n) {
	rhoinv = 1/rho[x];
    csq = GG1*(E[x] - .5*(px[x]*px[x] + py[x]*py[x] + pz[x]*pz[x])*rhoinv)*rhoinv;
    // Imogen's energy flux is unfortunately not positivity preserving
    if(csq < 0.0) csq = 0.0;
    dout[x] = sqrt(csq);
    x += dx;
    }

}


