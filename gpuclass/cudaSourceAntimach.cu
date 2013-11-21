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

#define BLOCKDIMX 16
#define BLOCKDIMY 16

__global__ void  cukern_AntiMach(double *rho, double *E, double *px, double *py, double *pz, int3 arraysize);

__constant__ __device__ double devLambda[2];

/*mass.gputag, ener.gputag, mom(1).gputag, mom(2).gputag, 1, run.time.dTime, xg.GPU_MemPtr, yg.GPU_MemPtr*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // At least 2 arguments expected
    // Input and result
    if ((nrhs!=7) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaApplyScalarPotential(rho, E, px, py, pz, gamma, m_max)\n");


    cudaCheckError("entering cudaSourceRotatingFrame");

    // Get source array info and create destination arrays
    ArrayMetadata amd;
    double **srcs = getGPUSourcePointers(prhs, &amd, 0, 4);

    double maxmach = *mxGetPr(prhs[6]); // get dt and dx to obey CFL such that .5 a tau^2 << h
    double polygamma = *mxGetPr(prhs[5]);

    dim3 gridsize, blocksize;
    int3 arraysize; arraysize.x = amd.dim[0]; arraysize.y = amd.dim[1]; arraysize.z = amd.dim[2];

    blocksize.x = BLOCKDIMX; blocksize.y = BLOCKDIMY; blocksize.z = 1;
    gridsize.x = arraysize.x / (blocksize.x); gridsize.x += ((blocksize.x) * gridsize.x < amd.dim[0]);
    gridsize.y = arraysize.z;
    gridsize.z = 1;

    double lambda[2];
    lambda[0] = maxmach;
    lambda[1] = polygamma*(polygamma-1.0);

    cudaMemcpyToSymbol(devLambda, &lambda[0], 2*sizeof(double), 0, cudaMemcpyHostToDevice);
    cukern_AntiMach<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], arraysize);

    cudaError_t epicFail = cudaGetLastError();
    if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, -1, "applyScalarPotential");

}

/* rho, E, Px, Py, Pz: arraysize-sized arrays */
#define M_0sq devLambda[0]
#define GG1 devLambda[1]

__global__ void  cukern_AntiMach(double *rho, double *E, double *px, double *py, double *pz, int3 arraysize)
{
/* strategy: XY files, fill in X direction, step in Y direction; griddim.y = Nz */
int myx = threadIdx.x + BLOCKDIMX*blockIdx.x;
int myy = threadIdx.y;
int myz = blockIdx.y;
int nx = arraysize.x; int ny = arraysize.y;

if(myx >= arraysize.x) return; 

int globaddr = myx + nx*(myy + ny*myz);

double locRho;
double locMom[3];
double locEner;
/*double inv_rsqr, xy;*/
double momsq;
double mach;

double dmomentum;

#ifdef DEBUGME
int stopme = (myz == 40) && (blockIdx.x == 12);
#endif

for(; myy < ny; myy += BLOCKDIMY) {

  locRho    = rho[globaddr];
  locMom[0] = px[globaddr];
  locMom[1] = py[globaddr];
  locMom[2] = pz[globaddr];
  locEner   = E[globaddr];
 
  // calculate local momentum density
  momsq = locMom[0]*locMom[0]+locMom[1]*locMom[1]+locMom[2]*locMom[2];

  // mach squared = v^2 / c_s^2 = p^2 / [rho (gg1 (E - .5 p^2 / rho) ) ]
  //                            = p^2 / [ gg1 ( rho E - .5 p^2) ]
  mach = momsq/(GG1*(locRho*locEner-.5*momsq));

  if(mach > M_0sq) { 
    mach = sqrt(mach/M_0sq) - 1; // Calculate mach-hat - 1 as our braking parameter
    // calculate rho * accel * dt (= dmomentum) * [ vector(momentum) * (ms/m0-1)^2 / |momentum| ]
    dmomentum = 1.0/(1.0 + mach*mach);

    // Apply braking force
    px[globaddr] = locMom[0] *dmomentum;
    py[globaddr] = locMom[1] *dmomentum;
    pz[globaddr] = locMom[2] *dmomentum;
    E[globaddr]  = locEner + .5*momsq*(dmomentum*dmomentum - 1.0)/locRho;
    }

  globaddr += nx*BLOCKDIMY;
  }
}

