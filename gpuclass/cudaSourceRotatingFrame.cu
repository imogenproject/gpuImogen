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

__global__ void  cukern_sourceRotatingFrame(double *rho, double *E, double *px, double *py, double *Rx, double *Ry, int3 arraysize);

__constant__ __device__ double devLambda[2];

/*mass.gputag, ener.gputag, mom(1).gputag, mom(2).gputag, 1, run.time.dTime, xg.GPU_MemPtr, yg.GPU_MemPtr*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // At least 2 arguments expected
    // Input and result
    if ((nrhs!=8) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaApplyScalarPotential(rho, E, px, py, omega, dt, xvector, yvector)\n");

  cudaCheckError("entering cudaSourceRotatingFrame");

    // Get source array info and create destination arrays
    ArrayMetadata amd;
    double **srcs = getGPUSourcePointers(prhs, &amd, 0, 3);

    ArrayMetadata xmd;
    double **xvec = getGPUSourcePointers(prhs, &xmd, 6, 6);
    ArrayMetadata ymd;
    double **yvec = getGPUSourcePointers(prhs, &ymd, 7, 7);

    dim3 gridsize, blocksize;
    int3 arraysize; arraysize.x = amd.dim[0]; arraysize.y = amd.dim[1]; arraysize.z = amd.dim[2];

    blocksize.x = BLOCKDIMX; blocksize.y = BLOCKDIMY; blocksize.z = 1;
    gridsize.x = arraysize.x / (blocksize.x); gridsize.x += ((blocksize.x) * gridsize.x < amd.dim[0]);
    gridsize.y = arraysize.z;
    gridsize.z = 1;

    double omega = *mxGetPr(prhs[4]);
    double dt    = *mxGetPr(prhs[5]);    
    double lambda[4];
    lambda[0] = omega;
    lambda[1] = dt;

    cudaMemcpyToSymbol(devLambda, &lambda[0], 2*sizeof(double), 0, cudaMemcpyHostToDevice);
    cukern_sourceRotatingFrame<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], xvec[0], yvec[0], arraysize);

    cudaError_t epicFail = cudaGetLastError();
    if(epicFail != cudaSuccess) cudaLaunchError(epicFail, blocksize, gridsize, &amd, -1, "applyScalarPotential");

}

/* 
 * a  = -[2 w X v + w X (w X r) ]
 * dv = -[2 w X v + w X (w X r) ] dt
 * dp = -rho dv = -rho [[2 w X v + w X (w X r) ] dt
 * dp = -[2 w X p + rho w X (w X r) ] dt
 *
 * w X p = |I  J  K | = <-w py, w px, 0> = u
 *         |0  0  w |
 *         |px py pz|
 *
 * w X r = <-w y, w x, 0> = s; 
 * w X s = |I   J  K| = <-w^2 x, -w^2 y, 0> = -w^2<x,y,0> = b
 *         |0   0  w|
 *         |-wy wx 0|
 * dp = -[2 u + rho b] dt
 *    = -[2 w<-py, px, 0> - rho w^2 <x, y, 0>] dt
 *    = w dt [2<py, -px> + rho w <x, y>] in going to static frame
 * 
 * dE = -v dot dp
 */
/* rho, E, Px, Py, Pz: arraysize-sized arrays
   omega: scalar
   Rx: [nx 1 1] sized array
   Ry: [ny 1 1] sized array */
#define OMEGA devLambda[0]
#define DT devLambda[1]
__global__ void  cukern_sourceRotatingFrame(double *rho, double *E, double *px, double *py, double *Rx, double *Ry, int3 arraysize)
{
/* strategy: XY files, fill in X direction, step in Y direction; griddim.y = Nz */
int myx = threadIdx.x + BLOCKDIMX*blockIdx.x;
int myy = threadIdx.y;
int myz = blockIdx.y;
int nx = arraysize.x; int ny = arraysize.y;

if(myx >= arraysize.x) return; 

int globaddr = myx + nx*(myy + ny*myz);

double locX = Rx[myx];
double locY;
double locRho;
double dmom; double dener;
double locMom[2];
/*double inv_rsqr, xy;*/

for(; myy < ny; myy += BLOCKDIMY) {
  locY = Ry[myy];

/*  inv_rsqr = 2.0/(locX*locX+locY*locY);
  xy = locX*locY;*/

  locRho = rho[globaddr];
  locMom[0] = px[globaddr];
  locMom[1] = py[globaddr];
 
/*  dmom = DT*OMEGA*(2*(x y px + x x py)/r^2 - rho w x); dpx */
  dmom         = DT*OMEGA*(2*locMom[1] + OMEGA*locX*locRho);
  px[globaddr] = locMom[0] + dmom;
  dener        = dmom*locMom[0] / locRho;

/*  dmom = DT*OMEGA*(-2*(x x px + x y py)/r^2 - rho w y); dpy */
  dmom         = DT*OMEGA*(-2*locMom[0] + OMEGA*locY*locRho);
  py[globaddr] = locMom[1] + dmom;
  E[globaddr] += dener + dmom*locMom[1] / locRho;

  globaddr += nx*BLOCKDIMY;
  }
}

