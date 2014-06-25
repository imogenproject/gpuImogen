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

/* THIS FUNCTION
   This function is used in source/source.m and introduces the fictitious forces which
   result from a rotating coordinate frame. The rotation axis is fixed at +Z-hat to
   reduce computational burden; The frame equations are given at the start of the
   kernel itself.
 */

__global__ void  cukern_sourceRotatingFrame(double *rho, double *E, double *px, double *py, double *xyvector);
//__global__ void  cukern_sourceRotatingFrame(double *rho, double *E, double *px, double *py, double *Rx, double *Ry, int3 arraysize);

__constant__ __device__ double devLambda[2];
__constant__ __device__ int devIntParams[3];

/*mass.gputag, ener.gputag, mom(1).gputag, mom(2).gputag, 1, run.time.dTime, xg.GPU_MemPtr, yg.GPU_MemPtr*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// At least 2 arguments expected
	// Input and result
	if ((nrhs!=7) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaApplyScalarPotential(rho, E, px, py, omega, dt, [xvector yvector])\n");

	CHECK_CUDA_ERROR("entering cudaSourceRotatingFrame");

	// Get source array info and create destination arrays
	ArrayMetadata amd;
	double **srcs = getGPUSourcePointers(prhs, &amd, 0, 3);

	ArrayMetadata xmd;
	double **xyvec = getGPUSourcePointers(prhs, &xmd, 6, 6);

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

	int hostIntParams[3] = {amd.dim[0], amd.dim[1], amd.dim[2]};

	cudaMemcpyToSymbol(devLambda, &lambda[0], 2*sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(devIntParams, &hostIntParams[0], 3*sizeof(int), 0, cudaMemcpyHostToDevice);
	cukern_sourceRotatingFrame<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], xyvec[0]);

	CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, -1, "applyScalarPotential");

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

#define NTH (BLOCKDIMX*BLOCKDIMY)

#define OMEGA devLambda[0]
#define DT devLambda[1]
__global__ void  cukern_sourceRotatingFrame(double *rho, double *E, double *px, double *py, double *Rvector)
{
	__shared__ double shar[4*BLOCKDIMX*BLOCKDIMY];
	//__shared__ double pxhf[BLOCKDIMX*BLOCKDIMY], pyhf[BLOCKDIMX*BLOCKDIMY];
	//__shared__ double px0[BLOCKDIMX*BLOCKDIMY], py0[BLOCKDIMX*BLOCKDIMY];

	/* strategy: XY files, fill in X direction, step in Y direction; griddim.y = Nz */
	int myx = threadIdx.x + BLOCKDIMX*blockIdx.x;
	int myy = threadIdx.y;
	int myz = blockIdx.y;
	int nx = devIntParams[0]; int ny = devIntParams[1];

	if(myx >= devIntParams[0]) return;

//	int globaddr = myx + nx*(myy + ny*myz);
	int tileaddr = myx + nx*(myy + ny*myz);
	rho += tileaddr; E += tileaddr; px += tileaddr; py += tileaddr;
	tileaddr = threadIdx.x + BLOCKDIMX*threadIdx.y;

	double locX = Rvector[myx]; Rvector += nx; // Advances this to the Y array for below
	double locY;
	double locRho;
	double dmom; double dener;
	double locMom[2];

	for(; myy < ny; myy += BLOCKDIMY) {
		locY = Rvector[myy];

		// Load original values to register
		//locRho = rho[globaddr];
		//locMom[0] = px[globaddr];
		//locMom[1] = py[globaddr];
		locRho = *rho;
		shar[tileaddr] = *px;
		shar[tileaddr+NTH] = *py;

		// Predict momenta at half-timestep using 1st order method
		dmom         = DT*OMEGA*(shar[tileaddr+NTH] + OMEGA*locX*locRho/2.0); // dmom = delta px
		dener = (shar[tileaddr]+dmom/2)*dmom/locRho;
		shar[tileaddr+2*NTH] = shar[tileaddr];

		dmom         = DT*OMEGA*(-shar[tileaddr] + OMEGA*locY*locRho/2.0); // dmom = delta py
		dener += (shar[tileaddr]+dmom/2)*dmom/locRho;
		shar[tileaddr+3*NTH] = shar[tileaddr+NTH] + dmom;

		// Now make full timestep update: Evalute f' using f(t_half)
		dmom         = DT*OMEGA*(2*shar[tileaddr+3*NTH] + OMEGA*locX*locRho);
		dener = (shar[tileaddr]+dmom/2)*dmom/locRho;
		*px = shar[tileaddr] + dmom;

		dmom         = DT*OMEGA*(-2*shar[tileaddr+2*NTH] + OMEGA*locY*locRho);
		dener += (shar[tileaddr+NTH]+dmom/2)*dmom/locRho;
		*py = shar[tileaddr+NTH] + dmom;

		// Change in energy is exactly the work done by force
		// Is exactly (p^2 / 2 rho) after minus before
		*E += dener;

		rho += nx*BLOCKDIMY;
		E += nx*BLOCKDIMY;
		px += nx*BLOCKDIMY;
		py += nx*BLOCKDIMY;
	}
}

