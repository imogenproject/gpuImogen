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

__global__ void cukern_FetchPartitionSubset1D(double *in, int nodeN, double *out, int partX0, int partNX);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// At least 2 arguments expected
	// Input and result
	if ((nrhs!=7) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaSourceRotatingFrame(rho, E, px, py, omega, dt, [xvector yvector])\n");

	CHECK_CUDA_ERROR("entering cudaSourceRotatingFrame");

	// Get source array info and create destination arrays
        MGArray fluid[4];
        int worked = MGA_accessMatlabArrays(prhs, 0, 3, &fluid[0]);

/* FIXME: accept this as a matlab array instead
 * FIXME: Transfer appropriate segments to __constant__ memory
 * FIXME: that seems the only reasonable way to avoid partitioning hell
 */
        MGArray xyvec;
        worked     = MGA_accessMatlabArrays(prhs, 6, 6, &xyvec);

	dim3 gridsize, blocksize;
	int3 arraysize;


	double omega = *mxGetPr(prhs[4]);
	double dt    = *mxGetPr(prhs[5]);
	double lambda[4];
	lambda[0] = omega;
	lambda[1] = dt;

	int i;

	double *devXYset[fluid->nGPUs];
	int sub[6];


	for(i = 0; i < fluid->nGPUs; i++) {
		cudaSetDevice(fluid->deviceID[i]);
		CHECK_CUDA_ERROR("cudaSetDevice");

		// Upload rotation parameters
		cudaMemcpyToSymbol(devLambda, &lambda[0], 2*sizeof(double), 0, cudaMemcpyHostToDevice);
		CHECK_CUDA_ERROR("memcpy to symbol");

		// Upload partition size
		calcPartitionExtent(fluid, i, &sub[0]);
		cudaMemcpyToSymbol(devIntParams, &sub[3], 3*sizeof(int), 0, cudaMemcpyHostToDevice);
		CHECK_CUDA_ERROR("memcpy to symbol");

		// Swipe the needed subsegments of the X/Y vectors from the supplied node-wide array
		cudaMalloc((void **)&devXYset[i], (sub[3]+sub[4])*sizeof(double));
		CHECK_CUDA_ERROR("cudaMalloc");

		blocksize = makeDim3(128, 1, 1);
                gridsize.x = ROUNDUPTO(sub[3], 128) / 128;
		gridsize.y = gridsize.z = 1;
		cukern_FetchPartitionSubset1D<<<gridsize, blocksize>>>(xyvec.devicePtr[i], fluid->dim[0], devXYset[i], sub[0], sub[3]);
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &xyvec, -1, "cukern_FetchPartitionSubset1D, X");
		gridsize.x = ROUNDUPTO(sub[4], 128) / 128;
		cukern_FetchPartitionSubset1D<<<gridsize, blocksize>>>(xyvec.devicePtr[i] + fluid->dim[0], fluid->dim[1], devXYset[i]+sub[3], sub[1], sub[4]);
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &xyvec, -1, "cukern_FetchPartitionSubset1D, Y");

		arraysize.x = sub[3]; arraysize.y = sub[4]; arraysize.z = sub[5];

		blocksize = makeDim3(BLOCKDIMX, BLOCKDIMY, 1);
		gridsize.x = ROUNDUPTO(arraysize.x, blocksize.x) / blocksize.x;
		gridsize.y = arraysize.z;
		gridsize.z = 1;

		cukern_sourceRotatingFrame<<<gridsize, blocksize>>>(
        	    fluid[0].devicePtr[i],
	            fluid[1].devicePtr[i],
        	    fluid[2].devicePtr[i],
	            fluid[3].devicePtr[i],
        	    devXYset[i]);
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, -1, "applyScalarPotential");

	}

	for(i = 0; i < fluid->nGPUs; i++) { 
		cudaFree(devXYset[i]);
		CHECK_CUDA_ERROR("cudaFree");
	}

        // Differencing has corrupted the energy and momentum halos: Restore them
	MGA_exchangeLocalHalos(fluid + 1, 3);

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

	double locX = Rvector[myx];
	Rvector += nx; // Advances this to the Y array for below
	double locY;
	double locRho;
	double dmom; double dener;
//	double locMom[2];

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
//		dener = (shar[tileaddr]+dmom/2)*dmom/locRho;
		shar[tileaddr+2*NTH] = shar[tileaddr] + dmom;

		dmom         = DT*OMEGA*(-shar[tileaddr] + OMEGA*locY*locRho/2.0); // dmom = delta py
//		dener += (shar[tileaddr]+dmom/2)*dmom/locRho;
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

/* Simple kernel:
 * Given in[0 ... (nodeN-1)], copies the segment in[partX0 ... (partX0 + partNX -1)] to out[0 ... (partNX-1)]
 * and helpfully wraps addresses circularly
 * invoke with gridDim.x * blockDim.x >= partNX
 */
__global__ void cukern_FetchPartitionSubset1D(double *in, int nodeN, double *out, int partX0, int partNX)
{
// calculate output address
int addrOut = threadIdx.x + blockDim.x * blockIdx.x;
if(addrOut >= partNX) return;

// Affine map back to input address
int addrIn = addrOut + partX0;
if(addrIn < 0) addrIn += partNX;

out[addrOut] = in[addrIn];
}
