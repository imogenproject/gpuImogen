/*
 * cudaSourceCylindricalTerms.cu
 *
 *  Created on: Jul 8, 2016
 *      Author: erik-k
 */

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

#include "cudaCommon.h"
#include "cudaSourceCylindricalTerms.h"

__device__ __constant__ double geoparam[8];
#define ARRAY_RIN geoparam[0]
#define ARRAY_DR  geoparam[1]

__device__ __constant__ int arrayparams[4];
#define ARRAY_NR arrayparams[0]
#define ARRAY_NPHI arrayparams[1]
#define ARRAY_NZ arrayparams[2]
#define ARRAY_SLABSIZE arrayparams[3]

__global__ void cukern_SourceCylindricalTerms(double *base, double dt);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if ((nrhs != 3) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaSourceCylindricalTerms(FluidManager, dt, run.geometry)\n");

    if(CHECK_CUDA_ERROR("entering cudaSourceCylindricalTerms") != SUCCESSFUL) { DROP_MEX_ERROR("Failed upon entry to cudaSourceScalarPotential."); }

    int status = SUCCESSFUL;

    // Get source array info and create destination arrays
    MGArray fluid[5];

    // Each partition uses the same common parameters
    GeometryParams geom = accessMatlabGeometryClass(prhs[2]);

    double dt = *mxGetPr(prhs[1]);
    double Rinner = geom.Rinner;

    int numFluids = mxGetNumberOfElements(prhs[0]);
	int fluidct;
	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		status = MGA_accessFluidCanister(prhs[0], fluidct, &fluid[0]);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) DROP_MEX_ERROR("cylindrical term source dumping: unable to access fluid.");

		status = sourcefunction_CylindricalTerms(&fluid[0], dt, &geom.h[0], Rinner);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) DROP_MEX_ERROR("cylindrical term source dumping: failed to apply source terms.");
	}
}

int sourcefunction_CylindricalTerms(MGArray *fluid, double dt, double *d3x, double Rinner)
{
	int returnCode = SUCCESSFUL;
	int apHost[4];

	int sub[6];

	int i;
	for(i = 0; i < fluid->nGPUs; i++) {
		calcPartitionExtent(fluid, i, &sub[0]);
		apHost[0] = sub[3];
		apHost[1] = sub[4];
		apHost[2] = sub[5];
		apHost[3] = fluid->slabPitch[i] / 8;

		double geo[2];
		geo[0] = Rinner;
		geo[1] = d3x[0];

		cudaSetDevice(fluid->deviceID[i]);
		returnCode = CHECK_CUDA_ERROR("Setting cuda device");
		if(returnCode != SUCCESSFUL) return returnCode;
		cudaMemcpyToSymbol(arrayparams, &apHost[0], 4*sizeof(int), 0, cudaMemcpyHostToDevice);
		returnCode = CHECK_CUDA_ERROR("Parameter constant upload");
		cudaMemcpyToSymbol(geoparam, &geo[0], 2*sizeof(double), 0, cudaMemcpyHostToDevice);
		returnCode = CHECK_CUDA_ERROR("Parameter constant upload");

		if(returnCode != SUCCESSFUL) return returnCode;

		dim3 blocksize; blocksize.x = blocksize.y = 16; blocksize.z = 1;
		dim3 gridsize; gridsize.x = ROUNDUPTO(sub[3], 16)/16;
		               gridsize.y = ROUNDUPTO(sub[4], 16)/16;
		               gridsize.z = 1;
		cukern_SourceCylindricalTerms<<<gridsize, blocksize>>>(fluid->devicePtr[i], dt);
		returnCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "Applying cylindrical source term: i=device#\n");
		if(returnCode != SUCCESSFUL) return returnCode;
	}

	return returnCode;
}

__global__ void cukern_SourceCylindricalTerms(double *base, double dt)
{

	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int z = 0;

	if(x >= ARRAY_NR) return;
	if(y >= ARRAY_NPHI) return;

	base += x + ARRAY_NR * y;

	int delta = ARRAY_NR * ARRAY_NPHI;

	double rho, u, v, w, E, P;
	double diffmom;
	double r = ARRAY_RIN + x*ARRAY_DR;

	while(z < ARRAY_NZ) {
		rho = base[0];
		E   = base[  ARRAY_SLABSIZE];
		u   = base[2*ARRAY_SLABSIZE];
		v   = base[3*ARRAY_SLABSIZE];

		diffmom = dt * v / (r * rho);

		base[2*ARRAY_SLABSIZE] = u + v * diffmom;
		base[3*ARRAY_SLABSIZE] = v - u * diffmom;

		z += 1; base += delta;
	}

}
