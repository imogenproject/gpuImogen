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

template <geometryType_t coords>
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

		int is_RZ = ((sub[4]==1)&&(sub[5]>1));

		cudaSetDevice(fluid->deviceID[i]);
		returnCode = CHECK_CUDA_ERROR("Setting cuda device");
		if(returnCode != SUCCESSFUL) return returnCode;
		cudaMemcpyToSymbol(arrayparams, &apHost[0], 4*sizeof(int), 0, cudaMemcpyHostToDevice);
		returnCode = CHECK_CUDA_ERROR("Parameter constant upload");
		cudaMemcpyToSymbol(geoparam, &geo[0], 2*sizeof(double), 0, cudaMemcpyHostToDevice);
		returnCode = CHECK_CUDA_ERROR("Parameter constant upload");

		if(returnCode != SUCCESSFUL) return returnCode;

		dim3 gridsize, blocksize;
		blocksize.x = blocksize.y = 16; blocksize.z = 1;
		gridsize.z = 1;

		if(is_RZ) {
			gridsize.x = ROUNDUPTO(sub[3], 16)/16;
			gridsize.y = ROUNDUPTO(sub[5], 16)/16;
			cukern_SourceCylindricalTerms<RZCYLINDRICAL><<<gridsize, blocksize>>>(fluid->devicePtr[i], dt);
		} else {
			gridsize.x = ROUNDUPTO(sub[3], 16)/16;
			gridsize.y = ROUNDUPTO(sub[4], 16)/16;
			cukern_SourceCylindricalTerms<CYLINDRICAL><<<gridsize, blocksize>>>(fluid->devicePtr[i], dt);
		}



		returnCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "Applying cylindrical source term: i=device#\n");
		if(returnCode != SUCCESSFUL) return returnCode;
	}

	return returnCode;
}

// The stage weights for 4th order Gauss-Legendre quadrature
#define GL4_A11 .25
#define GL4_A12 -0.038675134594812865529
#define GL4_A21 0.53867513459481286553
#define GL4_A22 .25
// The stage contribution weights
#define GL4_B1 .5
#define GL4_B2 .5
#define JACOBI_ITERS 2


template <geometryType_t coords>
__global__ void cukern_SourceCylindricalTerms(double *base, double dt)
{

	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int z = 0;

	if(x >= ARRAY_NR) return;

	base += x + ARRAY_NR * y;
	int delta = ARRAY_NR * ARRAY_NPHI;

	// For 3D, we run in R-Phi and step in Z. Otherwise, we do the whole (single RZ plane at once.
	switch(coords) {
	case CYLINDRICAL:
		if(y >= ARRAY_NPHI) return;
		z = 0;
		break;
	case RZCYLINDRICAL:
		if(y >= ARRAY_NZ) return;
		z = y;
		// We are stepping through many Zs at once, not just one
		delta = delta * blockDim.y*gridDim.y;
		break;
	}

	double rho, u, v, w, E, newP, vrA, vphiA, vrB, vphiB;
	double diffmom, diffmomB, scratch;
	double r = ARRAY_RIN + x*ARRAY_DR;

	int jacobiIter;

	while(z < ARRAY_NZ) {
		rho = base[0];

		u   = base[2*ARRAY_SLABSIZE]/rho; // vr
		v   = base[3*ARRAY_SLABSIZE]/rho; // vphi

		diffmom = dt * v / r;

		/* Make initial GL4 predictions */
		vrA   = u + GL4_A11* v * diffmom; // pr
		vphiA = v - GL4_A11* u * diffmom; // pphi

		vrB   = u + GL4_A12* v * diffmom; // pr
		vphiB = v - GL4_A12* u * diffmom; // pphi

		/* Take a few fixed point iterations */
		for(jacobiIter = 0; jacobiIter < JACOBI_ITERS; jacobiIter++) {
			diffmom = dt * vphiA / r;
			diffmomB= dt * vphiB / r;

			scratch = vrA;
			vrA   = u + GL4_A11* vphiA   * diffmom  + GL4_A12 * vphiB * diffmomB; // pr
			vphiA = v - GL4_A11* scratch * diffmom  + GL4_A12 * vrB   * diffmomB; // pphi

			diffmom = dt * vphiA / r;

			scratch = vrB;
			vrB   = u + GL4_A21* vphiA   * diffmom + GL4_A22 * vphiB   * diffmomB; // pr
			vphiB = v - GL4_A21* scratch * diffmom + GL4_A22 * scratch * diffmomB; // pphi
		}

		/* Finish the Runge-Kutta evaluation */
		u = rho*(u + .5*dt*(vphiA*vphiA + vphiB*vphiB)/r);
		v = rho*(v - .5*dt*(vphiA*vrA   + vphiB*vrB  )/r);

		// Analytically, the work integral is exactly zero so there is no source of Etotal or Einternal,
		// So the change in KE reflects itself as a source of Einternal
		// This can potentially get us into unhappiness
		E   = base[  ARRAY_SLABSIZE];
		w   = base[4*ARRAY_SLABSIZE];
		newP = E - .5*(u*u+v*v+w*w)/rho;
		if(newP <= 0.0) {
			base[ARRAY_SLABSIZE] = .5*(u*u+v*v+w*w)/rho + .00001*rho;
		}
		base[2*ARRAY_SLABSIZE] = u;
		base[3*ARRAY_SLABSIZE] = v;

		if(coords == RZCYLINDRICAL) {
			z += blockDim.y*gridDim.y;
		} else {
			z += 1;
		}
		base += delta;
	}

}
