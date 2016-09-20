/*
 * cudaSourceVTO.cu
 *
 *  Created on: Sep 8, 2016
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
#include "cudaSourceVTO.h"

__device__ __constant__ int arrayparams[4];
#define ARRAY_NUMEL arrayparams[0]
#define ARRAY_SLABSIZE arrayparams[1]

__global__ void cukern_SourceVacuumTaffyOperator(double *base, double dt, double alpha, double beta, double rhoCrit);

/* Calculate the whimsically named "vacuum taffy operator",
 *            [\rho  ]   [0                                                            ]
 * \partial_t [\rho v] ~ [-\alpha \rho v \theta(\rho - \rho_c)                         ]
 *            [Etot  ]   [(-\alpha \rho v.v - \beta (P - \rho T_0)\theta(\rho - \rho_c)]
 * if density is less than \rho_c, momentum exponentially decays to zero
 * and temperature decays to T_0.
 * i.e., regions evacuated below a sufficiently small density stop moving & become isothermal.
 *
 * This unphysical operation is implemented because the transient "puff" launched off the surface of
 * disk simulations leaves behind an intense rarefaction.
 *
 * Material originally at Keplerian equilibrium at small R migrates into this rarefaction,
 * which reduces its density until gravity switches off (must not vacuum matter onto grid),
 * which causes it to fly out, and suck more material in,
 * thereby effectively blowtorching the inner face of the disk.
 */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if ((nrhs != 2) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaSourceVTO(FluidManager, [dt alpha beta rho_crit])\n");

    if(CHECK_CUDA_ERROR("entering cudaSourceVTO") != SUCCESSFUL) { DROP_MEX_ERROR("Failed upon entry to cudaSourceVTO."); }

    int status = SUCCESSFUL;

    // Get source array info and create destination arrays
    MGArray fluid[5];

    double *fltargs = mxGetPr(prhs[1]);
    // FIXME: check number of elements

    double dt    = fltargs[0];
    double alpha = fltargs[1];
    double beta  = fltargs[2];
    double rho_c = fltargs[3];

    int numFluids = mxGetNumberOfElements(prhs[0]);
	int fluidct;
	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		status = MGA_accessFluidCanister(prhs[0], fluidct, &fluid[0]);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) DROP_MEX_ERROR("vacuum taffy operator dumping: unable to access fluid.");

		status = sourcefunction_VacuumTaffyOperator(&fluid[0], dt, alpha, beta, rho_c);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) DROP_MEX_ERROR("vacuum taffy operator dumping: failed to apply source terms.");
	}
}

int sourcefunction_VacuumTaffyOperator(MGArray *fluid, double dt, double alpha, double beta, double criticalDensity)
{
	int returnCode = SUCCESSFUL;
	int apHost[4];

	int sub[6];

	int i;
	for(i = 0; i < fluid->nGPUs; i++) {
		calcPartitionExtent(fluid, i, &sub[0]);
		apHost[0] = sub[3]*sub[4]*sub[5];
		apHost[1] = fluid->slabPitch[i] / 8;

		cudaSetDevice(fluid->deviceID[i]);
		returnCode = CHECK_CUDA_ERROR("Setting cuda device");
		if(returnCode != SUCCESSFUL) return returnCode;
		cudaMemcpyToSymbol(arrayparams, &apHost[0], 2*sizeof(int), 0, cudaMemcpyHostToDevice);
		returnCode = CHECK_CUDA_ERROR("Parameter constant upload");
		if(returnCode != SUCCESSFUL) return returnCode;

		dim3 blocksize(256, 1, 1);
		dim3 gridsize(32, 1, 1);

		if(blocksize.x*gridsize.x > fluid->partNumel[i]) {
			gridsize.x = fluid->partNumel[i] / blocksize.x / 2;
			if(gridsize.x < 1) gridsize.x = 1;
		}

		cukern_SourceVacuumTaffyOperator<<<gridsize, blocksize>>>(fluid->devicePtr[i], dt, alpha, beta, criticalDensity);
		returnCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "Applying cylindrical source term: i=device#\n");
		if(returnCode != SUCCESSFUL) return returnCode;
	}

	return returnCode;
}

__global__ void cukern_SourceVacuumTaffyOperator(double *base, double dt, double alpha, double beta, double rhoCrit)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;

	if(x >= ARRAY_NUMEL) return;

	base += x;
	int delta = blockDim.x*gridDim.x;

	double rho, u, v, w, E, P, v0, f0;
	f0 = exp(-alpha*dt); // FIXME: just pass this & exp(-beta dt) instead of alpha/beta/dt... DERP

	while(x < ARRAY_NUMEL) {
		rho = base[0];

		if(rho < rhoCrit) {
			// load remaining data
			E   = base[  ARRAY_SLABSIZE];
			u   = base[2*ARRAY_SLABSIZE];
			v   = base[3*ARRAY_SLABSIZE];
			w   = base[4*ARRAY_SLABSIZE];

			// compute squared momentum & decay factor
			v0 = u*u+v*v+w*w;
			f0 = exp(-alpha*dt);

			// decay momentum
			u=u*f0;
			v=v*f0;
			w=w*f0;

			// apply same change to Uinternal as well
			P = E - .5*v0/rho; // internal energy in P
			E = .5*v0*f0*f0/rho + P; // new Etotal

			// write vars back out
			base[  ARRAY_SLABSIZE] = E;
			base[2*ARRAY_SLABSIZE] = u;
			base[3*ARRAY_SLABSIZE] = v;
			base[4*ARRAY_SLABSIZE] = w;
		}

		// advance to next locations in array
		x += delta;
		base += delta;
	}

}
