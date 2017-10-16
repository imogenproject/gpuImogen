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
#define ARRAY_NX       arrayparams[0]
#define ARRAY_NYZ      arrayparams[1]
#define ARRAY_NUMEL    arrayparams[2]
#define ARRAY_SLABSIZE arrayparams[3]

__device__ __constant__ double devGeom[4];
#define GEOM_V0    devGeom[0]
#define GEOM_DV0    devGeom[1]
#define GEOM_V1    devGeom[2]
#define GEOM_DV1    devGeom[3]


__global__ void cukern_SourceVacuumTaffyOperator_IRF(double *base, double momfactor, double rhofactor, double rhoCrit, double rhoMin);

__global__ void cukern_SourceVacuumTaffyOperator_CylRRF(double *base, double momfactor, double rhofactor, double rhoCrit, double rhoMin);

/* Calculate the whimsically named "vacuum taffy operator",
 *	    [\rho  ]   [0	                                                    ]
 * \partial_t [\rho v] ~ [-\alpha \rho v \theta(\rho - \rho_c)		         ]
 *	    [Etot  ]   [(-\alpha \rho v.v - \beta (P - \rho T_0)\theta(\rho - \rho_c)]
 * if density is less than \rho_c, momentum exponentially decays to zero (in inertial frame)
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

    if ((nrhs != 3) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaSourceVTO(FluidManager, [dt alpha beta, omega], GeometryManager)\n");

    if(CHECK_CUDA_ERROR("entering cudaSourceVTO") != SUCCESSFUL) { DROP_MEX_ERROR("Failed before entry to cudaSourceVTO."); }

    int status = SUCCESSFUL;

    // Get source array info and create destination arrays
    MGArray fluid[5];

    double *fltargs = mxGetPr(prhs[1]);
    int N = mxGetNumberOfElements(prhs[1]);
    if(N != 4) {
	DROP_MEX_ERROR("Require [dt alpha beta frame_omega] in 2nd argument: Got other than 4 values\n");
    }

    double dt    = fltargs[0];
    double alpha = fltargs[1];
    double beta  = fltargs[2];
    double frameW= fltargs[3];
    double rho_c = 0;

    GeometryParams geo = accessMatlabGeometryClass(prhs[2]);

    int numFluids = mxGetNumberOfElements(prhs[0]);
	int fluidct;
	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		status = MGA_accessFluidCanister(prhs[0], fluidct, &fluid[0]);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) DROP_MEX_ERROR("vacuum taffy operator dumping: unable to access fluid.");

		const mxArray *flprop;
		flprop = mxGetProperty(prhs[0], fluidct, "MINMASS");
		if(flprop != NULL) {
			fltargs = mxGetPr(flprop);
			if(fltargs == NULL) {
				status = ERROR_NULL_POINTER;
			} else {
				rho_c = *fltargs;
			}
		} else {
			status = ERROR_INVALID_ARGS;
		}
		if(status != SUCCESSFUL) {
			PRINT_FAULT_HEADER;
			printf("Unable to fetch fluids(%i).MINMASS property! Abort run.\n", fluidct);
			PRINT_FAULT_FOOTER;
			DROP_MEX_ERROR("Crashing.\n");
		}

		status = sourcefunction_VacuumTaffyOperator(&fluid[0], dt, alpha, beta, frameW, rho_c, geo);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) DROP_MEX_ERROR("vacuum taffy operator dumping: failed to apply source terms.");
	}
}

int sourcefunction_VacuumTaffyOperator(MGArray *fluid, double dt, double alpha, double beta, double frameOmega, double minimumDensity, GeometryParams geo)
{
	int returnCode = SUCCESSFUL;
	int apHost[4];
	double hostGeomInfo[4];
	double criticalDensity = 5*minimumDensity;

	int sub[6];

	int i;
	for(i = 0; i < fluid->nGPUs; i++) {
		calcPartitionExtent(fluid, i, &sub[0]);
		apHost[0] = sub[3];
		apHost[1] = sub[4]*sub[5];
		apHost[2] = sub[3]*sub[4]*sub[5];
		apHost[3] = fluid->slabPitch[i] / 8;

		cudaSetDevice(fluid->deviceID[i]);
		returnCode = CHECK_CUDA_ERROR("Setting cuda device");
		if(returnCode != SUCCESSFUL) return returnCode;
		cudaMemcpyToSymbol((const void *)arrayparams, &apHost[0], 4*sizeof(int), 0, cudaMemcpyHostToDevice);
		returnCode = CHECK_CUDA_ERROR("Parameter constant upload");
		if(returnCode != SUCCESSFUL) return returnCode;
	
		hostGeomInfo[0] = -frameOmega*geo.x0;
		hostGeomInfo[1] = -frameOmega*geo.h[0];
		hostGeomInfo[2] = -frameOmega*geo.y0;
		hostGeomInfo[3] = -frameOmega*geo.h[1];
		cudaMemcpyToSymbol((const void *)devGeom, &hostGeomInfo[0], 4*sizeof(double), 0, cudaMemcpyHostToDevice);
	}

	for(i = 0; i < fluid->nGPUs; i++) {
		cudaSetDevice(fluid->deviceID[i]);
		calcPartitionExtent(fluid, i, &sub[0]);
		dim3 gridsize;
		gridsize.z = 1;
	
		// X grid spans X
		// Rest walks Y/Z
		dim3 blocksize(128, 1, 1);

		if(frameOmega != 0.0) {
			int n_yz = sub[4]*sub[5];
			gridsize.x = ROUNDUPTO(sub[3], 128)/128;
			gridsize.y = n_yz >= 32 ? 32 : n_yz;
			switch(geo.shape) {
			case SQUARE:
				PRINT_FAULT_HEADER;
				printf("VTO operator cannot currently handle rotating frame in square coordinates.\n");
				returnCode = ERROR_NOIMPLEMENT;
				PRINT_FAULT_FOOTER;
				break;
			case CYLINDRICAL:
				cukern_SourceVacuumTaffyOperator_CylRRF<<<gridsize, blocksize>>>(fluid->devicePtr[i], exp(-alpha*dt), exp(-beta*dt), criticalDensity, minimumDensity);
				break;
			}
		} else {
			gridsize.x = ROUNDUPTO(sub[3], 128)/128;
			gridsize.y = 1;
			cukern_SourceVacuumTaffyOperator_IRF<<<gridsize, blocksize>>>(fluid->devicePtr[i], exp(-alpha*dt), exp(-beta*dt), criticalDensity, minimumDensity);
		}
		if(returnCode != SUCCESSFUL) return returnCode;
		returnCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "Applying cylindrical source term: i=device#\n");
		if(returnCode != SUCCESSFUL) return returnCode;
	}

	return returnCode;
}


/* The more complex VTO must be aware of geometry to consider the rotation term when computing inertial rest
 * frame velocity to decay towards. */
__global__ void cukern_SourceVacuumTaffyOperator_CylRRF(double *base, double momfactor, double rhofactor, double rhoCrit, double rhoMin)
{
	// Cut off x threads at Nr
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	if(x >= ARRAY_NX) return;

	// Calculate the phi velocity for this thread's (fixed) R corresponding to inertial rest
	// A positive omega corresponds to negative vphi_rest
	double vphi_rest = GEOM_V0 + x*GEOM_DV0;

	// Note that microscopic Nphi*Nz was checked for before kernel invocation,
	// such that this will not overrun bounds
	base += x + ARRAY_NX*(threadIdx.y + blockDim.y*blockIdx.y);
	x = blockDim.y*blockIdx.y;

	double rho, u, v, w, E, f0, v0, P;

	momfactor /= rhofactor; // convert to velocity decay factor

	while(x < ARRAY_NYZ) {
		rho = base[0];

		if(rho < rhoCrit) {
			// load remaining data
			E   = base[  ARRAY_SLABSIZE];
			u   = base[2*ARRAY_SLABSIZE] / rho;
			v   = base[3*ARRAY_SLABSIZE] / rho;
			w   = base[4*ARRAY_SLABSIZE] / rho;

			// compute squared momentum & decay factor
			v0 = rho*(u*u+v*v+w*w); // velocity^2

			// decay velocity away
			u=u*momfactor;
			v = v - vphi_rest; // add frame velocity
			v *= momfactor;
			v = v + vphi_rest;
			w=w*momfactor;

			P = (E - .5*v0)/rho; // = T / (gamma-1)

			// decay away density?
			f0 = rho*rhofactor;

			if(f0 > rhoMin) {
				// exponentially so.
				rho = f0;
			} else {
				// only to the limit...
				rho = rhoMin;
			}

			// Decay temperature by rhofactor
			E = P*rhofactor*rho; // Eint = rho_new * T_new
			
			if(E < 1e-5 * rho) E = 1e-5 * rho; // Prevent T < 1e-5

			// Add new kinetic energy to new Etotal
			E += .5*rho*(u*u+v*v+w*w);

			// write vars back out
			base[0]		= rho;
			base[  ARRAY_SLABSIZE] = E;
			base[2*ARRAY_SLABSIZE] = u*rho;
			base[3*ARRAY_SLABSIZE] = v*rho;
			base[4*ARRAY_SLABSIZE] = w*rho;
		}

		// advance to next locations in array
		x += (blockDim.y*gridDim.y);
		base += ARRAY_NX*(blockDim.y*gridDim.y);
	}

}

/* The non-rotating-frame VTO is agnostic to coordinates and therefore simpler/faster */
__global__ void cukern_SourceVacuumTaffyOperator_IRF(double *base, double momfactor, double rhofactor, double rhoCrit, double rhoMin)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;

	if(x >= ARRAY_NUMEL) return;

	base += x;
	int delta = blockDim.x*gridDim.x;

	double rho, u, v, w, E, P, v0, f0;

	while(x < ARRAY_NUMEL) {
		rho = base[0];

		if(rho < rhoCrit) {
			// load remaining data
			E   = base[  ARRAY_SLABSIZE];
			u   = base[2*ARRAY_SLABSIZE];
			v   = base[3*ARRAY_SLABSIZE];
			w   = base[4*ARRAY_SLABSIZE];

			// compute squared momentum & decay factor
			v0 = (u*u+v*v+w*w)/(rho*rho); // velocity^2

			// decay velocity away
			u=u*momfactor;
			v=v*momfactor;
			w=w*momfactor;

			// Compute original Einternal
			P = E - .5*v0*rho; // internal energy in P

			// decay away density?
			f0 = rho*rhofactor;

			if(f0 > rhoMin) {
				// exponentially so.
				rho = f0;
				u*=rhofactor;
				v*=rhofactor;
				w*=rhofactor;
				// Decay Eint by that factor, and again (T decays at same rate)
				E = P*rhofactor;
			} else {
				// only to the limit...
				//f0 = rhoMin/rho;
				//u *= f0; v *= f0; w *= f0;
				E = P*rhofactor;
				//rho = rhoMin;
			}

			if(E < 1e-5 * rho) E = 1e-5 * rho;
			// Add new kinetic energy to Etotal
			E += .5*v0*momfactor*momfactor*rho;

			// write vars back out
			base[0]		= rho;
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
