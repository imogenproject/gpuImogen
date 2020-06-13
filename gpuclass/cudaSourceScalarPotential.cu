#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif
#ifndef NOMATLAB
#include "mex.h"
#endif

// CUDA
#include "cuda.h"

#include "cudaCommon.h"
#include "cudaGradientKernels.h"
#include "cudaSourceScalarPotential.h"

#define BLOCKDIMX 18
#define BLOCKDIMY 18

template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient3D(double *phi, double *f_x, double *f_y, double *f_z, int3 arraysize);

template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient2D(double *phi, double *fx, double *fy, int3 arraysize);

__global__ void  cukern_applyPotentialGradient3D(double *fluid, double *fx, double *fy, double *fz, unsigned int arrayNumel);
__global__ void  cukern_applyPotentialGradient2D(double *fluid, double *fx, double *fy, unsigned int arrayNumel);

__constant__ __device__ double devLambda[9];

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

#define RINNER devLambda[7]
#define DELTAR devLambda[8]

__constant__ __device__ unsigned int devSlabdim[3];

#ifdef STANDALONE_MEX_FUNCTION
int fetchMinDensity(mxArray *mxFluids, int fluidNum, double *rhoMin)
{
	int status = SUCCESSFUL;
	mxArray *flprop = mxGetProperty(mxFluids, fluidNum, "MINMASS");
	if(flprop != NULL) {
		rhoMin[0] = *((double *)mxGetPr(flprop));
	} else {
		PRINT_FAULT_HEADER;
		printf("Unable to access fluid(%i).MINMASS property.\n", fluidNum);
		PRINT_FAULT_FOOTER;
		status = ERROR_NULL_POINTER;
	}

	return status;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if ((nrhs!=4) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaApplyScalarPotential(FluidManager, phi, GeometryManager, [dt, rho_nograv, rho_fullgrav])\n");

    if(CHECK_CUDA_ERROR("entering cudaSourceScalarPotential") != SUCCESSFUL) { DROP_MEX_ERROR("Failed upon entry to cudaSourceScalarPotential."); }
    
    // Get source array info and create destination arrays
    MGArray fluid[5];
    MGArray phi;
    int worked = MGA_accessMatlabArrays(prhs, 1, 1, &phi);
    if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) { DROP_MEX_ERROR("Failed to access input arrays."); }

    // Each partition uses the same common parameters

    GeometryParams geom = accessMatlabGeometryClass(prhs[2]); // FIXME check for fail & return

    int ne = mxGetNumberOfElements(prhs[3]);
    if(ne != 3) {
    	printf("Input argument 3 has %i arguments, not three. Require precisely 3: [dt rho_nog rho_fullg]\n", ne);
    	DROP_MEX_ERROR("Crashing.");
    }

    double *sp = mxGetPr(prhs[3]);
    double dt         = sp[0]; /* dt */
    double rhoMinimum = sp[1]; /* minimum rho, rho_c */
    double rhoFull    = sp[2]; /* rho_g */

    int numFluids = mxGetNumberOfElements(prhs[0]);
    int fluidct;

    for(fluidct = 0; fluidct < numFluids; fluidct++) {
    	worked = MGA_accessFluidCanister(prhs[0], fluidct, &fluid[0]);
    	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;

    	mxArray *flprop = mxGetProperty(prhs[0], fluidct, "MINMASS");
    	if(flprop != NULL) {
    		rhoMinimum = *((double *)mxGetPr(flprop));
    	} else {
    		worked = ERROR_NULL_POINTER;
    		break;
    	}

    	worked = sourcefunction_ScalarPotential(&fluid[0], &phi, dt, geom, rhoMinimum, rhoFull);
    	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;
    }

    GridFluid fluids[numFluids];
    	for(fluidct = 0; fluidct < numFluids; fluidct++) {
    		worked = MGA_accessFluidCanister(prhs[0], fluidct, &fluids[fluidct].data[0]);
    		if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;
    		//fluids[fluidct].thermo = accessMatlabThermoDetails(mxGetProperty(prhs[0], fluidct, "thermoDetails"));
    		worked = fetchMinDensity((mxArray *)prhs[0], fluidct, &fluids[fluidct].rhoMin);
    	}

    if(worked != SUCCESSFUL) { DROP_MEX_ERROR("cudaSourceScalarPotential failed"); }

}
#endif

int sourcefunction_ScalarPotential(MGArray *fluid, MGArray *phi, double dt, GeometryParams geom, double minRho, double rhoFullGravity)
{
    double *dx = &geom.h[0];

    dim3 gridsize, blocksize;
    int3 arraysize;
    int i, sub[6];
    int worked;

    double lambda[9];
    lambda[0] = dt/(2.0*dx[0]);
    lambda[1] = dt/(2.0*dx[1]);
    lambda[2] = dt/(2.0*dx[2]);
    lambda[3] = minRho; /* minimum rho, rho_c */
    lambda[4] = rhoFullGravity; /* rho_g */

    lambda[5] = 1.0/(lambda[4] - lambda[3]); /* 1/(rho_g - rho_c) */
    lambda[6] = lambda[3]*lambda[5];

    lambda[7] = geom.Rinner;
    lambda[8] = dx[1];

    for(i = 0; i < fluid->nGPUs; i++) {
    	cudaSetDevice(fluid->deviceID[i]);
    	cudaMemcpyToSymbol((const void *)devLambda, lambda, 9*sizeof(double), 0, cudaMemcpyHostToDevice);
    	unsigned int sd[3];
    	sd[0] = (unsigned int)(fluid->slabPitch[i] / 8);
    	cudaMemcpyToSymbol((const void *)devSlabdim, sd, 1*sizeof(int), 0, cudaMemcpyHostToDevice);

    	worked = CHECK_CUDA_ERROR("cudaMemcpyToSymbol");

    	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;
    }

    if(worked != SUCCESSFUL) return worked;

    int isThreeD = (fluid->dim[2] > 1);

    MGArray gradientStorage;
    worked = MGA_allocSlab(fluid, &gradientStorage, 3);
    if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) return worked;
    worked = computeCentralGradient(phi, &gradientStorage, geom, 2, dt);
    if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) return worked;

    double *gs;
    // Iterate over all partitions, and here we GO!
    for(i = 0; i < fluid->nGPUs; i++) {
    	cudaSetDevice(fluid->deviceID[i]);
        calcPartitionExtent(fluid, i, sub);

        arraysize.x = sub[3]; arraysize.y = sub[4]; arraysize.z = sub[5];

        blocksize = makeDim3(BLOCKDIMX, BLOCKDIMY, 1);
        gridsize.x = arraysize.x / (blocksize.x - 2); gridsize.x += ((blocksize.x-2) * gridsize.x < arraysize.x);
        gridsize.y = arraysize.y / (blocksize.y - 2); gridsize.y += ((blocksize.y-2) * gridsize.y < arraysize.y);
        gridsize.z = 1;

        gs = gradientStorage.devicePtr[i];

        if(isThreeD) {
        	cukern_applyPotentialGradient3D<<<32, 256>>>(fluid[0].devicePtr[i], gs, gs+fluid->slabPitch[i]/8, gs+2*fluid->slabPitch[i]/8, fluid->partNumel[i]);
        	worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "cukern_applyPotentialGradient3D");
        	if(worked != SUCCESSFUL) break;
        } else {

        	cukern_applyPotentialGradient2D<<<32, 256>>>(fluid[0].devicePtr[i], gs, gs+fluid->slabPitch[i]/8, fluid->partNumel[i]);
        	worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "cukern_applyPotentialGradient2D");
        	if(worked != SUCCESSFUL) break;
        }
    }

    if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) return worked;
    worked = MGA_delete(&gradientStorage);

    return CHECK_IMOGEN_ERROR(worked);
}

/* dP = -rho grad(phi) dt
 * dE = -rho v \cdot grad(phi) dt
 * 
 * Exact integrals at fixed position:
 * P2 = P1 - rho grad(phi) t
 * E2 = E1 - P1 \cdot grad(phi) t + .5 rho grad(phi) \cdot grad(phi) t^2
 *    = E1 - dt grad(phi) \cdot ( P1 - .5 * rho * grad(phi) ) */
__global__ void  cukern_applyPotentialGradient3D(double *fluid, double *fx, double *fy, double *fz, unsigned int arrayNumel)
{
unsigned int globAddr = threadIdx.x + blockDim.x*blockIdx.x;

if(globAddr >= arrayNumel) return;

double deltaphi; // Store derivative of phi in one direction
double rhomin = devLambda[3];

double locrho, ener, mom;

for(; globAddr < arrayNumel; globAddr += blockDim.x*gridDim.x) {
	ener   = 0;
	locrho = fluid[globAddr]; // rho(z) -> rho
	if(locrho > rhomin) {
		mom      = fluid[globAddr + 2*devSlabdim[0]]; // load px(z) -> phiC
		deltaphi = fx[globAddr];
		if(locrho < RHOGRAV) { deltaphi *= (locrho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
		ener                               = -deltaphi*(mom-.5*locrho*deltaphi); // exact KE change
		fluid[globAddr + 2*devSlabdim[0]]     = mom - deltaphi*locrho;   // store px <- px - dt * rho dphi/dx;

		mom      = fluid[globAddr + 3*devSlabdim[0]]; // load py(z) -> phiC
		deltaphi = fy[globAddr];
		if(locrho < RHOGRAV) { deltaphi *= (locrho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
		ener                           -= deltaphi*(mom-.5*locrho*deltaphi); // exact KE change
		fluid[globAddr + 3*devSlabdim[0]]  = mom - deltaphi*locrho;   // store px <- px - dt * rho dphi/dx;

		mom      = fluid[globAddr + 4*devSlabdim[0]];
		deltaphi = fz[globAddr];
		if(locrho < RHOGRAV) { deltaphi *= (locrho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
		ener                           -= deltaphi*(mom-.5*locrho*deltaphi); // exact KE change
		fluid[globAddr + 4*devSlabdim[0]]  = mom - deltaphi*locrho;   // store px <- px - dt * rho dphi/dx;

		// Store changed kinetic energy
		fluid[globAddr + devSlabdim[0]]   += ener;
	}
}

}

/* dP = -rho grad(phi) dt
 * dE = -rho v \cdot grad(phi) dt
 * 
 * Exact integrals at fixed position:
 * P2 = P1 - rho grad(phi) t
 * E2 = E1 - P1 \cdot grad(phi) t + .5 rho grad(phi) \cdot grad(phi) t^2
 *    = E1 - dt grad(phi) \cdot ( P1 - .5 * rho * grad(phi) ) */
__global__ void  cukern_applyPotentialGradient2D(double *fluid, double *fx, double *fy, unsigned int arrayNumel)
{

unsigned int globAddr = threadIdx.x + blockDim.x*blockIdx.x;

if(globAddr >= arrayNumel) return;

double deltaphi; // Store derivative of phi in one direction
double rhomin = devLambda[3];

double locrho, ener, mom;

for(; globAddr < arrayNumel; globAddr += blockDim.x*gridDim.x) {
	ener   = 0;
	locrho = fluid[globAddr]; // rho(z) -> rho
	if(locrho > rhomin) {
		mom      = fluid[globAddr + 2*devSlabdim[0]]; // load px(z) -> phiC
		deltaphi = fx[globAddr];
		if(locrho < RHOGRAV) { deltaphi *= (locrho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
		ener                            = -deltaphi*(mom-.5*locrho*deltaphi); // exact KE change
		fluid[globAddr + 2*devSlabdim[0]]  = mom - deltaphi*locrho;   // store px <- px - dt * rho dphi/dx;

		mom      = fluid[globAddr + 3*devSlabdim[0]]; // load py(z) -> phiC
		deltaphi = fy[globAddr];
		if(locrho < RHOGRAV) { deltaphi *= (locrho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
		ener                           -= deltaphi*(mom-.5*locrho*deltaphi); // exact KE change
		fluid[globAddr + 3*devSlabdim[0]]  = mom - deltaphi*locrho;   // store px <- px - dt * rho dphi/dx;

		// Store change to kinetic energy
		fluid[globAddr + devSlabdim[0]]   += ener;
	}
}

}

