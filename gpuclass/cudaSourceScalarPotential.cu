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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if ((nrhs!=6) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaApplyScalarPotential(FluidManager, phi, dt, GeometryManager, rho_nograv, rho_fullgrav)\n");

    if(CHECK_CUDA_ERROR("entering cudaSourceScalarPotential") != SUCCESSFUL) { DROP_MEX_ERROR("Failed upon entry to cudaSourceScalarPotential."); }
    
    // Get source array info and create destination arrays
    MGArray fluid[5];
    MGArray phi;
    int worked = MGA_accessMatlabArrays(prhs, 1, 1, &phi);
    if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) { DROP_MEX_ERROR("Failed to access input arrays."); }

    // Each partition uses the same common parameters
    double dt = *mxGetPr(prhs[2]);
    GeometryParams geom = accessMatlabGeometryClass(prhs[3]); // FIXME check for fail & return
    double rhoMinimum = *mxGetPr(prhs[4]); /* minimum rho, rho_c */
    double rhoFull    = *mxGetPr(prhs[5]); /* rho_g */

    int numFluids = mxGetNumberOfElements(prhs[0]);
    int fluidct;
// FIXME require separate rhomin/rho_fullg per fluid becuase they will generally have distinct characteristic scales of density.
    for(fluidct = 0; fluidct < numFluids; fluidct++) {
    	worked = MGA_accessFluidCanister(prhs[0], fluidct, &fluid[0]);
    	// FIXME check if this barfed?
    	mxArray *flprop = mxGetProperty(prhs[0], fluidct, "MINMASS");
    	if(flprop != NULL) {
    		rhoMinimum = *((double *)mxGetPr(flprop));
    	} else {
    		worked = ERROR_NULL_POINTER;
    	}
    	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;
    	worked = sourcefunction_ScalarPotential(&fluid[0], &phi, dt, geom, rhoMinimum, rhoFull);
    	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;
    }

}

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

    double *gradMem[fluid->nGPUs];

    // Iterate over all partitions, and here we GO!
    for(i = 0; i < fluid->nGPUs; i++) {
    	cudaSetDevice(fluid->deviceID[i]);
        calcPartitionExtent(fluid, i, sub);

        cudaMalloc((void **)&gradMem[i], 3*sub[3]*sub[4]*sub[5]*sizeof(double));

        arraysize.x = sub[3]; arraysize.y = sub[4]; arraysize.z = sub[5];

        blocksize = makeDim3(BLOCKDIMX, BLOCKDIMY, 1);
        gridsize.x = arraysize.x / (blocksize.x - 2); gridsize.x += ((blocksize.x-2) * gridsize.x < arraysize.x);
        gridsize.y = arraysize.y / (blocksize.y - 2); gridsize.y += ((blocksize.y-2) * gridsize.y < arraysize.y);
        gridsize.z = 1;

        if(isThreeD) {
        	if(geom.shape == SQUARE) {
        		cukern_computeScalarGradient3D<SQUARE><<<gridsize, blocksize>>>(phi->devicePtr[i], gradMem[i], gradMem[i]+fluid->partNumel[i], gradMem[i]+fluid->partNumel[i]*2, arraysize);
        	}
        	if(geom.shape == CYLINDRICAL) {
        		cukern_computeScalarGradient3D<CYLINDRICAL><<<gridsize, blocksize>>>(phi->devicePtr[i], gradMem[i], gradMem[i]+fluid->partNumel[i], gradMem[i]+fluid->partNumel[i]*2, arraysize);
        	}
        	worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "cukern_computeScalarGradient3D (square)");
        	if(worked != SUCCESSFUL) break;

        	cukern_applyPotentialGradient3D<<<32, 256>>>(fluid[0].devicePtr[i], gradMem[i], gradMem[i]+fluid->partNumel[i], gradMem[i]+fluid->partNumel[i]*2, fluid->partNumel[i]);
        	worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "cukern_applyPotentialGradient3D");
        	if(worked != SUCCESSFUL) break;
        } else {
        	if(geom.shape == SQUARE) {
        		cukern_computeScalarGradient2D<SQUARE><<<gridsize, blocksize>>>(phi->devicePtr[i], gradMem[i], gradMem[i]+fluid->partNumel[i], arraysize);
        	}
        	if(geom.shape == CYLINDRICAL) {
        		cukern_computeScalarGradient2D<CYLINDRICAL><<<gridsize, blocksize>>>(phi->devicePtr[i], gradMem[i], gradMem[i]+fluid->partNumel[i], arraysize);
        	}
        	worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "cukern_computeScalarGradient2D");
        	if(worked != SUCCESSFUL) break;

        	cukern_applyPotentialGradient2D<<<32, 256>>>(fluid[0].devicePtr[i], gradMem[i], gradMem[i]+fluid->partNumel[i], fluid->partNumel[i]);
        	worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "cukern_applyPotentialGradient2D");
        	if(worked != SUCCESSFUL) break;
        }

	cudaFree((void *)gradMem[i]);

    }

    return CHECK_IMOGEN_ERROR(worked);

}

/*
 * dP = -rho grad(phi) dt
 * dE = -rho v \cdot grad(phi) dt
 * 
 * Exact integrals at fixed position:
 * P2 = P1 - rho grad(phi) t
 * E2 = E1 - P1 \cdot grad(phi) t + .5 rho grad(phi) \cdot grad(phi) t^2
 *    = E1 - dt grad(phi) \cdot ( P1 - .5 * rho * grad(phi) )
 */
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


/*
 * dP = -rho grad(phi) dt
 * dE = -rho v \cdot grad(phi) dt
 * 
 * Exact integrals at fixed position:
 * P2 = P1 - rho grad(phi) t
 * E2 = E1 - P1 \cdot grad(phi) t + .5 rho grad(phi) \cdot grad(phi) t^2
 *    = E1 - dt grad(phi) \cdot ( P1 - .5 * rho * grad(phi) )
 */
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


/* Compute the gradient of 3d array phi, store the results in f_x, f_y and f_z
 *
 *    In cylindrical geometry, f_x -> f_r,
 *                             f_y -> f_phi
 */
template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient3D(double *phi, double *fx, double *fy, double *fz, int3 arraysize)
{
int myLocAddr = threadIdx.x + BLOCKDIMX*threadIdx.y;

int myX = threadIdx.x + (BLOCKDIMX-2)*blockIdx.x - 1;
int myY = threadIdx.y + (BLOCKDIMY-2)*blockIdx.y - 1;

if((myX > arraysize.x) || (myY > arraysize.y)) return;

bool IWrite = (threadIdx.x > 0) && (threadIdx.x < (BLOCKDIMX-1)) && (threadIdx.y > 0) && (threadIdx.y < (BLOCKDIMY-1));
IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.y);

myX = (myX + arraysize.x) % arraysize.x;
myY = (myY + arraysize.y) % arraysize.y;

int globAddr = myX + arraysize.x*myY;

double deltaphi; // Store derivative of phi in one direction

__shared__ double phiA[BLOCKDIMX*BLOCKDIMY];
__shared__ double phiB[BLOCKDIMX*BLOCKDIMY];
__shared__ double phiC[BLOCKDIMX*BLOCKDIMY];

double *U; double *V; double *W;
double *temp;

U = phiA; V = phiB; W = phiC;

// Preload lower and middle planes
U[myLocAddr] = phi[globAddr + arraysize.x*arraysize.y*(arraysize.z-1)];
V[myLocAddr] = phi[globAddr];

__syncthreads();

int z;
int deltaz = arraysize.x*arraysize.y;
for(z = 0; z < arraysize.z; z++) {
  if(z >= arraysize.z - 1) deltaz = - arraysize.x*arraysize.y*(arraysize.z-1);

  if(IWrite) {
    deltaphi         = LAMX*(V[myLocAddr+1]-V[myLocAddr-1]);
    fx[globAddr]     = deltaphi; // store px <- px - dt * rho dphi/dx;
  }

  if(IWrite) {
  if(coords == SQUARE) {
    deltaphi         = LAMY*(V[myLocAddr+BLOCKDIMX]-V[myLocAddr-BLOCKDIMX]);
    }
    if(coords == CYLINDRICAL) {
    // In cylindrical coords, use dt/dphi * (delta-phi) / r to get d/dy
    deltaphi         = LAMY*(V[myLocAddr+BLOCKDIMX]-V[myLocAddr-BLOCKDIMX]) / (RINNER + DELTAR*myX);
    }
    fy[globAddr]     = deltaphi;
  }

  W[myLocAddr]       = phi[globAddr + deltaz]; // load phi(z+1) -> phiC
  __syncthreads();
  deltaphi           = LAMZ*(W[myLocAddr] - U[myLocAddr]);

  if(IWrite) {
    fz[globAddr]     = deltaphi;
  }

  temp = U; U = V; V = W; W = temp; // cyclically shift them back
  globAddr += arraysize.x * arraysize.y;

}

}

/* Compute the gradient of 2d array phi, store the results in f_x, f_y
 *    In cylindrical geometry, f_x -> f_r,
 *                             f_y -> f_phi
 */
template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient2D(double *phi, double *fx, double *fy, int3 arraysize)
{
	int myLocAddr = threadIdx.x + BLOCKDIMX*threadIdx.y;

	int myX = threadIdx.x + (BLOCKDIMX-2)*blockIdx.x - 1;
	int myY = threadIdx.y + (BLOCKDIMY-2)*blockIdx.y - 1;

	if((myX > arraysize.x) || (myY > arraysize.y)) return;

	bool IWrite = (threadIdx.x > 0) && (threadIdx.x < (BLOCKDIMX-1)) && (threadIdx.y > 0) && (threadIdx.y < (BLOCKDIMY-1));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.y);

	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.y) % arraysize.y;

	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction
	__shared__ double phiLoc[BLOCKDIMX*BLOCKDIMY];

	phiLoc[myLocAddr] = phi[globAddr];

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		// compute dt * (dphi/dx)
		deltaphi         = LAMX*(phiLoc[myLocAddr+1]-phiLoc[myLocAddr-1]);
		fx[globAddr] = deltaphi;

		// Calculate dt*(dphi/dy)
		if(coords == SQUARE) {
		deltaphi         = LAMY*(phiLoc[myLocAddr+BLOCKDIMX]-phiLoc[myLocAddr-BLOCKDIMX]);
		}
		if(coords == CYLINDRICAL) {
		// Converts d/dphi into physical distance based on R
		deltaphi         = LAMY*(phiLoc[myLocAddr+BLOCKDIMX]-phiLoc[myLocAddr-BLOCKDIMX]) / (RINNER + myX*DELTAR);
		}
		fy[globAddr]     = deltaphi;
	}

}




/* Compute the gradient of 2d array phi, store the results in f_x, f_y
 *    In cylindrical geometry, f_x -> f_r,
 *                             f_y -> f_phi
 */
__global__ void  cukern_computeScalarGradientRZ(double *phi, double *fx, double *fz, int3 arraysize)
{
	int myLocAddr = threadIdx.x + BLOCKDIMX*threadIdx.y;

	int myX = threadIdx.x + (BLOCKDIMX-2)*blockIdx.x - 1;
	int myY = threadIdx.y + (BLOCKDIMY-2)*blockIdx.y - 1;

	if((myX > arraysize.x) || (myY > arraysize.z)) return;

	bool IWrite = (threadIdx.x > 0) && (threadIdx.x < (BLOCKDIMX-1)) && (threadIdx.y > 0) && (threadIdx.y < (BLOCKDIMY-1));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.z);

	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.z) % arraysize.z;

	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction
	__shared__ double phiLoc[BLOCKDIMX*BLOCKDIMY];

	phiLoc[myLocAddr] = phi[globAddr];

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		// compute dt * (dphi/dx)
		deltaphi         = LAMX*(phiLoc[myLocAddr+1]-phiLoc[myLocAddr-1]);
		fx[globAddr] = deltaphi;

		// Calculate dt*(dphi/dy)
		deltaphi         = LAMZ*(phiLoc[myLocAddr+BLOCKDIMX]-phiLoc[myLocAddr-BLOCKDIMX]);
		fz[globAddr]     = deltaphi;
	}

}


