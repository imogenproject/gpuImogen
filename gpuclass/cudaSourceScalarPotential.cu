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
__global__ void  cukern_applyScalarPotential(double *rho, double *E, double *px, double *py, double *pz, double *phi, int3 arraysize);
/*mass.gputag, mom(1).gputag, mom(2).gputag, mom(3).gputag, ener.gputag, run.potentialField.gputag, 2*run.time.dTime);*/

template <geometryType_t coords>
__global__ void  cukern_applyScalarPotential_2D(double *rho, double *E, double *px, double *py, double *phi, int3 arraysize);

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


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if ((nrhs!=6) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaApplyScalarPotential(FluidManager, phi, dt, GeometryManager, rhomin, rho_fullg)\n");

    if(CHECK_CUDA_ERROR("entering cudaSourceScalarPotential") != SUCCESSFUL) { DROP_MEX_ERROR("Failed upon entry to cudaSourceScalarPotential."); }
    
    // Get source array info and create destination arrays
    MGArray fluid[5];
    MGArray phi;
    int worked = MGA_accessMatlabArrays(prhs, 1, 1, &phi);
    if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) { DROP_MEX_ERROR("Failed to access input arrays."); }

    // Each partition uses the same common parameters
    double dt = *mxGetPr(prhs[2]);
    GeometryParams geom = accessMatlabGeometryClass(prhs[3]);
    double rhoMinimum = *mxGetPr(prhs[4]); /* minimum rho, rho_c */
    double rhoFull    = *mxGetPr(prhs[5]); /* rho_g */

    int numFluids = mxGetNumberOfElements(prhs[0]);
    int fluidct;
// FIXME require separate rhomin/rho_fullg per fluid becuase they will generally have distinct characteristic scales of density.
    for(fluidct = 0; fluidct < numFluids; fluidct++) {
    	worked = MGA_accessFluidCanister(prhs[0], fluidct, &fluid[0]);
#if 0
    	mxArray *flprop = mxGetProperty(prhs[0], fluidct, "MINMASS");
    	if(flprop != NULL) {
    		rhoMinimum = *((double *)mxGetPr(flprop));
    	} else {
    		worked = ERROR_NULL_POINTER;
    	}
#endif
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
    	cudaMemcpyToSymbol(devLambda, lambda, 9*sizeof(double), 0, cudaMemcpyHostToDevice);
    	worked = CHECK_CUDA_ERROR("cudaMemcpyToSymbol");
    	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;
    }

    if(worked != SUCCESSFUL) return worked;

    int isThreeD = (fluid->dim[2] > 1);

    // Iterate over all partitions, and here we GO!
    for(i = 0; i < fluid->nGPUs; i++) {
    	cudaSetDevice(fluid->deviceID[i]);
        calcPartitionExtent(fluid, i, sub);


        arraysize.x = sub[3]; arraysize.y = sub[4]; arraysize.z = sub[5];

        blocksize = makeDim3(BLOCKDIMX, BLOCKDIMY, 1);
        gridsize.x = arraysize.x / (blocksize.x - 2); gridsize.x += ((blocksize.x-2) * gridsize.x < arraysize.x);
        gridsize.y = arraysize.y / (blocksize.y - 2); gridsize.y += ((blocksize.y-2) * gridsize.y < arraysize.y);
        gridsize.z = 1;

        if(isThreeD) {
	if(geom.shape == SQUARE) { 
        cukern_applyScalarPotential<SQUARE><<<gridsize, blocksize>>>(
            fluid[0].devicePtr[i],
            fluid[1].devicePtr[i],
            fluid[2].devicePtr[i],
            fluid[3].devicePtr[i],
            fluid[4].devicePtr[i],
            phi->devicePtr[i], arraysize);
	    }
	    if(geom.shape == CYLINDRICAL) {
        cukern_applyScalarPotential<CYLINDRICAL><<<gridsize, blocksize>>>(
            fluid[0].devicePtr[i],
            fluid[1].devicePtr[i],
            fluid[2].devicePtr[i],
            fluid[3].devicePtr[i],
            fluid[4].devicePtr[i],
            phi->devicePtr[i], arraysize);
}
        } else {
	if(geom.shape == SQUARE) {
        	cukern_applyScalarPotential_2D<SQUARE><<<gridsize, blocksize>>>(
        	            fluid[0].devicePtr[i],
        	            fluid[1].devicePtr[i],
        	            fluid[2].devicePtr[i],
        	            fluid[3].devicePtr[i],
        	            phi->devicePtr[i], arraysize);
			    }
			    if(geom.shape == CYLINDRICAL) {
        	cukern_applyScalarPotential_2D<CYLINDRICAL><<<gridsize, blocksize>>>(
        	            fluid[0].devicePtr[i],
        	            fluid[1].devicePtr[i],
        	            fluid[2].devicePtr[i],
        	            fluid[3].devicePtr[i],
        	            phi->devicePtr[i], arraysize);

			    }

        }
        worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "scalar potential kernel");
        if(worked != SUCCESSFUL) break;
    }

    return CHECK_IMOGEN_ERROR(worked);

}

/*
 * dP = -rho grad(phi) dt
 * dE = -rho v \cdot grad(phi) dt
 */
template <geometryType_t coords>
__global__ void  cukern_applyScalarPotential(double *rho, double *E, double *px, double *py, double *pz, double *phi, int3 arraysize)
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
double rhomin = devLambda[3];

__shared__ double phiA[BLOCKDIMX*BLOCKDIMY];
__shared__ double phiB[BLOCKDIMX*BLOCKDIMY];
__shared__ double phiC[BLOCKDIMX*BLOCKDIMY];

__shared__ double locrho[BLOCKDIMX*BLOCKDIMY];
__shared__ double ener[BLOCKDIMX*BLOCKDIMY];

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

  ener[myLocAddr]   = 0;
  locrho[myLocAddr] = rho[globAddr]; // rho(z) -> rho
  W[myLocAddr]      = px[globAddr]; // load px(z) -> phiC
  __syncthreads();

  if(IWrite && (locrho[myLocAddr] > rhomin)) {
    deltaphi         = LAMX*(V[myLocAddr+1]-V[myLocAddr-1]);
    if(locrho[myLocAddr] < RHOGRAV) { deltaphi *= (locrho[myLocAddr]*G1 - G2); } // reduce G for low density
    ener[myLocAddr] -= deltaphi*W[myLocAddr]; // ener -= dt * px * dphi/dx
    px[globAddr]     = W[myLocAddr] - deltaphi*locrho[myLocAddr]; // store px <- px - dt * rho dphi/dx;
  }

  W[myLocAddr] = py[globAddr]; // load py(z) -> phiC
  __syncthreads();
  if(IWrite && (locrho[myLocAddr] > rhomin)) {
  if(coords == SQUARE) {
    deltaphi         = LAMY*(V[myLocAddr+BLOCKDIMX]-V[myLocAddr-BLOCKDIMX]);
    }
    if(coords == CYLINDRICAL) {
    // In cylindrical coords, use dt/dphi * (delta-phi) / r to get d/dy
    deltaphi         = LAMY*(V[myLocAddr+BLOCKDIMX]-V[myLocAddr-BLOCKDIMX]) / (RINNER + DELTAR*myX);
    }
   if(locrho[myLocAddr] < RHOGRAV) { deltaphi *= (locrho[myLocAddr]*G1 - G2); } // reduce G for low density
    ener[myLocAddr] -= deltaphi*W[myLocAddr]; // ener -= dt * py * dphi/dy
    py[globAddr]     = W[myLocAddr] - deltaphi*locrho[myLocAddr]; // store py <- py - rho dphi/dy;
  }

  W[myLocAddr]       = phi[globAddr + deltaz]; // load phi(z+1) -> phiC
  __syncthreads();
  deltaphi           = LAMZ*(W[myLocAddr] - U[myLocAddr]);
  if(locrho[myLocAddr] < RHOGRAV) { deltaphi *= (locrho[myLocAddr]*G1 - G2); } // reduce G for low density
  __syncthreads();

  U[myLocAddr]       = pz[globAddr]; // load pz(z) -> phiA
  __syncthreads();
  if(IWrite && (locrho[myLocAddr] > rhomin)) {
    E[globAddr]     += ener[myLocAddr] - deltaphi*U[myLocAddr]; // Store E[x] <- ener - dt *pz * dphi/dz
    pz[globAddr]     = U[myLocAddr] - deltaphi*locrho[myLocAddr]; // store pz <- pz - rho dphi/dz;
  }

  temp = U; U = V; V = W; W = temp; // cyclically shift them back
  globAddr += arraysize.x * arraysize.y;

}

}


/*
 * dP = -rho grad(phi) dt
 * dE = -rho v \cdot grad(phi) dt
 * 
 * Exact integrals at fixed position:
 * P2 = P1 - rho grad(phi) t
 * E2 = E1 - P1 \cdot grad(phi) t + .5 rho grad(phi) \cdot grad(phi) t^2
 */
template <geometryType_t coords>
__global__ void  cukern_applyScalarPotential_2D(double *rho, double *E, double *px, double *py, double *phi, int3 arraysize)
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
	double rhomin = devLambda[3];
	double tmpMom;

	__shared__ double phiLoc[BLOCKDIMX*BLOCKDIMY];
	__shared__ double rhoLoc[BLOCKDIMX*BLOCKDIMY];
	double enerLoc = 0.0;

	rhoLoc[myLocAddr] = rho[globAddr]; // rho(z) -> rho
	phiLoc[myLocAddr] = phi[globAddr];

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite && (rhoLoc[myLocAddr] > rhomin)) {
		// compute dt * (dphi/dx)
		deltaphi         = LAMX*(phiLoc[myLocAddr+1]-phiLoc[myLocAddr-1]);
		// reduce coupling for low densities
		if(rhoLoc[myLocAddr] < RHOGRAV) { deltaphi *= (rhoLoc[myLocAddr]*G1 - G2); }
		// Load px
		tmpMom = px[globAddr];
		// Store delta-E due to change in x momentum: ener -= (dt * dphi/dx) * (px = rho vx) -= rho delta-phi
		enerLoc -= deltaphi*(tmpMom - .5*rhoLoc[myLocAddr]*deltaphi);
		// Update X momentum
		px[globAddr]     = tmpMom - deltaphi*rhoLoc[myLocAddr]; // store px <- px - dt * rho dphi/dx;
		// Calculate dt*(dphi/dy)
		if(coords == SQUARE) {
		deltaphi         = LAMY*(phiLoc[myLocAddr+BLOCKDIMX]-phiLoc[myLocAddr-BLOCKDIMX]);
		}
		if(coords == CYLINDRICAL) {
		// Converts d/dphi into physical distance based on R
		deltaphi         = LAMY*(phiLoc[myLocAddr+BLOCKDIMX]-phiLoc[myLocAddr-BLOCKDIMX]) / (RINNER + myX*DELTAR);
		}
		
		// reduce G for low density
		if(rhoLoc[myLocAddr] < RHOGRAV) { deltaphi *= (rhoLoc[myLocAddr]*G1 - G2); }
		// Load py
		tmpMom = py[globAddr];
		// Update global energy array with this & previous delta-E values
		E[globAddr] += enerLoc - deltaphi*(tmpMom - .5*rhoLoc[myLocAddr]*deltaphi); // ener -= dt * py * dphi/dy
		// Update Y momentum array
		py[globAddr]     = tmpMom - deltaphi*rhoLoc[myLocAddr]; // store py <- py - rho dphi/dy;
	}



}

