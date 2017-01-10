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

#define GRADBLOCKX 18
#define GRADBLOCKY 18

#define SRCBLOCKX 16
#define SRCBLOCKY 16

int sourcefunction_Composite(MGArray *fluid, MGArray *phi, MGArray *XYVectors, GeometryParams geom, double minRho, double rhoFullGravity, double omega, double dt);

__global__ void writeScalarToVector(double *x, long numel, double f);

template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient3D(double *phi, double *f_x, double *f_y, double *f_z, int3 arraysize);

template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient2D(double *phi, double *fx, double *fy, int3 arraysize);

__global__ void  cukern_computeScalarGradientRZ(double *phi, double *fx, double *fz, int3 arraysize);

template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient2D_h4(double *phi, double *fx, double *fy, int3 arraysize);

__global__ void  cukern_computeScalarGradientRZ_h4(double *phi, double *fx, double *fz, int3 arraysize);

__global__ void cukern_FetchPartitionSubset1D(double *in, int nodeN, double *out, int partX0, int partNX);

template <geometryType_t coords>
__global__ void  cukern_sourceComposite_IMP(double *fluidIn, double *Rvector, double *gravgrad, long pitch);

template <geometryType_t coords>
__global__ void  cukern_sourceComposite_RK4(double *fluidIn, double *Rvector, double *gravgrad, long pitch);


__constant__ __device__ double devLambda[12];

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

// __constant__ parameters for the rotating frame terms
#define OMEGA devLambda[9]
#define DT devLambda[10]
#define TWO_OMEGA_T devLambda[11]

__constant__ __device__ int devIntParams[3];


#define DERIV4

// How much to subtract from the derivative block
#ifdef DERIV4
#define FROMBLK 4
#else
#define FROMBLK 2
#endif


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// At least 2 arguments expected
	// Input and result
	if ((nrhs!=8) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaSourceRotatingFrame(FluidManager, phi, omega, dt, GeometryManager, rhomin, rho_fullg, [xvector yvector])\n");

	CHECK_CUDA_ERROR("entering cudaSourceRotatingFrame");

	// Get source array info and create destination arrays
	MGArray fluid[5], gravPot, xyvec;

	/* FIXME: accept this as a matlab array instead
	 * FIXME: Transfer appropriate segments to __constant__ memory
	 * FIXME: that seems the only reasonable way to avoid partitioning hell
	 */
	double omega = *mxGetPr(prhs[2]);
	double dt    = *mxGetPr(prhs[3]);
	double rhomin= *mxGetPr(prhs[5]);
	double rhofg = *mxGetPr(prhs[6]);
	GeometryParams geom = accessMatlabGeometryClass(prhs[4]);

	int status;

	status = MGA_accessMatlabArrays(prhs, 7, 7, &xyvec);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { DROP_MEX_ERROR("Failed to access X-Y vector."); }

	status = MGA_accessMatlabArrays(prhs, 1, 1, &gravPot);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { DROP_MEX_ERROR("Failed to access gravity potential array."); }

	dim3 gridsize, blocksize;
	int3 arraysize;

	int numFluids = mxGetNumberOfElements(prhs[0]);
	int fluidct;

	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		status = MGA_accessFluidCanister(prhs[0], fluidct, &fluid[0]);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;

		status = sourcefunction_Composite(&fluid[0], &gravPot, &xyvec, geom, rhomin, rhofg, omega, dt);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { DROP_MEX_ERROR("Failed to apply rotating frame source terms."); }
	}

}


int sourcefunction_Composite(MGArray *fluid, MGArray *phi, MGArray *XYVectors, GeometryParams geom, double minRho, double rhoFullGravity, double omega, double dt)
{
	dim3 gridsize, blocksize;
	int3 arraysize;

	double lambda[11];

	int i;
	int worked;

	double *devXYset[fluid->nGPUs];
	int sub[6];

    double *dx = &geom.h[0];
#ifdef DERIV4
    lambda[0] = dt/(12.0*dx[0]);
    lambda[1] = dt/(12.0*dx[1]);
    lambda[2] = dt/(12.0*dx[2]);
#else
    lambda[0] = dt/(2.0*dx[0]);
    lambda[1] = dt/(2.0*dx[1]);
    lambda[2] = dt/(2.0*dx[2]);
#endif
    lambda[3] = minRho; /* minimum rho, rho_c */
    lambda[4] = rhoFullGravity; /* rho_g */

    lambda[5] = 1.0/(lambda[4] - lambda[3]); /* 1/(rho_g - rho_c) */
    lambda[6] = lambda[3]*lambda[5];

    lambda[7] = geom.Rinner;
    lambda[8] = dx[1];

	lambda[9] = omega;
	lambda[10]= dt;

	int isThreeD = (fluid->dim[2] > 1);
	int isRZ = (fluid->dim[2] > 1) & (fluid->dim[1] == 1);

    for(i = 0; i < fluid->nGPUs; i++) {
    	cudaSetDevice(fluid->deviceID[i]);
    	cudaMemcpyToSymbol(devLambda, lambda, 11*sizeof(double), 0, cudaMemcpyHostToDevice);
    	worked = CHECK_CUDA_ERROR("cudaMemcpyToSymbol");
    	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;

    	calcPartitionExtent(fluid, i, &sub[0]);

    	cudaMemcpyToSymbol(devIntParams, &sub[3], 3*sizeof(int), 0, cudaMemcpyHostToDevice);
    	worked = CHECK_CUDA_ERROR("memcpy to symbol");
    	if(worked != SUCCESSFUL) break;
    }

    if(worked != SUCCESSFUL) return worked;

    double *gradMem[fluid->nGPUs];

    // Iterate over all partitions, and here we GO!
    for(i = 0; i < fluid->nGPUs; i++) {
		cudaSetDevice(fluid->deviceID[i]);
		worked = CHECK_CUDA_ERROR("cudaSetDevice");
		if(worked != SUCCESSFUL) break;

        calcPartitionExtent(fluid, i, sub);

        cudaMalloc((void **)&gradMem[i], 3*sub[3]*sub[4]*sub[5]*sizeof(double));

        arraysize.x = sub[3]; arraysize.y = sub[4]; arraysize.z = sub[5];

        blocksize = makeDim3(GRADBLOCKX, GRADBLOCKY, 1);
        gridsize.x = arraysize.x / (blocksize.x - FROMBLK); gridsize.x += ((blocksize.x-FROMBLK) * gridsize.x < arraysize.x);
        if(isRZ) {
        	gridsize.y = arraysize.z / (blocksize.y - FROMBLK); gridsize.y += ((blocksize.y-FROMBLK) * gridsize.y < arraysize.z);
        } else {
        	gridsize.y = arraysize.y / (blocksize.y - FROMBLK); gridsize.y += ((blocksize.y-FROMBLK) * gridsize.y < arraysize.y);
        }
        gridsize.z = 1;

        if(isThreeD) {
        	if(isRZ) {
        		cukern_computeScalarGradientRZ_h4<<<gridsize, blocksize>>>(phi->devicePtr[i], gradMem[i], gradMem[i] + 2*fluid->partNumel[i],  arraysize);
        		writeScalarToVector<<<32, 256>>>(gradMem[i]+fluid->partNumel[i], fluid->partNumel[i], 0.0);
        	} else {
        		if(geom.shape == SQUARE) {
        			cukern_computeScalarGradient3D<SQUARE><<<gridsize, blocksize>>>(phi->devicePtr[i], gradMem[i], gradMem[i]+fluid->partNumel[i], gradMem[i]+fluid->partNumel[i]*2, arraysize); }
        		if(geom.shape == CYLINDRICAL) {
        			cukern_computeScalarGradient3D<CYLINDRICAL><<<gridsize, blocksize>>>(phi->devicePtr[i], gradMem[i], gradMem[i]+fluid->partNumel[i], gradMem[i]+fluid->partNumel[i]*2, arraysize); }
        	}
        } else {
        	if(geom.shape == SQUARE) {
        		cukern_computeScalarGradient2D_h4<SQUARE><<<gridsize, blocksize>>>(phi->devicePtr[i], gradMem[i], gradMem[i]+fluid->partNumel[i], arraysize); }
        	if(geom.shape == CYLINDRICAL) {
        		cukern_computeScalarGradient2D_h4<CYLINDRICAL><<<gridsize, blocksize>>>(phi->devicePtr[i], gradMem[i], gradMem[i]+fluid->partNumel[i], arraysize); }

        	writeScalarToVector<<<32, 256>>>(gradMem[i]+2*fluid->partNumel[i], fluid->partNumel[i], 0.0);

        }
        worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "cukern_computeScalarGradient");

		// This section extracts the portions of the supplied partition-cloned [X;Y] vector relevant to the current partition
		cudaMalloc((void **)&devXYset[i], (sub[3]+sub[4])*sizeof(double));
		worked = CHECK_CUDA_ERROR("cudaMalloc");
		if(worked != SUCCESSFUL) break;

		blocksize = makeDim3(128, 1, 1);
		gridsize.x = ROUNDUPTO(sub[3], 128) / 128;
		gridsize.y = gridsize.z = 1;
		cukern_FetchPartitionSubset1D<<<gridsize, blocksize>>>(XYVectors->devicePtr[i], fluid->dim[0], devXYset[i], sub[0], sub[3]);
		worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, XYVectors, i, "cukern_FetchPartitionSubset1D, X");
		if(worked != SUCCESSFUL) break;

		gridsize.x = ROUNDUPTO(sub[4], 128) / 128;
		cukern_FetchPartitionSubset1D<<<gridsize, blocksize>>>(XYVectors->devicePtr[i] + fluid->dim[0], fluid->dim[1], devXYset[i]+sub[3], sub[1], sub[4]);
		worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, XYVectors, i, "cukern_FetchPartitionSubset1D, Y");
		if(worked != SUCCESSFUL) break;

		// Prepare to launch the solver itself!
		arraysize.x = sub[3]; arraysize.y = sub[4]; arraysize.z = sub[5];

		blocksize = makeDim3(SRCBLOCKX, SRCBLOCKY, 1);
		gridsize.x = ROUNDUPTO(arraysize.x, blocksize.x) / blocksize.x;
		if(isRZ) {
			gridsize.y = 1;
		} else {
			gridsize.y = arraysize.z;
		}
		gridsize.z = 1;

		if(isRZ) {
			if(geom.shape == SQUARE) {
				cukern_sourceComposite_RK4<RZSQUARE><<<gridsize, blocksize>>>(fluid->devicePtr[i], XYVectors->devicePtr[i], gradMem[i], fluid->slabPitch[i]/8);
			} else {
				cukern_sourceComposite_RK4<RZCYLINDRICAL><<<gridsize, blocksize>>>(fluid->devicePtr[i], XYVectors->devicePtr[i], gradMem[i], fluid->slabPitch[i]/8);
			}
		} else {
			if(geom.shape == SQUARE) {
				cukern_sourceComposite_IMP<SQUARE><<<gridsize, blocksize>>>(fluid->devicePtr[i], XYVectors->devicePtr[i], gradMem[i], fluid->slabPitch[i]/8);
			} else {
				cukern_sourceComposite_IMP<CYLINDRICAL><<<gridsize, blocksize>>>(fluid->devicePtr[i], XYVectors->devicePtr[i], gradMem[i], fluid->slabPitch[i]/8);
			}
		}

		worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "cukernSourceComposite");
		if(worked != SUCCESSFUL) break;
    }

    int j; // This will halt at the stage failed upon if CUDA barfed above
    for(j = 0; j < i; j++) {
		cudaFree((void *)gradMem[j]);
		cudaFree((void *)devXYset[j]);
    }

    // Don't bother checking cudaFree if we already have an error caused above, it was just trying to clean up the barf
    if(worked == SUCCESSFUL)
    	worked = CHECK_CUDA_ERROR("cudaFree");

    return CHECK_IMOGEN_ERROR(worked);

}



/* Compute the gradient of 3d array phi, store the results in f_x, f_y and f_z
 *
 *    In cylindrical geometry, f_x -> f_r,
 *                             f_y -> f_phi
 */
template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient3D(double *phi, double *fx, double *fy, double *fz, int3 arraysize)
{
int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

int myX = threadIdx.x + (GRADBLOCKX-2)*blockIdx.x - 1;
int myY = threadIdx.y + (GRADBLOCKY-2)*blockIdx.y - 1;

if((myX > arraysize.x) || (myY > arraysize.y)) return;

bool IWrite = (threadIdx.x > 0) && (threadIdx.x < (GRADBLOCKX-1)) && (threadIdx.y > 0) && (threadIdx.y < (GRADBLOCKY-1));
IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.y);

myX = (myX + arraysize.x) % arraysize.x;
myY = (myY + arraysize.y) % arraysize.y;

int globAddr = myX + arraysize.x*myY;

double deltaphi; // Store derivative of phi in one direction

__shared__ double phiA[GRADBLOCKX*GRADBLOCKY];
__shared__ double phiB[GRADBLOCKX*GRADBLOCKY];
__shared__ double phiC[GRADBLOCKX*GRADBLOCKY];

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
    deltaphi         = LAMY*(V[myLocAddr+GRADBLOCKX]-V[myLocAddr-GRADBLOCKX]);
    }
    if(coords == CYLINDRICAL) {
    // In cylindrical coords, use dt/dphi * (delta-phi) / r to get d/dy
    deltaphi         = LAMY*(V[myLocAddr+GRADBLOCKX]-V[myLocAddr-GRADBLOCKX]) / (RINNER + DELTAR*myX);
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
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	int myX = threadIdx.x + (GRADBLOCKX-2)*blockIdx.x - 1;
	int myY = threadIdx.y + (GRADBLOCKY-2)*blockIdx.y - 1;

	if((myX > arraysize.x) || (myY > arraysize.y)) return;

	bool IWrite = (threadIdx.x > 0) && (threadIdx.x < (GRADBLOCKX-1)) && (threadIdx.y > 0) && (threadIdx.y < (GRADBLOCKY-1));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.y);

	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.y) % arraysize.y;

	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction
	__shared__ double phiLoc[GRADBLOCKX*GRADBLOCKY];

	phiLoc[myLocAddr] = phi[globAddr];

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		// compute dt * (dphi/dx)
		deltaphi         = LAMX*(phiLoc[myLocAddr+1]-phiLoc[myLocAddr-1]);
		fx[globAddr] = deltaphi;

		// Calculate dt*(dphi/dy)
		if(coords == SQUARE) {
		deltaphi         = LAMY*(phiLoc[myLocAddr+GRADBLOCKX]-phiLoc[myLocAddr-GRADBLOCKX]);
		}
		if(coords == CYLINDRICAL) {
		// Converts d/dphi into physical distance based on R
		deltaphi         = LAMY*(phiLoc[myLocAddr+GRADBLOCKX]-phiLoc[myLocAddr-GRADBLOCKX]) / (RINNER + myX*DELTAR);
		}
		fy[globAddr]     = deltaphi;
	}

}

/* Compute the gradient of 2d array phi, store the results in f_x, f_y
 *    In cylindrical geometry, f_x -> f_r,
 *                             f_y -> f_phi
 */
template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient2D_h4(double *phi, double *fx, double *fy, int3 arraysize)
{
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	int myX = threadIdx.x + (GRADBLOCKX-4)*blockIdx.x - 2;
	int myY = threadIdx.y + (GRADBLOCKY-4)*blockIdx.y - 2;

	if((myX > arraysize.x) || (myY > arraysize.y)) return;

	bool IWrite = (threadIdx.x > 1) && (threadIdx.x < (GRADBLOCKX-2)) && (threadIdx.y > 1) && (threadIdx.y < (GRADBLOCKY-2));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.y);

	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.y) % arraysize.y;

	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction
	__shared__ double phiLoc[GRADBLOCKX*GRADBLOCKY];

	phiLoc[myLocAddr] = phi[globAddr];

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		// compute dt * (dphi/dx)
		deltaphi         = LAMX*(-phiLoc[myLocAddr+2] + 8*phiLoc[myLocAddr+1] - 8*phiLoc[myLocAddr-1] + phiLoc[myLocAddr-2]);
		fx[globAddr] = deltaphi;

		// Calculate dt*(dphi/dy)
		if(coords == SQUARE) {
		deltaphi         = LAMY*(-phiLoc[myLocAddr+2*GRADBLOCKX] + 8*phiLoc[myLocAddr+1*GRADBLOCKX] - 8*phiLoc[myLocAddr-1*GRADBLOCKX] + phiLoc[myLocAddr-2*GRADBLOCKX]);
		}
		if(coords == CYLINDRICAL) {
		// Converts d/dphi into physical distance based on R
		deltaphi         = LAMY*(-phiLoc[myLocAddr+2*GRADBLOCKX] + 8*phiLoc[myLocAddr+1*GRADBLOCKX] - 8*phiLoc[myLocAddr-1*GRADBLOCKX] + phiLoc[myLocAddr-2*GRADBLOCKX])/(RINNER + myX*DELTAR);
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
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	int myX = threadIdx.x + (GRADBLOCKX-2)*blockIdx.x - 1;
	int myY = threadIdx.y + (GRADBLOCKY-2)*blockIdx.y - 1;

	if((myX > arraysize.x) || (myY > arraysize.z)) return;

	bool IWrite = (threadIdx.x > 0) && (threadIdx.x < (GRADBLOCKX-1)) && (threadIdx.y > 0) && (threadIdx.y < (GRADBLOCKY-1));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.z);

	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.z) % arraysize.z;

	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction
	__shared__ double phiLoc[GRADBLOCKX*GRADBLOCKY];

	phiLoc[myLocAddr] = phi[globAddr];

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		// compute dt * (dphi/dx)
		deltaphi         = LAMX*(phiLoc[myLocAddr+1]-phiLoc[myLocAddr-1]);
		fx[globAddr] = deltaphi;

		// Calculate dt*(dphi/dz)
		deltaphi         = LAMZ*(phiLoc[myLocAddr+GRADBLOCKX]-phiLoc[myLocAddr-GRADBLOCKX]);
		fz[globAddr]     = deltaphi;
	}

}

/* Compute the gradient of 2d array phi, store the results in f_x, f_y
 *    In cylindrical geometry, f_x -> f_r,
 *                             f_y -> f_phi
 */
__global__ void  cukern_computeScalarGradientRZ_h4(double *phi, double *fx, double *fz, int3 arraysize)
{
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	int myX = threadIdx.x + (GRADBLOCKX-4)*blockIdx.x - 2;
	int myY = threadIdx.y + (GRADBLOCKY-4)*blockIdx.y - 2;

	if((myX > arraysize.x) || (myY > arraysize.z)) return;

	bool IWrite = (threadIdx.x > 1) && (threadIdx.x < (GRADBLOCKX-2)) && (threadIdx.y > 1) && (threadIdx.y < (GRADBLOCKY-2));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.z);

	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.z) % arraysize.z;

	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction
	__shared__ double phiLoc[GRADBLOCKX*GRADBLOCKY];

	phiLoc[myLocAddr] = phi[globAddr];

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		// compute dt * (dphi/dx)
		deltaphi         = LAMX*(-phiLoc[myLocAddr+2] + 8*phiLoc[myLocAddr+1] - 8*phiLoc[myLocAddr-1] + phiLoc[myLocAddr-2]);
		fx[globAddr] = deltaphi;

		// Calculate dt*(dphi/dz)
		deltaphi         = LAMZ*(-phiLoc[myLocAddr+2*GRADBLOCKX] + 8*phiLoc[myLocAddr+1*GRADBLOCKX] - 8*phiLoc[myLocAddr-1*GRADBLOCKX] + phiLoc[myLocAddr-2*GRADBLOCKX]);
		fz[globAddr]     = deltaphi;
	}

}


__global__ void cukern_FetchPartitionSubset1D(double *in, int nodeN, double *out, int partX0, int partNX);

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

#define JACOBI_ITER_MAX 5
#define NTH (SRCBLOCKX*SRCBLOCKY)

template <geometryType_t coords>
__global__ void  cukern_sourceComposite_IMP(double *fluidIn, double *Rvector, double *gravgrad, long pitch)
{
	__shared__ double shar[4*SRCBLOCKX*SRCBLOCKY];
	//__shared__ double px0[SRCBLOCKX*SRCBLOCKY], py0[SRCBLOCKX*SRCBLOCKY];

	/* strategy: XY files, fill in X direction, step in Y direction; griddim.y = Nz */
	int myx = threadIdx.x + SRCBLOCKX*blockIdx.x;
	int myy = threadIdx.y;
	int myz = blockIdx.y;
	int nx = devIntParams[0];
	int ny;
	if((coords == SQUARE) || (coords == CYLINDRICAL)) { // Not RZ coords
		ny = devIntParams[1];
	} else {
		ny = devIntParams[2];
	}

	if(myx >= devIntParams[0]) return; // return if x >= nx

	// Compute global index at the start
	int tileaddr = myx + nx*(myy + ny*myz);
	fluidIn += tileaddr;
	gravgrad += tileaddr;

	tileaddr = threadIdx.x + SRCBLOCKX*threadIdx.y;

	double locX = Rvector[myx];
	Rvector += nx; // Advances this to the Y array for below

	double locY;
	if((coords == CYLINDRICAL) || (coords == RZCYLINDRICAL)) locY = 0.0;

	double locRho, deltaphi;
	double vdel, dener;

	double vx0, vy0, vz0, vphi_combined;

	int jacobiIters;

	for(; myy < ny; myy += SRCBLOCKY) {
		// Only in square XY or XYZ coordinates must we account for a centripetal term in the the 2-direction
		if((coords == SQUARE) || (coords == RZSQUARE)) {
			locY = Rvector[myy];
		}

		locRho = *fluidIn;
		vx0 = fluidIn[2*pitch] / locRho; // convert to vr
		vy0 = fluidIn[3*pitch] / locRho; // convert to vy/vphi
		vz0 = fluidIn[4*pitch] / locRho;
		shar[tileaddr] = vx0;
		shar[tileaddr+NTH] = vy0;

		// Repeatedly perform fixed point iterations to solve the combined time differential operators
		// This yields the implicit Euler value for the midpoint (t = 0.5) if successful
		for(jacobiIters = 0; jacobiIters < JACOBI_ITER_MAX; jacobiIters++) {
			if((coords == SQUARE) || (coords == RZSQUARE)) {
				// Rotating frame contribution, vx
				vdel          = DT*OMEGA*(OMEGA*locX + 2.0*shar[tileaddr+NTH]); // delta-vx
				// Gravity gradient contribution, vx
				deltaphi      = gravgrad[0];
				if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
				vdel         -= deltaphi;
				// store predicted value for vx
				shar[tileaddr+2*NTH] = vx0 + .5*vdel;

				// rotating frame contribution, vy
				vdel          = -DT*OMEGA*(OMEGA*locY - 2*shar[tileaddr]);
				// gravity gradient contribution, vy
				deltaphi = gravgrad[pitch];
				if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
				vdel         -= deltaphi;
				// store predicted delta for vy
				shar[tileaddr+3*NTH] = vy0 + .5*vdel;
			} else {
				// Rotating frame contribution + cylindrical contribution, pr
				vphi_combined = OMEGA*locX + shar[tileaddr+NTH];
				vdel          = DT*vphi_combined*vphi_combined / locX;
				// Gravity gradient contribution, pr
				deltaphi      = gravgrad[0];
				if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
				vdel         -= deltaphi;
				// store predicted value for pr
				shar[tileaddr+2*NTH] = vx0 + .5*vdel;

				// rotating frame contribution, ptheta
				vphi_combined = shar[tileaddr+NTH] + 2*locX*OMEGA;
				vdel          = -DT*shar[tileaddr]*vphi_combined / locX;
				// gravity gradient contribution, ptheta
				deltaphi = gravgrad[pitch];
				if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
				vdel         -= deltaphi;
				// store predicted delta for ptheta
				shar[tileaddr+3*NTH] = vy0 + .5*vdel;
			}

			__syncthreads();
			shar[tileaddr]     = shar[tileaddr+2*NTH];
			shar[tileaddr+NTH] = shar[tileaddr+3*NTH];
			__syncthreads();

		}

		// Compute minus the original XY/R-theta kinetic energy density
		dener = -(vx0*vx0+vy0*vy0+vz0*vz0);

		if((coords == SQUARE) || (coords == RZSQUARE)) {
			// Rotating frame contribution, vx
			vdel          = DT*OMEGA*(OMEGA*locX + 2.0*shar[tileaddr+2*NTH]); // delta-vx
			// Gravity gradient contribution, vx
			deltaphi      = gravgrad[0];
			if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
			vdel         -= deltaphi;
			// store value for vx
			vx0 += vdel;

			// rotating frame contribution, vy
			vdel          = -DT*OMEGA*(OMEGA*locY - 2*shar[tileaddr+NTH]);
			// gravity gradient contribution, vy
			deltaphi = gravgrad[pitch];
			if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
			vdel         -= deltaphi;
			// store delta for vy
			vy0 += vdel;
		} else {
			// Rotating frame contribution + cylindrical contribution, pr
			vphi_combined = OMEGA*locX + shar[tileaddr+NTH];
			vdel          = DT*vphi_combined*vphi_combined/locX;
			// Gravity gradient contribution, pr
			deltaphi      = gravgrad[0];
			if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
			vdel         -= deltaphi;
			// store predicted value for pr
			vx0 += vdel;

			// rotating frame contribution, ptheta
			vphi_combined = shar[tileaddr+NTH] + 2*locX*OMEGA;
			vdel          = -DT*shar[tileaddr]*vphi_combined/locX;
			// gravity gradient contribution, ptheta
			deltaphi = gravgrad[pitch];
			if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
			vdel         -= deltaphi;
			// store predicted delta for ptheta
			vy0 += vdel;
		}
		
		// Only a linear force in the Z direction: No need to iterate: Exact solution available
		deltaphi = gravgrad[2*pitch];
		if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
		vz0 -= deltaphi;

		// Add the new XY/R-theta kinetic energy density
		dener += vx0*vx0+vy0*vy0+vz0*vz0;

		fluidIn[2*pitch] = vx0 * locRho;
		fluidIn[3*pitch] = vy0 * locRho;
		fluidIn[4*pitch] = vz0 * locRho;
		// Change in total energy is exactly the work done by forces
		fluidIn[pitch] += .5*locRho*dener;

		// Hop pointers forward
		fluidIn += nx*SRCBLOCKY;
		gravgrad+= nx*SRCBLOCKY;
	}
}

template <geometryType_t coords>
__global__ void  cukern_sourceComposite_RK4(double *fluidIn, double *Rvector, double *gravgrad, long pitch)
{
	__shared__ double shar[4*SRCBLOCKX*SRCBLOCKY];
	//__shared__ double px0[SRCBLOCKX*SRCBLOCKY], py0[SRCBLOCKX*SRCBLOCKY];

	/* strategy: XY files, fill in X direction, step in Y direction; griddim.y = Nz */
	int myx = threadIdx.x + SRCBLOCKX*blockIdx.x;
	int myy = threadIdx.y;
	int myz = blockIdx.y;
	int nx = devIntParams[0];
	int ny;
	if((coords == SQUARE) || (coords == CYLINDRICAL)) { // Not RZ coords
		ny = devIntParams[1];
	} else {
		ny = devIntParams[2];
	}

	if(myx >= devIntParams[0]) return; // return if x >= nx

	// Compute global index at the start
	int tileaddr = myx + nx*(myy + ny*myz);
	fluidIn += tileaddr;
	gravgrad += tileaddr;

	tileaddr = threadIdx.x + SRCBLOCKX*threadIdx.y;

	double locX = Rvector[myx];
	Rvector += nx; // Advances this to the Y array for below

	double locY;
	if((coords == CYLINDRICAL) || (coords == RZCYLINDRICAL)) locY = 0.0;

	double locRho, deltaphi;
	double vdel, dener;

	double vx0, vy0, vxS, vyS, vphi_combined;

	int stageCount; double alpha, beta;
	alpha = 1.0/6.0;
	beta = 0.5; 

	for(; myy < ny; myy += SRCBLOCKY) {
		// Only in square XY or XYZ coordinates must we account for a centripetal term in the the 2-direction
		if((coords == SQUARE) || (coords == RZSQUARE)) {
			locY = Rvector[myy];
		}

		locRho = *fluidIn;
		vx0 = fluidIn[2*pitch] / locRho; // convert to vr
		vy0 = fluidIn[3*pitch] / locRho; // convert to vy/vphi

		shar[tileaddr] = vxS = vx0;
		shar[tileaddr+NTH] = vyS = vy0;

		for(stageCount = 0; stageCount < 4; stageCount++) {

		if((coords == SQUARE) || (coords == RZSQUARE)) {
			// Rotating frame contribution, vx
			vdel          = DT*OMEGA*(OMEGA*locX + 2.0*vyS); // delta-vx
			// Gravity gradient contribution, vx
			deltaphi      = gravgrad[0];
			if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
			vdel         -= deltaphi;
			// store predicted value for vx
			shar[tileaddr+2*NTH] = vx0 + beta*vdel;
			// Accumulate delta
			shar[tileaddr]      += alpha*vdel;

			// rotating frame contribution, vy
			vdel          = -DT*OMEGA*(OMEGA*locY - 2*vxS);
			// gravity gradient contribution, vy
			deltaphi = gravgrad[pitch];
			if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
			vdel         -= deltaphi;
			// store predicted delta for vy
			shar[tileaddr+3*NTH] = vy0 + beta*vdel;
			// Accumulate delta
			shar[tileaddr]      += alpha*vdel;
		} else {
			// Rotating frame contribution + cylindrical contribution, pr
			vphi_combined = OMEGA*locX + shar[tileaddr+NTH];
			vdel          = DT*vphi_combined*vphi_combined / locX;
			// Gravity gradient contribution, pr
			deltaphi      = gravgrad[0];
			if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
			vdel         -= deltaphi;
			// store predicted value for vr
			shar[tileaddr+2*NTH] = vx0 + beta*vdel;
			// Accumulate delta
			shar[tileaddr]      += alpha*vdel;

			// rotating frame contribution, ptheta
			vphi_combined = shar[tileaddr+NTH] + 2*locX*OMEGA;
			vdel          = -DT*shar[tileaddr]*vphi_combined / locX;
			// gravity gradient contribution, ptheta
			deltaphi = gravgrad[pitch];
			if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
			vdel         -= deltaphi;
			// store predicted delta for vtheta
			shar[tileaddr+3*NTH] = vy0 + beta*vdel;
			// Accumulate delta
			shar[tileaddr]      += alpha*vdel;
		}

		__syncthreads();
		vxS = shar[tileaddr + 2*NTH];
		vyS = shar[tileaddr + 3*NTH];
		__syncthreads();

		switch(stageCount) {
			case 0: alpha = 1.0/3.0; break;
			case 1: beta = 1.0; break;
			case 2: alpha = 1.0/6.0; break;
		}

		}

		vphi_combined = fluidIn[4*pitch] / locRho; // vz...

		dener = -(vx0*vx0+vy0*vy0+vphi_combined*vphi_combined);

		deltaphi = gravgrad[2*pitch];
		if(locRho < RHOGRAV) { deltaphi *= (locRho*G1 - G2); } // G smoothly -> 0 as rho -> RHO_MIN
		vphi_combined -= deltaphi;

		// Download the final values from shmem
		vxS = shar[tileaddr];
		vyS = shar[tileaddr + NTH];

		// Add the new XY/R-theta kinetic energy density
		dener += vxS*vxS+vyS*vyS+vphi_combined*vphi_combined;

		fluidIn[2*pitch] = vxS * locRho;
		fluidIn[3*pitch] = vyS * locRho;
		fluidIn[4*pitch] = vphi_combined * locRho;

		// Change in total energy is exactly the work done by forces
		fluidIn[pitch] += .5*locRho*dener;

		// Hop pointers forward
		fluidIn += nx*SRCBLOCKY;
		gravgrad+= nx*SRCBLOCKY;
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

/* Converts the fluid slab array from conservative
 * 		[rho, Etotal, px, py, pz]
 * variables to
 * 		[rho, Einternal, vx, vy, vz]
 * primitive variables which may be more suited for some computations. */
__global__ void cukern_cvtToPrimitiveVars(double *fluid, long partNumel, long pitch)
{
	unsigned int globAddr = threadIdx.x + blockDim.x*blockIdx.x;

	if(globAddr >= partNumel) return;

	double rhoinv, p[3], Etot;

	fluid += globAddr;

	for(; globAddr < partNumel; globAddr += blockDim.x*gridDim.x) {
		rhoinv = 1.0/fluid[0];
		Etot = fluid[pitch];
		p[0] = fluid[2*pitch];
		p[1] = fluid[3*pitch];
		p[2] = fluid[4*pitch];

		fluid[2*pitch] = p[0]*rhoinv;
		fluid[3*pitch] = p[1]*rhoinv;
		fluid[4*pitch] = p[2]*rhoinv;

		Etot -= .5*(p[0]*p[0]+p[1]*p[1]+p[2]*p[2])*rhoinv;
		fluid[pitch] = Etot;

		fluid += blockDim.x*gridDim.x;
	}
}


/* Converts the fluid slab array from primitive
 * 		[rho, Einternal, vx, vy, vz]
 * variables to conservative
 * 		[rho, Etotal, px, py, pz]
 * variables which are mandatory for conservative flux differencing */
__global__ void cukern_cvtToConservativeVars(double *fluid, long partNumel, long pitch)
{
	unsigned int globAddr = threadIdx.x + blockDim.x*blockIdx.x;

	if(globAddr >= partNumel) return;

	double rho, v[3], Eint;

	fluid += globAddr;

	for(; globAddr < partNumel; globAddr += blockDim.x*gridDim.x) {
		rho = fluid[0];
		Eint = fluid[pitch];
		v[0] = fluid[2*pitch];
		v[1] = fluid[3*pitch];
		v[2] = fluid[4*pitch];

		fluid[2*pitch] = v[0]*rho;
		fluid[3*pitch] = v[1]*rho;
		fluid[4*pitch] = v[2]*rho;

		Eint += .5*(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])*rho;
		fluid[pitch] = Eint;

		fluid += blockDim.x*gridDim.x;
	}

}

// FIXME implement cvtGasdustToBarydelta

// FIXME implement cvtBarydeltaToGasdust

// Needed with the gradient calculator in 2D.
// Vomits the value f into array x, from x[0] to x[numel-1]
__global__ void writeScalarToVector(double *x, long numel, double f)
{
	long a = threadIdx.x + blockDim.x*blockIdx.x;

	for(; a < numel; a+= blockDim.x*gridDim.x) {
		x[a] = f;

	}

}
