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
#include "nvToolsExt.h"

#include "cudaCommon.h"
#include "cudaSourceScalarPotential.h"
#include "cudaGradientKernels.h"

#define SRCBLOCKX 16
#define SRCBLOCKY 16

int sourcefunction_Composite(MGArray *fluid, MGArray *phi, MGArray *XYVectors, GeometryParams geom, double rhoNoG, double rhoFullGravity, double dt, int spaceOrder, int temporalOrder, MGArray *storageBuffer);

__global__ void cukern_FetchPartitionSubset1D(double *in, int nodeN, double *out, int partX0, int partNX);

template <geometryType_t coords>
__global__ void  cukern_sourceComposite_IMP(double *fluidIn, double *Rvector, double *gravgrad, long pitch);

template <geometryType_t coords>
__global__ void  cukern_sourceComposite_RK4(double *fluidIn, double *Rvector, double *gravgrad, long pitch);

// This will probably be slow as balls but should provide a golden standard of accuracy
template <geometryType_t coords>
__global__ void  cukern_sourceComposite_GL4(double *fluidIn, double *Rvector, double *gravgrad, long pitch);

template <geometryType_t coords>
__global__ void  cukern_sourceComposite_GL6(double *fluidIn, double *Rvector, double *gravgrad, long pitch);

__constant__ __device__ double devLambda[12];

#define LAMX devLambda[0]
#define LAMY devLambda[1]
#define LAMZ devLambda[2]

// Define: F = -beta * rho * grad(phi)
// rho_g = density for full effect of gravity 
// rho_c = minimum density to feel gravity at all
// beta = { rho_g < rho         : 1 (NORMAL GRAVITY)                }
//        { rho_c < rho < rho_g : [(rho-rho_c)/(rho_g-rho_c)]^2 }
//        {         rho < rho_c : 0                                 }

// This provides a continuous (though not differentiable at rho = rho_g) way to surpress gravitation of the background fluid
// The original process of cutting gravity off below a critical density a few times the minimum
// density is believed to cause "blowups" at the inner edge of circular flow profiles due to being
// discontinuous. If even smoothness is insufficient and smooth differentiability is required,
// a more-times-continuous profile can be constructed, but let's not go there unless forced.

// Density below which we force gravity effects to zero
#define RHO_FULLG devLambda[3]
#define RHO_NOG devLambda[4]

// 1 / (rho_fullg - rho_nog)
#define G1 devLambda[5]

// rho_nog / (rho_fullg - rho_nog)
#define G2 devLambda[6]
#define RINNER devLambda[7]
#define DELTAR devLambda[8]

// __constant__ parameters for the rotating frame terms
#define OMEGA devLambda[9]
#define DT devLambda[10]
#define TWO_OMEGA_T devLambda[11]

__constant__ __device__ int devIntParams[3];



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// At least 2 arguments expected
	// Input and result
	if ((nrhs!=5) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaTestSourceComposite(FluidManager, phi, GeometryManager, [rhomin, rho_fullg, dt, spaceorder], [xvector yvector])\n");

	CHECK_CUDA_ERROR("entering cudaSourceRotatingFrame");

	// Get source array info and create destination arrays
	MGArray fluid[5], gravPot, xyvec;

	/* FIXME: accept this as a matlab array instead
	 * FIXME: Transfer appropriate segments to __constant__ memory
	 * FIXME: that seems the only reasonable way to avoid partitioning hell
	 */
	double *scalars = mxGetPr(prhs[3]);
	if(mxGetNumberOfElements(prhs[3]) != 5) {
		PRINT_FAULT_HEADER;
		printf("The 4th argument must be a five element vector: [rho_nog, rho_fullg, dt, space order, temporal order]. It contains %lui elements.\n", mxGetNumberOfElements(prhs[3]));
		PRINT_FAULT_FOOTER;
		DROP_MEX_ERROR("Invalid arguments, brah!");
	}

	double rhonog= scalars[0];
	double rhofg = scalars[1];
	double dt    = scalars[2];
	int spaceOrder    = (int)scalars[3];
	int timeOrder     = (int)scalars[4];
	GeometryParams geom = accessMatlabGeometryClass(prhs[2]);

	int status;

	status = MGA_accessMatlabArrays(prhs, 4, 4, &xyvec);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { DROP_MEX_ERROR("Failed to access X-Y vector."); }

	if(spaceOrder != 0) {
		status = MGA_accessMatlabArrays(prhs, 1, 1, &gravPot);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { DROP_MEX_ERROR("Failed to access gravity potential array."); }
	}
	dim3 gridsize, blocksize;

	int numFluids = mxGetNumberOfElements(prhs[0]);
	int fluidct;

	// Allocate one buffer to be used if we have multiple fluids
	MGArray tempSlab;
	tempSlab.nGPUs = -1; // nonallocated marker

	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		status = MGA_accessFluidCanister(prhs[0], fluidct, &fluid[0]);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;

		mxArray *q = derefXatNdotAdotB(prhs[0], fluidct, "MINMASS", NULL);
		double rhomin = *mxGetPr(q);
		double rhonog= rhomin * 4; // FIXME this is a test hack
		double rhofg = rhomin * 4.1;


		status = sourcefunction_Composite(&fluid[0], &gravPot, &xyvec, geom, rhonog, rhofg, dt, spaceOrder, timeOrder, &tempSlab);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { DROP_MEX_ERROR("Failed to apply rotating frame source terms."); }
	}

	MGA_delete(&tempSlab);

}


int sourcefunction_Composite(MGArray *fluid, MGArray *phi, MGArray *XYVectors, GeometryParams geom, double rhoNoG, double rhoFullGravity, double dt, int spaceOrder, int timeOrder, MGArray *storageBuffer)
{
#ifdef USE_NVTX
	nvtxRangePush(__FUNCTION__);
#endif
	dim3 gridsize, blocksize;
	int3 arraysize;

	double lambda[11];

	int i;
	int worked;

	double *devXYset[fluid->nGPUs];
	int sub[6];

	double *dx = &geom.h[0];

	lambda[3] = rhoFullGravity;
	lambda[4] = rhoNoG;

    lambda[5] = 1.0/(rhoFullGravity - rhoNoG);
    lambda[6] = rhoNoG/(rhoFullGravity - rhoNoG);

    lambda[7] = geom.Rinner; // This is actually overwritten per partition below
    lambda[8] = dx[1];

	lambda[9] = geom.frameOmega;
	lambda[10]= dt;

	//int isThreeD = (fluid->dim[2] > 1);
	int isRZ = (fluid->dim[2] > 1) & (fluid->dim[1] == 1);

	MGArray gradslab;
	gradslab.nGPUs = -1;
	int usingLocalStorage = 0;
	// if we get no buffer then allocate local storage
	if(storageBuffer == NULL) {
		usingLocalStorage = 1;
		storageBuffer = &gradslab;
	}

	if(storageBuffer->nGPUs == -1) { // need to allocate it
		#ifdef USE_NVTX
		nvtxMark("cudaTestSourceComposite.cu:182 large malloc 3 slabs");
		#endif
		worked = MGA_allocSlab(phi, storageBuffer, 3);
		if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) return worked;
	}
	MGArray *gs = storageBuffer;

	worked = computeCentralGradient(phi, gs, geom, spaceOrder, dt);
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) return worked;

    for(i = 0; i < fluid->nGPUs; i++) {
    	cudaSetDevice(fluid->deviceID[i]);
    	calcPartitionExtent(fluid, i, &sub[0]);

    	lambda[7] = geom.Rinner + dx[0] * sub[0]; // Innermost cell coord may change per-partition

    	cudaMemcpyToSymbol((const void *)devLambda, lambda, 11*sizeof(double), 0, cudaMemcpyHostToDevice);
    	worked = CHECK_CUDA_ERROR("cudaMemcpyToSymbol");
    	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;

    	cudaMemcpyToSymbol((const void *)devIntParams, &sub[3], 3*sizeof(int), 0, cudaMemcpyHostToDevice);
    	worked = CHECK_CUDA_ERROR("memcpy to symbol");
    	if(worked != SUCCESSFUL) break;

    	cudaMalloc((void **)&devXYset[i], (sub[3]+sub[4])*sizeof(double));
    	worked = CHECK_CUDA_ERROR("malloc devXYset");
    	if(worked != SUCCESSFUL) break;
    }

    if(worked != SUCCESSFUL) return worked;

    double *fpi;

    // Iterate over all partitions, and here we GO!
    for(i = 0; i < fluid->nGPUs; i++) {
		cudaSetDevice(fluid->deviceID[i]);
		worked = CHECK_CUDA_ERROR("cudaSetDevice");
		if(worked != SUCCESSFUL) break;

        calcPartitionExtent(fluid, i, sub);

        arraysize.x = sub[3]; arraysize.y = sub[4]; arraysize.z = sub[5];

        fpi = fluid->devicePtr[i]; // save some readability below...

        // This section extracts the portions of the supplied partition-cloned [X;Y] vector relevant to the current partition
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
        gridsize.y = (isRZ) ? 1 : arraysize.z;
        gridsize.z = 1;

        switch(timeOrder) {
        case 2:
        	if(isRZ) {
        		if(geom.shape == SQUARE) {
        			cukern_sourceComposite_IMP<RZSQUARE><<<gridsize, blocksize>>>(fpi, devXYset[i], gs->devicePtr[i], fluid->slabPitch[i]/8);
        		} else {
        			cukern_sourceComposite_IMP<RZCYLINDRICAL><<<gridsize, blocksize>>>(fpi, devXYset[i], gs->devicePtr[i], fluid->slabPitch[i]/8);
        		}
        	} else {
        		if(geom.shape == SQUARE) {
        			cukern_sourceComposite_IMP<SQUARE><<<gridsize, blocksize>>>(fpi, devXYset[i], gs->devicePtr[i], fluid->slabPitch[i]/8);
        		} else {
        			cukern_sourceComposite_IMP<CYLINDRICAL><<<gridsize, blocksize>>>(fpi, devXYset[i], gs->devicePtr[i], fluid->slabPitch[i]/8);
        		}
        	}
        	break;
        case 4:
        	if(isRZ) {
        		if(geom.shape == SQUARE) {
        			worked = ERROR_NOIMPLEMENT; break;
        			//cukern_sourceComposite_GL4<RZSQUARE><<<gridsize, blocksize>>>(fpi, devXYset[i], gs->devicePtr[i], fluid->slabPitch[i]/8);
        		} else {
        			cukern_sourceComposite_GL4<RZCYLINDRICAL><<<gridsize, blocksize>>>(fpi, devXYset[i], gs->devicePtr[i], fluid->slabPitch[i]/8);
        		}
        	} else {
        		if(geom.shape == SQUARE) {
        			worked = ERROR_NOIMPLEMENT; break;
        			//cukern_sourceComposite_GL4<SQUARE><<<gridsize, blocksize>>>(fpi, devXYset[i], gs->devicePtr[i], fluid->slabPitch[i]/8);
        		} else {
        			cukern_sourceComposite_GL4<CYLINDRICAL><<<gridsize, blocksize>>>(fpi, devXYset[i], gs->devicePtr[i], fluid->slabPitch[i]/8);
        		}
        	}
        	break;
        case 6:
        	if(isRZ) {
        		if(geom.shape == SQUARE) {
        			worked = ERROR_NOIMPLEMENT; break;
        			//cukern_sourceComposite_GL6<RZSQUARE><<<gridsize, blocksize>>>(fpi, devXYset[i], gs->devicePtr[i], fluid->slabPitch[i]/8);
        		} else {
        			cukern_sourceComposite_GL6<RZCYLINDRICAL><<<gridsize, blocksize>>>(fpi, devXYset[i], gs->devicePtr[i], fluid->slabPitch[i]/8);
        		}
        	} else {
        		if(geom.shape == SQUARE) {
					worked = ERROR_NOIMPLEMENT; break;
        			//cukern_sourceComposite_GL6<SQUARE><<<gridsize, blocksize>>>(fpi, devXYset[i], gs->devicePtr[i], fluid->slabPitch[i]/8);
        		} else {
        			cukern_sourceComposite_GL6<CYLINDRICAL><<<gridsize, blocksize>>>(fpi, devXYset[i], gs->devicePtr[i], fluid->slabPitch[i]/8);
        		}
        	}
        	break;
        default:
        	PRINT_FAULT_HEADER;
        	printf("Source function requires a temporal order of 2 (implicit midpt), 4 (Gauss-Legendre 4th order) or 6 (GL-6th): Received %i\n", timeOrder);
        	PRINT_FAULT_FOOTER;
        	break;
        }

        if(worked != SUCCESSFUL) break;
        worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "cukernSourceComposite");
        if(worked != SUCCESSFUL) break;
    }

    if(worked != SUCCESSFUL) return worked;

    worked = MGA_exchangeLocalHalos(fluid, 5);
    if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) return worked;

    int j; // This will halt at the stage failed upon if CUDA barfed above
    #ifdef USE_NVTX
    	nvtxMark("Freeing devXYset");
    #endif
    for(j = 0; j < i; j++) {
    	cudaFree((void *)devXYset[j]);
    }

	if(usingLocalStorage) {
		#ifdef USE_NVTX
		nvtxMark("cudaTestSourceComposite.cu:323 large free");
		#endif
		MGA_delete(gs);
	}

    // Don't bother checking cudaFree if we already have an error caused above, it was just trying to clean up the barf
    if(worked == SUCCESSFUL)
    	worked = CHECK_CUDA_ERROR("cudaFree");

#ifdef USE_NVTX
	nvtxRangePop();
#endif

    return CHECK_IMOGEN_ERROR(worked);

}

/* The equations of motion for a rotating frame:
 *
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

#define JACOBI_ITER_MAX 3
#define NTH (SRCBLOCKX*SRCBLOCKY)

/* Solves the combined equations of a rotating frame and gravity,
 *
 * d/dt[ px ] = - rho (2 w X v + w X (w X r)).xhat - rho dphi/dx
 *     [ py ] = - rho (2 w X v + w X (w X r)).yhat - rho dphi/dy
 *     [ pz ] = - rho (2 w X v + w X (w X r)).zhat - rho dphi/dz
 *     [ E ]  = p.dp/2
 *
 * in either SQUARE or CYLINDRICAL coordinates using the implicit midpoint method,
 *
 *     y_half = y_0 + .5 dt f(y_half);
 *     y_1    = y_0 + dt    f(y_half);
 *
 * The implicit equations are iterated using JACOBI_ITER_MAX Jacobi steps updating vx then vy.
 * Frame rotation is always in the z-hat direction so no nonlinearity appears in the z direction.
 */
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
				vdel         -= deltaphi;
				// store predicted value for vx
				shar[tileaddr+2*NTH] = vx0 + .5*vdel;

				// rotating frame contribution, vy
				vdel          = -DT*OMEGA*(OMEGA*locY - 2*shar[tileaddr]);
				// gravity gradient contribution, vy
				deltaphi = gravgrad[pitch];
				vdel         -= deltaphi;
				// store predicted delta for vy
				shar[tileaddr+3*NTH] = vy0 + .5*vdel;
			} else {
				// Rotating frame contribution + cylindrical contribution, pr
				vphi_combined = OMEGA*locX + shar[tileaddr+NTH];
				vdel          = DT*vphi_combined*vphi_combined / locX; // a = (vphi + r*W)^2 / r
				// Gravity gradient contribution, pr
				deltaphi      = gravgrad[0];
				vdel         -= deltaphi;
				// store predicted value for pr
				shar[tileaddr+2*NTH] = vx0 + .5*vdel;

				// rotating frame contribution, ptheta
				vphi_combined = shar[tileaddr+NTH] + 2*locX*OMEGA; // a = -vr vphi - 2 vr w
				vdel          = -DT*shar[tileaddr]*vphi_combined / locX;
				// gravity gradient contribution, ptheta
				deltaphi = gravgrad[pitch];
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
			vdel         -= deltaphi;
			// store value for vx
			vx0 += vdel;

			// rotating frame contribution, vy
			vdel          = -DT*OMEGA*(OMEGA*locY - 2*shar[tileaddr+NTH]);
			// gravity gradient contribution, vy
			deltaphi = gravgrad[pitch];
			vdel         -= deltaphi;
			// store delta for vy
			vy0 += vdel;
		} else {
			// Rotating frame contribution + cylindrical contribution, pr
			vphi_combined = OMEGA*locX + shar[tileaddr+NTH];
			vdel          = DT*vphi_combined*vphi_combined/locX;
			// Gravity gradient contribution, pr
			deltaphi      = gravgrad[0];
			vdel         -= deltaphi;
			// store predicted value for pr
			vx0 += vdel;

			// rotating frame contribution, ptheta
			vphi_combined = shar[tileaddr+NTH] + 2*locX*OMEGA;
			vdel          = -DT*shar[tileaddr]*vphi_combined/locX;
			// gravity gradient contribution, ptheta
			deltaphi = gravgrad[pitch];
			vdel         -= deltaphi;
			// store predicted delta for ptheta
			vy0 += vdel;
		}
		
		// Only a linear force in the Z direction: No need to iterate: Exact solution available
		deltaphi = gravgrad[2*pitch];
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

/* These coefficients define the Butcher tableau of the
 * 4th order Gauss-Legendre quadrature method */
#define GL4_C1 0.2113248654051871344705659794271924
#define GL4_C2 0.7886751345948128655294340205728076
#define GL4_A11 .25
#define GL4_A12 -0.03867513459481286552943402057280764
#define GL4_A21 0.5386751345948128655294340205728076
#define GL4_A22 .25

/* Solves the combined equations of a rotating frame and gravity in either
 * SQUARE or CYLINDRICAL coordinates using 4th order Gauss-Legendre quadrature.
 * This requires simultaneous solution of 2N equations at 2 intermediate points,
 * for N=2 (vx and vy) followed by evaluation of the output sum.
 *
 * The implicit solve makes a forward Euler starter prediction before
 * applying Jacobi iterations to update in the order
 *     vx1, vy1, vx2, vy2
 * for up to JACOBI_MAX_ITER times.
 */
template <geometryType_t coords>
__global__ void  cukern_sourceComposite_GL4(double *fluidIn, double *Rvector, double *gravgrad, long pitch)
{
	__shared__ double shar[4*SRCBLOCKX*SRCBLOCKY];

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

	double vxA, vxB, vyA, vyB;

	double q1, q2; // temp vars?

	int jacobiIters;

	for(; myy < ny; myy += SRCBLOCKY) {
		// Only in square XY or XYZ coordinates must we account for a centripetal term in the the 2-direction
		if((coords == SQUARE) || (coords == RZSQUARE)) {
			locY = Rvector[myy];
		}

		locRho = *fluidIn;
		vxA = fluidIn[2*pitch] / locRho; // convert to vr
		vyA = fluidIn[3*pitch] / locRho; // convert to vy/vphi
		shar[tileaddr] = vxA;
		shar[tileaddr+NTH] = vyA;

		// Generate a 1st order prediction for what the values will be using fwd euler
		// This is worth roughly 1 iteration but as can be seen will take way less time
		if((coords == SQUARE) || (coords == RZSQUARE)) {
			q1 = DT*OMEGA*(OMEGA*locX + 2.0*vyA) - gravgrad[0]; // delta-vx
			q2 =  -DT*OMEGA*(OMEGA*locY - 2*vxA) - gravgrad[pitch]; // delta-vy

			vxB = vxA + GL4_C2 * q1;
			vyB = vxB + GL4_C2 * q2;

			vxA += GL4_C1 * q1;
			vyA += GL4_C1 * q2;
		} else {
			q1 = OMEGA*locX + vyA;
			q2 = -vxA*(vyA + 2*OMEGA*locX);

			deltaphi      = gravgrad[0];
			vxB  = vxA + GL4_C2*(DT*q1*q1/locX - deltaphi);
			vxA += GL4_C1*(DT*q1*q1/locX - deltaphi);

			deltaphi = gravgrad[pitch];
			vyB  = vyA + GL4_C2*(DT*q2/locX - deltaphi);
			vyA += GL4_C1*(DT*q2/locX - deltaphi);
		}

		// Repeatedly perform fixed point iterations to solve the combined differential operators
		for(jacobiIters = 0; jacobiIters < JACOBI_ITER_MAX; jacobiIters++) {
			if((coords == SQUARE) || (coords == RZSQUARE)) {
			/////////////////
			/////////////// ruh-roh
			///////////////
			} else {
				// Rotating frame contribution + cylindrical contribution, vr, step A
				q1 = OMEGA*locX + vyA;
				q2 = OMEGA*locX + vyB;
				// Gravity gradient contribution, vr, step A
				deltaphi      = gravgrad[0];
				// Improve estimates for radial velocity
				vdel         = -GL4_C1*deltaphi + DT*(q1*q1*GL4_A11 + q2*q2*GL4_A12)/locX;
				vxA = shar[tileaddr] + vdel;
				vdel         = -GL4_C2*deltaphi + DT*(q1*q1*GL4_A21 + q2*q2*GL4_A22)/locX;
				vxB = shar[tileaddr] + vdel;

				// Load azimuthal gravity gradient
				deltaphi = gravgrad[pitch];

				q2 = vxB*(vyB+2*locX*OMEGA);
				// Note we leave the GL quadrature coefficient off and can reuse q2

				//vdel          = -DT*(GL4_A11*2*locX*OMEGA*vxA+GL4_A12*q2)/locX - GL4_C1 * deltaphi;
				//vyA = (shar[tileaddr + NTH] + vdel)/(1+DT*GL4_A11*vxA/locX)
				vdel          = -DT*(GL4_A11*2*locX*OMEGA*vxA+GL4_A12*q2) - GL4_C1 * deltaphi * locX;
				vyA = (shar[tileaddr + NTH]*locX + vdel)/(locX+DT*GL4_A11*vxA);

				q1 = vxA*(vyA+2*locX*OMEGA);
				vdel          = -DT*(GL4_A21*q1+GL4_A22*2*locX*OMEGA*vxB) - GL4_C2 * deltaphi * locX;
				vyB = (shar[tileaddr+NTH]*locX + vdel)/(locX + DT*GL4_A11*vxA);
			}

		}

		// Compute minus the original kinetic energy density
		q1 = shar[tileaddr];
		q2 = shar[tileaddr+NTH];
		dener = -(q1*q1+q2*q2);
		q1 = fluidIn[4*pitch] / locRho;
		dener -= q1*q1;

		if((coords == SQUARE) || (coords == RZSQUARE)) {
		///////////////
		//////////// ruh-roh
		/////////////
		} else {
			// evaluate final Vr
			q1 = OMEGA*locX + vyA;
			q2 = OMEGA*locX + vyB;
			deltaphi = gravgrad[0];
			shar[tileaddr] = shar[tileaddr] - deltaphi + .5*DT*(q1*q1+q2*q2)/locX;

			// evalute final Vphi
			deltaphi = gravgrad[pitch];
			shar[tileaddr+NTH] = shar[tileaddr+NTH] - deltaphi - .5*DT*(vxA*(vyA+2*OMEGA*locX)+vxB*(vyB+2*OMEGA*locX))/locX;
		}
		vxA = shar[tileaddr];
		vyA = shar[tileaddr+NTH];

		// Only a linear force in the Z direction: No need to iterate: Exact solution available
		deltaphi = gravgrad[2*pitch];
		q1 = fluidIn[4*pitch] / locRho - deltaphi;

		// Add the new XY/R-theta kinetic energy density
		dener += (vxA*vxA + vyA*vyA + q1*q1);

		fluidIn[2*pitch] = vxA * locRho;
		fluidIn[3*pitch] = vyA * locRho;
		fluidIn[4*pitch] = q1  * locRho;
		// Change in total energy is exactly the work done by forces
		fluidIn[pitch] += .5*locRho*dener;

		// Hop pointers forward
		fluidIn += nx*SRCBLOCKY;
		gravgrad+= nx*SRCBLOCKY;
	}
}

#define GL6_C1 0.28918148932210804453
#define GL6_C2 .5
#define GL6_C3 0.71081851067789195547
#define GL6_A11 0.13888888888888888889
#define GL6_A21 0.30026319498086459244
#define GL6_A31 0.26798833376246945173
#define GL6_A12 -0.035976667524938903456
#define GL6_A22 0.22222222222222222222
#define GL6_A32 0.4804211119693833479
#define GL6_A13 0.0097894440153083260496
#define GL6_A23 -0.02248541720308681466
#define GL6_A33 0.13888888888888888889
#define GL6_B1 0.27777777777777777778
#define GL6_B2 0.44444444444444444444
#define GL6_B3 0.27777777777777777778
/* Solves the combined equations of a rotating frame and gravity
 * in either SQUARE or CYLINDRICAL coordinates using 6th order
 * Gauss-Legendre quadrature: This requires simultaneous self-consistent
 * solution of 3N equations at 3 intermediate points, for N=2 (vx and vy)
 * followed by evaluation of the output sum.
 *
 * The implicit solve makes a forward Euler starter prediction before
 * applying Jacobi iterations to update in the order
 *     vx1, vx2, vx3, vy1, vy2, vy3
 * for up to JACOBI_MAX_ITER times.
 */
template <geometryType_t coords>
__global__ void  cukern_sourceComposite_GL6(double *fluidIn, double *Rvector, double *gravgrad, long pitch)
{
	__shared__ double shar[6*SRCBLOCKX*SRCBLOCKY];
	//__shared__ double px0[SRCBLOCKX*SRCBLOCKY], py0[SRCBLOCKX*SRCBLOCKY];

	// strategy: XY files, fill in X direction, step in Y direction; griddim.y = Nz
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

	double vxA, vxB, vxC, vyA, vyB, vyC;

	double q1, q2, q3; // temp vars?

	int jacobiIters;

	for(; myy < ny; myy += SRCBLOCKY) {
		// Only in square XY or XYZ coordinates must we account for a centripetal term in the the 2-direction
		if((coords == SQUARE) || (coords == RZSQUARE)) {
			locY = Rvector[myy];
		}

		locRho = *fluidIn;
		vxA = fluidIn[2*pitch] / locRho; // convert to vr
		vyA = fluidIn[3*pitch] / locRho; // convert to vy/vphi
		shar[tileaddr] = vxA;
		shar[tileaddr+NTH] = vyA;

		// Generate a 1st order prediction for what the values will be using fwd euler
		// This is worth roughly 1 iteration but as can be seen will take way less time
		if((coords == SQUARE) || (coords == RZSQUARE)) {
		/////
		/////
		/////
		} else {
			q1 = OMEGA*locX + vyA;
			q2 = -vxA*(vyA + 2*OMEGA*locX);

			deltaphi      = gravgrad[0];
			vxC  = vxA + GL6_C3*(DT*q1*q1/locX - deltaphi);
			vxB  = vxA + GL6_C2*(DT*q1*q1/locX - deltaphi);
			vxA +=       GL6_C1*(DT*q1*q1/locX - deltaphi);

			deltaphi = gravgrad[pitch];
			vyC  = vyA + GL6_C3*(DT*q2/locX - deltaphi);
			vyB  = vyA + GL6_C2*(DT*q2/locX - deltaphi);
			vyA +=       GL6_C1*(DT*q2/locX - deltaphi);
		}

		// Repeatedly perform fixed point iterations to solve the combined time differential operators
		// This yields the implicit Euler value for the midpoint (t = 0.5) if successful
		for(jacobiIters = 0; jacobiIters < JACOBI_ITER_MAX; jacobiIters++) {
			if((coords == SQUARE) || (coords == RZSQUARE)) {
			///////////////
			/////////////// ruh-roh
			///////////////
			} else {
				// Rotating frame contribution + cylindrical contribution, Vr:
				// Depends only on Vtheta... improve all estimates for Vr now:
				q1 = OMEGA*locX + vyA;
				q2 = OMEGA*locX + vyB;
				q3 = OMEGA*locX + vyC;
				// Gravity gradient contribution, vr
				deltaphi      = gravgrad[0];

				vdel         = -GL6_C1*deltaphi + DT*(q1*q1*GL6_A11 + q2*q2*GL6_A12+q3*q3*GL6_A13)/locX;
				vxA = shar[tileaddr] + vdel;
				vdel         = -GL6_C2*deltaphi + DT*(q1*q1*GL6_A21 + q2*q2*GL6_A22+q3*q3*GL6_A23)/locX;
				vxB = shar[tileaddr] + vdel;
				vdel         = -GL6_C3*deltaphi + DT*(q1*q1*GL6_A31 + q2*q2*GL6_A32+q3*q3*GL6_A33)/locX;
				vxC = shar[tileaddr] + vdel;

				// gravity gradient contribution, vtheta
				deltaphi = gravgrad[pitch];
				// rotating frame contribution, vtheta
				q1 = vxA*(vyA+2*locX*OMEGA);
				q2 = vxB*(vyB+2*locX*OMEGA);
				q3 = vxC*(vyC+2*locX*OMEGA);
				vdel          = -DT*(GL6_A11*q1 + GL6_A12*q2+GL6_A13*q3)/locX - GL6_C1 * deltaphi;
				vyA = shar[tileaddr+NTH] + vdel;

				// update q1 & improve vyB
				q1 = vxA*(vyA+2*locX*OMEGA);
				vdel          = -DT*(GL6_A21*q1 + GL6_A22*q2+GL6_A23*q3)/locX - GL6_C2 * deltaphi;
				vyB = shar[tileaddr+NTH] + vdel;

				// update q2 & improve vyC
				q2 = vxB*(vyB+2*locX*OMEGA);
				vdel          = -DT*(GL6_A31*q1 + GL6_A32*q2+GL6_A33*q3)/locX - GL6_C3 * deltaphi;
				vyC = shar[tileaddr+NTH] + vdel;
			}

		}

		// Compute minus the original kinetic energy density
		q1 = shar[tileaddr];
		q2 = shar[tileaddr+NTH];
		dener = -(q1*q1+q2*q2);
		q1 = fluidIn[4*pitch] / locRho;
		dener -= q1*q1;

		if((coords == SQUARE) || (coords == RZSQUARE)) {
		///////////////
		//////////// ruh-roh
		/////////////
		} else {
			// evaluate final Vr
			q1 = OMEGA*locX + vyA;
			q2 = OMEGA*locX + vyB;
			q3 = OMEGA*locX + vyC;
			deltaphi = gravgrad[0];
			shar[tileaddr] = shar[tileaddr] - deltaphi + DT*(GL6_B1*q1*q1 + GL6_B2*q2*q2 + GL6_B3*q3*q3)/locX;

			// evalute final Vphi
			q1 = vxA*(vyA+2*OMEGA*locX);
			q2 = vxB*(vyB+2*OMEGA*locX);
			q3 = vxC*(vyC+2*OMEGA*locX);
			deltaphi = gravgrad[pitch];
			shar[tileaddr+NTH] = shar[tileaddr+NTH] - deltaphi - DT*(GL6_B1*q1 + GL6_B2*q2 + GL6_B3*q3)/locX;
		}
		vxA = shar[tileaddr];
		vyA = shar[tileaddr+NTH];

		// Only a linear force in the Z direction: No need to iterate: Exact solution available
		deltaphi = gravgrad[2*pitch];
		q1 = fluidIn[4*pitch] / locRho - deltaphi;

		// Add the new XY/R-theta kinetic energy density
		dener += (vxA*vxA + vyA*vyA + q1*q1);

		fluidIn[2*pitch] = vxA * locRho;
		fluidIn[3*pitch] = vyA * locRho;
		fluidIn[4*pitch] = q1  * locRho;
		// Change in total energy is exactly the work done by forces
		fluidIn[pitch] += .5*locRho*dener;

		// Hop pointers forward
		fluidIn += nx*SRCBLOCKY;
		gravgrad+= nx*SRCBLOCKY;
	}
}

/* Solves the combined equations of a rotating frame and gravity
 * in either SQUARE or CYLINDRICAL coordinates using the well-known
 * 4th order explicit multistage method of Runge & Kutta.
 */
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
			vdel         -= deltaphi;
			// store predicted value for vx
			shar[tileaddr+2*NTH] = vx0 + beta*vdel;
			// Accumulate delta
			shar[tileaddr]      += alpha*vdel;

			// rotating frame contribution, vy
			vdel          = -DT*OMEGA*(OMEGA*locY - 2*vxS);
			// gravity gradient contribution, vy
			deltaphi = gravgrad[pitch];
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
if(addrIn < 0) {
	double delta = in[1]-in[0];
	out[addrOut] = in[0]+delta*addrIn;
} else {
	if(addrIn >= nodeN) {
		double delta = in[1]-in[0];
		out[addrOut] = in[nodeN-1] + (addrIn-nodeN+1)*delta;
	} else {
		out[addrOut] = in[addrIn];
	}
}

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
