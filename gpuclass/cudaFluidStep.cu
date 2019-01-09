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
#include "nvToolsExt.h"

// MPI
#include "mpi.h"
#include "mpi_common.h"

// Only uncomment this if you plan to debug this file.
// Causes fluid solvers to emit arrays of debug variables back to Matlab
//#define DEBUGMODE

// Local defs
#include "cudaCommon.h"
#include "cudaStatics.h"
#include "cudaFluidStep.h"
#ifdef DEBUGMODE
#include "debug_inserts.h"
#endif

// If defined, HLLC kernels check and enforce minimum density/temperature limits
#define CFD_ENFORCE_MINVALS

// FUNCTION PLACES
// HLLC 1st order step: L.700
// HLLC 2nd order step: L.1000
// HLL step:            L.1500
// XinJin step:         L.2000

/* THIS FUNCTION
This function calculates a f
irst order accurate upwind step of the conserved transport part of the
Euler equations (CFD or MHD) which is used as the half-step predictor in the Runge-Kutta timestep

The 1D fluid equations solved are the conserved transport equations,
     | rho |         | px                       |
     | px  |         | vx px + P - bx^2         |
d/dt | py  | = -d/dx | vx py     - bx by        |
     | pz  |         | vx pz     - bx bz        |
     | E   |         | vx (E+P)  - bx (B dot v) |

with auxiliary equations
  vx = px / rho
  P  = (gamma-1)e + .5*B^2 = thermal pressure + magnetic pressure
  e  = E - .5*(p^2)/rho - .5*(B^2)
and the understanding that d/dx refers to conservative finite
volume difference and d/dt is the discretized ODE method-of-
lines time derivative.

In general thermal pressure is an arbitrary positive function of e, however the ideal gas
law is built into Imogen in multiple locations and significant re-checking would be needed
if it were to be generalized.

The hydro functions solve the same equations with B set to <0,0,0> which simplifies
and considerably speeds up the process. */

template <unsigned int fluxDirection>
__global__ void  __launch_bounds__(128, 6) cukern_DustRiemann_1storder(double *Qin, double *Qout, double lambda);

template <unsigned int fluxDirection>
__global__ void __launch_bounds__(128, 6) cukern_DustRiemann_2ndorder(double *Qin, double *Qout, double lambda);

template <unsigned int PCswitch>
__global__ void cukern_HLL_step(double *Qin, double *Qstore, double lambda);

template <unsigned int fluxDirection>
__global__ void cukern_HLLC_1storder(double *Qin, double *Qout, double lambda);
template <unsigned int fluxDirection>
__global__ void cukern_HLLC_2ndorder(double *Qin, double *Qout, double lambda);

template <unsigned int PCswitch>
__global__ void cukern_XinJinMHD_step(double *Qbase, double *Qstore, double *mag, double *Cfreeze, double lambda);

template <unsigned int PCswitch>
__global__ void cukern_XinJinHydro_step(double *Qbase, double *Qstore, double *Cfreeze, double lambda);

// FIXME: pressure is now solved for by the XJ kernel, this fcn need only compute the freeze speed.
/* Stopgap until I manage to stuff pressure solvers into all the predictors... */
__global__ void cukern_PressureSolverHydro(double *state, double *gasPressure);
__global__ void cukern_PressureFreezeSolverHydro(double *state, double *gasPressure, double *Cfreeze);

__global__ void cukern_PressureFreezeSolverMHD(double *state, double *totalPressure, double *mag, double *Cfreeze);

typedef struct { double Vleft, Vright; } SpeedBounds;
__device__ SpeedBounds computeEinfeldtBounds(double rhoL, double vL, double PL, double rhoR, double vR, double PR, double deltaVsq);

#define BLOCKLEN 28
#define BLOCKLENP2 30
#define BLOCKLENP4 32

#define YBLOCKS 4
#define FREEZE_NY 4

#ifdef FLOATFLUX
__constant__ __device__ float fluidQtys[12];
#else
__constant__ __device__ double fluidQtys[12];
#endif

//#define LIMITERFUNC fluxLimiter_Zero
//#define LIMITERFUNC fluxLimiter_minmod
#define LIMITERFUNC fluxLimiter_Ospre
//#define LIMITERFUNC fluxLimiter_VanLeer
//#define LIMITERFUNC fluxLimiter_superbee

//#define SLOPEFUNC slopeLimiter_vanAlbada
//#define SLOPEFUNC slopeLimiter_MC
#define SLOPEFUNC slopeLimiter_Ospre
//#define SLOPEFUNC slopeLimiter_Zero
//#define SLOPEFUNC slopeLimiter_minmod
//#define SLOPEFUNC slopeLimiter_VanLeer

#define FLUID_GAMMA   fluidQtys[0]
#define FLUID_GM1     fluidQtys[1]
#define FLUID_GG1     fluidQtys[2]
#define FLUID_MINMASS fluidQtys[3]
#define FLUID_MINTEMP fluidQtys[4]
#define MHD_PRESS_B   fluidQtys[5]
#define MHD_CS_B      fluidQtys[6]
#define FLUID_GOVERGM1 fluidQtys[7]
// The following define the center of the innermost
// cell of the cylindrical annulus and the radial step size
#define CYLGEO_RINI   fluidQtys[8]
#define CYLGEO_DR     fluidQtys[9]

__constant__ __device__ int    arrayParams[4];
#define DEV_NX arrayParams[0]
#define DEV_NY arrayParams[1]
#define DEV_NZ arrayParams[2]
#define DEV_SLABSIZE arrayParams[3]

cudaError_t invokeFluidKernel(FluidMethods algo, int stepdirect, int order, dim3 gridsize, dim3 blocksize, double *fluidBase, double *tempmem, double dt);

#ifdef STANDALONE_MEX_FUNCTION
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	int wanted_nlhs = 0;
#ifdef DEBUGMODE
	wanted_nlhs = 1;
#endif

	// Input and result
	if ((nrhs!=10) || (nlhs != wanted_nlhs)) {
		if(wanted_nlhs == 0) mexErrMsgTxt("Wrong number of arguments: need cudaFluidStep([dt, purehydro, fluid gamma, run.fluid.MINMASS, method, direction], rho, E, px, py, pz, bx, by, bz, geometry)\n");
		if(wanted_nlhs == 1) mexErrMsgTxt("Wrong number of arguments. cudaFluidStep was compiled for debug and requires\n dbgArrays = cudaFluidStep([dt, purehydro, fluid gamma, run.fluid.MINMASS, method, direction], rho, E, px, py, pz, bx, by, bz, geometry)\n");
	}

	CHECK_CUDA_ERROR("entering cudaFluidStep");

	double *scalars = mxGetPr(prhs[0]);

	double dt = scalars[0];
	int hydroOnly = (int)scalars[1];
	double gamma  = scalars[2];
	double rhomin = scalars[3];
	int method    = (int)scalars[4];
	int stepdir   = (int)scalars[5];

	MGArray fluid[5], mag[3];
	int worked = MGA_accessMatlabArrays(prhs, 1, 5, &fluid[0]);

	// The sanity checker tended to barf on the 9 [allzeros] that represent "no array" before.
	if(hydroOnly == false) worked = MGA_accessMatlabArrays(prhs, 6, 8, &mag[0]);

	ParallelTopology topology;
	FluidStepParams stepParameters;

	stepParameters.geometry = accessMatlabGeometryClass(prhs[9]);
	const mxArray *mxtopo = mxGetProperty(prhs[9], 0, "topology");
	topoStructureToC(mxtopo, &topology);

	stepParameters.dt      = dt;
	stepParameters.onlyHydro   = 1;
	stepParameters.thermoGamma = gamma;
	stepParameters.minimumRho  = rhomin;
	//stepParameters.stepMethod  = method;
	switch(method) { // FIXME: this should use a single fcn like flux_ML_iface's
	case 1:
		stepParameters.stepMethod = METHOD_HLL; break;
	case 2:
		stepParameters.stepMethod = METHOD_HLLC; break;
	case 3:
		stepParameters.stepMethod = METHOD_XINJIN; break;
	}

	stepParameters.stepDirection = stepdir;

#ifdef DEBUGMODE
	performFluidUpdate_1D(&fluid[0], stepParameters, &topology, NULL, plhs);
#else
	performFluidUpdate_1D(&fluid[0], stepParameters, &topology, NULL);
#endif
}

#endif
// STANDALONE_MEX_FUNCTION

#ifdef DEBUGMODE
int performFluidUpdate_1D(MGArray *fluid, FluidStepParams params, ParallelTopology* topo, MGArray *tmpst, mxArray **dbOutput)
#else
int performFluidUpdate_1D(MGArray *fluid, FluidStepParams params, ParallelTopology* topo, MGArray *tmpst)
#endif
{
#ifdef USE_NVTX
	nvtxRangePush(__FUNCTION__);
#endif

	CHECK_CUDA_ERROR("entering cudaFluidStep");

	// If we don't get an array, we make our own
	MGArray localTmpStorage;
	// mark it as requiring allocation
	// if localTmpStorage.nGPUs != -1, we also know to free it after
	// whereas if we were passed storage, we do not
	localTmpStorage.nGPUs = -1;

	if(tmpst == NULL) {
		tmpst = &localTmpStorage;
	}

	int hydroOnly = params.onlyHydro;

	double lambda     = params.dt / params.geometry.h[params.stepDirection-1];

	double gamma = params.thermoGamma;
	double rhomin= params.minimumRho;

	// We let the callers just tell us X/Y/Z (~ directions 1/2/3),
	// At this point we know enough to map these to different templates
	// (e.g. the two Theta direction fluxes depend on current array orientation)
	int stepdirect = params.stepDirection;
	if(params.geometry.shape == CYLINDRICAL) {
		if(stepdirect == FLUX_X) stepdirect = FLUX_RADIAL;
		if(stepdirect == FLUX_Y) {
			if(fluid->currentPermutation[1] == 1) stepdirect = FLUX_THETA_213;
			if(fluid->currentPermutation[1] == 3) stepdirect = FLUX_THETA_231;
			if(fluid->currentPermutation[1] == 2) DROP_MEX_ERROR("Fatal: Misordered coordinates! Cannot flux Y if Y is not linear in mem.");
		}
	}

	int returnCode = SUCCESSFUL;

	/* Precalculate thermodynamic values which we'll dump to __constant__ mem
	 */
#ifdef FLOATFLUX
	float gamHost[12];
#else
	double gamHost[12];
#endif
	gamHost[0] = gamma;
	gamHost[1] = gamma-1.0;
	gamHost[2] = gamma*(gamma-1.0);
	gamHost[3] = rhomin;
	// Calculation of minimum internal energy for adiabatic fluid:
	// assert     cs > cs_min
	//     g P / rho > g rho_min^(g-1)
	// (g-1) e / rho > rho_min^(g-1)
	//             e > rho rho_min^(g-1)/(g-1)
	gamHost[4] = 1e-5 / (gamma-1);//powl(rhomin, gamma-1.0)/(gamma-1.0); - minimum einternal
	gamHost[5] = 1.0 - .5*gamma;
	gamHost[6] = ALFVEN_CSQ_FACTOR - .5*(gamma-1.0)*gamma;
	gamHost[7] = gamma/(gamma-1.0); // pressure to energy flux conversion for ideal gas adiabatic EoS
	// Even for gamma=5/3, soundspeed is very weakly dependent on density (cube root) for adiabatic fluid

	// NOTE: camHost[8] is updated per partition
	if(params.geometry.shape == CYLINDRICAL) {
		gamHost[8] = params.geometry.Rinner;
		gamHost[9] = params.geometry.h[params.stepDirection-1];
	}

	if(fluid->dim[0] < 3) return SUCCESSFUL;

	// Temporary storage for RK method, to be allocated per-GPU
	double *wStepValues[fluid->nGPUs];

	int i, sub[6];

	// Host array parameters, to be uploaded to const memory per-GPU
	int haParams[4];
	for(i = 0; i < fluid->nGPUs; i++) {
		cudaSetDevice(fluid->deviceID[i]);

		calcPartitionExtent(fluid, i, &sub[0]);

		// NOTE: IMPORTANT: This alters the preceived layout for RZ launches!!!!!!!!!!!!
		if((sub[4] == 1) && (sub[5] > 1)) {
			// R-Z simulation: pretend Z doesn't exist to improve exe efficiency here
			sub[4] = sub[5]; sub[5] = 1;
		}
		haParams[0] = sub[3];
		haParams[1] = sub[4];
		haParams[2] = sub[5];

		/* This is aligned on 256 so we _can_ safely divide by 8
		 * We _have_  to because the cuda code does (double *) + SLABSIZE */
		haParams[3] = fluid->slabPitch[i] / sizeof(double);

		// Inner cell radius must be set per partition
		gamHost[8] = params.geometry.Rinner + sub[0] * gamHost[9];

		cudaMemcpyToSymbol((const void *)arrayParams, &haParams[0], 4*sizeof(int), 0, cudaMemcpyHostToDevice);
#ifdef FLOATFLUX
		cudaMemcpyToSymbol((const void *)fluidQtys, &gamHost[0], 10*sizeof(float), 0, cudaMemcpyHostToDevice);
#else
		cudaMemcpyToSymbol((const void *)fluidQtys, &gamHost[0], 10*sizeof(double), 0, cudaMemcpyHostToDevice);
#endif
	}
	returnCode = CHECK_CUDA_ERROR("Parameter upload");
	if(returnCode != SUCCESSFUL) return returnCode;

	dim3 arraysize, blocksize, gridsize;

	if(hydroOnly == 1) {
		// Switches between various prediction steps
		switch(params.stepMethod) {
		case METHOD_XINJIN: {

			// Allocate memory for the half timestep's output
			int numarrays = 6 ;
			if(tmpst->nGPUs == -1) {
				returnCode = MGA_allocSlab(fluid, tmpst, numarrays);
				//returnCode = grabTemporaryMemory(&wStepValues[0], fluid, numarrays);
				if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;
				int qc;
				// Just copy the pointers so we don't have to rewrite any of this crap
				for(qc = 0; qc < fluid->nGPUs; qc++) { wStepValues[qc] = tmpst->devicePtr[qc]; }
			}

			arraysize = makeDim3(&fluid[0].dim[0]);
			blocksize = makeDim3(BLOCKLENP4, YBLOCKS, 1);

			MGArray ref = fluid[0];
			ref.dim[0] = ref.nGPUs;
			ref.haloSize = 0;
			MGArray *cfreeze[2];
			returnCode = MGA_allocArrays(&cfreeze[0], 1, &ref);
			if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;

			// Compute pressure & local freezing speed
			for(i = 0; i < fluid->nGPUs; i++) {
				calcPartitionExtent(fluid, i, &sub[0]);
				arraysize = makeDim3(sub[3],sub[4],sub[5]);

				dim3 cfblk = makeDim3(64, 4, 1);
				dim3 cfgrid = makeDim3(ROUNDUPTO(arraysize.y,4)/4, arraysize.z, 1);

				cudaSetDevice(fluid->deviceID[i]);
				cukern_PressureFreezeSolverHydro<<<cfgrid, cfblk>>>(fluid->devicePtr[i], wStepValues[i] + (5*fluid->slabPitch[i])/8, cfreeze[0]->devicePtr[i]);
				returnCode = CHECK_CUDA_LAUNCH_ERROR(cfblk, cfgrid, fluid, hydroOnly, "In cudaFluidStep: cukern_PressureFreezeSolverHydro");
				if(returnCode != SUCCESSFUL) return returnCode;
			}

			// Compute global freezing speeds
			cfreeze[1] = NULL;
			MGA_globalReduceDimension(cfreeze[0], &cfreeze[1], MGA_OP_MAX, 1, 0, 1, topo);

			gridsize.x = ROUNDUPTO(arraysize.x, blocksize.x) / blocksize.x;

			// Invoke half timestep
			for(i = 0; i < fluid->nGPUs; i++) {
				calcPartitionExtent(fluid, i, &sub[0]);
				arraysize = makeDim3(sub[3],sub[4],sub[5]);

				blocksize = makeDim3(32, (arraysize.y > 1) ? 4 : 1, 1);
				gridsize = makeDim3(ROUNDUPTO(arraysize.x,blocksize.x - 4)/(blocksize.x-4), arraysize.z, 1);
// FIXME: This will apparently dump if the arraysize.x < 28? Wat???
				cudaSetDevice(fluid->deviceID[i]);
				switch(stepdirect) {
				case FLUX_X: cukern_XinJinHydro_step<0+RK_PREDICT><<<gridsize, blocksize>>>(fluid->devicePtr[i], wStepValues[i], cfreeze[1]->devicePtr[i], 0.25*lambda); break;
				case FLUX_Y: cukern_XinJinHydro_step<2+RK_PREDICT><<<gridsize, blocksize>>>(fluid->devicePtr[i], wStepValues[i], cfreeze[1]->devicePtr[i], 0.25*lambda); break;
				case FLUX_Z: cukern_XinJinHydro_step<4+RK_PREDICT><<<gridsize, blocksize>>>(fluid->devicePtr[i], wStepValues[i], cfreeze[1]->devicePtr[i], 0.25*lambda); break;
				case FLUX_RADIAL: cukern_XinJinHydro_step<6+RK_PREDICT><<<gridsize, blocksize>>>(fluid->devicePtr[i], wStepValues[i], cfreeze[1]->devicePtr[i], 0.25*lambda); break;
				case FLUX_THETA_213: cukern_XinJinHydro_step<8+RK_PREDICT><<<gridsize, blocksize>>>(fluid->devicePtr[i], wStepValues[i], cfreeze[1]->devicePtr[i], 0.25*lambda); break;
				case FLUX_THETA_231: cukern_XinJinHydro_step<10+RK_PREDICT><<<gridsize, blocksize>>>(fluid->devicePtr[i], wStepValues[i], cfreeze[1]->devicePtr[i], 0.25*lambda); break;
				}
				returnCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: cukern_XinJinHydro_step prediction step");
				if(returnCode != SUCCESSFUL) return returnCode;
			}

#ifdef DBG_FIRSTORDER // 1st-order testing: Dumps upwinded values straight back to output arrays
			printf("WARNING: Operating at first order for debug purposes!\n");
			for(i = 0; i < fluid->nGPUs; i++ ) {
				cudaSetDevice(fluid->deviceID[i]);
				cudaMemcpy(fluid[0].devicePtr[i], wStepValues[i], 5*fluid[0].slabPitch[i], cudaMemcpyDeviceToDevice);
			}

#ifdef DEBUGMODE // If in first-order debug mode, intermediate values were dumped to wStepValues to be returned
			returnDebugArray(fluid, 6, wStepValues, dbOutput);
#endif
#else // runs at 2nd order

			for(i = 0; i < fluid->nGPUs; i++) {
				calcPartitionExtent(fluid, i, &sub[0]);
				arraysize = makeDim3(sub[3],sub[4],sub[5]);

				dim3 cfblk = makeDim3(64, 4, 1);
				dim3 cfgrid = makeDim3(ROUNDUPTO(arraysize.y,4)/4, arraysize.z, 1);

				cudaSetDevice(fluid->deviceID[i]);
				cukern_PressureFreezeSolverHydro<<<cfgrid, cfblk>>>(wStepValues[i], wStepValues[i] + (5*fluid->slabPitch[i])/8, cfreeze[0]->devicePtr[i]);
				returnCode = CHECK_CUDA_LAUNCH_ERROR(cfblk, cfgrid, fluid, hydroOnly, "In cudaFluidStep: cukern_PressureFreezeSolverHydro");
				if(returnCode != SUCCESSFUL) return returnCode;
			}

			MGA_globalReduceDimension(cfreeze[0], &cfreeze[1], MGA_OP_MAX, 1, 0, 1, topo);

			for(i = 0; i < fluid->nGPUs; i++) {
				calcPartitionExtent(fluid, i, &sub[0]);
				arraysize = makeDim3(sub[3],sub[4],sub[5]);

				blocksize = makeDim3(32, (arraysize.y > 1) ? 4 : 1, 1);
				gridsize = makeDim3(ROUNDUPTO(arraysize.x,blocksize.x - 4)/(blocksize.x - 4), arraysize.z, 1);

				cudaSetDevice(fluid->deviceID[i]);

				switch(stepdirect) {
				case FLUX_X: cukern_XinJinHydro_step<0+RK_CORRECT><<<gridsize, blocksize>>>(fluid->devicePtr[i], wStepValues[i], cfreeze[1]->devicePtr[i], 0.5*lambda); break;
				case FLUX_Y: cukern_XinJinHydro_step<2+RK_CORRECT><<<gridsize, blocksize>>>(fluid->devicePtr[i], wStepValues[i], cfreeze[1]->devicePtr[i], 0.5*lambda); break;
				case FLUX_Z: cukern_XinJinHydro_step<4+RK_CORRECT><<<gridsize, blocksize>>>(fluid->devicePtr[i], wStepValues[i], cfreeze[1]->devicePtr[i], 0.5*lambda); break;
				case FLUX_RADIAL: cukern_XinJinHydro_step<6+RK_CORRECT><<<gridsize, blocksize>>>(fluid->devicePtr[i], wStepValues[i], cfreeze[1]->devicePtr[i], 0.5*lambda); break;
				case FLUX_THETA_213: cukern_XinJinHydro_step<8+RK_CORRECT><<<gridsize, blocksize>>>(fluid->devicePtr[i], wStepValues[i], cfreeze[1]->devicePtr[i], 0.5*lambda); break;
				case FLUX_THETA_231: cukern_XinJinHydro_step<10+RK_CORRECT><<<gridsize, blocksize>>>(fluid->devicePtr[i], wStepValues[i], cfreeze[1]->devicePtr[i], 0.5*lambda); break;
				}
				returnCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: cukern_XinJinHydro_step prediction step");
				if(returnCode != SUCCESSFUL) return returnCode;
			}
#endif // still have to avoid memory leak regardless of order we ran at
			MGA_delete(cfreeze[0]);
			MGA_delete(cfreeze[1]);

			if(localTmpStorage.nGPUs != -1) {
				returnCode = MGA_delete(&localTmpStorage);
				if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;
			}

			//returnCode = releaseTemporaryMemory(&wStepValues[0], fluid);

		} break;
		case METHOD_HLL:
		case METHOD_HLLC: {
			int numarrays;
#ifdef DEBUGMODE
			numarrays = 6 + DBG_NUMARRAYS;
#else
#ifdef USE_RK3
			numarrays = 11;
#else
			numarrays = 6;
#endif
#endif

#ifdef USE_SSPRK
#define TSFACTOR 1.0
#else
#ifdef USE_RK3
#define TSFACTOR 1.0
#else
#define TSFACTOR 0.5
#endif
#endif

			if(tmpst->nGPUs == -1) {
				returnCode = MGA_allocSlab(fluid, tmpst, 6);
				//returnCode = grabTemporaryMemory(&wStepValues[0], fluid, numarrays);
				if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;
			}

			// Just copy the pointers so we don't have to rewrite any of this crap
			int qc;
			for(qc = 0; qc < fluid->nGPUs; qc++) { wStepValues[qc] = tmpst->devicePtr[qc]; }


			//returnCode = grabTemporaryMemory(&wStepValues[0], fluid, numarrays);
			//if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;

			// Launch zee kernels
			for(i = 0; i < fluid->nGPUs; i++) {
				cudaSetDevice(fluid->deviceID[i]);

				// Find out the size of the partition
				calcPartitionExtent(fluid, i, sub);
				if((sub[4] == 1) && (sub[5] > 1)) {
					sub[4] = sub[5]; sub[5] = 1;
				}

				gridsize.x = (sub[3]/BLOCKLEN); gridsize.x += 1*(gridsize.x*BLOCKLEN < sub[3]);
				gridsize.y = sub[5];
				blocksize = makeDim3(32, YBLOCKS, 1);
				if(sub[4] < YBLOCKS) {
					blocksize.y = sub[4];
				}

				// Fire off the fluid update step
				if(params.stepMethod == METHOD_HLL) {
					cukern_PressureSolverHydro<<<256, 128>>>(fluid[0].devicePtr[i], wStepValues[i] + 5*fluid->slabPitch[i]/sizeof(double) );
					CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: cukern_PressureSolverHydro");
#ifdef DEBUGMODE
				returnDebugArray(fluid, 1, wStepValues, dbOutput);
#endif
				}
				cudaError_t ohboy = invokeFluidKernel(params.stepMethod, stepdirect, 1, gridsize, blocksize, fluid->devicePtr[i], wStepValues[i], TSFACTOR*lambda);
				returnCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: cukern_HLLC_1storder");
#undef TSFACTOR
				if(returnCode != SUCCESSFUL) return returnCode;


#ifdef DBG_FIRSTORDER // Run at 1st order: dump first order values straight back to output arrays
				printf("WARNING: Operating at first order for debug purposes!\n");
				for(i = 0; i < fluid->nGPUs; i++) {
					cudaSetDevice(fluid->deviceID[i]);
					cudaMemcpy(fluid[0].devicePtr[i], wStepValues[i], 5*fluid[0].slabPitch[i], cudaMemcpyDeviceToDevice);
				}
#else // runs at higher order

				// FIXME awful hack
			}

			//
			MGArray fluidB[5];
			int qwer;
			for(i = 0; i < fluid->nGPUs; i++) {
				for(qwer = 0; qwer < 5; qwer++) {
					fluidB[qwer] = fluid[qwer];
					fluidB[qwer].devicePtr[i] = wStepValues[i] + fluid->slabPitch[i]*qwer/sizeof(double);
				}
			}

			returnCode = setFluidBoundary(&fluidB[0], fluid->matlabClassHandle, &params.geometry, params.stepDirection);
			if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;

			for(i = 0; i < fluid->nGPUs; i++) {
				cudaSetDevice(fluid->deviceID[i]);

				// Find out the size of the partition
				calcPartitionExtent(fluid, i, sub);
				if((sub[4] == 1) && (sub[5] > 1)) {
					sub[4] = sub[5]; sub[5] = 1;
				}

				gridsize.x = (sub[3]/BLOCKLEN); gridsize.x += 1*(gridsize.x*BLOCKLEN < sub[3]);
				gridsize.y = sub[5];
				blocksize = makeDim3(32, YBLOCKS, 1);
				if(sub[4] < YBLOCKS) {
					blocksize.y = sub[4];
				}

				if(params.stepMethod == METHOD_HLL) {
					cukern_PressureSolverHydro<<<256, 128>>>(wStepValues[i], wStepValues[i] + 5*fluid->slabPitch[i]/sizeof(double) );
					CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: cukern_PressureSolverHydro");
				}
				cudaError_t ohboy = invokeFluidKernel(params.stepMethod, stepdirect, 2, gridsize, blocksize, fluid->devicePtr[i], wStepValues[i], lambda);
				returnCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: cukern_HLLC_2ndorder");

#ifdef USE_RK3
				ohboy = invokeFluidKernel(params.stepMethod, stepdirect, 3, gridsize, blocksize, fluid->devicePtr[i], wStepValues[i]+5*fluid->slabPitch[i]/8, lambda);
				returnCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: cukern_HLLC_3rdorder");
#endif


				if(returnCode != SUCCESSFUL) return returnCode;
#ifdef DBG_SECONDORDER
				returnDebugArray(fluid, 6, wStepValues, dbOutput);
#endif

#endif
			}

			if(localTmpStorage.nGPUs != -1) {
				// If we allocated this locally, free it now
				returnCode = MGA_delete(&localTmpStorage);
				if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;
			}
			// otherwise reuse it for future steps
			// returnCode = releaseTemporaryMemory(&wStepValues[0], fluid);

			if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;
		} break;
		}

		if(fluid->partitionDir == PARTITION_X) returnCode = MGA_exchangeLocalHalos(fluid, 5);
		if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;

	} else { // If MHD case
		//		int MGA_globalPancakeReduce(MGArray *in, MGArray *out, MPI_Op operate, int dir, int partitionOnto, int redistribute, const mxArray *topo)
		//worked = MGA_globalPancakeReduce(MGArray *in, MGArray *out, MPI_Op operate, int dir, int partitionOnto, int redistribute, const mxArray *topo)
		//cfSync(hostptrs[9], fluid[0].dim[1]*fluid[0].dim[2], prhs[13]); // prhs[13] is the parallel topo
		/*CHECK_CUDA_ERROR("In cudaFluidStep: first mhd c_f sync");

		cukern_XinJinMHD_step<RK_PREDICT><<<gridsize, blocksize>>>(wStepValues[0], hostptrs[9], 0.25*lambda, arraySize.x, arraySize.y, fluid->numel);
		returnCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: mhd predict step");
		if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;

		dim3 cfblk;  cfblk.x = 64; cfblk.y = 4; cfblk.z = 1;
		dim3 cfgrid; cfgrid.x = arraySize.y / 4; cfgrid.y = arraySize.z; cfgrid.z = 1;
		cfgrid.x += 1*(4*cfgrid.x < arraySize.y);
		cukern_PressureFreezeSolverMHD<<<cfgrid, cfblk>>>(wStepValues[0], hostptrs[9], arraySize.x, arraySize.y, fluid->numel);
		returnCode = CHECK_CUDA_LAUNCH_ERROR(cfblk, cfgrid, fluid, hydroOnly, "In cudaFluidStep: cukern_PressureFreezeSolverMHD");
		if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;
		//cfSync(hostptrs[9], fluid[0].dim[1]*fluid[0].dim[2], prhs[13]);
		CHECK_CUDA_ERROR("In cudaFluidStep: second mhd c_f sync");
		cukern_XinJinMHD_step<RK_CORRECT><<<gridsize, blocksize>>>(wStepValues[0], hostptrs[9], 0.5*lambda, arraySize.x, arraySize.y, fluid->numel);
		returnCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: mhd TVD step");
		if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode; */
	}


#ifdef USE_NVTX
	nvtxRangePop();
#endif

	return SUCCESSFUL;

}

cudaError_t invokeFluidKernel(FluidMethods algo, int stepdirect, int order, dim3 gridsize, dim3 blocksize, double *fluidBase, double *tempmem, double dt)
{

	/* NOTE: Be very careful re: the constant prefactors on dt here
	 * Some of these (e.g. SSPRK stage 2, and RK3 stage 3) also have prefactors embedded in the kernel!
	 */
	if(algo == METHOD_HLL) {
		if(order == 1) {
			switch(stepdirect) {
			case FLUX_X: cukern_HLL_step<RK_PREDICT+0><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_Y: cukern_HLL_step<RK_PREDICT+2><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_Z: cukern_HLL_step<RK_PREDICT+4><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_RADIAL: cukern_HLL_step<RK_PREDICT+6><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_THETA_213: cukern_HLL_step<RK_PREDICT+8><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_THETA_231: cukern_HLL_step<RK_PREDICT+10><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			}
		} else {
			switch(stepdirect) {
			case FLUX_X: cukern_HLL_step<RK_CORRECT+0><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_Y: cukern_HLL_step<RK_CORRECT+2><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_Z: cukern_HLL_step<RK_CORRECT+4><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_RADIAL: cukern_HLL_step<RK_CORRECT+6><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_THETA_213: cukern_HLL_step<RK_CORRECT+8><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_THETA_231: cukern_HLL_step<RK_CORRECT+10><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			}
		}
	}
	if(algo == METHOD_HLLC) {
		switch(order) {
		case 1: // predictor
#ifdef USE_SSPRK
			switch(stepdirect) {
			case FLUX_X: cukern_HLLC_2ndorder<FLUX_X+TIMESCHEME_SSPRK_A><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_Y: cukern_HLLC_2ndorder<FLUX_Y+TIMESCHEME_SSPRK_A><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_Z: cukern_HLLC_2ndorder<FLUX_Z+TIMESCHEME_SSPRK_A><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_RADIAL: cukern_HLLC_2ndorder<FLUX_RADIAL+TIMESCHEME_SSPRK_A><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_THETA_213: cukern_HLLC_2ndorder<FLUX_THETA_213+TIMESCHEME_SSPRK_A><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_THETA_231: cukern_HLLC_2ndorder<FLUX_THETA_231+TIMESCHEME_SSPRK_A><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			}
#else
#ifdef USE_RK3
			switch(stepdirect) {
			case FLUX_X: cukern_HLLC_2ndorder<FLUX_X+TIMESCHEME_RK3_A><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_Y: cukern_HLLC_2ndorder<FLUX_Y+TIMESCHEME_RK3_A><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_Z: cukern_HLLC_2ndorder<FLUX_Z+TIMESCHEME_RK3_A><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_RADIAL: cukern_HLLC_2ndorder<FLUX_RADIAL+TIMESCHEME_RK3_A><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_THETA_213: cukern_HLLC_2ndorder<FLUX_THETA_213+TIMESCHEME_RK3_A><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			case FLUX_THETA_231: cukern_HLLC_2ndorder<FLUX_THETA_231+TIMESCHEME_RK3_A><<<gridsize, blocksize>>>(fluidBase, tempmem, dt); break;
			}
#else
			switch(stepdirect) {
			case FLUX_X: cukern_HLLC_1storder<FLUX_X><<<gridsize, blocksize>>>(fluidBase, tempmem, .5*dt); break;
			case FLUX_Y: cukern_HLLC_1storder<FLUX_Y><<<gridsize, blocksize>>>(fluidBase, tempmem, .5*dt); break;
			case FLUX_Z: cukern_HLLC_1storder<FLUX_Z><<<gridsize, blocksize>>>(fluidBase, tempmem, .5*dt); break;
			case FLUX_RADIAL: cukern_HLLC_1storder<FLUX_RADIAL><<<gridsize, blocksize>>>(fluidBase, tempmem, .5*dt); break;
			case FLUX_THETA_213: cukern_HLLC_1storder<FLUX_THETA_213><<<gridsize, blocksize>>>(fluidBase, tempmem, .5*dt); break;
			case FLUX_THETA_231: cukern_HLLC_1storder<FLUX_THETA_231><<<gridsize, blocksize>>>(fluidBase, tempmem, .5*dt); break;
			}
#endif
#endif
		break;
		case 2: // corrector
#ifdef USE_SSPRK
			switch(stepdirect) {
			case FLUX_X: cukern_HLLC_2ndorder<FLUX_X+TIMESCHEME_SSPRK_B><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_Y: cukern_HLLC_2ndorder<FLUX_Y+TIMESCHEME_SSPRK_B><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_Z: cukern_HLLC_2ndorder<FLUX_Z+TIMESCHEME_SSPRK_B><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_RADIAL: cukern_HLLC_2ndorder<FLUX_RADIAL+TIMESCHEME_SSPRK_B><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_THETA_213: cukern_HLLC_2ndorder<FLUX_THETA_213+TIMESCHEME_SSPRK_B><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_THETA_231: cukern_HLLC_2ndorder<FLUX_THETA_231+TIMESCHEME_SSPRK_B><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			}
#else
#ifdef USE_RK3
			switch(stepdirect) {
			case FLUX_X: cukern_HLLC_2ndorder<FLUX_X+TIMESCHEME_RK3_B><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_Y: cukern_HLLC_2ndorder<FLUX_Y+TIMESCHEME_RK3_B><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_Z: cukern_HLLC_2ndorder<FLUX_Z+TIMESCHEME_RK3_B><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_RADIAL: cukern_HLLC_2ndorder<FLUX_RADIAL+TIMESCHEME_RK3_B><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_THETA_213: cukern_HLLC_2ndorder<FLUX_THETA_213+TIMESCHEME_RK3_B><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_THETA_231: cukern_HLLC_2ndorder<FLUX_THETA_231+TIMESCHEME_RK3_B><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			}
#else
			switch(stepdirect) {
			case FLUX_X: cukern_HLLC_2ndorder<FLUX_X+TIMESCHEME_RK2><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_Y: cukern_HLLC_2ndorder<FLUX_Y+TIMESCHEME_RK2><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_Z: cukern_HLLC_2ndorder<FLUX_Z+TIMESCHEME_RK2><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_RADIAL: cukern_HLLC_2ndorder<FLUX_RADIAL+TIMESCHEME_RK2><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_THETA_213: cukern_HLLC_2ndorder<FLUX_THETA_213+TIMESCHEME_RK2><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			case FLUX_THETA_231: cukern_HLLC_2ndorder<FLUX_THETA_231+TIMESCHEME_RK2><<<gridsize, blocksize>>>(tempmem, fluidBase, dt); break;
			}
#endif
#endif
			break;
			case 3: // RK3 final stage
#ifdef USE_RK3
				switch(stepdirect) {
				case FLUX_X: cukern_HLLC_2ndorder<FLUX_X+TIMESCHEME_RK3_C><<<gridsize, blocksize>>>(tempmem, fluidBase, 2*dt); break;
				case FLUX_Y: cukern_HLLC_2ndorder<FLUX_Y+TIMESCHEME_RK3_C><<<gridsize, blocksize>>>(tempmem, fluidBase, 2*dt); break;
				case FLUX_Z: cukern_HLLC_2ndorder<FLUX_Z+TIMESCHEME_RK3_C><<<gridsize, blocksize>>>(tempmem, fluidBase, 2*dt); break;
				case FLUX_RADIAL: cukern_HLLC_2ndorder<FLUX_RADIAL+TIMESCHEME_RK3_C><<<gridsize, blocksize>>>(tempmem, fluidBase, 2*dt); break;
				case FLUX_THETA_213: cukern_HLLC_2ndorder<FLUX_THETA_213+TIMESCHEME_RK3_C><<<gridsize, blocksize>>>(tempmem, fluidBase, 2*dt); break;
				case FLUX_THETA_231: cukern_HLLC_2ndorder<FLUX_THETA_231+TIMESCHEME_RK3_C><<<gridsize, blocksize>>>(tempmem, fluidBase, 2*dt); break;
				}
#endif
				break;
		}
	}
	return cudaSuccess;
}

#ifdef FLOATFLUX
#define SQRTFUNC sqrtf
#else
#define SQRTFUNC sqrt
#endif

// Sometimes nsight gets stupid about parsing the nVidia headers and I'm tired of this
// crap about how __syncthreads is "undeclared."
extern __device__ __device_builtin__ void                   __syncthreads(void);

// These tell the HLL and HLLC solvers how to dereference their shmem blocks in convenient shorthand
#define BOS0 (0*BLOCKLENP4)
#define BOS1 (1*BLOCKLENP4)
#define BOS2 (2*BLOCKLENP4)
#define BOS3 (3*BLOCKLENP4)
#define BOS4 (4*BLOCKLENP4)
#define BOS5 (5*BLOCKLENP4)
#define BOS6 (6*BLOCKLENP4)
#define BOS7 (7*BLOCKLENP4)
#define BOS8 (8*BLOCKLENP4)
#define BOS9 (9*BLOCKLENP4)

// for first-order HLLC kernel
#define N_SHMEM_BLOCKS_FO 5

template <unsigned int fluxDirection>
__global__ void __launch_bounds__(128, 6) cukern_HLLC_1storder(double *Qin, double *Qout, double lambda)
{
	// Create center, look-left and look-right indexes
	int IC = threadIdx.x;
	int IL = threadIdx.x - 1;
	IL += 1*(IL < 0);
	int IR = threadIdx.x + 1;
	IR -= BLOCKLENP4*(IR > (BLOCKLENP4-1));

	// Advance by the Y index
	IC += N_SHMEM_BLOCKS_FO*BLOCKLENP4*threadIdx.y;
	IL += N_SHMEM_BLOCKS_FO*BLOCKLENP4*threadIdx.y;
	IR += N_SHMEM_BLOCKS_FO*BLOCKLENP4*threadIdx.y;


#ifdef FLOATFLUX
	float A, B, C, D, E, F, G, H;
	__shared__ float shblk[YBLOCKS*N_SHMEM_BLOCKS_FO*BLOCKLENP4];
	float *shptr = &shblk[IC];
	float cylgeomA, cylgeomB, cylgeomC;
#else
	/* Declare shared variable array */
	__shared__ double shblk[YBLOCKS*N_SHMEM_BLOCKS_FO*BLOCKLENP4];
	double *shptr = &shblk[IC];
	double A, B, C, D, E, F, G, H;
	double cylgeomA, cylgeomB, cylgeomC;
#endif
	/* My x index: thread + blocksize block, wrapped circularly */
	//int thisThreadPonders  = (threadIdx.x > 0) && (threadIdx.x < blockDim.x-1);
	int thisThreadDelivers = (threadIdx.x >= 2) && (threadIdx.x <= (BLOCKLENP4-3));

	int x0 = threadIdx.x + (BLOCKLEN)*blockIdx.x - 2;
	if(x0 < 0) x0 += DEV_NX; // left wraps to right edge
	if(x0 > (DEV_NX+1)) return; // More than 2 past right returns
	if(x0 > (DEV_NX-1)) { x0 -= DEV_NX; thisThreadDelivers = 0; } // past right must wrap around to left

	/* If doing cylindrical geometry... */
	// Compute multiple scale factors for radial direction fluxes
	if(fluxDirection == FLUX_RADIAL) { // cylindrical, R direction
		A = CYLGEO_RINI + x0 * CYLGEO_DR; // r_center
		cylgeomA = (A - .5*CYLGEO_DR) / (A);
		cylgeomB = (A + .5*CYLGEO_DR) / (A);
		cylgeomC = CYLGEO_DR / A;
	}
	// The kern will step through r, so we have to add to R and compute 1.0/R
	if(fluxDirection == FLUX_THETA_213) {
		cylgeomA = threadIdx.y*CYLGEO_DR + CYLGEO_RINI;
		}
	// The threads will step through z, so R is fixed and we can rescale the flux factor once:
	// We just scale lambda ( = dt / dtheta) by 1/r_c
	if(fluxDirection == FLUX_THETA_231) {
		lambda /= (blockIdx.y*CYLGEO_DR + CYLGEO_RINI);
	}

	/* Do some index calculations */
	x0 += DEV_NX*(DEV_NY*blockIdx.y + threadIdx.y); /* This block is now positioned to start at its given (x,z) coordinate */
	int j = threadIdx.y;
	Qin += x0;
	Qout += x0;

	int fluxmode;

	for(; j < DEV_NY; j += blockDim.y) {
		/* LOAD VARIABLES: CONSTANT APPROXIMATION -> NO RECONSTRUCTION STEP */	
		A = Qin[0             ]; /* rho; Load the q_i variables */
#ifdef HLLC_ENFORCE_MINVALS
		if(A < FLUID_MINMASS) {
			A = Qin[0] = FLUID_MINMASS;
		}
#endif
		B = Qin[  DEV_SLABSIZE]; /* E */
		switch(fluxDirection) {
		case FLUX_X: /* Variables are in normal order [px py pz] */
		case FLUX_RADIAL:
			C = Qin[2*DEV_SLABSIZE]; /* Px */
			D = Qin[3*DEV_SLABSIZE]; /* Py */
			E = Qin[4*DEV_SLABSIZE]; /* Pz */
			break;
		case FLUX_Y: /* Slabs are in order [py px pz] */
		case FLUX_THETA_213:
		case FLUX_THETA_231:
			C = Qin[3*DEV_SLABSIZE]; /* Px */
			D = Qin[2*DEV_SLABSIZE]; /* Py */
			E = Qin[4*DEV_SLABSIZE]; /* Pz */
			break;
		case FLUX_Z: /* Slabs are in order [pz py px] */
			C = Qin[4*DEV_SLABSIZE]; /* Px */
			D = Qin[3*DEV_SLABSIZE]; /* Py */
			E = Qin[2*DEV_SLABSIZE]; /* Pz */
			break;
		}

		/* Convert to primitive variables */
		F = 1.0/A;			

		B -= .5*(C*C+D*D+E*E)*F; /* internal energy */
#ifdef HLLC_ENFORCE_MINVALS
		if(B*F < FLUID_MINTEMP) {
			B = A*FLUID_MINTEMP;
			Qin[DEV_SLABSIZE] = .5*(C*C+D*D+E*E)*F + B;
		}
#endif

		C *= F; /* velocity */
		D *= F;
		E *= F;

		shptr[BOS0] = A; /* Upload rho  to shmem */
		shptr[BOS1] = FLUID_GM1*B; /* pressure */
		shptr[BOS2] = C; /* vx */
		shptr[BOS3] = D; /* vy */
		shptr[BOS4] = E; /* vz */

		__syncthreads();

		/* CALCULATE WAVESPEED ESTIMATE */
		C -= shblk[IR+BOS2]; /* Calculate Vleft - Vright */
		D -= shblk[IR+BOS3];
		E -= shblk[IR+BOS4];

		C = C*C+D*D+E*E; /* velocity jump, squared */

		// The compiler is observed to successfully inline this in all cases
		SpeedBounds sb = computeEinfeldtBounds(shblk[IC+BOS0], shblk[IC+BOS2], shblk[IC+BOS1], shblk[IR+BOS0], shblk[IR+BOS2], shblk[IR+BOS1], C);
		C = sb.Vleft;
		D = sb.Vright;
#ifdef DBG_FIRSTORDER
		DBGSAVE(0, C); DBGSAVE(2, D);
#endif

		A = shptr[BOS2]; /* vx_le */
		B = shblk[IR+BOS2]; /* vx_re */
		/* HLLC METHOD: COMPUTE SPEED IN STAR REGION */
		/* Now we have lambda_min in C and lambda_max in D; A, B and F are free. */
		__syncthreads(); // I don't think this is necessary...

		A = shptr[BOS0]*(C-A); // rho*(S-v) left
		B = shblk[IR+BOS0]*(D-B); // rho*(S-v) right

		/* Batten et al, Eqn 34 for S_M:
		Speed of HLLC contact
		 S_M =   rho_r Vr(Sr-Vr) - Pr - rho_l Vl(Sl-Vl) + Pl
			 -------------------------------------------
		                 rho_r(Sr-vr) - rho_l(Sl-vl)          */
		E = (B*shblk[IR+BOS2] - shblk[IR+BOS1] - A*shptr[BOS2] + shptr[BOS1]) / (B-A);
		__syncthreads();
		#ifdef DBG_FIRSTORDER
			DBGSAVE(1, E);
		#endif 
		
		if(E > 0) {
			if(C > 0) fluxmode = 0; // Fleft
			else fluxmode = 2; // F*left
		} else {
			if(D > 0) fluxmode = 3; // F*right
			else fluxmode = 1; // Fright
		}

		#ifdef DBG_FIRSTORDER
			DBGSAVE(3,1.0*fluxmode);
		#endif

		switch(fluxmode) {
			case 0:
				A = shptr[BOS0]; // rho
				B = shptr[BOS2]; // vx
				C = shptr[BOS3]; // vy
				D = shptr[BOS4]; // vz
				E = .5*(B*B+C*C+D*D)*A; // .5 v^2
				F = shptr[BOS1]; // P

				A *= B; // rho vx           = RHO FLUX
				C *= A; // rho vx vy        = PY FLUX
				D *= A; // rho vx vz        = PZ FLUX
				G = A*B+F; // rho vx vx + P = PX FLUX
				B *= (E+FLUID_GOVERGM1*F);//= ENERGY FLUX
				break;
			case 2: 
/* Case 3 differs from case 2 only in that C <-> D (Sleft for Sright) and loading right cell's left edge vs left cell's right edge */
				A = shptr[BOS0]; // rho
				B = shptr[BOS2]; // vx
				D = A*(C-B)/(C-E);  // rho*beta
				F = shptr[BOS3]; // vy
				H = shptr[BOS4]; // vz
				G = .5*(B*B+F*F+H*H); // .5v^2
				A = D*E; // RHO FLUX
				F *= A; // PY FLUX
				H *= A; // PZ FLUX

				C *= (E-B); // S(Sstar-vx)
				B = shptr[BOS1] + D*(C+E*B); // PX FLUX = P + rho beta (S(Sstar-vx) + Sstar*vx)
				G = D*E*(FLUID_GOVERGM1 * shptr[BOS1]/shptr[BOS0] + G + C);
				break;
			case 3:
				A = shblk[IR+BOS0]; // rho
				B = shblk[IR+BOS2]; // vx
				C = A*(D-B)/(D-E);  // rho*beta
				F = shblk[IR+BOS3]; // vy
				H = shblk[IR+BOS4]; // vz
				G = .5*(B*B+F*F+H*H); // .5v^2
				A = C*E; // RHO FLUX
				F *= A; // PY FLUX
				H *= A; // PZ FLUX

				D *= (E-B); // S(Sstar-vx)
				B = shblk[IR+BOS1] + C*(D+E*B); // PX FLUX = P + rho beta (S(Sstar-vx) + Sstar*vx)
				G = C*E*(FLUID_GOVERGM1 * shblk[IR+BOS1]/shblk[IR+BOS0] + G + D);
				break;
			case 1:
				A = shblk[IR+BOS0]; // rho
				B = shblk[IR+BOS2]; // vx
				C = shblk[IR+BOS3]; // vy
				D = shblk[IR+BOS4]; // vz
				E = .5*(B*B+C*C+D*D)*A; // .5 v^2
				F = shblk[IR+BOS1]; // P

				A *= B; // rho vx           = RHO FLUX
				C *= A; // rho vx vy        = PY FLUX
				D *= A; // rho vx vz        = PZ FLUX
				G = A*B+F; // rho vx vx + P = PX FLUX
				B *= (E+FLUID_GOVERGM1*F); //    = ENERGY FLUX
				break;

		} // case 0,1: [A B G C D] = [rho E px py pz] flux
		  // case 2,3: [A G B F H] = [rho E px py pz] flux

		if(fluxDirection == FLUX_RADIAL) {
			E = shptr[BOS1];
		} // If computing cylindrical radial flux save pressure to compute P dt / r geometric term.

		__syncthreads();
		switch(fluxmode) {
			case 0:
			case 1:
				shptr[BOS0] = A;
				shptr[BOS1] = B;
				shptr[BOS2] = G;
				shptr[BOS3] = C;
				shptr[BOS4] = D;
			break;
			case 2:
			case 3:
				shptr[BOS0] = A;
				shptr[BOS1] = G;
				shptr[BOS2] = B;
				shptr[BOS3] = F;
				shptr[BOS4] = H;
			break;
		}

		__syncthreads();
		// All fluxes are now uploaded to shmem:
		// shptr[BOS0, 1, 2, 3, 4] = flux of [rho, E, px, py, pz]

#ifdef DBG_FIRSTORDER
	DBGSAVE(4, shptr[BOS2]); // px flux
	DBGSAVE(5, shptr[BOS1]); // E flux
#endif

		if(thisThreadDelivers) {
			switch(fluxDirection) {
			case FLUX_X:
				Qout[0]              = Qin[0           ]   - lambda * ((double)shptr[BOS0]-(double)shblk[IL+BOS0]);
				Qout[DEV_SLABSIZE]   = Qin[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS1]-(double)shblk[IL+BOS1]);
				Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS2]-(double)shblk[IL+BOS2]);
				Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS3]-(double)shblk[IL+BOS3]);
				Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS4]-(double)shblk[IL+BOS4]);
				break;
			case FLUX_Y:
				Qout[0]              = Qin[0           ]   - lambda * ((double)shptr[BOS0]-(double)shblk[IL+BOS0]);
				Qout[DEV_SLABSIZE]   = Qin[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS1]-(double)shblk[IL+BOS1]);
				Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS2]-(double)shblk[IL+BOS2]);
				Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS3]-(double)shblk[IL+BOS3]);
				Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS4]-(double)shblk[IL+BOS4]);
				break;
			case FLUX_Z:
				Qout[0]              = Qin[0           ]   - lambda * ((double)shptr[BOS0]-(double)shblk[IL+BOS0]);
				Qout[DEV_SLABSIZE]   = Qin[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS1]-(double)shblk[IL+BOS1]);
				Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS2]-(double)shblk[IL+BOS2]);
				Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS3]-(double)shblk[IL+BOS3]);
				Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS4]-(double)shblk[IL+BOS4]);
				break;
			case FLUX_THETA_231:// NOTE lambda was rescaled in this path to lambda / r_center
				Qout[0]              = Qin[0           ]   - lambda * ((double)shptr[BOS0]-(double)shblk[IL+BOS0]);
				Qout[DEV_SLABSIZE]   = Qin[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS1]-(double)shblk[IL+BOS1]);
				Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS2]-(double)shblk[IL+BOS2]);
				Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS3]-(double)shblk[IL+BOS3]);
				Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS4]-(double)shblk[IL+BOS4]);
				break;
			case FLUX_RADIAL:
				Qout[0]              = Qin[0           ]   - lambda*(cylgeomB*(double)shptr[BOS0] - cylgeomA*(double)shblk[IL+BOS0]);
				Qout[DEV_SLABSIZE]   = Qin[DEV_SLABSIZE]   - lambda*(cylgeomB*(double)shptr[BOS1] - cylgeomA*(double)shblk[IL+BOS1]);
				Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * (  (double)(cylgeomB*shptr[BOS2])
																		-(double)(cylgeomA*shblk[IL+BOS2])
																		- cylgeomC*E); // (P dt / r) source term
				Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * ((double)(cylgeomB*shptr[BOS3])-(double)(cylgeomA*shblk[IL+BOS3]));
				Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * ((double)(cylgeomB*shptr[BOS4])-(double)(cylgeomA*shblk[IL+BOS4]));
				break;
			case FLUX_THETA_213: // p_theta
				Qout[0]              = Qin[0           ]   - lambda * ((double)shptr[BOS0]-(double)shblk[IL+BOS0]) / cylgeomA;
				Qout[DEV_SLABSIZE]   = Qin[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS1]-(double)shblk[IL+BOS1]) / cylgeomA;
				Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS2]-(double)shblk[IL+BOS2]) / cylgeomA;
				Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS3]-(double)shblk[IL+BOS3]) / cylgeomA;
				Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS4]-(double)shblk[IL+BOS4]) / cylgeomA;
				break;
			}
		}

		Qin += blockDim.y*DEV_NX;
		Qout += blockDim.y*DEV_NX;
		// Move the r_center coordinate out as the y-aligned blocks march in x (=r)
		if(fluxDirection == FLUX_THETA_213) {
			cylgeomA += YBLOCKS*CYLGEO_DR;
		}
	}

}

// Second-order HLLC uses 10 blocks
#define N_SHMEM_BLOCKS_SO 10

template <unsigned int fluxScheme>
__global__ void cukern_HLLC_2ndorder(double *Qin, double *Qout, double lambda)
{
	// These are built off templated vars, they'll be optimized away
	unsigned int fluxDirection = fluxScheme & 7; // 1-6 = x/y/z/r/theta213/theta231

	// Create center, look-left and look-right indexes
	int IC = threadIdx.x;
	int IL = threadIdx.x - 1;
	IL += 1*(IL < 0);
	int IR = threadIdx.x + 1;
	IR -= BLOCKLENP4*(IR > (BLOCKLENP4-1));

	// Advance by the Y index
	IC += N_SHMEM_BLOCKS_SO*BLOCKLENP4*threadIdx.y;
	IL += N_SHMEM_BLOCKS_SO*BLOCKLENP4*threadIdx.y;
	IR += N_SHMEM_BLOCKS_SO*BLOCKLENP4*threadIdx.y;

	/* Declare shared variable array */
#ifdef FLOATFLUX
	__shared__ float shblk[YBLOCKS*N_SHMEM_BLOCKS_SO*BLOCKLENP4];
		float *shptr = &shblk[IC];
		float A, B, C, D, E, F, G, H;
		float cylgeomA, cylgeomB, cylgeomC;
#else
	__shared__ double shblk[YBLOCKS*N_SHMEM_BLOCKS_SO*BLOCKLENP4];
	double *shptr = &shblk[IC];
	double A, B, C, D, E, F, G, H;
	double cylgeomA, cylgeomB, cylgeomC;
#endif

	/* My x index: thread + blocksize block, wrapped circularly */
	//int thisThreadPonders  = (threadIdx.x > 0) && (threadIdx.x < blockDim.x-1);
	int thisThreadDelivers = (threadIdx.x >= 2) && (threadIdx.x <= (BLOCKLENP4-3));

	int x0 = threadIdx.x + (BLOCKLEN)*blockIdx.x - 2;
	if(x0 < 0) x0 += DEV_NX; // left wraps to right edge
	if(x0 > (DEV_NX+1)) return; // More than 2 past right returns
	if(x0 > (DEV_NX-1)) { x0 -= DEV_NX; thisThreadDelivers = 0; } // past right must wrap around to left

	/* If doing cylindrical geometry... */
	// Compute multiple scale factors for radial direction fluxes
	if(fluxDirection == FLUX_RADIAL) { // cylindrical, R direction
		A = CYLGEO_RINI + x0 * CYLGEO_DR; // r_center
		cylgeomA = (A - .5*CYLGEO_DR) / (A);
		cylgeomB = (A + .5*CYLGEO_DR) / (A);
		cylgeomC = 0.5 * CYLGEO_DR / A; // NOTE: We use 0.5 here instead of 1.0 because we are inserting P = Pleft + Pright
										// At the fluxer stage to recover the average from the left/right values
	}
	// The kern will step through r, so we have to add to R and compute 1.0/R
	if(fluxDirection == FLUX_THETA_213) {
		cylgeomA = threadIdx.y*CYLGEO_DR + CYLGEO_RINI;
	}
	// The kerns will step through z so R is fixed and we compute it once
	// We just scale lambda ( = dt / dtheta) by 1/r_c
	if(fluxDirection == FLUX_THETA_231) {
		lambda /= (blockIdx.y*CYLGEO_DR + CYLGEO_RINI);
	}

	/* Do some index calculations */
	x0 += DEV_NX*(DEV_NY*blockIdx.y + threadIdx.y); /* This block is now positioned to start at its given (x,z) coordinate */
	int j = threadIdx.y;
	Qin += x0;
	Qout += x0;

	int fluxmode;

	for(; j < DEV_NY; j += blockDim.y) {
		/* LOAD VARIABLES AND PERFORM MUSCL RECONSTRUCTION */	
		A = Qin[0             ]; /* Load the q_i variables */
#ifdef HLLC_ENFORCE_MINVALS
		if(A < FLUID_MINMASS) {
			A = Qin[0] = FLUID_MINMASS;
		}
#endif
		B = Qin[  DEV_SLABSIZE];
		switch(fluxDirection) {
		case FLUX_X: /* Variables are in normal order [px py pz] */
		case FLUX_RADIAL:
			C = Qin[2*DEV_SLABSIZE]; /* Px */
			D = Qin[3*DEV_SLABSIZE]; /* Py */
			E = Qin[4*DEV_SLABSIZE]; /* Pz */
			break;
		case FLUX_Y: /* Slabs are in order [py px pz] */
		case FLUX_THETA_213:
		case FLUX_THETA_231:
			C = Qin[3*DEV_SLABSIZE]; /* Px */
			D = Qin[2*DEV_SLABSIZE]; /* Py */
			E = Qin[4*DEV_SLABSIZE]; /* Pz */
			break;
		case FLUX_Z: /* Slabs are in order [pz py px] */
			C = Qin[4*DEV_SLABSIZE]; /* Px */
			D = Qin[3*DEV_SLABSIZE]; /* Py */
			E = Qin[2*DEV_SLABSIZE]; /* Pz */
			break;
		}
		F = 1.0/A;			

		B -= .5*(C*C+D*D+E*E)*F; /* internal energy */
#ifdef HLLC_ENFORCE_MINVALS
		if(B*F < FLUID_MINTEMP) {
			B = A*FLUID_MINTEMP;
			Qin[DEV_SLABSIZE] = .5*(C*C+D*D+E*E)*F + B;
		}
#endif
#ifdef RECONSTRUCT_VELOCITY
		C *= F; /* velocity */
		D *= F;
		E *= F;
#endif

		shptr[BOS0] = A; /* Upload to shmem: rho, epsilon, vx, vy, vz */
		shptr[BOS2] = B;
		shptr[BOS4] = C; // store one of v or mom
		shptr[BOS6] = D;
		shptr[BOS8] = E;

		__syncthreads();
		/* Perform monotonic reconstruction */

		F = A - shblk[IL+BOS0];           // take backwards derivative
		shptr[BOS1] = F;               // upload
		__syncthreads();
		F = SLOPEFUNC(F, shblk[IR+BOS1]); // Compute monotonic slope
		__syncthreads();
		shptr[BOS0] = A-F;             // apply left/right corrections
		shptr[BOS1] = A+F;

		F = B - shblk[IL+BOS2]; 
		shptr[BOS3] = F;
		__syncthreads();
		F = SLOPEFUNC(F, shblk[IR+BOS3]);
		__syncthreads();
		shptr[BOS2] = FLUID_GM1*(B-F); // store PRESSURE, not epsilon
		shptr[BOS3] = FLUID_GM1*(B+F);

		F = C - shblk[IL+BOS4];
		shptr[BOS5] = F;
		__syncthreads();
		F = SLOPEFUNC(F, shblk[IR+BOS5]);
		__syncthreads();
		shptr[BOS4] = C-F;
		shptr[BOS5] = C+F;

		F = D - shblk[IL+BOS6]; 
		shptr[BOS7] = F;
		__syncthreads();
		F = SLOPEFUNC(F, shblk[IR+BOS7]);
		__syncthreads();
		shptr[BOS6] = D-F;
		shptr[BOS7] = D+F;

		F = E - shblk[IL+BOS8];
		shptr[BOS9] = F;
		__syncthreads();
		F = SLOPEFUNC(F, shblk[IR+BOS9]);
		__syncthreads();
		shptr[BOS8] = E-F;
		shptr[BOS9] = E+F;
		__syncthreads();

#ifdef RECONSTRUCT_VELOCITY
		// Nothing more to do, the approximate RP solver expects v here
#else
		shptr[BOS4] /= shptr[BOS0]; // V right side
		shptr[BOS6] /= shptr[BOS0];
		shptr[BOS8] /= shptr[BOS0];

		shptr[BOS5] /= shptr[BOS1];
		shptr[BOS7] /= shptr[BOS1];
		shptr[BOS9] /= shptr[BOS1];
		__syncthreads();
#endif

		/* CALCULATE WAVESPEED ESTIMATE */
		C = shptr[BOS5]-shblk[IR+BOS4]; /* Calculate Vleft - Vright */
		D = shptr[BOS7]-shblk[IR+BOS6];
		E = shptr[BOS9]-shblk[IR+BOS8];

		C = C*C+D*D+E*E; /* velocity jump, squared */

		// The compiler is observed to successfully inline this in all templates
		SpeedBounds sb = computeEinfeldtBounds(shblk[IC+BOS1], shblk[IC+BOS5], shblk[IC+BOS3], shblk[IR+BOS0], shblk[IR+BOS4], shblk[IR+BOS2], C);
		C = sb.Vleft;
		D = sb.Vright;

#ifdef DBG_SECONDORDER
		DBGSAVE(0, C);
		DBGSAVE(2, D);
#endif

		/* HLLC METHOD: COMPUTE SPEED IN STAR REGION */
		/* Now we have lambda_min in C and lambda_max in D; A, B and F are free. */
		__syncthreads(); // I don't think this is necessary...
		A = shblk[IC+BOS5]; /* vx_le */
		B = shblk[IR+BOS4]; /* vx_re */

		A = shptr[BOS1]*(C-A); // rho_le*(Sleft - vx_le)
		B = shblk[IR+BOS0]*(D-B); // rho_re*(Sright- vx_re)
		/* Batten et al, Eqn 34 for S_M:
		Speed of HLLC contact
		 S_M =   rho_r Vr(Sr-Vr) - Pr - rho_l Vl(Sl-Vl) + Pl
			 -------------------------------------------
		                 rho_r(Sr-vr) - rho_l(Sl-vl)          */
		E = (B*shblk[IR+BOS4] - shblk[IR+BOS2] - A*shptr[BOS5] + shptr[BOS3]) / (B-A);
		__syncthreads();
#ifdef DBG_SECONDODER
DBGSAVE(1, E);
#endif

		if(E > 0) {
			if(C > 0) fluxmode = 0; // Fleft
			else fluxmode = 2; // F*left
		} else {
			if(D > 0) fluxmode = 3; // F*right
			else fluxmode = 1; // Fright
		}

		switch(fluxmode) {
			case 0:
				A = shptr[BOS1]; // rho
				B = shptr[BOS5]; // vx
				C = shptr[BOS7]; // vy
				D = shptr[BOS9]; // vz
				E = .5*(B*B+C*C+D*D)*A; // .5 rho v^2 = KE
				F = shptr[BOS3]; // P

				A *= B; // rho vx           = RHO FLUX
				C *= A; // rho vx vy        = PY FLUX
				D *= A; // rho vx vz        = PZ FLUX
				G = A*B+F; // rho vx vx + P = PX FLUX
				B *= (E+FLUID_GOVERGM1*F); //    = ENERGY FLUX
				break;
			case 2: 
/* Case two differs from case 1 only in that C <-> D (Sleft for Sright) and loading right cell's left edge vs left cell's right edge */
				A = shptr[BOS1]; // rho
				B = shptr[BOS5]; // vx
				D = A*(C-B)/(C-E);  // rho*beta
				F = shptr[BOS7]; // vy
				H = shptr[BOS9]; // vz
				G = .5*(B*B+F*F+H*H); // .5v^2
				A = D*E; // RHO FLUX
				F *= A; // PY FLUX
				H *= A; // PZ FLUX

				C *= (E-B); // S(Sstar-vx)
				B = shptr[BOS3] + D*(C+E*B); // PX FLUX = P + rho beta (S(Sstar-vx) + Sstar*vx)
				G = D*E*(FLUID_GOVERGM1 * shptr[BOS3]/shptr[BOS1] + G + C);
				break;
			case 3:
				A = shblk[IR+BOS0]; // rho
				B = shblk[IR+BOS4]; // vx
				C = A*(D-B)/(D-E);  // rho*beta
				F = shblk[IR+BOS6]; // vy
				H = shblk[IR+BOS8]; // vz
				G = .5*(B*B+F*F+H*H); // .5v^2
				A = C*E; // RHO FLUX
				F *= A; // PY FLUX
				H *= A; // PZ FLUX

				D *= (E-B); // S(Sstar-vx)
				B = shblk[IR+BOS2] + C*(D+E*B); // PX FLUX = P + rho beta (S(Sstar-vx) + Sstar*vx)
				G = C*E*(FLUID_GOVERGM1 * shblk[IR+BOS2]/shblk[IR+BOS0] + G + D);
				break;
			case 1:
				A = shblk[IR+BOS0]; // rho
				B = shblk[IR+BOS4]; // vx
				C = shblk[IR+BOS6]; // vy
				D = shblk[IR+BOS8]; // vz
				E = .5*(B*B+C*C+D*D)*A; // .5 rho v^2
				F = shblk[IR+BOS2]; // P

				A *= B; // rho vx           = RHO FLUX
				C *= A; // rho vx vy        = PY FLUX
				D *= A; // rho vx vz        = PZ FLUX
				G = A*B+F; // rho vx vx + P = PX FLUX
				B *= (E+FLUID_GOVERGM1*F); //    = ENERGY FLUX
				break;

		}

		__syncthreads();
		switch(fluxmode) {
			case 0:
			case 1:
				shptr[BOS7] = A;
				shptr[BOS5] = B;
				shptr[BOS6] = G;
				shptr[BOS8] = C;
				shptr[BOS9] = D;
			break;
			case 2:
			case 3:
				shptr[BOS7] = A;
				shptr[BOS5] = G;
				shptr[BOS6] = B;
				shptr[BOS8] = F;
				shptr[BOS9] = H;
			break;
		}

#ifdef DBG_SECONDORDER
//DBGSAVE(3, shptr[BOS7]); // rho
//DBGSAVE(4, shptr[BOS6]); // px
//DBGSAVE(5, shptr[BOS5]); // E
#endif
		// The code does not break without this because we use 32xN size threadblocks and
		// warps execute synchronously, but cuda-memcheck --racecheck complains bitterly
		__syncthreads();

		if(thisThreadDelivers) {
			switch(fluxScheme) {
			case FLUX_X + TIMESCHEME_RK2:
			Qout[0]              -= lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]);
			Qout[DEV_SLABSIZE]   -= lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]);
			Qout[2*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]);
			Qout[3*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]);
			Qout[4*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]);
			break;
			case FLUX_Y + TIMESCHEME_RK2:
			Qout[0]              -= lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]);
			Qout[DEV_SLABSIZE]   -= lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]);
			Qout[3*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]);
			Qout[2*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]);
			Qout[4*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]);
			break;
			case FLUX_Z + TIMESCHEME_RK2:
			Qout[0]              -= lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]);
			Qout[DEV_SLABSIZE]   -= lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]);
			Qout[4*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]);
			Qout[3*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]);
			Qout[2*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]);
			break;
			case FLUX_RADIAL + TIMESCHEME_RK2:
			Qout[0]              -= lambda * (cylgeomB*(double)shptr[BOS7]-cylgeomA*(double)shblk[IL+BOS7]);
			Qout[DEV_SLABSIZE]   -= lambda * (cylgeomB*(double)shptr[BOS5]-cylgeomA*(double)shblk[IL+BOS5]);
			Qout[2*DEV_SLABSIZE] -= lambda * (cylgeomB*(double)shptr[BOS6]
			                                 -cylgeomA*(double)shblk[IL+BOS6]
			                                 -cylgeomC*(shptr[BOS2]+shptr[BOS3])); // (P dt / r) source term);
			Qout[3*DEV_SLABSIZE] -= lambda * (cylgeomB*(double)shptr[BOS8]-cylgeomA*(double)shblk[IL+BOS8]); // Note from earlier in fcn,
			Qout[4*DEV_SLABSIZE] -= lambda * (cylgeomB*(double)shptr[BOS9]-cylgeomA*(double)shblk[IL+BOS9]); // cylgeomC is rescaled by 1/2 since we average cell's left & right pressures
			break;
			case FLUX_THETA_213 + TIMESCHEME_RK2:
			Qout[0]              -= lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]) / cylgeomA;
			Qout[DEV_SLABSIZE]   -= lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]) / cylgeomA;
			Qout[3*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]) / cylgeomA;
			Qout[2*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]) / cylgeomA;
			Qout[4*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]) / cylgeomA;
			break;
			case FLUX_THETA_231 + TIMESCHEME_RK2: // NOTE in this case lambda was rescaled to lambda / r_center
			Qout[0]              -= lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]);
			Qout[DEV_SLABSIZE]   -= lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]);
			Qout[3*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]);
			Qout[2*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]);
			Qout[4*DEV_SLABSIZE] -= lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]);
			break;
			// SSPRK PREDICTOR TIME SCHEME (out = in + tau f'(in))
			case FLUX_X + TIMESCHEME_SSPRK_A:
			Qout[0]              = Qin[0]            - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]);
			Qout[DEV_SLABSIZE]   = Qin[DEV_SLABSIZE] - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]);
			Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE]-lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]);
			Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE]-lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]);
			Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE]-lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]);
			break;
			case FLUX_Y + TIMESCHEME_SSPRK_A:
			Qout[0]              = Qin[0]            - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]);
			Qout[DEV_SLABSIZE]   = Qin[DEV_SLABSIZE] - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]);
			Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE]-lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]);
			Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE]-lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]);
			Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE]-lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]);
			break;
			case FLUX_Z + TIMESCHEME_SSPRK_A:
			Qout[0]              = Qin[0]            - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]);
			Qout[DEV_SLABSIZE]   = Qin[DEV_SLABSIZE] - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]);
			Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE]-lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]);
			Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE]-lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]);
			Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE]-lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]);
			break;
			case FLUX_RADIAL + TIMESCHEME_SSPRK_A:
			Qout[0]              = Qin[0]             -lambda * (cylgeomB*(double)shptr[BOS7]-cylgeomA*(double)shblk[IL+BOS7]);
			Qout[DEV_SLABSIZE]   = Qin[DEV_SLABSIZE]  -lambda * (cylgeomB*(double)shptr[BOS5]-cylgeomA*(double)shblk[IL+BOS5]);
			Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE]-lambda * (cylgeomB*(double)shptr[BOS6]
			                                                    -cylgeomA*(double)shblk[IL+BOS6]
			                                                    -cylgeomC*(shptr[BOS2]+shptr[BOS3])); // (P dt / r) source term);
			Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE]-lambda * (cylgeomB*(double)shptr[BOS8]-cylgeomA*(double)shblk[IL+BOS8]); // Note from earlier in fcn,
			Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE]-lambda * (cylgeomB*(double)shptr[BOS9]-cylgeomA*(double)shblk[IL+BOS9]); // cylgeomC is rescaled by 1/2 since we average cell's left & right pressures
			break;
			case FLUX_THETA_213 + TIMESCHEME_SSPRK_A:
			Qout[0]              = Qin[0]             -lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]) / cylgeomA;
			Qout[DEV_SLABSIZE]   = Qin[DEV_SLABSIZE]  -lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]) / cylgeomA;
			Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE]-lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]) / cylgeomA;
			Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE]-lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]) / cylgeomA;
			Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE]-lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]) / cylgeomA;
			break;
			case FLUX_THETA_231 + TIMESCHEME_SSPRK_A: // NOTE in this case lambda was rescaled to lambda / r_center
			Qout[0]              = Qin[0]             -lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]);
			Qout[DEV_SLABSIZE]   = Qin[DEV_SLABSIZE]  -lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]);
			Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE]-lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]);
			Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE]-lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]);
			Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE]-lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]);
			break;
			// EXPLICIT TRAPEZOID CORRECTOR TIME SCHEME (out = .5*[in + out + tau*f'(in)] )
			case FLUX_X + TIMESCHEME_SSPRK_B:
			Qout[0]              = .5*(Qin[0]              + Qout[0]              - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]));
			Qout[DEV_SLABSIZE]   = .5*(Qin[DEV_SLABSIZE]   + Qout[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]));
			Qout[2*DEV_SLABSIZE] = .5*(Qin[2*DEV_SLABSIZE] + Qout[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]));
			Qout[3*DEV_SLABSIZE] = .5*(Qin[3*DEV_SLABSIZE] + Qout[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]));
			Qout[4*DEV_SLABSIZE] = .5*(Qin[4*DEV_SLABSIZE] + Qout[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]));
			break;
			case FLUX_Y + TIMESCHEME_SSPRK_B:
			Qout[0]              = .5*(Qin[0]              + Qout[0]              - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]));
			Qout[DEV_SLABSIZE]   = .5*(Qin[DEV_SLABSIZE]   + Qout[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]));
			Qout[3*DEV_SLABSIZE] = .5*(Qin[3*DEV_SLABSIZE] + Qout[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]));
			Qout[2*DEV_SLABSIZE] = .5*(Qin[2*DEV_SLABSIZE] + Qout[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]));
			Qout[4*DEV_SLABSIZE] = .5*(Qin[4*DEV_SLABSIZE] + Qout[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]));

			break;
			case FLUX_Z + TIMESCHEME_SSPRK_B:
			Qout[0]              = .5*(Qin[0]              + Qout[0]              - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]));
			Qout[DEV_SLABSIZE]   = .5*(Qin[DEV_SLABSIZE]   + Qout[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]));
			Qout[4*DEV_SLABSIZE] = .5*(Qin[4*DEV_SLABSIZE] + Qout[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]));
			Qout[3*DEV_SLABSIZE] = .5*(Qin[3*DEV_SLABSIZE] + Qout[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]));
			Qout[2*DEV_SLABSIZE] = .5*(Qin[2*DEV_SLABSIZE] + Qout[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]));
			break;
			// Because we use qin to calc derivs, we must write qin = .5(qin + qout + f'(qin))
			case FLUX_RADIAL + TIMESCHEME_SSPRK_B:
			Qout[0]              = .5*(Qin[0]            + Qout[0]             - lambda * (cylgeomB*(double)shptr[BOS7]-cylgeomA*(double)shblk[IL+BOS7]));
			Qout[DEV_SLABSIZE]   = .5*(Qin[DEV_SLABSIZE]  +Qout[DEV_SLABSIZE]  - lambda * (cylgeomB*(double)shptr[BOS5]-cylgeomA*(double)shblk[IL+BOS5]));
			Qout[2*DEV_SLABSIZE] = .5*(Qin[2*DEV_SLABSIZE]+Qout[2*DEV_SLABSIZE] - lambda * (cylgeomB*(double)shptr[BOS6]
			                                                                               -cylgeomA*(double)shblk[IL+BOS6]
			                                                                               -cylgeomC*(shptr[BOS2]+shptr[BOS3]))); // (P dt / r) source term);
			Qout[3*DEV_SLABSIZE] = .5*(Qin[3*DEV_SLABSIZE]+Qout[3*DEV_SLABSIZE] - lambda * (cylgeomB*(double)shptr[BOS8]-cylgeomA*(double)shblk[IL+BOS8])); // Note from earlier in fcn,
			Qout[4*DEV_SLABSIZE] = .5*(Qin[4*DEV_SLABSIZE]+Qout[4*DEV_SLABSIZE] - lambda * (cylgeomB*(double)shptr[BOS9]-cylgeomA*(double)shblk[IL+BOS9])); // cylgeomC is rescaled by 1/2 since we average cell's left & right pressures
			break;
			case FLUX_THETA_213 + TIMESCHEME_SSPRK_B:
			Qout[0]              = .5*(Qout[0]            + Qin[0]            - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]) / cylgeomA);
			Qout[DEV_SLABSIZE]   = .5*(Qout[DEV_SLABSIZE] + Qin[DEV_SLABSIZE] - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]) / cylgeomA);
			Qout[3*DEV_SLABSIZE] = .5*(Qout[3*DEV_SLABSIZE]+Qin[3*DEV_SLABSIZE]-lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]) / cylgeomA);
			Qout[2*DEV_SLABSIZE] = .5*(Qout[2*DEV_SLABSIZE]+Qin[2*DEV_SLABSIZE]-lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]) / cylgeomA);
			Qout[4*DEV_SLABSIZE] = .5*(Qout[4*DEV_SLABSIZE]+Qin[4*DEV_SLABSIZE]-lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]) / cylgeomA);
			break;
			case FLUX_THETA_231 + TIMESCHEME_SSPRK_B: // NOTE in this case lambda was rescaled to lambda / r_center
			Qout[0]              = .5*(Qout[0]            + Qin[0]            - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]));
			Qout[DEV_SLABSIZE]   = .5*(Qout[DEV_SLABSIZE]  +Qin[DEV_SLABSIZE]  -lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]));
			Qout[3*DEV_SLABSIZE] = .5*(Qout[3*DEV_SLABSIZE]+Qin[3*DEV_SLABSIZE]-lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]));
			Qout[2*DEV_SLABSIZE] = .5*(Qout[2*DEV_SLABSIZE]+Qin[2*DEV_SLABSIZE]-lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]));
			Qout[4*DEV_SLABSIZE] = .5*(Qout[4*DEV_SLABSIZE]+Qin[4*DEV_SLABSIZE]-lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]));
			break;

			// remember 'in' is tempmem and 'out' is y0
			// RK3 STAGE TWO: (in[5-9] = .75 * out + .25 * in[0-4] + .25*tau*f'(in)] )
			case FLUX_X + TIMESCHEME_RK3_B:
			Qin[5*DEV_SLABSIZE] = .75*Qout[0]              + .25*(Qin[0]              - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]));
			Qin[6*DEV_SLABSIZE] = .75*Qout[DEV_SLABSIZE]   + .25*(Qin[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]));
			Qin[7*DEV_SLABSIZE] = .75*Qout[2*DEV_SLABSIZE] + .25*(Qin[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]));
			Qin[8*DEV_SLABSIZE] = .75*Qout[3*DEV_SLABSIZE] + .25*(Qin[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]));
			Qin[9*DEV_SLABSIZE] = .75*Qout[4*DEV_SLABSIZE] + .25*(Qin[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]));
			break;
			case FLUX_Y + TIMESCHEME_RK3_B:
			Qin[5*DEV_SLABSIZE] = .75*Qout[0]              + .25*(Qin[0]              - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]));
			Qin[6*DEV_SLABSIZE] = .75*Qout[DEV_SLABSIZE]   + .25*(Qin[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]));
			Qin[8*DEV_SLABSIZE] = .75*Qout[3*DEV_SLABSIZE] + .25*(Qin[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]));
			Qin[7*DEV_SLABSIZE] = .75*Qout[2*DEV_SLABSIZE] + .25*(Qin[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]));
			Qin[9*DEV_SLABSIZE] = .75*Qout[4*DEV_SLABSIZE] + .25*(Qin[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]));
			break;
			case FLUX_Z + TIMESCHEME_RK3_B:
			Qin[5*DEV_SLABSIZE] = .75*Qout[0]              + .25*(Qin[0]              - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]));
			Qin[6*DEV_SLABSIZE] = .75*Qout[DEV_SLABSIZE]   + .25*(Qin[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]));
			Qin[9*DEV_SLABSIZE] = .75*Qout[4*DEV_SLABSIZE] + .25*(Qin[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]));
			Qin[8*DEV_SLABSIZE] = .75*Qout[3*DEV_SLABSIZE] + .25*(Qin[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]));
			Qin[7*DEV_SLABSIZE] = .75*Qout[2*DEV_SLABSIZE] + .25*(Qin[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]));
			break;

			case FLUX_RADIAL + TIMESCHEME_RK3_B:
			Qin[5*DEV_SLABSIZE] = .75*Qout[0]              + .25*(Qin[0]             - lambda * (cylgeomB*(double)shptr[BOS7]-cylgeomA*(double)shblk[IL+BOS7]));
			Qin[6*DEV_SLABSIZE] = .75*Qout[DEV_SLABSIZE]   + .25*(Qin[DEV_SLABSIZE]  - lambda * (cylgeomB*(double)shptr[BOS5]-cylgeomA*(double)shblk[IL+BOS5]));
			Qin[7*DEV_SLABSIZE] = .75*Qout[2*DEV_SLABSIZE] + .25*(Qin[2*DEV_SLABSIZE] - lambda * (cylgeomB*(double)shptr[BOS6]
			                                                                              -cylgeomA*(double)shblk[IL+BOS6]
			                                                                              -cylgeomC*(shptr[BOS2]+shptr[BOS3]))); // (P dt / r) source term);
			Qin[8*DEV_SLABSIZE] = .75*Qout[3*DEV_SLABSIZE] + .25*(Qin[3*DEV_SLABSIZE] - lambda * (cylgeomB*(double)shptr[BOS8]-cylgeomA*(double)shblk[IL+BOS8])); // Note from earlier in fcn,
			Qin[9*DEV_SLABSIZE] = .75*Qout[4*DEV_SLABSIZE] + .25*(Qin[4*DEV_SLABSIZE] - lambda * (cylgeomB*(double)shptr[BOS9]-cylgeomA*(double)shblk[IL+BOS9])); // cylgeomC is rescaled by 1/2 since we average cell's left & right pressures
			break;
			case FLUX_THETA_213 + TIMESCHEME_RK3_B:
			Qin[5*DEV_SLABSIZE] = .75*Qout[0]              + .25*(Qin[0]            - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]) / cylgeomA);
			Qin[6*DEV_SLABSIZE] = .75*Qout[DEV_SLABSIZE]   + .25*(Qin[DEV_SLABSIZE] - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]) / cylgeomA);
			Qin[8*DEV_SLABSIZE] = .75*Qout[3*DEV_SLABSIZE] + .25*(Qin[3*DEV_SLABSIZE]-lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]) / cylgeomA);
			Qin[7*DEV_SLABSIZE] = .75*Qout[2*DEV_SLABSIZE] + .25*(Qin[2*DEV_SLABSIZE]-lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]) / cylgeomA);
			Qin[9*DEV_SLABSIZE] = .75*Qout[4*DEV_SLABSIZE] + .25*(Qin[4*DEV_SLABSIZE]-lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]) / cylgeomA);
			break;
			case FLUX_THETA_231 + TIMESCHEME_RK3_B: // NOTE in this case lambda was rescaled to lambda / r_center
			Qin[5*DEV_SLABSIZE] = .75*Qout[0]              + .25*(Qin[0]            - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]));
			Qin[6*DEV_SLABSIZE] = .75*Qout[DEV_SLABSIZE]   + .25*(Qin[DEV_SLABSIZE]  -lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]));
			Qin[8*DEV_SLABSIZE] = .75*Qout[3*DEV_SLABSIZE] + .25*(Qin[3*DEV_SLABSIZE]-lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]));
			Qin[7*DEV_SLABSIZE] = .75*Qout[2*DEV_SLABSIZE] + .25*(Qin[2*DEV_SLABSIZE]-lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]));
			Qin[9*DEV_SLABSIZE] = .75*Qout[4*DEV_SLABSIZE] + .25*(Qin[4*DEV_SLABSIZE]-lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]));
			break;


			// remember 'in' is tempmem and 'out' is y0
			// this stage must be passed &tempmem[5*DEV_SLABSIZE]!
			// RK3 STAGE THREE: out = .333 * out + .666 * (in[0-4] + tau*f'(in)] )
			// WARNING this must have lambda doubled as can be seen below!
			case FLUX_X + TIMESCHEME_RK3_C:
			Qout[0]              = (Qout[0]              + 2*Qin[0]              - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]))/3.0;
			Qout[DEV_SLABSIZE]   = (Qout[DEV_SLABSIZE]   + 2*Qin[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]))/3.0;
			Qout[2*DEV_SLABSIZE] = (Qout[2*DEV_SLABSIZE] + 2*Qin[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]))/3.0;
			Qout[3*DEV_SLABSIZE] = (Qout[3*DEV_SLABSIZE] + 2*Qin[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]))/3.0;
			Qout[4*DEV_SLABSIZE] = (Qout[4*DEV_SLABSIZE] + 2*Qin[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]))/3.0;
			break;
			case FLUX_Y + TIMESCHEME_RK3_C:
			Qout[0]              = (Qout[0]              + 2*Qin[0]              - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]))/3.0;
			Qout[DEV_SLABSIZE]   = (Qout[DEV_SLABSIZE]   + 2*Qin[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]))/3.0;
			Qout[3*DEV_SLABSIZE] = (Qout[3*DEV_SLABSIZE] + 2*Qin[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]))/3.0;
			Qout[2*DEV_SLABSIZE] = (Qout[2*DEV_SLABSIZE] + 2*Qin[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]))/3.0;
			Qout[4*DEV_SLABSIZE] = (Qout[4*DEV_SLABSIZE] + 2*Qin[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]))/3.0;

			break;
			case FLUX_Z + TIMESCHEME_RK3_C:
			Qout[0]              = (Qout[0]              + 2*Qin[0]              - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]))/3.0;
			Qout[DEV_SLABSIZE]   = (Qout[DEV_SLABSIZE]   + 2*Qin[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]))/3.0;
			Qout[4*DEV_SLABSIZE] = (Qout[4*DEV_SLABSIZE] + 2*Qin[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]))/3.0;
			Qout[3*DEV_SLABSIZE] = (Qout[3*DEV_SLABSIZE] + 2*Qin[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]))/3.0;
			Qout[2*DEV_SLABSIZE] = (Qout[2*DEV_SLABSIZE] + 2*Qin[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]))/3.0;
			break;
			// Because we use qin to calc derivs, we must write qin = .5(qin + qout + f'(qin))
			case FLUX_RADIAL + TIMESCHEME_RK3_C:
			Qout[0]              = (Qout[0]              + 2*Qin[0]              - lambda * (cylgeomB*(double)shptr[BOS7]-cylgeomA*(double)shblk[IL+BOS7]))/3.0;
			Qout[DEV_SLABSIZE]   = (Qout[DEV_SLABSIZE]   + 2*Qin[DEV_SLABSIZE]   - lambda * (cylgeomB*(double)shptr[BOS5]-cylgeomA*(double)shblk[IL+BOS5]))/3.0;
			Qout[2*DEV_SLABSIZE] = (Qout[2*DEV_SLABSIZE] + 2*Qin[2*DEV_SLABSIZE] - lambda * (cylgeomB*(double)shptr[BOS6]
			                                                                                -cylgeomA*(double)shblk[IL+BOS6]
			                                                                                -cylgeomC*(shptr[BOS2]+shptr[BOS3])))/3.0; // (P dt / r) source term);
			Qout[3*DEV_SLABSIZE] = (Qout[3*DEV_SLABSIZE] + 2*Qin[3*DEV_SLABSIZE] - lambda * (cylgeomB*(double)shptr[BOS8]-cylgeomA*(double)shblk[IL+BOS8]))/3.0; // Note from earlier in fcn,
			Qout[4*DEV_SLABSIZE] = (Qout[4*DEV_SLABSIZE] + 2*Qin[4*DEV_SLABSIZE] - lambda * (cylgeomB*(double)shptr[BOS9]-cylgeomA*(double)shblk[IL+BOS9]))/3.0; // cylgeomC is rescaled by 1/2 since we average cell's left & right pressures
			break;
			case FLUX_THETA_213 + TIMESCHEME_RK3_C:
			Qout[0]              = (Qout[0]              + 2*Qin[0]              - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]) / cylgeomA)/3.0;
			Qout[DEV_SLABSIZE]   = (Qout[DEV_SLABSIZE]   + 2*Qin[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]) / cylgeomA)/3.0;
			Qout[3*DEV_SLABSIZE] = (Qout[3*DEV_SLABSIZE] + 2*Qin[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]) / cylgeomA)/3.0;
			Qout[2*DEV_SLABSIZE] = (Qout[2*DEV_SLABSIZE] + 2*Qin[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]) / cylgeomA)/3.0;
			Qout[4*DEV_SLABSIZE] = (Qout[4*DEV_SLABSIZE] + 2*Qin[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]) / cylgeomA)/3.0;
			break;
			case FLUX_THETA_231 + TIMESCHEME_RK3_C: // NOTE in this case lambda was rescaled to lambda / r_center
			Qout[0]              = (Qout[0]              + 2*Qin[0]              - lambda * ((double)shptr[BOS7]-(double)shblk[IL+BOS7]))/3.0;
			Qout[DEV_SLABSIZE]   = (Qout[DEV_SLABSIZE]   + 2*Qin[DEV_SLABSIZE]   - lambda * ((double)shptr[BOS5]-(double)shblk[IL+BOS5]))/3.0;
			Qout[3*DEV_SLABSIZE] = (Qout[3*DEV_SLABSIZE] + 2*Qin[3*DEV_SLABSIZE] - lambda * ((double)shptr[BOS6]-(double)shblk[IL+BOS6]))/3.0;
			Qout[2*DEV_SLABSIZE] = (Qout[2*DEV_SLABSIZE] + 2*Qin[2*DEV_SLABSIZE] - lambda * ((double)shptr[BOS8]-(double)shblk[IL+BOS8]))/3.0;
			Qout[4*DEV_SLABSIZE] = (Qout[4*DEV_SLABSIZE] + 2*Qin[4*DEV_SLABSIZE] - lambda * ((double)shptr[BOS9]-(double)shblk[IL+BOS9]))/3.0;
			break;

			}
		}

		Qin += blockDim.y*DEV_NX;
		Qout += blockDim.y*DEV_NX;

		// Move the r_center coordinate out as the y-aligned blocks march in x (=r)
		if(fluxDirection == FLUX_THETA_213) {
			cylgeomA += YBLOCKS*CYLGEO_DR;
		}
		__syncthreads();

	}

}

#define N_SHMEM_BLOCKS 6

#define HLL_LEFT 0
#define HLL_HLL  1
#define HLL_RIGHT 2

#define HLLTEMPLATE_XDIR 0
#define HLLTEMPLATE_YDIR 2
#define HLLTEMPLATE_ZDIR 4

__device__ SpeedBounds computeEinfeldtBounds(double rhoL, double vL, double PL, double rhoR, double vR, double PR, double deltaVsq)
{
	if(1) {
		// The intended, optimized, manifestly Galilean invariant formulation
		double csql = FLUID_GAMMA*PL/rhoL;
		double csqr = FLUID_GAMMA*PR/rhoR;

		// Compute roe coefficients once
		double Fa = SQRTFUNC(rhoR/rhoL);
		double Fb = 1.0/(1.0+Fa);

		double Croe = ((csql + Fa*csqr) + .5*FLUID_GM1*deltaVsq*Fa*Fb)*Fb;

		SpeedBounds sb;

		double Vroe = (vL + Fa*vR)*Fb;
		Croe = Vroe-SQRTFUNC(Croe); // Roe-avg sound speed finally

		sb.Vleft = vL - SQRTFUNC(csql); // sonic speeds
		sb.Vright = vR + SQRTFUNC(csqr);

		sb.Vleft = (Croe < sb.Vleft) ? Croe : sb.Vleft;

		Croe = 2*Vroe - Croe;

		sb.Vright = (Croe > sb.Vright) ? Croe : sb.Vright;
		return sb;
	} else {
		// This is a completely naive implementation of the Einfeldt bound
		double hl = .5*vL*vL + FLUID_GOVERGM1 * PL / rhoL;
		double hr = .5*vR*vR + FLUID_GOVERGM1 * PR / rhoR;

		double hbar = (sqrt(rhoL)*hl + sqrt(rhoR)*hr)/(sqrt(rhoL)+sqrt(rhoR));
		double vbar = (sqrt(rhoL)*vL + sqrt(rhoR)*vR)/(sqrt(rhoL)+sqrt(rhoR));

		double cbar = vbar - FLUID_GM1 * (hbar - .5*vbar*vbar);
		SpeedBounds sb;

		sb.Vleft = vL - sqrt(FLUID_GAMMA*PL/rhoL);
		sb.Vleft = (sb.Vleft < cbar) ? sb.Vleft : cbar;
		cbar = -cbar + 2*vbar;
		sb.Vright= vR + sqrt(FLUID_GAMMA*PR/rhoR);
		sb.Vright = (sb.Vright > cbar) ? sb.Vright : cbar;
		return sb;
	}
}


template <unsigned int PCswitch>
__global__ void cukern_HLL_step(double *Qin, double *Qstore, double lambda)
{
	// Create center, rotate-left and rotate-right indexes
	int IC = threadIdx.x;
	int IL = threadIdx.x - 1;
	IL += 1*(IL < 0);
	int IR = threadIdx.x + 1;
	IR -= BLOCKLENP4*(IR > (BLOCKLENP4-1));

        // Advance by the Y index
 	IC += N_SHMEM_BLOCKS*BLOCKLENP4*threadIdx.y;
	IL += N_SHMEM_BLOCKS*BLOCKLENP4*threadIdx.y;
	IR += N_SHMEM_BLOCKS*BLOCKLENP4*threadIdx.y;

	/* Declare variable arrays */
	__shared__ double shblk[YBLOCKS*N_SHMEM_BLOCKS*BLOCKLENP4];
	double Ale, Ble, Cle, Dle, Ele;
	double Are, Bre, Cre, Dre, Ere;
	int HLL_FluxMode;
	double Sleft, Sright, Utilde, Atilde;
	double Fa, Fb; /* temp vars */

	/* My x index: thread + blocksize block, wrapped circularly */
	int thisThreadDelivers = (threadIdx.x >= 2) && (threadIdx.x <= (BLOCKLENP4-3));

	int x0 = threadIdx.x + (BLOCKLEN)*blockIdx.x - 2;
	if(x0 < 0) x0 += DEV_NX; // left wraps to right edge
	if(x0 > (DEV_NX+1)) return; // More than 2 past right returns
	if(x0 > (DEV_NX-1)) { x0 -= DEV_NX; thisThreadDelivers = 0; } // past right must wrap around to left

	double cylgeomA, cylgeomB, cylgeomC;
	// If doing cylindrical geometry...
	// Compute multiple scale factors for radial direction fluxes
	int fluxDirection = (PCswitch / 2) + 1; // will be evaluted away @ compile time

	if(fluxDirection == FLUX_RADIAL) { // cylindrical, R direction
		Fa = CYLGEO_RINI + x0 * CYLGEO_DR; // r_center
		cylgeomA = lambda* (Fa - .5*CYLGEO_DR) / (Fa);
		cylgeomB = lambda* (Fa + .5*CYLGEO_DR) / (Fa);
		cylgeomC = lambda* CYLGEO_DR / Fa;
	}
	// The kern will step through r, so we have to add to R and compute 1.0/R
	if(fluxDirection == FLUX_THETA_213) {
		cylgeomA = threadIdx.y*CYLGEO_DR + CYLGEO_RINI;
	}
	// The kerns will step through z so R is fixed and we compute it once
	// We just scale lambda ( = dt / dtheta) by 1/r_c
	if(fluxDirection == FLUX_THETA_231) {
		lambda /= (blockIdx.y*CYLGEO_DR + CYLGEO_RINI);
	}

	/* Do some index calculations */
	x0 += DEV_NX*(DEV_NY*blockIdx.y + threadIdx.y); /* This block is now positioned to start at its given (x,z) coordinate */
	int j = threadIdx.y;

	Qin += x0;
	Qstore += x0;

	for(; j < DEV_NY; j += blockDim.y) {

		if((PCswitch & 1) == RK_PREDICT) {
			/* If making prediction use simple 0th order "reconstruction." */
			Ale = Are = Qin[0*DEV_SLABSIZE]; /* load rho */

			switch(fluxDirection) {
			case FLUX_X:
			case FLUX_RADIAL:
				Bre = Qin[2*DEV_SLABSIZE]; /* load px as px */
				Dre = Qin[3*DEV_SLABSIZE]; // py = py
				Ere = Qin[4*DEV_SLABSIZE]; // pz = pz
				break;
			case FLUX_Y:
			case FLUX_THETA_213:
			case FLUX_THETA_231:
				Bre = Qin[3*DEV_SLABSIZE]; /* load py as px */
				Dre = Qin[2*DEV_SLABSIZE]; // px = py
				Ere = Qin[4*DEV_SLABSIZE]; // pz = pz
				break;
			case FLUX_Z:
				Bre = Qin[4*DEV_SLABSIZE]; /* load pz as px */
				Dre = Qin[2*DEV_SLABSIZE]; // py = px
				Ere = Qin[3*DEV_SLABSIZE]; // pz = py;
				break;
			}

			// We calculated the gas pressure into temp array # 6 before calling
			Cle = Cre = Qstore[5*DEV_SLABSIZE]; /* load gas pressure */
			Ble = Bre / Ale; /* Calculate vx */
			Dle = Dre / Ale; // vy
			Ele = Ere / Ale; // vz
#ifdef DBG_FIRSTORDER
		DBSAVE_HLL(0, Ale);
		DBSAVE_HLL(1, Ble);
		DBSAVE_HLL(2, Cle);
		DBSAVE_HLL(3, Dle);
		DBSAVE_HLL(4, Ele);
		#endif

#ifndef USE_SSPRK
			Bre = Ble; // naive 0th order construction
			Dre = Dle;
			Ere = Ele;
#else
			// SSPRK methods requires that every time derivative calculation be equally spatially accurate
			// So we must use slope limited reconstruction here
			shblk[IC + BOS0] = Ale;
			shblk[IC + BOS1] = Ble;
			shblk[IC + BOS2] = Cle;
			shblk[IC + BOS3] = Dle;
			shblk[IC + BOS4] = Ele;
			__syncthreads();

			/*************** BEGIN SECTION 2 compute backward differences */
			Are = Ale - shblk[IL + BOS0];
			Bre = Ble - shblk[IL + BOS1];
			Cre = Cle - shblk[IL + BOS2];
			Dre = Dle - shblk[IL + BOS3];
			Ere = Ele - shblk[IL + BOS4];
			__syncthreads();

			/*************** BEGIN SECTION 3 upload backward differences */
			shblk[IC + BOS0] = Are;
			shblk[IC + BOS1] = Bre;
			shblk[IC + BOS2] = Cre;
			shblk[IC + BOS3] = Dre;
			shblk[IC + BOS4] = Ere;
			__syncthreads();

			/*************** BEGIN SECTION 4 compute slope using my BD & right's BD */
			Fa = SLOPEFUNC(Are, shblk[IR + BOS0]);
			Are = Ale + Fa;
			Ale -= Fa;

			Fa = SLOPEFUNC(Bre, shblk[IR + BOS1]);
			Bre = Ble + Fa;
			Ble -= Fa;

			Fa = SLOPEFUNC(Cre, shblk[IR + BOS2]);
			Cre = Cle + Fa;
			Cle -= Fa;

			Fa = SLOPEFUNC(Dre, shblk[IR + BOS3]);
			Dre = Dle + Fa;
			Dle -= Fa;

			Fa = SLOPEFUNC(Ere, shblk[IR + BOS4]);
			Ere = Ele + Fa;
			Ele -= Fa;
#endif
		} else {
			/* If making correction, perform linear MUSCL reconstruction */
			Ale = Qstore[0*DEV_SLABSIZE]; /* load rho */
			switch(fluxDirection) {
			case FLUX_X:
			case FLUX_RADIAL:
				Bre = Qstore[2*DEV_SLABSIZE]; /* load px as px */
				Dre = Qstore[3*DEV_SLABSIZE]; // py = py
				Ere = Qstore[4*DEV_SLABSIZE]; // pz = pz
				break;
			case FLUX_Y:
			case FLUX_THETA_213:
			case FLUX_THETA_231:
				Bre = Qstore[3*DEV_SLABSIZE]; /* load py as px */
				Dre = Qstore[2*DEV_SLABSIZE]; // px = py
				Ere = Qstore[4*DEV_SLABSIZE]; // pz = pz
				break;
			case FLUX_Z:
				Bre = Qstore[4*DEV_SLABSIZE]; /* load pz as px */
				Dre = Qstore[2*DEV_SLABSIZE]; // py = px
				Ere = Qstore[3*DEV_SLABSIZE]; // pz = py;
				break;
			}

			Cle = Qstore[5*DEV_SLABSIZE]; /* load pressure */
			Ble = Bre / Ale; /* Calculate vx */
			Dle = Dre / Ale;
			Ele = Ere / Ale;

			shblk[IC + BOS0] = Ale;
			shblk[IC + BOS1] = Ble;
			shblk[IC + BOS2] = Cle;
			shblk[IC + BOS3] = Dle;
			shblk[IC + BOS4] = Ele;
			__syncthreads();

			/*************** BEGIN SECTION 2 compute backward differences */
			Are = Ale - shblk[IL + BOS0];
			Bre = Ble - shblk[IL + BOS1];
			Cre = Cle - shblk[IL + BOS2];
			Dre = Dle - shblk[IL + BOS3];
			Ere = Ele - shblk[IL + BOS4];
			__syncthreads();

			/*************** BEGIN SECTION 3 upload backward differences */
			shblk[IC + BOS0] = Are;
			shblk[IC + BOS1] = Bre;
			shblk[IC + BOS2] = Cre;
			shblk[IC + BOS3] = Dre;
			shblk[IC + BOS4] = Ere;
			__syncthreads();

			/*************** BEGIN SECTION 4 compute slope using my BD & right's BD */
			Fa = SLOPEFUNC(Are, shblk[IR + BOS0]);
			Are = Ale + Fa;
			Ale -= Fa;

			Fa = SLOPEFUNC(Bre, shblk[IR + BOS1]);
			Bre = Ble + Fa;
			Ble -= Fa;

			Fa = SLOPEFUNC(Cre, shblk[IR + BOS2]);
			Cre = Cle + Fa;
			Cle -= Fa;

			Fa = SLOPEFUNC(Dre, shblk[IR + BOS3]);
			Dre = Dle + Fa;
			Dle -= Fa;

			Fa = SLOPEFUNC(Ere, shblk[IR + BOS4]);
			Ere = Ele + Fa;
			Ele -= Fa;
		}

		__syncthreads();
		/* Rotate the [le_i-1 re_i-1][le_i re_i][le_i+1 re_i+1] variables one left
		 * so that each cell stores [re_i le_i+1]
		 * and thus each thread deals with F_i+1/2 */
		shblk[IC + BOS0] = Ale;
		shblk[IC + BOS1] = Ble;
		shblk[IC + BOS2] = Cle;
		shblk[IC + BOS3] = Dle;
		shblk[IC + BOS4] = Ele;

		__syncthreads();

		Ale = Are; Are = shblk[IR + BOS0];
		shblk[IC + BOS5] = Ale; // store this for later...
		Ble = Bre; Bre = shblk[IR + BOS1];
		Cle = Cre; Cre = shblk[IR + BOS2];
		Dle = Dre; Dre = shblk[IR + BOS3];
		Ele = Ere; Ere = shblk[IR + BOS4];

		/* Get  velocity jump speeds */
		Utilde = (Bre-Ble);
		Sleft  = (Dre-Dle);
		Sright = (Ere-Ele);

		SpeedBounds sb = computeEinfeldtBounds(Ale, Ble, Cle, Are, Bre, Cre, Utilde*Utilde+Sleft*Sleft+Sright*Sright);
		Sleft = sb.Vleft;
		Sright= sb.Vright;

		/* Compute the reciprocal used by the HLL flux mode many times, in advance */
		Atilde = 1/(Sright - Sleft);

		__syncthreads();

		Fa = Ale*Ble; // Raw mass fluxes
		Fb = Are*Bre;

		/* Determine where our flux originates from (left, HLL, or right regions) */
		HLL_FluxMode = HLL_HLL;
		if(Sleft >= 0) HLL_FluxMode = HLL_LEFT;
		if(Sright<= 0) HLL_FluxMode = HLL_RIGHT;

		/* Calculate the mass and momentum fluxes */
		switch(HLL_FluxMode) {
		case HLL_LEFT:
			shblk[IC + BOS0] = Fa; // mass flux
			shblk[IC + BOS1] = Ble * (FLUID_GOVERGM1*Cle + .5*Ale*(Ble*Ble+Dle*Dle+Ele*Ele)); // ener flux
			shblk[IC + BOS2] = Fa*Ble + Cle; // px flux
			shblk[IC + BOS3] = Fa*Dle; // py flux
			shblk[IC + BOS4] = Fa*Ele; // pz flux
			break;
		case HLL_HLL:

			shblk[IC + BOS0] = (Sright*(Ble-Sleft)*Ale     - Sleft*(Bre-Sright)*Are    )*Atilde;
			shblk[IC + BOS2] = (Sright*(Ble-Sleft)*Fa      - Sleft*(Bre-Sright)*Fb + Sright*Cle - Sleft*Cre)*Atilde;
			shblk[IC + BOS3] = (Sright*(Ble-Sleft)*Ale*Dle - Sleft*(Bre-Sright)*Are*Dre)*Atilde;
			shblk[IC + BOS4] = (Sright*(Ble-Sleft)*Ale*Ele - Sleft*(Bre-Sright)*Are*Ere)*Atilde;
			// Compute the other fluxes first, now we can trash Fa and Fb
			Fa = .5*Ale*(Ble*Ble+Dle*Dle+Ele*Ele); // KE densities
			Fb = .5*Are*(Bre*Bre+Dre*Dre+Ere*Ere);
			shblk[IC + BOS1] = (Sright*(Ble-Sleft)*Fa - Sleft*(Bre-Sright)*Fb); // HLL KE flux
			shblk[IC + BOS1] = (shblk[IC + BOS1] + (FLUID_GOVERGM1*(Sright*Cle*Ble-Sleft*Cre*Bre) + Sleft*Sright*(Cre-Cle)/FLUID_GM1))*Atilde;
			break;
		case HLL_RIGHT:
			shblk[IC + BOS0] = Fb; // rho flux
			shblk[IC + BOS1] = Bre * (FLUID_GOVERGM1*Cre + .5*Are*(Bre*Bre+Dre*Dre+Ere*Ere)); // ener flu
			shblk[IC + BOS2] = Fb*Bre + Cre; // px flux
			shblk[IC + BOS3] = Fb*Dre; // py flux
			shblk[IC + BOS4] = Fb*Ere; // pz flux
			break;
		}

		__syncthreads();

		if(thisThreadDelivers) {
			if((PCswitch & 1) == RK_PREDICT) {
				switch(fluxDirection) {
				case FLUX_X:
#ifdef USE_SSPRK
					Qstore[0*DEV_SLABSIZE] = Qin[0]              - lambda * (shblk[IC+BOS0] - shblk[IL+BOS0]);
					Qstore[1*DEV_SLABSIZE] = Qin[DEV_SLABSIZE]   - lambda * (shblk[IC+BOS1] - shblk[IL+BOS1]);
					Qstore[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * (shblk[IC+BOS2] - shblk[IL+BOS2]);
					Qstore[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * (shblk[IC+BOS3] - shblk[IL+BOS3]);
					Qstore[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * (shblk[IC+BOS4] - shblk[IL+BOS4]);
#else
					Qstore[0*DEV_SLABSIZE] = Ale                 - lambda * (shblk[IC+BOS0] - shblk[IL+BOS0]);
					Qstore[1*DEV_SLABSIZE] = Qin[DEV_SLABSIZE]   - lambda * (shblk[IC+BOS1] - shblk[IL+BOS1]);
					Qstore[2*DEV_SLABSIZE] = Ale*Ble             - lambda * (shblk[IC+BOS2] - shblk[IL+BOS2]);
					Qstore[3*DEV_SLABSIZE] = Ale*Dle             - lambda * (shblk[IC+BOS3] - shblk[IL+BOS3]);
					Qstore[4*DEV_SLABSIZE] = Ale*Ele             - lambda * (shblk[IC+BOS4] - shblk[IL+BOS4]);
#endif
					break;
				case FLUX_Y:
#ifdef USE_SSPRK
					Qstore[0*DEV_SLABSIZE] = Qin[0]              - lambda * (shblk[IC+BOS0] - shblk[IL+BOS0]);
					Qstore[1*DEV_SLABSIZE] = Qin[DEV_SLABSIZE]   - lambda * (shblk[IC+BOS1] - shblk[IL+BOS1]);
					Qstore[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * (shblk[IC+BOS2] - shblk[IL+BOS2]);
					Qstore[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * (shblk[IC+BOS3] - shblk[IL+BOS3]);
					Qstore[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * (shblk[IC+BOS4] - shblk[IL+BOS4]);
#else
					Qstore[0*DEV_SLABSIZE] = Ale                 - lambda * (shblk[IC+BOS0] - shblk[IL+BOS0]);
					Qstore[1*DEV_SLABSIZE] = Qin[DEV_SLABSIZE]   - lambda * (shblk[IC+BOS1] - shblk[IL+BOS1]);
					Qstore[3*DEV_SLABSIZE] = Ale*Ble             - lambda * (shblk[IC+BOS2] - shblk[IL+BOS2]);
					Qstore[2*DEV_SLABSIZE] = Ale*Dle             - lambda * (shblk[IC+BOS2] - shblk[IL+BOS3]);
					Qstore[4*DEV_SLABSIZE] = Ale*Ele             - lambda * (shblk[IC+BOS4] - shblk[IL+BOS4]);
#endif
					break;
				case FLUX_Z:
#ifdef USE_SSPRK
					Qstore[0*DEV_SLABSIZE] = Qin[0]              - lambda * (shblk[IC+BOS0] - shblk[IL+BOS0]);
					Qstore[1*DEV_SLABSIZE] = Qin[DEV_SLABSIZE]   - lambda * (shblk[IC+BOS1] - shblk[IL+BOS1]);
					Qstore[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * (shblk[IC+BOS2] - shblk[IL+BOS2]);
					Qstore[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * (shblk[IC+BOS3] - shblk[IL+BOS3]);
					Qstore[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * (shblk[IC+BOS4] - shblk[IL+BOS4]);
#else
					Qstore[0*DEV_SLABSIZE] = Ale                 - lambda * (shblk[IC+BOS0] - shblk[IL+BOS0]);
					Qstore[1*DEV_SLABSIZE] = Qin[DEV_SLABSIZE]   - lambda * (shblk[IC+BOS1] - shblk[IL+BOS1]);
					Qstore[4*DEV_SLABSIZE] = Ale*Ble             - lambda * (shblk[IC+BOS2] - shblk[IL+BOS2]);
					Qstore[2*DEV_SLABSIZE] = Ale*Dle             - lambda * (shblk[IC+BOS2] - shblk[IL+BOS3]);
					Qstore[3*DEV_SLABSIZE] = Ale*Ele             - lambda * (shblk[IC+BOS4] - shblk[IL+BOS4]);
#endif
					break;
				case FLUX_THETA_231: // This looks unaltered but lambda (= dt/dtheta) was actually rescaled with different 1/r_c's at start
#ifdef USE_SSPRK
					Qstore[0*DEV_SLABSIZE] = Qin[0]              - lambda * (shblk[IC+BOS0] - shblk[IL+BOS0]);
					Qstore[1*DEV_SLABSIZE] = Qin[DEV_SLABSIZE]   - lambda * (shblk[IC+BOS1] - shblk[IL+BOS1]);
					Qstore[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * (shblk[IC+BOS2] - shblk[IL+BOS2]);
					Qstore[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * (shblk[IC+BOS3] - shblk[IL+BOS3]);
					Qstore[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * (shblk[IC+BOS4] - shblk[IL+BOS4]);
#else
					Qstore[0*DEV_SLABSIZE] = Ale               - lambda * (shblk[IC+BOS0] - shblk[IL+BOS0]);
					Qstore[1*DEV_SLABSIZE] = Qin[DEV_SLABSIZE] - lambda * (shblk[IC+BOS1] - shblk[IL+BOS1]);
					Qstore[3*DEV_SLABSIZE] = Ale*Ble           - lambda * (shblk[IC+BOS2] - shblk[IL+BOS2]);
					Qstore[2*DEV_SLABSIZE] = Ale*Dle           - lambda * (shblk[IC+BOS3] - shblk[IL+BOS3]);
					Qstore[4*DEV_SLABSIZE] = Ale*Ele           - lambda * (shblk[IC+BOS4] - shblk[IL+BOS4]);
#endif
					break;
				case FLUX_RADIAL:
#ifdef USE_SSPRK
					Qstore[0*DEV_SLABSIZE] = Qin[0]              -  (cylgeomB*shblk[IC+BOS0] - cylgeomA*shblk[IL+BOS0]);
					Qstore[1*DEV_SLABSIZE] = Qin[DEV_SLABSIZE]   -  (cylgeomB*shblk[IC+BOS1] - cylgeomA*shblk[IL+BOS1]);
					Qstore[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] -  (cylgeomB*shblk[IC+BOS2] - cylgeomA*shblk[IL+BOS2]
					                                                                - Qstore[5*DEV_SLABSIZE]*cylgeomC);
					Qstore[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] -  (cylgeomB*shblk[IC+BOS3] - cylgeomA*shblk[IL+BOS3]);
					Qstore[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] -  (cylgeomB*shblk[IC+BOS4] - cylgeomA*shblk[IL+BOS4]);
#else
					Qstore[0*DEV_SLABSIZE] = Ale               -  (cylgeomB*shblk[IC+BOS0] - cylgeomA*shblk[IL+BOS0]);
					Qstore[1*DEV_SLABSIZE] = Qin[DEV_SLABSIZE] -  (cylgeomB*shblk[IC+BOS1] - cylgeomA*shblk[IL+BOS1]);
					Qstore[2*DEV_SLABSIZE] = Ale*Ble           -  ( cylgeomB*shblk[IC+BOS2]
					                                              - cylgeomA*shblk[IL+BOS2]
					                                              - Cle*cylgeomC);
					Qstore[3*DEV_SLABSIZE] = Ale*Dle           -  (cylgeomB*shblk[IC+BOS3] - cylgeomA*shblk[IL+BOS3]);
					Qstore[4*DEV_SLABSIZE] = Ale*Ele           -  (cylgeomB*shblk[IC+BOS4] - cylgeomA*shblk[IL+BOS4]);
#endif
					break;
				case FLUX_THETA_213:
#ifdef USE_SSPRK
					Qstore[0*DEV_SLABSIZE] = Qin[0]              - lambda * (shblk[IC+BOS0] - shblk[IL+BOS0]) / cylgeomA;
					Qstore[1*DEV_SLABSIZE] = Qin[DEV_SLABSIZE]   - lambda * (shblk[IC+BOS1] - shblk[IL+BOS1]) / cylgeomA;
					Qstore[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * (shblk[IC+BOS2] - shblk[IL+BOS2]) / cylgeomA;
					Qstore[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * (shblk[IC+BOS3] - shblk[IL+BOS3]) / cylgeomA;
					Qstore[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * (shblk[IC+BOS4] - shblk[IL+BOS4]) / cylgeomA;
#else
					Qstore[0*DEV_SLABSIZE] = Ale                  - lambda * (shblk[IC+BOS0] - shblk[IL+BOS0]) / cylgeomA;
					Qstore[1*DEV_SLABSIZE] = Qin[DEV_SLABSIZE]    - lambda * (shblk[IC+BOS1] - shblk[IL+BOS1]) / cylgeomA;
					Qstore[3*DEV_SLABSIZE] = Ale*Ble              - lambda * (shblk[IC+BOS2] - shblk[IL+BOS2]) / cylgeomA;
					Qstore[2*DEV_SLABSIZE] = Ale*Dle              - lambda * (shblk[IC+BOS3] - shblk[IL+BOS3]) / cylgeomA;
					Qstore[4*DEV_SLABSIZE] = Ale*Ele              - lambda * (shblk[IC+BOS4] - shblk[IL+BOS4]) / cylgeomA;
#endif
					break;
				}
			} else {
				switch(fluxDirection) {
				case FLUX_X:
#ifdef USE_SSPRK
					Qin[0*DEV_SLABSIZE] = .5*(Qin[0*DEV_SLABSIZE] + Qstore[0*DEV_SLABSIZE] - lambda*(shblk[IC+BOS0]-shblk[IL+BOS0])); // SSPRK2
					Qin[1*DEV_SLABSIZE] = .5*(Qin[1*DEV_SLABSIZE] + Qstore[1*DEV_SLABSIZE] - lambda*(shblk[IC+BOS1]-shblk[IL+BOS1])); // SSPRK2
					Qin[2*DEV_SLABSIZE] = .5*(Qin[2*DEV_SLABSIZE] + Qstore[2*DEV_SLABSIZE] - lambda*(shblk[IC+BOS2]-shblk[IL+BOS2])); // SSPRK2
					Qin[3*DEV_SLABSIZE] = .5*(Qin[3*DEV_SLABSIZE] + Qstore[3*DEV_SLABSIZE] - lambda*(shblk[IC+BOS3]-shblk[IL+BOS3])); // SSPRK2
					Qin[4*DEV_SLABSIZE] = .5*(Qin[4*DEV_SLABSIZE] + Qstore[4*DEV_SLABSIZE] - lambda*(shblk[IC+BOS4]-shblk[IL+BOS4])); // SSPRK2
#else
					Qin[0*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS0] - shblk[IL+BOS0]);
					Qin[1*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS1] - shblk[IL+BOS1]);
					Qin[2*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS2] - shblk[IL+BOS2]);
					Qin[3*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS3] - shblk[IL+BOS3]);
					Qin[4*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS4] - shblk[IL+BOS4]);
#endif
					break;
				case FLUX_Y:
#ifdef USE_SSPRK
					Qin[0*DEV_SLABSIZE] = .5*(Qin[0*DEV_SLABSIZE] + Qstore[0*DEV_SLABSIZE] - lambda*(shblk[IC+BOS0]-shblk[IL+BOS0])); // SSPRK2
					Qin[1*DEV_SLABSIZE] = .5*(Qin[1*DEV_SLABSIZE] + Qstore[1*DEV_SLABSIZE] - lambda*(shblk[IC+BOS1]-shblk[IL+BOS1])); // SSPRK2
					Qin[3*DEV_SLABSIZE] = .5*(Qin[3*DEV_SLABSIZE] + Qstore[3*DEV_SLABSIZE] - lambda*(shblk[IC+BOS2]-shblk[IL+BOS2])); // SSPRK2
					Qin[2*DEV_SLABSIZE] = .5*(Qin[2*DEV_SLABSIZE] + Qstore[2*DEV_SLABSIZE] - lambda*(shblk[IC+BOS3]-shblk[IL+BOS3])); // SSPRK2
					Qin[4*DEV_SLABSIZE] = .5*(Qin[4*DEV_SLABSIZE] + Qstore[4*DEV_SLABSIZE] - lambda*(shblk[IC+BOS4]-shblk[IL+BOS4])); // SSPRK2
#else
					Qin[0*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS0]- shblk[IL+BOS0]);
					Qin[1*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS1]- shblk[IL+BOS1]);
					Qin[3*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS2]- shblk[IL+BOS2]);
					Qin[2*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS3]- shblk[IL+BOS3]);
					Qin[4*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS4]- shblk[IL+BOS4]);
#endif
					break;
				case FLUX_Z:
#ifdef USE_SSPRK
					Qin[0*DEV_SLABSIZE] = .5*(Qin[0*DEV_SLABSIZE] + Qstore[0*DEV_SLABSIZE] - lambda*(shblk[IC+BOS0]-shblk[IL+BOS0])); // SSPRK2
					Qin[1*DEV_SLABSIZE] = .5*(Qin[1*DEV_SLABSIZE] + Qstore[1*DEV_SLABSIZE] - lambda*(shblk[IC+BOS1]-shblk[IL+BOS1])); // SSPRK2
					Qin[4*DEV_SLABSIZE] = .5*(Qin[4*DEV_SLABSIZE] + Qstore[4*DEV_SLABSIZE] - lambda*(shblk[IC+BOS2]-shblk[IL+BOS2])); // SSPRK2
					Qin[2*DEV_SLABSIZE] = .5*(Qin[2*DEV_SLABSIZE] + Qstore[2*DEV_SLABSIZE] - lambda*(shblk[IC+BOS3]-shblk[IL+BOS3])); // SSPRK2
					Qin[3*DEV_SLABSIZE] = .5*(Qin[3*DEV_SLABSIZE] + Qstore[3*DEV_SLABSIZE] - lambda*(shblk[IC+BOS4]-shblk[IL+BOS4])); // SSPRK2
#else
					Qin[0*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS0]- shblk[IL+BOS0]);
					Qin[1*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS1]- shblk[IL+BOS1]);
					Qin[4*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS2]- shblk[IL+BOS2]);
					Qin[2*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS3]- shblk[IL+BOS3]);
					Qin[3*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS4]- shblk[IL+BOS4]);
#endif
					break;
				case FLUX_THETA_231: // This looks unaltered but lambda (= dt/dtheta) was actually rescaled with different 1/r_c's at start
#ifdef USE_SSPRK
					Qin[0*DEV_SLABSIZE] = .5*(Qin[0*DEV_SLABSIZE] + Qstore[0*DEV_SLABSIZE] - lambda*(shblk[IC+BOS0]-shblk[IL+BOS0])); // SSPRK2
					Qin[1*DEV_SLABSIZE] = .5*(Qin[1*DEV_SLABSIZE] + Qstore[1*DEV_SLABSIZE] - lambda*(shblk[IC+BOS1]-shblk[IL+BOS1])); // SSPRK2
					Qin[3*DEV_SLABSIZE] = .5*(Qin[3*DEV_SLABSIZE] + Qstore[3*DEV_SLABSIZE] - lambda*(shblk[IC+BOS2]-shblk[IL+BOS2])); // SSPRK2
					Qin[2*DEV_SLABSIZE] = .5*(Qin[2*DEV_SLABSIZE] + Qstore[2*DEV_SLABSIZE] - lambda*(shblk[IC+BOS3]-shblk[IL+BOS3])); // SSPRK2
					Qin[4*DEV_SLABSIZE] = .5*(Qin[4*DEV_SLABSIZE] + Qstore[4*DEV_SLABSIZE] - lambda*(shblk[IC+BOS4]-shblk[IL+BOS4])); // SSPRK2
#else
					Qin[0*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS0]- shblk[IL+BOS0]);
					Qin[1*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS1]- shblk[IL+BOS1]);
					Qin[3*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS2]- shblk[IL+BOS2]);
					Qin[2*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS3]- shblk[IL+BOS3]);
					Qin[4*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS4]- shblk[IL+BOS4]);
#endif
					break;
				case FLUX_RADIAL:
#ifdef USE_SSPRK
					Qin[0*DEV_SLABSIZE] = .5*(Qin[0*DEV_SLABSIZE] + Qstore[0*DEV_SLABSIZE] - (cylgeomB*shblk[IC+BOS0] - cylgeomA*shblk[IL+BOS0]));
					Qin[1*DEV_SLABSIZE] = .5*(Qin[1*DEV_SLABSIZE] + Qstore[1*DEV_SLABSIZE] - (cylgeomB*shblk[IC+BOS1] - cylgeomA*shblk[IL+BOS1]));
					Qin[2*DEV_SLABSIZE] = .5*(Qin[2*DEV_SLABSIZE] + Qstore[2*DEV_SLABSIZE] - (cylgeomB*shblk[IC+BOS2] - cylgeomA*shblk[IL+BOS2]
					                                                                      - Qstore[5*DEV_SLABSIZE]*cylgeomC));
 					Qin[3*DEV_SLABSIZE] = .5*(Qin[3*DEV_SLABSIZE] + Qstore[3*DEV_SLABSIZE] - (cylgeomB*shblk[IC+BOS3] - cylgeomA*shblk[IL+BOS3]));
					Qin[4*DEV_SLABSIZE] = .5*(Qin[4*DEV_SLABSIZE] + Qstore[4*DEV_SLABSIZE] - (cylgeomB*shblk[IC+BOS4] - cylgeomA*shblk[IL+BOS4]));

#else
					// Realized these can just use cylgeom* multiplied by lambda at the start
					Qin[0*DEV_SLABSIZE] -= (cylgeomB*shblk[IC+BOS0] - cylgeomA*shblk[IL+BOS0]);
					Qin[1*DEV_SLABSIZE] -= (cylgeomB*shblk[IC+BOS1] - cylgeomA*shblk[IL+BOS1]);
					Qin[2*DEV_SLABSIZE] -= (cylgeomB*shblk[IC+BOS2] - cylgeomA*shblk[IL+BOS2] - Qstore[5*DEV_SLABSIZE]*cylgeomC);
					Qin[3*DEV_SLABSIZE] -= (cylgeomB*shblk[IC+BOS3] - cylgeomA*shblk[IL+BOS3]);
					Qin[4*DEV_SLABSIZE] -= (cylgeomB*shblk[IC+BOS4] - cylgeomA*shblk[IL+BOS4]);
#endif
					break;
				case FLUX_THETA_213:
#ifdef USE_SSPRK
					Qin[0*DEV_SLABSIZE] = .5*(Qin[0*DEV_SLABSIZE] + Qstore[0*DEV_SLABSIZE] - lambda*(shblk[IC+BOS0]-shblk[IL+BOS0]) / cylgeomA); // SSPRK2
					Qin[1*DEV_SLABSIZE] = .5*(Qin[1*DEV_SLABSIZE] + Qstore[1*DEV_SLABSIZE] - lambda*(shblk[IC+BOS1]-shblk[IL+BOS1]) / cylgeomA); // SSPRK2
					Qin[3*DEV_SLABSIZE] = .5*(Qin[3*DEV_SLABSIZE] + Qstore[3*DEV_SLABSIZE] - lambda*(shblk[IC+BOS2]-shblk[IL+BOS2]) / cylgeomA); // SSPRK2
					Qin[2*DEV_SLABSIZE] = .5*(Qin[2*DEV_SLABSIZE] + Qstore[2*DEV_SLABSIZE] - lambda*(shblk[IC+BOS3]-shblk[IL+BOS3]) / cylgeomA); // SSPRK2
					Qin[4*DEV_SLABSIZE] = .5*(Qin[4*DEV_SLABSIZE] + Qstore[4*DEV_SLABSIZE] - lambda*(shblk[IC+BOS4]-shblk[IL+BOS4]) / cylgeomA); // SSPRK2
#else
					Qin[0*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS0]- shblk[IL+BOS0]) / cylgeomA;
					Qin[1*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS1]- shblk[IL+BOS1]) / cylgeomA;
					Qin[3*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS2]- shblk[IL+BOS2]) / cylgeomA;
					Qin[2*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS3]- shblk[IL+BOS3]) / cylgeomA;
					Qin[4*DEV_SLABSIZE] -= lambda * (shblk[IC+BOS4]- shblk[IL+BOS4]) / cylgeomA;
#endif
					break;
				} // end switch
			} // end predict/correct step
		}

//		x0 += blockDim.y*DEV_NX;
		Qstore += blockDim.y * DEV_NX;
		Qin    += blockDim.y * DEV_NX;
		// Move the r_center coordinate out as the y-aligned blocks march in x (=r)
		if(fluxDirection == FLUX_THETA_213) {
			cylgeomA += YBLOCKS*CYLGEO_DR;
		}
		__syncthreads();
	}

}

#undef DBGSAVE
#define DBGSAVE(n, x) if(thisThreadDelivers) { Qstore[((n)+6)*(1024*64)] = (x); }

template <unsigned int PCswitch>
__global__ void cukern_XinJinHydro_step(double *Qbase, double *Qstore, double *Cfreeze, double lambda)
{
	if(threadIdx.y >= DEV_NY) return;

	// Create center, rotate-left and rotate-right indexes
	int IC = threadIdx.x;
	int IL = threadIdx.x - 1;
	IL += 1*(IL < 0);
	int IR = threadIdx.x + 1;
	IR -= 1*(IL > (BLOCKLENP4-1));

	IC += 4*BLOCKLENP4*threadIdx.y;
	IL += 4*BLOCKLENP4*threadIdx.y;
	IR += 4*BLOCKLENP4*threadIdx.y;

	/* Declare variable arrays */
	__shared__ double shblk[2*YBLOCKS*4*BLOCKLENP4];
	double Q[5]; double prop[5];
	double P, C_f, vx, w;

	/* My x index: thread + blocksize block, wrapped circularly */
	//int thisThreadPonders  = (threadIdx.x > 0) && (threadIdx.x < blockDim.x-1);
	int thisThreadDelivers = (threadIdx.x >= 2) && (threadIdx.x <= BLOCKLENP4-3);

	int x0 = threadIdx.x + (BLOCKLEN)*blockIdx.x - 2;
	if(x0 < 0) x0 += DEV_NX; // left wraps to right edge
	if(x0 > (DEV_NX+1)) return; // More than 2 past right returns
	if(x0 > (DEV_NX-1)) { x0 -= DEV_NX; thisThreadDelivers = 0; } // past right must wrap around to left

	double cylgeomA, cylgeomB, cylgeomC;

	/* If doing cylindrical geometry... */
	// Compute multiple scale factors for radial direction fluxes
	int fluxDirection = (PCswitch / 2) + 1; // Will be optimized away during template compilation

	if(fluxDirection == FLUX_RADIAL) { // cylindrical, R direction
		P = CYLGEO_RINI + x0 * CYLGEO_DR; // r_center
		cylgeomA = (P - .5*CYLGEO_DR) / (P);
		cylgeomB = (P + .5*CYLGEO_DR) / (P);
		cylgeomC = 2.0*CYLGEO_DR / P;
		// NOTE: We scale cylgeomC by a factor of 2 because the fluxes, to avoid a divide-by-2,
		// are all computed too large by a factor of 2, and to compensate the CPU scales lambda by 0.5,
		// Thus we must double the P dt / r coefficient to maintain proper ratio.
	}
	// The kern will step through r, so we have to add to R and compute 1.0/R
	if(fluxDirection == FLUX_THETA_213) {
		cylgeomA = threadIdx.y*CYLGEO_DR + CYLGEO_RINI;
	}
	// The kerns will step through z so R is fixed and we compute it once
	// We just scale lambda ( = dt / dtheta) by 1/r_c
	if(fluxDirection == FLUX_THETA_231) {
		lambda /= (blockIdx.y*CYLGEO_DR + CYLGEO_RINI);
	}

	/* Do some index calculations */
	x0 += DEV_NX*(DEV_NY*blockIdx.y + threadIdx.y); /* This block is now positioned to start at its given (x,z) coordinate */
	int j = threadIdx.y;
	int i;

	for(; j < DEV_NY; j += YBLOCKS) {
		C_f = Cfreeze[j + DEV_NY*blockIdx.y];
		if((PCswitch & 1) == RK_PREDICT) {
			// Load from init inputs, write to Qstore[]
			Q[0] = Qbase[x0];
			Q[1] = Qbase[x0+  DEV_SLABSIZE];
			Q[2] = Qbase[x0+2*DEV_SLABSIZE];
			Q[3] = Qbase[x0+3*DEV_SLABSIZE];
			Q[4] = Qbase[x0+4*DEV_SLABSIZE];
		} else {
			// Load from qstore, update init inputs
			Q[0] = Qstore[x0];
			Q[1] = Qstore[x0+  DEV_SLABSIZE];
			Q[2] = Qstore[x0+2*DEV_SLABSIZE];
			Q[3] = Qstore[x0+3*DEV_SLABSIZE];
			Q[4] = Qstore[x0+4*DEV_SLABSIZE];
		}

		P  = FLUID_GM1 * (Q[1] - .5*(Q[4]*Q[4]+Q[3]*Q[3]+Q[2]*Q[2])/Q[0]);

		switch(fluxDirection) {
		case FLUX_X: case FLUX_RADIAL:
			vx = Q[2] / Q[0]; break;
		case FLUX_Y: case FLUX_THETA_213: case FLUX_THETA_231:
			vx = Q[3] / Q[0]; break;
		case FLUX_Z:
			vx = Q[4] / Q[0]; break;
		}

		for(i = 0; i < 5; i++) {
			/* Calculate raw fluxes for rho, E, px, py, pz in order: */
			/* Permute which things we use to calculate the fluxes here 
			in order to avoid having to rearrange the memory loads which
			causes problems when this loop iterates over the memory slabs in order. */
			switch(fluxDirection) {
			case FLUX_X:
			case FLUX_RADIAL:
			switch(i) {
			case 0: w = Q[2];          break;
			case 1: w = vx*(Q[1] + P); break;
			case 2: w = vx*Q[2] + P;   break;
			case 3: w = vx*Q[3];       break;
			case 4: w = vx*Q[4];       break;
			} break;
			case FLUX_Y:
			case FLUX_THETA_213:
			case FLUX_THETA_231:
			switch(i) {
			case 0: w = Q[3];          break;
			case 1: w = vx*(Q[1] + P); break;
			case 2: w = vx*Q[2];       break;
			case 3: w = vx*Q[3] + P;   break;
			case 4: w = vx*Q[4];       break;
			} break;
			case FLUX_Z:
			switch(i) {
			case 0: w = Q[4];          break;
			case 1: w = vx*(Q[1] + P); break;
			case 2: w = vx*Q[2];       break;
			case 3: w = vx*Q[3];       break;
			case 4: w = vx*Q[4] + P;   break;
			} break;
			}

			shblk[IC + BOS0] = (C_f*Q[i] - w); /* Cell's leftgoing  flux */
			shblk[IC + BOS1] = (C_f*Q[i] + w); /* Cell's rightgoing flux */
			__syncthreads();

			if((PCswitch & 1) == RK_CORRECT) {
				/* Entertain two flux corrections */
				shblk[IC + BOS2] = 0.5*(shblk[IC + BOS0] - shblk[IL + BOS0]); /* Bkwd deriv of leftgoing flux */
				shblk[IC + BOS3] = 0.5*(shblk[IC + BOS1] - shblk[IL + BOS1]); /* Bkwd deriv of ritegoing flux */
				__syncthreads();

				/* Impose TVD limiter */
				shblk[IC + BOS0] -= LIMITERFUNC(shblk[IC+BOS2], shblk[IR+BOS2]); // leftgoing flux extrap'd to left edge
				shblk[IC + BOS1] += LIMITERFUNC(shblk[IC+BOS3], shblk[IR+BOS3]); // ritegoing flux extrap'd to rite edge
				__syncthreads();
			}

			if(thisThreadDelivers) {
				if((PCswitch & 1) == RK_PREDICT) {
					switch(fluxDirection) {
					case FLUX_X:
					case FLUX_Y:
					case FLUX_Z:
					case FLUX_THETA_231: // This looks the same but we rescaled lambda = dt/dtheta by 1/r above
						prop[i] = Q[i] - lambda * ( shblk[IC+BOS1]- shblk[IL+BOS1] -
								shblk[IR+BOS0]+ shblk[IC+BOS0]); break;
					case FLUX_RADIAL:
						prop[i] = Q[i] - lambda * ( cylgeomB*shblk[IC+BOS1]  // rightgoing
						                          - cylgeomA*shblk[IL+BOS1]
						                          - cylgeomB*shblk[IR+BOS0]  // leftgoing
						                          + cylgeomA*shblk[IC+BOS0]);
						if(i == 2) prop[i] += lambda*cylgeomC*P;
						break;
					case FLUX_THETA_213:
						prop[i] = Q[i] - lambda * ( shblk[IC+BOS1]- shblk[IL+BOS1] -
								shblk[IR+BOS0]+ shblk[IC+BOS0])/cylgeomA;
						break;
					}
				} else {
					switch(fluxDirection) {
					case FLUX_X:
					case FLUX_Y:
					case FLUX_Z:
					case FLUX_THETA_231: // This looks the same but we rescaled lambda = dt/dtheta by 1/r above
						prop[i] = Qbase[x0 + i*DEV_SLABSIZE] - lambda * ( shblk[IC+BOS1]- shblk[IL+BOS1] -
								shblk[IR+BOS0]+ shblk[IC+BOS0]); break;
					case FLUX_RADIAL:
						// ABAB - err = +15e-3
						// ABBA - err = +1e-3
						// BAAB - err = +6e-3
						// BABA - err = -6e-3
						// BBAA - err = +6e-3
						// AABB - err = +6e-3
						prop[i] = Qbase[x0 + i*DEV_SLABSIZE] - lambda * ( cylgeomB*shblk[IC+BOS1]
						                                                - cylgeomA*shblk[IL+BOS1]
						                                                - cylgeomB*shblk[IR+BOS0]
						                                                + cylgeomA*shblk[IC+BOS0]);
						if(i == 2) prop[i] += lambda*cylgeomC*P;
						break;
					case FLUX_THETA_213:
						prop[i] = Qbase[x0 + i*DEV_SLABSIZE] - lambda * ( shblk[IC+BOS1]- shblk[IL+BOS1] -
								shblk[IR+BOS0]+ shblk[IC+BOS0])/cylgeomA;
						break;
					}
				}
			}

			__syncthreads();
		}

		if(thisThreadDelivers) {
			// Enforce density positivity
			prop[0] = (prop[0] < FLUID_MINMASS) ? FLUID_MINMASS : prop[0];

			// Calculate kinetic energy density and enforce minimum pressure
			w = .5*(prop[2]*prop[2] + prop[3]*prop[3] + prop[4]*prop[4])/prop[0];
			if((prop[1] - w) < prop[0]*FLUID_MINTEMP) {
				prop[1] = w + prop[0]*FLUID_MINTEMP;
			}

			if((PCswitch & 1) == RK_PREDICT) {
				Qstore[x0]                = prop[0];
				Qstore[x0+DEV_SLABSIZE]   = prop[1];
				Qstore[x0+2*DEV_SLABSIZE] = prop[2];
				Qstore[x0+3*DEV_SLABSIZE] = prop[3];
				Qstore[x0+4*DEV_SLABSIZE] = prop[4];
			} else {
				Qbase[x0]                = prop[0];
				Qbase[x0+DEV_SLABSIZE]   = prop[1];
				Qbase[x0+2*DEV_SLABSIZE] = prop[2];
				Qbase[x0+3*DEV_SLABSIZE] = prop[3];
				Qbase[x0+4*DEV_SLABSIZE] = prop[4];
			}
		}

		x0 += YBLOCKS*DEV_NX;
		// Move the r_center coordinate out as the y-aligned blocks march in x (=r)
		if(fluxDirection == FLUX_THETA_213) {
			cylgeomA += YBLOCKS*CYLGEO_DR;
		}

		__syncthreads();
	}

}


template <unsigned int PCswitch>
__global__ void cukern_XinJinMHD_step(double *Qbase, double *Qstore, double *mag, double *Cfreeze, double lambda)
{
	// Create center, rotate-left and rotate-right indexes
	int IC = threadIdx.x;
	int IL = threadIdx.x - 1;
	IL += 1*(IL < 0);
	int IR = threadIdx.x + 1;
	IR -= 1*(IL > 15);

	IC += 4*16*threadIdx.y;
	IL += 4*16*threadIdx.y;
	IR += 4*16*threadIdx.y;

	/* Declare variable arrays */
	__shared__ double shblk[2*YBLOCKS*4*BLOCKLENP4];
	double Q[5]; double prop[5]; double B[3];
	double P, C_f, vx, w;

	/* My x index: thread + blocksize block, wrapped circularly */
	//int thisThreadPonders  = (threadIdx.x > 0) && (threadIdx.x < blockDim.x-1);
	int thisThreadDelivers = (threadIdx.x >= 2) && (threadIdx.x <= 13);

	int x0 = threadIdx.x + (BLOCKLEN)*blockIdx.x - 2;
	if(x0 < 0) x0 += DEV_NX; // left wraps to right edge
	if(x0 > (DEV_NX+1)) return; // More than 2 past right returns
	if(x0 > (DEV_NX-1)) { x0 -= DEV_NX; thisThreadDelivers = 0; } // past right must wrap around to left

	/* Do some index calculations */
	x0 += DEV_NX*(DEV_NY*blockIdx.y + threadIdx.y); /* This block is now positioned to start at its given (x,z) coordinate */
	int j = threadIdx.y;
	int i;

	for(; j < DEV_NY; j += YBLOCKS) {
		C_f = Cfreeze[j + DEV_NY*blockIdx.y];

		if((PCswitch & 1) == RK_PREDICT) {
			// Load from init inputs, write to Qstore[]
			Q[0] = Qbase[x0];
			Q[1] = Qbase[x0+  DEV_SLABSIZE];
			Q[2] = Qbase[x0+2*DEV_SLABSIZE];
			Q[3] = Qbase[x0+3*DEV_SLABSIZE];
			Q[4] = Qbase[x0+4*DEV_SLABSIZE];
		} else {
			// Load from qstore, update init inputs
			Q[0] = Qstore[x0];
			Q[1] = Qstore[x0+  DEV_SLABSIZE];
			Q[2] = Qstore[x0+2*DEV_SLABSIZE];
			Q[3] = Qstore[x0+3*DEV_SLABSIZE];
			Q[4] = Qstore[x0+4*DEV_SLABSIZE];
		}

		B[0] = mag[x0 + 0]; // Bx
		B[1] = mag[x0 + DEV_SLABSIZE]; // By
		B[2] = mag[x0 + 2*DEV_SLABSIZE]; // Bz

		// calculate total pressure P
		P = 0; // FIXME this needs to be calc'd
		P  = FLUID_GM1 * (Q[1] - .5*(Q[4]*Q[4]+Q[3]*Q[3]+Q[2]*Q[2])/Q[0] - 0.5*(B[0]*B[0]+B[1]*B[1]+B[2]*B[2]));

		switch(PCswitch & 6) {
				case 0: vx = Q[2] / Q[0]; break;
				case 2: vx = Q[3] / Q[0]; break;
				case 4: vx = Q[4] / Q[0]; break;
				}

		for(i = 0; i < 5; i++) {
			/* Calculate raw fluxes for rho, E, px, py, pz in order: */
			switch(PCswitch & 6) {
			case 0: switch(i) {
			case 0: w = Q[2]; break;
			case 1: w = (vx * (Q[1] + P) - B[0]*(Q[2]*B[0]+Q[3]*B[1]+Q[4]*B[2])/Q[0] ); break;
			case 2: w = (vx*Q[2] + P - B[0]*B[0]); break;
			case 3: w = (vx*Q[3]     - B[0]*B[1]); break;
			case 4: w = (vx*Q[4]     - B[0]*B[2]); break;
			}; break;
			case 2: switch(i) {
			case 0: w = Q[3]; break;
			case 1: w = (vx * (Q[1] + P) - B[1]*(Q[2]*B[0]+Q[3]*B[1]+Q[4]*B[2])/Q[0] ); break;
			case 2: w = (vx*Q[2]     - B[1]*B[0]); break;
			case 3: w = (vx*Q[3] + P - B[1]*B[1]); break;
			case 4: w = (vx*Q[4]     - B[1]*B[2]); break;
			}; break;
			case 4: switch(i) {
			case 0: w = Q[4]; break;
			case 1: w = (vx * (Q[1] + P) - B[2]*(Q[2]*B[0]+Q[3]*B[1]+Q[4]*B[2])/Q[0] ); break;
			case 2: w = (vx*Q[2]     - B[2]*B[0]); break;
			case 3: w = (vx*Q[3]     - B[2]*B[1]); break;
			case 4: w = (vx*Q[4] + P - B[2]*B[2]); break;
			}; break;

			}
			shblk[IC + BOS0] = (C_f*Q[i] - w); /* Left  going flux */
			shblk[IC + BOS1] = (C_f*Q[i] + w); /* Right going flux */
			__syncthreads();

			if(PCswitch == RK_CORRECT) {
				/* Entertain two flux corrections */
				shblk[IC + BOS2] = 0.5*(shblk[IC + BOS0] - shblk[IL + BOS0]); /* Deriv of leftgoing flux */
				shblk[IC + BOS3] = 0.5*(shblk[IC + BOS1] - shblk[IL + BOS1]); /* Deriv of ritegoing flux */
				__syncthreads();

				/* Impose TVD limiter */
				shblk[IC + BOS0] -= LIMITERFUNC(shblk[IC+BOS2], shblk[IR+BOS2]);
				shblk[IC + BOS1] += LIMITERFUNC(shblk[IC+BOS3], shblk[IR+BOS3]);
				__syncthreads();
			}

			if(thisThreadDelivers) {
				if(PCswitch == RK_PREDICT) {
					prop[i] = Q[i] - lambda * ( shblk[IC+BOS1]- shblk[IL+BOS1] - shblk[IR+BOS0]+ shblk[IC+BOS0]);
				} else {
					prop[i] = Qstore[x0 + i*DEV_SLABSIZE] - lambda * ( shblk[IC+BOS1]- shblk[IL+BOS1] - shblk[IR+BOS0]+ shblk[IC+BOS0]);
				}
			}

			__syncthreads();
		}

		if(thisThreadDelivers) {
			// Enforce density positivity
			prop[0] = (prop[0] < FLUID_MINMASS) ? FLUID_MINMASS : prop[0];

			// Calculate kinetic+magnetic energy density and enforce minimum pressure
			w = .5*(prop[2]*prop[2] + prop[3]*prop[3] + prop[4]*prop[4])/prop[0] + .5*(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
			if((prop[1] - w) < prop[0]*FLUID_MINTEMP) {
				prop[1] = w + prop[0]*FLUID_MINTEMP;
			}

			if(PCswitch == RK_PREDICT) {
				Qstore[x0                 ] = prop[0];
				Qstore[x0 +   DEV_SLABSIZE] = prop[1];
				Qstore[x0 + 2*DEV_SLABSIZE] = prop[2];
				Qstore[x0 + 3*DEV_SLABSIZE] = prop[3];
				Qstore[x0 + 4*DEV_SLABSIZE] = prop[4];
			} else {
				Qbase[x0               ] = prop[0];
				Qbase[x0+  DEV_SLABSIZE] = prop[1];
				Qbase[x0+2*DEV_SLABSIZE] = prop[2];
				Qbase[x0+3*DEV_SLABSIZE] = prop[3];
				Qbase[x0+4*DEV_SLABSIZE] = prop[4];
			}
		}

		x0 += YBLOCKS*DEV_NX;
		__syncthreads();
	}

}

/* Read Qstore and calculate pressure in it */
/* invoke with nx threads and nb blocks, whatever's best for the arch */
/* Note, state and gasPressure are not necessarily separate allocations
 * they will, in fact, usually be the first 5 slabs of the fluid state & the sixth, respectively
 * However all reads/writes are safely nonoverlapping
 */
__global__ void cukern_PressureSolverHydro(double *state, double *gasPressure)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;

	double rho, E, z, momsq, P;

	int hx = blockDim.x*gridDim.x;
	int DAN = DEV_NX * DEV_NY * DEV_NZ;

	while(x < DAN) {
		rho = state[x      ];
		#ifdef CFD_ENFORCE_MINVALS
		if(rho < FLUID_MINMASS) {
			P = FLUID_MINMASS/rho;
			rho = state[x] = FLUID_MINMASS;
			// invent mass whole-cloth but hold velocity/temperature constant
			E   = state[x + DEV_SLABSIZE] = P*state[x + DEV_SLABSIZE];
			z   = state[x+2*DEV_SLABSIZE] = P*state[x+2*DEV_SLABSIZE];
			momsq = z*z;
			z   = state[x+3*DEV_SLABSIZE] = P*state[x+3*DEV_SLABSIZE];
			momsq += z*z;
			z   = state[x+4*DEV_SLABSIZE] = P*state[x+4*DEV_SLABSIZE];
			momsq += z*z;
		} else {
		#endif
		E   = state[x + DEV_SLABSIZE];
		z   = state[x+2*DEV_SLABSIZE];
		momsq = z*z;
		z   = state[x+3*DEV_SLABSIZE];
		momsq += z*z;
		z   = state[x+4*DEV_SLABSIZE];
		momsq += z*z;
		#ifdef CFD_ENFORCE_MINVALS
		}
		#endif

		P = E - .5*momsq/rho;

#ifdef CFD_ENFORCE_MINVALS
		if(P < rho * FLUID_MINTEMP) {
			P = FLUID_MINTEMP*rho; // actually e_internal
			state[x+DEV_SLABSIZE] = (.5*momsq/rho) + P;
			gasPressure[x] = P * FLUID_GM1;
		} else {
			gasPressure[x] = P * FLUID_GM1;
		}
#endif

		x += hx;
	}

}


/* Invoke with [64 x N] threads and [ny/N nz 1] blocks */
__global__ void cukern_PressureFreezeSolverHydro(double *state, double *gasPressure, double *Cfreeze)
{
	__shared__ double Cfshared[64*FREEZE_NY];

	if(threadIdx.y + blockIdx.x*FREEZE_NY >= DEV_NY) return;

	state += threadIdx.x + DEV_NX*(threadIdx.y + blockIdx.x*FREEZE_NY + DEV_NY*blockIdx.y);
	gasPressure += threadIdx.x + DEV_NX*(threadIdx.y + blockIdx.x*FREEZE_NY + DEV_NY*blockIdx.y);

	int x = threadIdx.x;
	int i = threadIdx.x + 64*threadIdx.y;

	double invrho, px, psq, P, locCf, cmax;

	Cfshared[i] = 0.0;
	cmax = 0.0;

	for(; x < DEV_NX; x += 64) {
		invrho = 1.0/state[0]; // load inverse of density
		psq = state[3*DEV_SLABSIZE]; // accumulate p^2 and end with px
		px =  state[4*DEV_SLABSIZE];
		psq = psq*psq + px*px;
		px = state[2*DEV_SLABSIZE];
		psq += px*px;

		// Store pressure
		*gasPressure = P = (state[DEV_SLABSIZE] - .5*psq*invrho)*FLUID_GM1;

		locCf = fabs(px)*invrho + sqrt(FLUID_GAMMA*P*invrho);

		cmax = (locCf > cmax) ? locCf : cmax; // As we hop down the X direction, track the max C_f encountered

		state += 64;
		gasPressure += 64;
	}

	Cfshared[i] = cmax;

	// Perform a reduction
	if(threadIdx.x >= 32) return;
	__syncthreads();
	locCf = Cfshared[i+32];
	Cfshared[i] = cmax = (locCf > cmax) ? locCf : cmax;

	if(threadIdx.x >= 16) return;
	__syncthreads();
	locCf = Cfshared[i+16];
	Cfshared[i] = cmax = (locCf > cmax) ? locCf : cmax;

	if(threadIdx.x >= 8) return;
	__syncthreads();
	locCf = Cfshared[i+8];
	Cfshared[i] = cmax = (locCf > cmax) ? locCf : cmax;

	if(threadIdx.x >= 4) return;
	__syncthreads();
	locCf = Cfshared[i+4];
	Cfshared[i] = cmax = (locCf > cmax) ? locCf : cmax;

	if(threadIdx.x >= 2) return;
	__syncthreads();
	locCf = Cfshared[i+2];
	Cfshared[i] = cmax = (locCf > cmax) ? locCf : cmax;

	if(threadIdx.x >= 1) return;
	__syncthreads();
	locCf = Cfshared[i+1];
	cmax = (locCf > cmax) ? locCf : cmax;

	// Index into the freezing speed array
	x = (threadIdx.y + FREEZE_NY * blockIdx.x) + DEV_NY*(blockIdx.y);

	Cfreeze[x] = cmax;
}


/* Invoke with [64 x N] threads and [ny/N nz 1] blocks */\
/* Reads magnetic field from inputPointers[5, 6, 7][x] */
__global__ void cukern_PressureFreezeSolverMHD(double *state, double *mag, double *totalPressure, double *Cfreeze)
{
	__shared__ double Cfshared[64*FREEZE_NY];

	if(threadIdx.y + blockIdx.x*FREEZE_NY >= DEV_NY) return;

	int delta = threadIdx.x + DEV_NX*(threadIdx.y + blockIdx.x*FREEZE_NY + DEV_NY*blockIdx.y);

	state         += delta;
	mag           += delta;
    totalPressure += delta;

	int x = threadIdx.x;
	int i = threadIdx.x + 64*threadIdx.y;

	double invrho, px, psq, locCf, cmax;
	double b, bsq;

	Cfshared[i] = 0.0;
	cmax = 0.0;

	for(; x < DEV_NX; x += 64) {
		invrho = 1.0/state[0]; // load inverse of density
		psq = state[3*DEV_SLABSIZE]; // accumulate p^2 and end with px
		px =  state[4*DEV_SLABSIZE];
		psq = psq*psq + px*px;
		px = state[2*DEV_SLABSIZE];
		psq += px*px;

		b = mag[0]; // Accumulate B.B into bsq
		bsq = mag[DEV_SLABSIZE];
		bsq = bsq*bsq + b*b;
		b = mag[2*DEV_SLABSIZE];
		bsq = bsq + b*b;

		b = state[DEV_SLABSIZE] - .5*psq*invrho; // Etot - KE

		// Store pressure
		totalPressure[0] = FLUID_GM1 *b + MHD_PRESS_B*bsq;

		// Find the fast MS wavespeed
		locCf = fabs(px)*invrho + sqrt((FLUID_GG1*b + MHD_CS_B*bsq)*invrho);

		cmax = (locCf > cmax) ? locCf : cmax; // As we hop down the X direction, track the max C_f encountered
		state         += 64;
		mag           += 64;
		totalPressure += 64;
	}

	Cfshared[i] = cmax;

	// Perform a reduction
	if(threadIdx.x >= 32) return;
	__syncthreads();
	locCf = Cfshared[i+32];
	Cfshared[i] = cmax = (locCf > cmax) ? locCf : cmax;

	if(threadIdx.x >= 16) return;
	__syncthreads();
	locCf = Cfshared[i+16];
	Cfshared[i] = cmax = (locCf > cmax) ? locCf : cmax;

	if(threadIdx.x >= 8) return;
	__syncthreads();
	locCf = Cfshared[i+8];
	Cfshared[i] = cmax = (locCf > cmax) ? locCf : cmax;

	if(threadIdx.x >= 4) return;
	__syncthreads();
	locCf = Cfshared[i+4];
	Cfshared[i] = cmax = (locCf > cmax) ? locCf : cmax;

	if(threadIdx.x >= 2) return;
	__syncthreads();
	locCf = Cfshared[i+2];
	Cfshared[i] = cmax = (locCf > cmax) ? locCf : cmax;

	if(threadIdx.x >= 1) return;
	__syncthreads();
	locCf = Cfshared[i+1];
	cmax = (locCf > cmax) ? locCf : cmax;

	// Index into the freezing speed array
	x = (threadIdx.y + FREEZE_NY * blockIdx.x) + DEV_NY*(blockIdx.y);

	Cfreeze[x] = cmax;
}
