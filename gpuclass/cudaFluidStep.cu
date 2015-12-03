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

#include "cudaCommon.h" // This defines the getGPUSourcePointers and makeGPUDestinationArrays utility functions

// Only uncomment this if you plan to debug this file.
//#define DEBUGMODE

#include "mpi.h"
#include "parallel_halo_arrays.h"
//#include "mpi_common.h"

#ifdef DEBUGMODE
    #include "debug_inserts.h"
#endif

#define RK_PREDICT 0
#define RK_CORRECT 1

#define STEPTYPE_XINJIN 1
#define STEPTYPE_HLL 2
#define STEPTYPE_HLLC 3
//#define STEPTYPE_AUSM 4

#define FLUX_X 1
#define FLUX_Y 2
#define FLUX_Z 3


/* THIS FUNCTION
This function calculates a first order accurate upwind step of the conserved transport part of the 
Euler equations (CFD or MHD) which is used as the half-step predictor in the Runge-Kutta timestep

For simple CFD, the magnetic terms are identically zero.

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

In general thermal pressure is an arbitrary positive function of e, however the ideal gas
law is built into Imogen in multiple locations and significant re-checking would be needed
if it were to be generalized.

The hydro functions solve the same equations with B set to <0,0,0> which simplifies
and considerably speeds up the process.

 */

//__device__ void __syncthreads(void);

/* This is my handwritten assembly version of the Osher function
 * It is not now observed to execute faster than the -O2 version of same.
 */

__device__ __noinline__ double slopeLimiter_Osher_asm(double A, double B)
{
double retval;
asm(    ".reg .f64 s;\n\t"      // hold sum
        ".reg .f64 p;\n\t"       // hold product
        ".reg .f64 q;\n\t"       // hold quotient"
        ".reg .pred isneg;\n\t"  // predicate for positivity
        "mul.f64        s, %1, %2;\n\t" // Store AB in s
        "mov.f64        %0, 0d0000000000000000;\n\t" // Load output register with zero
        "setp.le.f64    isneg, s, 0d0000000000000000;\n\t" // isneg is true if AB <= 0
        "@isneg bra     endline;\n\t" // Hop past the computation to follow
        "neg.f64        p, s;\n\t"      // store -AB in p
        "add.f64        s, %1, %2;\n\t" // Store A+B in s
        "fma.rn.f64     q, s, s, p;\n\t" // Store (A+B)^2 - AB = AA + AB + BB in q
        "mul.f64        p, p, 0dBFE8000000000000;\n\t" // store -.75(-AB) = .75AB in p
        "mul.f64        p, p, s;\n\t" // Store p s  = (.75 A B) (A+B) in p
        "div.rn.f64     %0, p, q;\n\t" // Store that / the quotient in return register
        "endline:\n\t" : "=d"(retval) : "d"(A), "d"(B) );
return retval;
}

pParallelTopology topoStructureToC(const mxArray *prhs);
void cfSync(double *cfArray, int cfNumel, const mxArray *topo);

__global__ void replicateFreezeArray(double *freezeIn, double *freezeOut, int ncopies, int ny, int nz);
__global__ void reduceFreezeArray(double *freezeClone, double *freeze, int nx, int ny, int nz);

__global__ void cukern_AUSM_firstorder_uniform(double *P, double *Qout, double lambdaQtr, int nx, int ny, int devArrayNumel);

template <unsigned int PCswitch>
__global__ void cukern_AUSM_step(double *Qstore, double lambda, int nx, int ny);

template <unsigned int PCswitch>
__global__ void cukern_HLL_step(double *Qin, double *Qstore, double lambda, int nx, int ny);

template <unsigned int fluxDirection>
__global__ void cukern_HLLC_1storder(double *Qin, double *Qout, double lambda);
template <unsigned int fluxDirection>
__global__ void cukern_HLLC_2ndorder(double *Qin, double *Qout, double lambda);

template <unsigned int PCswitch>
__global__ void cukern_XinJinMHD_step(double *Qstore, double *Cfreeze, double lambda, int nx, int ny, int devArrayNumel);

template <unsigned int PCswitch>
__global__ void cukern_XinJinHydro_step(double *Qstore, double *Cfreeze, double lambda, int nx, int ny, int devArrayNumel);

/* Stopgap until I manage to stuff pressure solvers into all the predictors... */
__global__ void cukern_PressureSolverHydro(double *state, double *gasPressure, int devArrayNumel);
__global__ void cukern_PressureFreezeSolverHydro(double *state, double *gasPressure, double *Cfreeze, int nx, int ny, int devArrayNumel);

// FIXME: this needs to be rewritten like the above...
__global__ void cukern_PressureFreezeSolverMHD(double *Qstore, double *Cfreeze, int nx, int ny, int devArrayNumel);

#define BLOCKLEN 28
#define BLOCKLENP2 30
#define BLOCKLENP4 32

#define YBLOCKS 4
#define FREEZE_NY 4

#define PTRS_PER_KERN 10
__constant__ __device__ double *inputPointers[PTRS_PER_KERN*MAX_GPUS_USED];
__constant__ __device__ double fluidQtys[8];

//#define LIMITERFUNC fluxLimiter_Zero
#define LIMITERFUNC fluxLimiter_minmod
//#define LIMITERFUNC fluxLimiter_Osher
//#define LIMITERFUNC fluxLimiter_VanLeer
//#define LIMITERFUNC fluxLimiter_superbee

#define SLOPEFUNC slopeLimiter_Osher
//#define SLOPEFUNC slopeLimiter_Zero
//#define SLOPEFUNC slopeLimiter_minmod
//#define SLOPEFUNC slopeLimiter_VanLeer

#define FLUID_GAMMA   fluidQtys[0]
#define FLUID_GM1     fluidQtys[1]
#define FLUID_GG1     fluidQtys[2]
#define FLUID_MINMASS fluidQtys[3]
#define FLUID_MINEINT fluidQtys[4]
#define MHD_PRESS_B   fluidQtys[5]
#define MHD_CS_B      fluidQtys[6]
#define FLUID_GOVERGM1 fluidQtys[7]



__constant__ __device__ int    arrayParams[4];
#define DEV_NX arrayParams[0]
#define DEV_NY arrayParams[1]
#define DEV_NZ arrayParams[2]
#define DEV_SLABSIZE arrayParams[3]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	int wanted_nlhs = 0;
	#ifdef DEBUGMODE
		wanted_nlhs = 1;
	#endif
	
	// Input and result
	if ((nrhs!=14) || (nlhs != wanted_nlhs)) mexErrMsgTxt("Wrong number of arguments: need cudaFluidStep(rho, E, px, py, pz, bx, by, bz, Ptot, c_f, lambda, purehydro?, fluid gamma)\n");

	CHECK_CUDA_ERROR("entering cudaFluidStep");

	int hydroOnly;
	hydroOnly = (int)*mxGetPr(prhs[11]);

	MGArray fluid[5], mag[3], PC[2];
	int worked = MGA_accessMatlabArrays(prhs, 0, 4, &fluid[0]);

	//cudaStream_t *GPUStreamList = getGPUTypeStreams(prhs[0]);

	// The sanity checker tended to barf on the 9 [allzeros] that represent "no array" before.
	if(hydroOnly == false) worked = MGA_accessMatlabArrays(prhs, 5, 7, &mag[0]);
	worked = MGA_accessMatlabArrays(prhs, 8, 9, &PC[0]);

	double lambda     = *mxGetPr(prhs[10]);

	double *thermo = mxGetPr(prhs[12]);
	double gamma = thermo[0];
	double rhomin= thermo[1];

	int steptype = (int)thermo[2];
	unsigned int stepdirect= (unsigned int)thermo[3];

	/* To be uploads to fluidQtys[] constant memory array */
	double gamHost[8];
	gamHost[0] = gamma;
	gamHost[1] = gamma-1.0;
	gamHost[2] = gamma*(gamma-1.0);
	gamHost[3] = rhomin;
	// Calculation of minimum internal energy for adiabatic fluid:
	// assert     cs > cs_min
	//     g P / rho > g rho_min^(g-1)
	// (g-1) e / rho > rho_min^(g-1)
	//             e > rho rho_min^(g-1)/(g-1)
	gamHost[4] = powl(rhomin, gamma-1.0)/(gamma-1.0);
	gamHost[5] = 1.0 - .5*gamma;
	gamHost[6] = ALFVEN_CSQ_FACTOR - .5*(gamma-1.0)*gamma;
	gamHost[7] = gamma/(gamma-1.0); // pressure to energy flux conversion for ideal gas adiabatic EoS
	// Even for gamma=5/3, soundspeed is very weakly dependent on density (cube root) for adiabatic fluid

	dim3 arraySize = makeDim3(&fluid[0].dim[0]);
	dim3 blocksize = makeDim3(BLOCKLENP4, YBLOCKS, 1);
	dim3 gridsize;

	double *wStepValues[fluid->nGPUs];

	int i, j, sub[6];

	int haParams[4];
	double *hostptrs[PTRS_PER_KERN];
	for(i = 0; i < fluid->nGPUs; i++) {
		cudaSetDevice(fluid->deviceID[i]);

		calcPartitionExtent(fluid, i, &sub[0]);
		haParams[0] = sub[3];
		haParams[1] = sub[4];
		haParams[2] = sub[5];
		/* This is aligned on 256 so we _can_ safely divide by 8
		 * We _have_  to because the cuda code does (double *) + SLABSIZE */
		haParams[3] = fluid->slabPitch[i] / sizeof(double);

		// Copy [rho, px, py, pz, E, bx, by, bz, P] array pointers to inputPointers
		/* FIXME: This sucks but the HLL kernel depends on it and I
		 * FIXME: seriously don't have time to rewrite it presently */

		for(j = 0; j < 5; j++)  hostptrs[j+0] = fluid[j].devicePtr[i];

		hostptrs[2           ] = fluid[1+stepdirect].devicePtr[i];
		hostptrs[1+stepdirect] = fluid[2           ].devicePtr[i];

		for(j = 0; j < 3; j++) hostptrs[j+5] =   mag[j].devicePtr[i];
		for(j = 0; j < 2; j++) hostptrs[j+8] =    PC[j].devicePtr[i];

		cudaMemcpyToSymbol(arrayParams, &haParams[0], 4*sizeof(int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(fluidQtys, &gamHost[0], 8*sizeof(double), 0, cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(inputPointers, hostptrs, PTRS_PER_KERN*sizeof(double *), 0, cudaMemcpyHostToDevice);
	}

	if(hydroOnly == 1) {
		// Switches between various prediction steps
		switch(steptype) {
		case STEPTYPE_XINJIN: {
			// Allocate memory for the half timestep's output
			int numarrays = 6 ;
			for(i = 0; i < fluid->nGPUs; i++) {
				cudaSetDevice(fluid->deviceID[i]);
				CHECK_CUDA_ERROR("cudaSetDevice()");
				cudaMalloc((void **)&wStepValues[i], numarrays*fluid->slabPitch[i]*sizeof(double)); // [rho px py pz E P]_.5dt
				CHECK_CUDA_ERROR("In cudaFluidStep: halfstep malloc");
			}

// Step 1: Compute freezing speed

// Step 2: Compute


			cfSync(hostptrs[9], fluid[0].dim[1]*fluid[0].dim[2], prhs[13]);

			cukern_XinJinHydro_step<RK_PREDICT><<<gridsize, blocksize>>>(wStepValues[0], hostptrs[9], 0.25*lambda, arraySize.x, arraySize.y, (int)fluid->numel);
			CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: cukern_XinJinHydro_step prediction step");

#ifdef DBG_FIRSTORDER // Run at 1st order: dump upwind values straight back to output arrays
#ifdef DEBUGMODE
			printf("WARNING: Operating at first order for debug purposes!\n");
			cudaMemcpy(fluid[0].devicePtr[i], wStepValues[i], 5*fluid[0].numel*sizeof(double), cudaMemcpyDeviceToDevice);
			//returnDebugArray(fluid, 6, wStepValues, plhs);
#endif
#else // runs at 2nd order

			dim3 cfblk;  cfblk.x = 64; cfblk.y = 4; cfblk.z = 1;
			dim3 cfgrid; cfgrid.x = arraySize.y / 4; cfgrid.y = arraySize.z; cfgrid.z = 1;
			cfgrid.x += 1*(4*cfgrid.x < arraySize.y);

			cukern_PressureFreezeSolverHydro<<<cfgrid, cfblk>>>(wStepValues[0], wStepValues[0] + (5*fluid->slabPitch[i])/8, hostptrs[9], arraySize.x, arraySize.y, fluid->numel);
			CHECK_CUDA_LAUNCH_ERROR(cfblk, cfgrid, fluid, hydroOnly, "In cudaFluidStep: cukern_PressureFreezeSolverHydro");
			cfSync(hostptrs[9], fluid[0].dim[1]*fluid[0].dim[2], prhs[13]);
			CHECK_CUDA_ERROR("In cudaFluidStep: second hd c_f sync");
			cukern_XinJinHydro_step<RK_CORRECT><<<gridsize, blocksize>>>(wStepValues[0], hostptrs[9], 0.5*lambda, arraySize.x, arraySize.y, fluid->numel);
			CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: cukern_XinJinHydro_step corrector step");
#endif
		} break;
		case STEPTYPE_HLL:
		case STEPTYPE_HLLC: {
			int numarrays;
			#ifdef DEBUGMODE
			numarrays = 6 + DBG_NUMARRAYS;
			#else
			numarrays = 6;
			#endif

			// Allocate memory for the half timestep's output
			for(i = 0; i < fluid->nGPUs; i++) {
				cudaSetDevice(fluid->deviceID[i]);
				CHECK_CUDA_ERROR("cudaSetDevice()");
				cudaMalloc((void **)&wStepValues[i], numarrays*fluid->slabPitch[i]); // [rho px py pz E P]_.5dt
				CHECK_CUDA_ERROR("In cudaFluidStep: halfstep malloc");
			}

			// Launch zee kernels
			for(i = 0; i < fluid->nGPUs; i++) {
				cudaSetDevice(fluid->deviceID[i]);

				// Find out the size of the partition
				calcPartitionExtent(fluid, i, sub);
				gridsize.x = (sub[3]/BLOCKLEN); gridsize.x += 1*(gridsize.x*BLOCKLEN < sub[3]);
				gridsize.y = sub[5];

				// Fire off the fluid update step
				if(steptype == STEPTYPE_HLL) {
					cukern_PressureSolverHydro<<<32, 256>>>(fluid[0].devicePtr[i], wStepValues[i] + 5*haParams[3], fluid->partNumel[i]);
					CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: cukern_PressureSolverHydro");
					cukern_HLL_step<RK_PREDICT><<<gridsize, blocksize>>>(fluid[0].devicePtr[i], wStepValues[i], .5*lambda, sub[3], sub[4]);
				} else switch(stepdirect) {
				case 1: cukern_HLLC_1storder<FLUX_X><<<gridsize, blocksize>>>(fluid[0].devicePtr[i], wStepValues[i], .5*lambda); break;
				case 2: cukern_HLLC_1storder<FLUX_Y><<<gridsize, blocksize>>>(fluid[0].devicePtr[i], wStepValues[i], .5*lambda); break;
				case 3: cukern_HLLC_1storder<FLUX_Z><<<gridsize, blocksize>>>(fluid[0].devicePtr[i], wStepValues[i], .5*lambda); break;
				}
				CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: cukern_HLLC_1storder");

				#ifdef DBG_FIRSTORDER // Run at 1st order: dump upwind values straight back to output arrays
					#ifdef DEBUGMODE
					printf("WARNING: Operating at first order for debug purposes!\n");
					cudaMemcpy(fluid[0].devicePtr[i], wStepValues[i], 5*fluid[0].slabPitch[0], cudaMemcpyDeviceToDevice);
					returnDebugArray(fluid, 6, wStepValues, plhs);
					#endif
				#else // runs at 2nd order

				if(steptype == STEPTYPE_HLL) {
					cukern_PressureSolverHydro<<<32, 256>>>(wStepValues[i], wStepValues[i] + 5*haParams[3], fluid->partNumel[i]);
					CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: cukern_PressureSolverHydro");
					cukern_HLL_step<RK_CORRECT><<<gridsize, blocksize>>>(fluid[0].devicePtr[i], wStepValues[i], lambda, sub[3], sub[4]);
				} else switch(stepdirect) {
				case 1: cukern_HLLC_2ndorder<FLUX_X><<<gridsize, blocksize>>>(wStepValues[i], fluid[0].devicePtr[i], lambda); break;
				case 2: cukern_HLLC_2ndorder<FLUX_Y><<<gridsize, blocksize>>>(wStepValues[i], fluid[0].devicePtr[i], lambda); break;
				case 3: cukern_HLLC_2ndorder<FLUX_Z><<<gridsize, blocksize>>>(wStepValues[i], fluid[0].devicePtr[i], lambda); break;
				}
				CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: cukern_HLL_step correction step");
					#ifdef DBG_SECONDORDER
                                        returnDebugArray(fluid, 6, wStepValues, plhs);
					#endif

				#endif
			}
			for(i = 0; i < fluid->nGPUs; i++) {
				// Release the memory taken for this step
				cudaSetDevice(fluid->deviceID[i]);
				CHECK_CUDA_ERROR("cudaSetDevice();");
				cudaFree(wStepValues[i]);
				CHECK_CUDA_ERROR("cudaFree()");
			}
		} break;
		}
		if(fluid->partitionDir == PARTITION_X) MGA_exchangeLocalHalos(fluid, 5);
		CHECK_CUDA_ERROR("after exchange halos");

		// We never got this unscrewed so it's disabled even though the 1st order AUSM code is below
		//		cukern_AUSM_step<RK_CORRECT><<<gridsize, blocksize>>>(wStepValues, lambda, arraySize.x, arraySize.y);
	} else {
		cfSync(hostptrs[9], fluid[0].dim[1]*fluid[0].dim[2], prhs[13]);
		CHECK_CUDA_ERROR("In cudaFluidStep: first mhd c_f sync");

		cukern_XinJinMHD_step<RK_PREDICT><<<gridsize, blocksize>>>(wStepValues[0], hostptrs[9], 0.25*lambda, arraySize.x, arraySize.y, fluid->numel);
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: mhd predict step");

		dim3 cfblk;  cfblk.x = 64; cfblk.y = 4; cfblk.z = 1;
		dim3 cfgrid; cfgrid.x = arraySize.y / 4; cfgrid.y = arraySize.z; cfgrid.z = 1;
		cfgrid.x += 1*(4*cfgrid.x < arraySize.y);
		cukern_PressureFreezeSolverMHD<<<cfgrid, cfblk>>>(wStepValues[0], hostptrs[9], arraySize.x, arraySize.y, fluid->numel);
		CHECK_CUDA_LAUNCH_ERROR(cfblk, cfgrid, fluid, hydroOnly, "In cudaFluidStep: cukern_PressureFreezeSolverMHD");
		cfSync(hostptrs[9], fluid[0].dim[1]*fluid[0].dim[2], prhs[13]);
		CHECK_CUDA_ERROR("In cudaFluidStep: second mhd c_f sync");
		cukern_XinJinMHD_step<RK_CORRECT><<<gridsize, blocksize>>>(wStepValues[0], hostptrs[9], 0.5*lambda, arraySize.x, arraySize.y, fluid->numel);
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, hydroOnly, "In cudaFluidStep: mhd TVD step");
	}

}

void cfSync(double *cfArray, int cfNumel, const mxArray *topo)
{
	pParallelTopology topology = topoStructureToC(topo);

	/* Reverse silly Fortran memory ordering */
	int d = 0;
	int dmax = topology->nproc[d];

	MPI_Comm commune = MPI_Comm_f2c(topology->comm);
	int r0; MPI_Comm_rank(commune, &r0);

	double *storeA = (double*)malloc(sizeof(double)*cfNumel);
	double *storeB = (double*)malloc(sizeof(double)*cfNumel);

	cudaMemcpy(storeA, cfArray, cfNumel*sizeof(double), cudaMemcpyDeviceToHost);

	//MPI_Status amigoingtodie;

	/* FIXME: This is a temporary hack
   FIXME: The creation of these communicators should be done once,
   FIXME: by PGW, at start time. */
	int dimprocs[dmax];
	int proc0, procstep;
	switch(d) { /* everything here is Wrong because fortran is Wrong */
	case 0: /* i0 = nx Y + nx ny Z, step = 1 -> nx ny */
		/* x dimension: step = ny nz, i0 = z + nz y */
		proc0 = topology->coord[2] + topology->nproc[2]*topology->coord[1];
		procstep = topology->nproc[2]*topology->nproc[1];
		break;
	case 1: /* i0 = x + nx ny Z, step = nx */
		/* y dimension: step = nz, i0 = z + nx ny x */
		proc0 = topology->coord[2] + topology->nproc[2]*topology->nproc[1]*topology->coord[0];
		procstep = topology->nproc[2];
		break;
	case 2: /* i0 = x + nx Y, step = nx ny */
		/* z dimension: i0 = nz y + nz ny x, step = 1 */
		proc0 = topology->nproc[2]*(topology->coord[1] + topology->nproc[1]*topology->coord[0]);
		procstep = 1;
		break;
	}
	int j;
	for(j = 0; j < dmax; j++) {
		dimprocs[j] = proc0 + j*procstep;
	}

	MPI_Group worldgroup, dimgroup;
	MPI_Comm dimcomm;
	/* r0 has our rank in the world group */
	MPI_Comm_group(commune, &worldgroup);
	MPI_Group_incl(worldgroup, dmax, dimprocs, &dimgroup);
	/* Create communicator for this dimension */
	MPI_Comm_create(commune, dimgroup, &dimcomm);

	/* Perform the reduce */
	MPI_Allreduce((void *)storeA, (void *)storeB, cfNumel, MPI_DOUBLE, MPI_MAX, dimcomm);

	MPI_Barrier(dimcomm);
	/* Clean up */
	MPI_Group_free(&dimgroup);
	MPI_Comm_free(&dimcomm);

	cudaMemcpy(cfArray, storeB, cfNumel*sizeof(double), cudaMemcpyHostToDevice);

	free(storeA); free(storeB);

	return;

}

/* Invoke with [8 1] threadblock and [ny nz] grid */
__global__ void replicateFreezeArray(double *freezeIn, double *freezeOut, int ncopies, int ny, int nz)
{
	int y = blockIdx.x;
	int z = blockIdx.y;
	__shared__ double c;

	if(threadIdx.x == 0) {
		c = freezeIn[y + ny*z];
	}
	__syncthreads();
	int x;
	for(x = threadIdx.x; x < ncopies; x += blockDim.x) {
		freezeOut[x + ncopies*(y+ny*z)] = c;
	}
}

/* Invoke with [x 1 1] threads and [ny nz] grid */
__global__ void reduceFreezeArray(double *freezeClone, double *freeze, int nx, int ny, int nz)
{
	__shared__ double locarr[16];
	int y = blockIdx.x;
	int z = blockIdx.y;
	int tix = threadIdx.x;

	double phi = 0.0;
	freezeClone += nx*(y+ny*z);
	int i;
	for(i = 0; i < nx; i += 16) {
		phi = (freeze[i+tix] > phi)? freeze[i+tix] : phi;
	}
	__syncthreads();

	locarr[tix] = phi;

	if(tix < 8) locarr[tix] = (locarr[tix+8] > locarr[tix])? locarr[tix+8] : locarr[tix];
	__syncthreads();
	if(tix < 4) locarr[tix] = (locarr[tix+4] > locarr[tix])? locarr[tix+4] : locarr[tix];
	__syncthreads();
	if(tix < 2) locarr[tix] = (locarr[tix+2] > locarr[tix])? locarr[tix+2] : locarr[tix];
	__syncthreads();
	if(tix == 0) freeze[y+ny*z] = (locarr[1] > locarr[0])? locarr[1] : locarr[0];

}

// These tell the HLL and HLLC solvers how to dereference their shmem blocks
// 16 threads, 8 circshifted shmem arrays
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

#define N_SHMEM_BLOCKS 5


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
	IC += N_SHMEM_BLOCKS*BLOCKLENP4*threadIdx.y;
	IL += N_SHMEM_BLOCKS*BLOCKLENP4*threadIdx.y;
	IR += N_SHMEM_BLOCKS*BLOCKLENP4*threadIdx.y;

	/* Declare shared variable array */
	__shared__ double shblk[YBLOCKS*N_SHMEM_BLOCKS*BLOCKLENP4];
	double *shptr = &shblk[IC];

	/* Declare tons of doubles and hope optimizer can sort this out */
	double A, B, C, D, E, F, G, H;

	/* My x index: thread + blocksize block, wrapped circularly */
	//int thisThreadPonders  = (threadIdx.x > 0) && (threadIdx.x < blockDim.x-1);
	int thisThreadDelivers = (threadIdx.x >= 2) && (threadIdx.x <= (BLOCKLENP4-3));

	int x0 = threadIdx.x + (BLOCKLEN)*blockIdx.x - 2;
	if(x0 < 0) x0 += DEV_NX; // left wraps to right edge
	if(x0 > (DEV_NX+1)) return; // More than 2 past right returns
	if(x0 > (DEV_NX-1)) { x0 -= DEV_NX; thisThreadDelivers = 0; } // past right must wrap around to left

	/* Do some index calculations */
	x0 += DEV_NX*(DEV_NY*blockIdx.y + threadIdx.y); /* This block is now positioned to start at its given (x,z) coordinate */
	int j = threadIdx.y;
	Qin += x0;
	Qout += x0;

	int fluxmode;

	for(; j < DEV_NY; j += blockDim.y) {
		/* LOAD VARIABLES: CONSTANT APPROXIMATION -> NO RECONSTRUCTION STEP */	
		A = Qin[0             ]; /* rho; Load the q_i variables */
		B = Qin[  DEV_SLABSIZE]; /* E */
		switch(fluxDirection) {
		case FLUX_X: /* Variables are in normal order [px py pz] */
			C = Qin[2*DEV_SLABSIZE]; /* Px */
			D = Qin[3*DEV_SLABSIZE]; /* Py */
			E = Qin[4*DEV_SLABSIZE]; /* Pz */
			break;
		case FLUX_Y: /* Slabs are in order [py px pz] */
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

		A = shptr[BOS0]; /* rho_le */
		B = shblk[IR+BOS0]; /* rho_re */
		F = sqrt(B/A); /* = batten et al's R */

		A = .5*C*F*FLUID_GM1/((1.0+F)*(1.0+F)); // A now contains velocity jump term from Batten eqn 51 (\overline{c}^2)

		C = shptr[BOS1]/shptr[BOS0]; /* Pl/rhol = csq_l / gamma */
		D = shblk[IR+BOS1]/B; /* Pr/rhor = csq_r / gamma */

		E = FLUID_GAMMA * (C + F*D) / (1.0+F) + A; /* Batten eqn 51 implemented */

		E = sqrt(E); /* Batten \overline{c} */

		C = sqrt(FLUID_GAMMA * C); // left c_s
		D = sqrt(FLUID_GAMMA * D); // right c_s

		A = shptr[BOS2]; /* vx_le */
		B = shblk[IR+BOS2]; /* vx_re */

		F = (A + F*B)/(1.0+F); /* vx_tilde */

		C = A - C;
		D = B + D;

		F -= E; 
		C = (C < F) ? C : F; /* left bound: min[u_l - c_l, u_roe - c_roe] */

		F = F + 2.0*E;
		D = (D > F) ? D : F; /* right bound: max[u_r + c_r, u_roe + c_roe] */

		#ifdef DBG_FIRSTORDER
			DBGSAVE(0, C); DBGSAVE(2, D);
		#endif

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
				B *= (E+FLUID_GOVERGM1*F); //    = ENERGY FLUX
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
		// shptr[BOS0, 1, 2, 3, 4] = flux of [rho, E, px, py, pz

#ifdef DBG_FIRSTORDER
	DBGSAVE(4, shptr[BOS2]); // px flux
	DBGSAVE(5, shptr[BOS1]); // E flux
#endif

		if(thisThreadDelivers) {
			Qout[0]              = Qin[0           ] - lambda * (shptr[BOS0]-shblk[IL+BOS0]);
			Qout[DEV_SLABSIZE]   = Qin[  DEV_SLABSIZE] - lambda * (shptr[BOS1]-shblk[IL+BOS1]);
			switch(fluxDirection) {
			case FLUX_X:
				Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * (shptr[BOS2]-shblk[IL+BOS2]);
				Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * (shptr[BOS3]-shblk[IL+BOS3]);
				Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * (shptr[BOS4]-shblk[IL+BOS4]);
				break;
			case FLUX_Y:
				Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * (shptr[BOS2]-shblk[IL+BOS2]);
				Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * (shptr[BOS3]-shblk[IL+BOS3]);
				Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * (shptr[BOS4]-shblk[IL+BOS4]);
				break;
			case FLUX_Z:
				Qout[4*DEV_SLABSIZE] = Qin[4*DEV_SLABSIZE] - lambda * (shptr[BOS2]-shblk[IL+BOS2]);
				Qout[3*DEV_SLABSIZE] = Qin[3*DEV_SLABSIZE] - lambda * (shptr[BOS3]-shblk[IL+BOS3]);
				Qout[2*DEV_SLABSIZE] = Qin[2*DEV_SLABSIZE] - lambda * (shptr[BOS4]-shblk[IL+BOS4]);
				break;
			}
		}

		Qin += blockDim.y*DEV_NX;
		Qout += blockDim.y*DEV_NX;
	}

}

#undef N_SHMEM_BLOCKS
#define N_SHMEM_BLOCKS 10

template <unsigned int fluxDirection>
__global__ void cukern_HLLC_2ndorder(double *Qin, double *Qout, double lambda)
{
	// Create center, look-left and look-right indexes
	int IC = threadIdx.x;
	int IL = threadIdx.x - 1;
	IL += 1*(IL < 0);
	int IR = threadIdx.x + 1;
	IR -= BLOCKLENP4*(IR > (BLOCKLENP4-1));

	// Advance by the Y index
	IC += N_SHMEM_BLOCKS*BLOCKLENP4*threadIdx.y;
	IL += N_SHMEM_BLOCKS*BLOCKLENP4*threadIdx.y;
	IR += N_SHMEM_BLOCKS*BLOCKLENP4*threadIdx.y;

	/* Declare shared variable array */
	__shared__ double shblk[YBLOCKS*N_SHMEM_BLOCKS*BLOCKLENP4];
	double *shptr = &shblk[IC];

	/* Declare tons of doubles and hope optimizer can sort this out */
	double A, B, C, D, E, F, G, H;

	/* My x index: thread + blocksize block, wrapped circularly */
	//int thisThreadPonders  = (threadIdx.x > 0) && (threadIdx.x < blockDim.x-1);
	int thisThreadDelivers = (threadIdx.x >= 2) && (threadIdx.x <= (BLOCKLENP4-3));

	int x0 = threadIdx.x + (BLOCKLEN)*blockIdx.x - 2;
	if(x0 < 0) x0 += DEV_NX; // left wraps to right edge
	if(x0 > (DEV_NX+1)) return; // More than 2 past right returns
	if(x0 > (DEV_NX-1)) { x0 -= DEV_NX; thisThreadDelivers = 0; } // past right must wrap around to left

	/* Do some index calculations */
	x0 += DEV_NX*(DEV_NY*blockIdx.y + threadIdx.y); /* This block is now positioned to start at its given (x,z) coordinate */
	int j = threadIdx.y;
	Qin += x0;
	Qout += x0;

	int fluxmode;

	for(; j < DEV_NY; j += blockDim.y) {
		/* LOAD VARIABLES AND PERFORM MUSCL RECONSTRUCTION */	
		A = Qin[0             ]; /* Load the q_i variables */
		B = Qin[  DEV_SLABSIZE];
		switch(fluxDirection) {
		case FLUX_X: /* Variables are in normal order [px py pz] */
			C = Qin[2*DEV_SLABSIZE]; /* Px */
			D = Qin[3*DEV_SLABSIZE]; /* Py */
			E = Qin[4*DEV_SLABSIZE]; /* Pz */
			break;
		case FLUX_Y: /* Slabs are in order [py px pz] */
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
		C *= F; /* velocity */
		D *= F;
		E *= F;

		shptr[BOS0] = A; /* Upload to shmem: rho, epsilon, vx, vy, vz */
		shptr[BOS2] = B;
		shptr[BOS4] = C;
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

		/* CALCULATE WAVESPEED ESTIMATE */
		C = shptr[BOS5]-shblk[IR+BOS4]; /* Calculate Vleft - Vright */
                D = shptr[BOS7]-shblk[IR+BOS6];
                E = shptr[BOS9]-shblk[IR+BOS8];

                C = C*C+D*D+E*E; /* velocity jump, squared */

                A = shptr[BOS1]; /* rho_le */
                B = shblk[IR+BOS0]; /* rho_re */
                F = sqrt(B/A); /* = batten et al's R */

                A = .5*C*F*FLUID_GM1/((1.0+F)*(1.0+F)); // A now contains velocity jump term from Batten eqn 51 (\overline{c}^2)

                C = shptr[BOS3]/shptr[BOS1]; /* Pl/rhol = csq_l / gamma */
                D = shblk[IR+BOS2]/shblk[IR+BOS0]; /* Pr/rhor = csq_r / gamma */

                E = FLUID_GAMMA * (C + F*D) / (1.0+F) + A; /* Batten eqn 51 implemented */

                E = sqrt(E); /* Batten \overline{c} */

		C = sqrt(FLUID_GAMMA * C); // left c_s
		D = sqrt(FLUID_GAMMA * D); // right c_s

		A = shptr[BOS5]; /* vx_le */
		B = shblk[IR+BOS4]; /* vx_re */

		F = (A + F*B)/(1.0+F); /* vx_tilde */

		C = A - C; // (vx-cs)_le
		D = B + D; // (vx+cs)_re

		F -= E;    // (vbar - cbar)
		C = (C < F) ? C : F; /* left bound: min[u_l - c_l, u_roe - c_roe] */

		F = F + 2.0*E; // (vbar + cbar)
		D = (D > F) ? D : F; /* right bound: max[u_r + c_r, u_roe + c_roe] */

#ifdef DBG_SECONDORDER
DBGSAVE(0, C);
DBGSAVE(2, D);
#endif

		/* HLLC METHOD: COMPUTE SPEED IN STAR REGION */
		/* Now we have lambda_min in C and lambda_max in D; A, B and F are free. */
		__syncthreads(); // I don't think this is necessary...

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

		if(thisThreadDelivers) {
			Qout[0]              -= lambda * (shptr[BOS7]-shblk[IL+BOS7]);
			Qout[DEV_SLABSIZE]   -= lambda * (shptr[BOS5]-shblk[IL+BOS5]);
			switch(fluxDirection) {
			case FLUX_X:
				Qout[2*DEV_SLABSIZE] -= lambda * (shptr[BOS6]-shblk[IL+BOS6]);
				Qout[3*DEV_SLABSIZE] -= lambda * (shptr[BOS8]-shblk[IL+BOS8]);
				Qout[4*DEV_SLABSIZE] -= lambda * (shptr[BOS9]-shblk[IL+BOS9]);
				break;
			case FLUX_Y:
				Qout[3*DEV_SLABSIZE] -= lambda * (shptr[BOS6]-shblk[IL+BOS6]);
				Qout[2*DEV_SLABSIZE] -= lambda * (shptr[BOS8]-shblk[IL+BOS8]);
				Qout[4*DEV_SLABSIZE] -= lambda * (shptr[BOS9]-shblk[IL+BOS9]);
				break;
			case FLUX_Z:
				Qout[4*DEV_SLABSIZE] -= lambda * (shptr[BOS6]-shblk[IL+BOS6]);
				Qout[3*DEV_SLABSIZE] -= lambda * (shptr[BOS8]-shblk[IL+BOS8]);
				Qout[2*DEV_SLABSIZE] -= lambda * (shptr[BOS9]-shblk[IL+BOS9]);
				break;
			}
		}

		Qin += blockDim.y*DEV_NX;
		Qout += blockDim.y*DEV_NX;
		__syncthreads();

	}

}

#undef N_SHMEM_BLOCKS
#define N_SHMEM_BLOCKS 10

#define HLL_LEFT 0
#define HLL_HLL  1
#define HLL_RIGHT 2
template <unsigned int PCswitch>
__global__ void cukern_HLL_step(double *Qin, double *Qstore, double lambda, int nx, int ny)
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
	double Ale, Ble, Cle;
	double Are, Bre, Cre;
	int HLL_FluxMode;
	double Sleft, Sright, Utilde, Atilde;
	double Fa, Fb; /* temp vars */

	/* My x index: thread + blocksize block, wrapped circularly */
	int thisThreadDelivers = (threadIdx.x >= 2) && (threadIdx.x <= (BLOCKLENP4-3));

	int x0 = threadIdx.x + (BLOCKLEN)*blockIdx.x - 2;
	if(x0 < 0) x0 += nx; // left wraps to right edge
	if(x0 > (nx+1)) return; // More than 2 past right returns
	if(x0 > (nx-1)) { x0 -= nx; thisThreadDelivers = 0; } // past right must wrap around to left

	/* Do some index calculations */
	x0 += nx*(ny*blockIdx.y + threadIdx.y); /* This block is now positioned to start at its given (x,z) coordinate */
	int j = threadIdx.y;

	for(; j < ny; j += blockDim.y) {

		if(PCswitch == RK_PREDICT) {
			/* If making prediction use simple 0th order "reconstruction." */
			Ale = Are = Qin[0*DEV_SLABSIZE + x0]; /* load rho */
			Bre =       Qin[2*DEV_SLABSIZE + x0]; /* load px */
			// We calculated the gas pressure into temp array # 6 before calling
			Cle = Cre = Qstore[5*DEV_SLABSIZE + x0]; /* load gas pressure */
			Ble = Bre / Ale; /* Calculate vx */
			Bre = Ble;
		} else {
			/* If making correction, perform linear MUSCL reconstruction */
			Ale = Qstore[x0 + 0*DEV_SLABSIZE]; /* load rho */
			Bre = Qstore[x0 + 2*DEV_SLABSIZE]; /* load px */
			Cle = Qstore[x0 + 5*DEV_SLABSIZE]; /* load pressure */
			Ble = Bre / Ale; /* Calculate vx */

			shblk[IC + BOS0] = Ale;
			shblk[IC + BOS1] = Ble;
			shblk[IC + BOS2] = Cle;
			__syncthreads();

			/*************** BEGIN SECTION 2 */
			Are = Ale - shblk[IL + BOS0];
			Bre = Ble - shblk[IL + BOS1];
			Cre = Cle - shblk[IL + BOS2];
			__syncthreads();

			/*************** BEGIN SECTION 3 */
			shblk[IC + BOS0] = Are;
			shblk[IC + BOS1] = Bre;
			shblk[IC + BOS2] = Cre;
			__syncthreads();

			/*************** BEGIN SECTION 4 */
			Fa = SLOPEFUNC(Are, shblk[IR + BOS0]);
			Are = Ale + Fa;
			Ale -= Fa;
			Fa = SLOPEFUNC(Bre, shblk[IR + BOS1]);
			Bre = Ble + Fa;
			Ble -= Fa;
			Fa = SLOPEFUNC(Cre, shblk[IR + BOS2]);
			Cre = Cle + Fa;
			Cle -= Fa;
		}
		// up to here uses 40 regs
		__syncthreads();
		/* Rotate the [le_i-1 re_i-1][le_i re_i][le_i+1 re_i+1] variables one left
		 * so that each cell stores [re_i le_i+1]
		 * and thus each thread deals with F_i+1/2 */
		shblk[IC + BOS0] = Ale;
		shblk[IC + BOS1] = Are;
		shblk[IC + BOS2] = Ble;
		shblk[IC + BOS4] = Cle;

		__syncthreads();

		Ale = sqrt(Are); Are = sqrt(shblk[IR + BOS0]);
		Ble = Bre; Bre = shblk[IR + BOS2];
		Cle = Cre; Cre = shblk[IR + BOS4];
		/* Calculation may now proceed based only on register variables! */

		//48 registers
		/* Get Roe-average particle x speed */
		Utilde = (Ale*Ble + Are*Bre)/(Ale+Are);

		/* Get Roe-average sonic speed and take our S_+- estimate for HLL */
		Sleft  = sqrt(FLUID_GAMMA*Cle);
		Sright = sqrt(FLUID_GAMMA*Cre);
		Atilde = (Sleft+Sright)/(Ale+Are);
		Sleft  = Utilde - Atilde;
		Sright = Utilde + Atilde;
		// 48 regs

		/* We always divide by 1/2a for flux mode HLL_HLL so save some calculations from here out */
		Atilde = .5/Atilde;

		// Load non-square-rooted density back up down here after using Ale/Are as scratch
		Ale = shblk[IC + BOS1]; Are = shblk[IR + BOS0];
		/* Accumulate 2*kinetic energy */
		__syncthreads();

		Fa = Ale*Ble; // Raw mass flux
		Fb = Are*Bre;

		shblk[IC + BOS0] = Fa;
		shblk[IC + BOS2] = Fb;

		shblk[IC + BOS6] = Ble*Fa; // Raw (convective) momentum flux, also to be used for pressure calculation
		shblk[IC + BOS7] = Bre*Fb;

		/* Determine where our flux originates from (Uleft, Uhll, or Uright) */
		HLL_FluxMode = HLL_HLL;
		if(Sleft > 0) HLL_FluxMode = HLL_LEFT;
		if(Sright< 0) HLL_FluxMode = HLL_RIGHT;
		// 50 regs

		/* Calculate the mass and momentum fluxes */
		switch(HLL_FluxMode) {
		case HLL_LEFT:  shblk[IC + BOS1] = Fa;
		shblk[IC + BOS3] = Fa*Ble + Cle; break;
		case HLL_HLL:   shblk[IC + BOS1] = (Sright*Fa - Sleft*Fb + Sleft*Sright*(Are-Ale))*Atilde;
		shblk[IC + BOS3] = (Sright*(Ble*Fa+Cle) - Sleft*(Bre*Fb+Cre) + Sleft*Sright*(Fb-Fa))*Atilde; break;
		case HLL_RIGHT: shblk[IC + BOS1] = Fb;
		shblk[IC + BOS3] = Fb*Bre + Cre; break;
		}
		// 52 registers
		/* Transfer Atilde to shmem, freeing Utilde/Atilde as Ule/Ure pair */
		shblk[IC + BOS5] = Atilde;

		__syncthreads();

		shblk[IC + BOS2] = (shblk[IL + BOS1]-shblk[IC + BOS1]);
		shblk[IC + BOS4] = (shblk[IL + BOS3]-shblk[IC + BOS3]);

		/* Flux density and momentum... for prediction we explicitly did not use MUSCL and
                   therefore Ale = Acentered. */
		if(thisThreadDelivers) {
			if(PCswitch == RK_PREDICT) {
				Qstore[x0                 ] = Ale + lambda * shblk[IC + BOS2];
				Qstore[x0 + 2*DEV_SLABSIZE] = Fa  + lambda * shblk[IC + BOS4];
			} else {
				Qin[x0 + 0*DEV_SLABSIZE] += lambda * shblk[IC + BOS2];
				Qin[x0 + 2*DEV_SLABSIZE] += lambda * shblk[IC + BOS4];
			}
		}

		__syncthreads();
		// 55 registers
		if(PCswitch == RK_PREDICT) {
			/* If making prediction use simple 0th order "reconstruction." */
			Fa     = Qin[x0 + 3*DEV_SLABSIZE]; /* load py */
			Utilde = Qin[x0 + 4*DEV_SLABSIZE]; /* load pz */
		} else {
			/* If making correction, perform 1st order MUSCL reconstruction */
			Fa =     Qstore[x0 + 3*DEV_SLABSIZE]; /* load py */
			Utilde = Qstore[x0 + 4*DEV_SLABSIZE]; /* load pz */

			shblk[IC + BOS0] = Fa;
			shblk[IC + BOS2] = Utilde;
			__syncthreads();

			/*************** BEGIN SECTION 2 */
			Fb = Fa - shblk[IL + BOS0];
			Atilde = Utilde - shblk[IL + BOS2];
			__syncthreads();

			/*************** BEGIN SECTION 3 */

			shblk[IC + BOS0] = Fb;
			shblk[IC + BOS2] = Atilde;
			__syncthreads();

			/*************** BEGIN SECTION 4 */
			/* Take the speed hit and use shmem #4 to avoid eating more registers */
			shblk[IC + BOS4] = SLOPEFUNC(Fb, shblk[IR + BOS0]);
			Fb = Fa + shblk[IC + BOS4];
			Fa -= shblk[IC + BOS4];
			shblk[IC + BOS4] = SLOPEFUNC(Atilde, shblk[IR + BOS2]);
			Atilde = Utilde + shblk[IC + BOS4];
			Utilde -= shblk[IC + BOS4];
		}
		/* Rotate py and pz to the left so cell i has L/R values of interface i+1/2 */
		__syncthreads();
		//55 registers
		shblk[IC + BOS0] = Fa;
		shblk[IC + BOS1] = Utilde;

		__syncthreads();
		if(PCswitch == RK_PREDICT) {

			Fb = shblk[IR + BOS0];
			Atilde = shblk[IR + BOS1];
		} else {
			Fa = Fb; Fb = shblk[IR + BOS0];
			Utilde = Atilde; Atilde = shblk[IR + BOS1];
		}

		shblk[IC + BOS6] = .5*(shblk[IC + BOS6] + (Fa*Fa + Utilde*Utilde)/Ale);
		shblk[IC + BOS7] = .5*(shblk[IC + BOS7] + (Fb*Fb + Atilde*Atilde)/Are);

		switch(HLL_FluxMode) {
		case HLL_LEFT:  shblk[IC + BOS2] = Fa * Ble; /* py flux */
		shblk[IC + BOS3] = Utilde * Ble; /* pz flux */
		shblk[IC + BOS4] = Ble * (shblk[IC + BOS6] + FLUID_GOVERGM1*Cle);
		break; /* E flux */
		case HLL_HLL:   shblk[IC + BOS2] = (Sright*(Fa*Ble) - Sleft*(Fb*Bre) + Sleft*Sright*(Fb-Fa))*shblk[IC + BOS5];
		shblk[IC + BOS3] = (Sright*(Utilde*Ble) - Sleft*(Atilde*Bre) + Sleft*Sright*(Atilde-Utilde)) * shblk[IC + BOS5];
		shblk[IC + BOS4] = (Sright*Ble*(shblk[IC + BOS6] + FLUID_GOVERGM1*Cle) - Sleft*Bre*(shblk[IC + BOS7] + FLUID_GOVERGM1*Cre) + Sleft*Sright*(shblk[IC + BOS7] + 1.5*Cre - shblk[IC + BOS6] - 1.5*Cle))*shblk[IC + BOS5];
		break;
		case HLL_RIGHT:	shblk[IC + BOS2] = Fb * Bre;
		shblk[IC + BOS3] = Atilde * Bre;
		shblk[IC + BOS4] = Bre * (shblk[IC + BOS7] + FLUID_GOVERGM1*Cre);
		break;

		}

		__syncthreads(); /* shmem 2: py flux, shmem3: pz flux, shmem 4: E flux */

		if(thisThreadDelivers) {
			if(PCswitch == RK_PREDICT) {
				Qstore[x0 +   DEV_SLABSIZE] = Qin[x0 + 1*DEV_SLABSIZE] + lambda*(shblk[IL + BOS4]-shblk[IC + BOS4]);
				Qstore[x0 + 3*DEV_SLABSIZE] = Fa                   + lambda*(shblk[IL + BOS2]-shblk[IC + BOS2]);
				Qstore[x0 + 4*DEV_SLABSIZE] = Utilde               + lambda*(shblk[IL + BOS3]-shblk[IC + BOS3]);
			} else {
				Qin[x0 + 1*DEV_SLABSIZE] += lambda*(shblk[IL + BOS4]-shblk[IC + BOS4]);
				Qin[x0 + 3*DEV_SLABSIZE] += lambda*(shblk[IL + BOS2]-shblk[IC + BOS2]);
				Qin[x0 + 4*DEV_SLABSIZE] += lambda*(shblk[IL + BOS3]-shblk[IC + BOS3]);
			}
		}

		x0 += blockDim.y*nx;
		__syncthreads();
	}

}

#undef DBGSAVE
#define DBGSAVE(n, x) if(thisThreadDelivers) { Qstore[((n)+6)*(1024*64)] = (x); }

template <unsigned int PCswitch>
__global__ void cukern_XinJinHydro_step(double *Qstore, double *Cfreeze, double lambda, int nx, int ny, int devArrayNumel)
{
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
	if(x0 < 0) x0 += nx; // left wraps to right edge
	if(x0 > (nx+1)) return; // More than 2 past right returns
	if(x0 > (nx-1)) { x0 -= nx; thisThreadDelivers = 0; } // past right must wrap around to left

	/* Do some index calculations */
	x0 += nx*(ny*blockIdx.y + threadIdx.y); /* This block is now positioned to start at its given (x,z) coordinate */
	int j = threadIdx.y;
	int i;

	for(; j < ny; j += YBLOCKS) {
		C_f = Cfreeze[j + ny*blockIdx.y];

		if(PCswitch == RK_PREDICT) {
			// Load from init inputs, write to Qstore[]
			Q[0] = inputPointers[0][x0];
			Q[1] = inputPointers[1][x0];
			Q[2] = inputPointers[2][x0];
			Q[3] = inputPointers[3][x0];
			Q[4] = inputPointers[4][x0];
			P    = inputPointers[8][x0];
		} else {
			// Load from qstore, update init inputs
			Q[0] = Qstore[x0];
			Q[1] = Qstore[x0+  devArrayNumel];
			Q[2] = Qstore[x0+2*devArrayNumel];
			Q[3] = Qstore[x0+3*devArrayNumel];
			Q[4] = Qstore[x0+4*devArrayNumel];
			P    = Qstore[x0+5*devArrayNumel];
		}

		vx = Q[2] / Q[0];

		for(i = 0; i < 5; i++) {
			/* Calculate raw fluxes for rho, E, px, py, pz in order: */
			switch(i) {
			case 0: w = Q[2];          break;
			case 1: w = vx*(Q[1] + P); break;
			case 2: w = vx*Q[2] + P;   break;
			case 3: w = vx*Q[3];       break;
			case 4: w = vx*Q[4];       break;
			}

			shblk[IC + BOS0] = (C_f*Q[i] - w); /* Left  going flux */
			shblk[IC + BOS1] = (C_f*Q[i] + w); /* Right going flux */
			__syncthreads();

			if(PCswitch == RK_CORRECT) {
				/* Entertain two flux corrections */
				shblk[IC + BOS2] = (shblk[IL + BOS0] - shblk[IC + BOS0]) / 2.0; /* Deriv of leftgoing flux */
				shblk[IC + BOS3] = (shblk[IC + BOS1] - shblk[IL + BOS1]) / 2.0; /* Deriv of ritegoing flux */
				__syncthreads();

				/* Impose TVD limiter */
				shblk[IC + BOS0] += LIMITERFUNC(shblk[IC+BOS2], shblk[IR+BOS2]);
				shblk[IC + BOS1] += LIMITERFUNC(shblk[IR+BOS3], shblk[IC+BOS3]);
				__syncthreads();
			}

			if(thisThreadDelivers) {
				if(PCswitch == RK_PREDICT) {
					prop[i] = Q[i] - lambda * ( shblk[IC+BOS1]- shblk[IL+BOS1] -
							shblk[IR+BOS0]+ shblk[IC+BOS0]);
				} else {
					prop[i] = inputPointers[i][x0] - lambda * ( shblk[IC+BOS1]- shblk[IL+BOS1] -
							shblk[IR+BOS0]+ shblk[IC+BOS0]);
				}
			}

			__syncthreads();
		}

		if(thisThreadDelivers) {
			// Enforce density positivity
			prop[0] = (prop[0] < FLUID_MINMASS) ? FLUID_MINMASS : prop[0];

			// Calculate kinetic energy density and enforce minimum pressure
			w = .5*(prop[2]*prop[2] + prop[3]*prop[3] + prop[4]*prop[4])/prop[0];
			if((prop[1] - w) < prop[0]*FLUID_MINEINT) {
				prop[1] = w + prop[0]*FLUID_MINEINT;
			}

			if(PCswitch == RK_PREDICT) {
				Qstore[x0] = prop[0];
				Qstore[x0+devArrayNumel] = prop[1];
				Qstore[x0+2*devArrayNumel] = prop[2];
				Qstore[x0+3*devArrayNumel] = prop[3];
				Qstore[x0+4*devArrayNumel] = prop[4];
			} else {
				inputPointers[0][x0] = prop[0];
				inputPointers[1][x0] = prop[1];
				inputPointers[2][x0] = prop[2];
				inputPointers[3][x0] = prop[3];
				inputPointers[4][x0] = prop[4];
			}
		}

		x0 += YBLOCKS*nx;
		__syncthreads();
	}

}

template <unsigned int PCswitch>
__global__ void cukern_XinJinMHD_step(double *Qstore, double *Cfreeze, double lambda, int nx, int ny, int devArrayNumel)
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
	if(x0 < 0) x0 += nx; // left wraps to right edge
	if(x0 > (nx+1)) return; // More than 2 past right returns
	if(x0 > (nx-1)) { x0 -= nx; thisThreadDelivers = 0; } // past right must wrap around to left

	/* Do some index calculations */
	x0 += nx*(ny*blockIdx.y + threadIdx.y); /* This block is now positioned to start at its given (x,z) coordinate */
	int j = threadIdx.y;
	int i;

	for(; j < ny; j += YBLOCKS) {
		C_f = Cfreeze[j + ny*blockIdx.y];

		if(PCswitch == RK_PREDICT) {
			// Load from init inputs, write to Qstore[]
			Q[0] = inputPointers[0][x0];
			Q[1] = inputPointers[1][x0];
			Q[2] = inputPointers[2][x0];
			Q[3] = inputPointers[3][x0];
			Q[4] = inputPointers[4][x0];
			P    = inputPointers[8][x0];
		} else {
			// Load from qstore, update init inputs
			Q[0] = Qstore[x0];
			Q[1] = Qstore[x0+  devArrayNumel];
			Q[2] = Qstore[x0+2*devArrayNumel];
			Q[3] = Qstore[x0+3*devArrayNumel];
			Q[4] = Qstore[x0+4*devArrayNumel];
			P    = Qstore[x0+5*devArrayNumel];
		}
		B[0] = inputPointers[5][x0]; // Bx
		B[1] = inputPointers[6][x0]; // By
		B[2] = inputPointers[7][x0]; // Bz
		vx = Q[2] / Q[0];

		for(i = 0; i < 5; i++) {
			/* Calculate raw fluxes for rho, E, px, py, pz in order: */
			switch(i) {
			case 0: w = Q[2]; break;
			case 1: w = (vx * (Q[1] + P) - B[0]*(Q[2]*B[0]+Q[3]*B[1]+Q[4]*B[2])/Q[0] ); break;
			case 2: w = (vx*Q[2] + P - B[0]*B[0]); break;
			case 3: w = (vx*Q[3]     - B[0]*B[1]); break;
			case 4: w = (vx*Q[4]     - B[0]*B[2]); break;
			}

			shblk[IC + BOS0] = (C_f*Q[i] - w); /* Left  going flux */
			shblk[IC + BOS1] = (C_f*Q[i] + w); /* Right going flux */
			__syncthreads();

			if(PCswitch == RK_CORRECT) {
				/* Entertain two flux corrections */
				shblk[IC + BOS2] = (shblk[IL + BOS0] - shblk[IC + BOS0]) / 2.0; /* Deriv of leftgoing flux */
				shblk[IC + BOS3] = (shblk[IC + BOS1] - shblk[IL + BOS1]) / 2.0; /* Deriv of ritegoing flux */
				__syncthreads();

				/* Impose TVD limiter */
				shblk[IC + BOS0] += LIMITERFUNC(shblk[IC+BOS2], shblk[IR+BOS2]);
				shblk[IC + BOS1] += LIMITERFUNC(shblk[IR+BOS3], shblk[IC+BOS3]);
				__syncthreads();
			}

			if(thisThreadDelivers) {
				if(PCswitch == RK_PREDICT) {
					prop[i] = Q[i] - lambda * ( shblk[IC+BOS1]- shblk[IL+BOS1] -
							shblk[IR+BOS0]+ shblk[IC+BOS0]);
				} else {
					prop[i] = inputPointers[i][x0] - lambda * ( shblk[IC+BOS1]- shblk[IL+BOS1] -
							shblk[IR+BOS0]+ shblk[IC+BOS0]);
				}
			}

			__syncthreads();
		}

		if(thisThreadDelivers) {
			// Enforce density positivity
			prop[0] = (prop[0] < FLUID_MINMASS) ? FLUID_MINMASS : prop[0];

			// Calculate kinetic+magnetic energy density and enforce minimum pressure
			w = .5*(prop[2]*prop[2] + prop[3]*prop[3] + prop[4]*prop[4])/prop[0] + .5*(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
			if((prop[1] - w) < prop[0]*FLUID_MINEINT) {
				prop[1] = w + prop[0]*FLUID_MINEINT;
			}

			if(PCswitch == RK_PREDICT) {
				Qstore[x0] = prop[0];
				Qstore[x0+devArrayNumel] = prop[1];
				Qstore[x0+2*devArrayNumel] = prop[2];
				Qstore[x0+3*devArrayNumel] = prop[3];
				Qstore[x0+4*devArrayNumel] = prop[4];
			} else {
				inputPointers[0][x0] = prop[0];
				inputPointers[1][x0] = prop[1];
				inputPointers[2][x0] = prop[2];
				inputPointers[3][x0] = prop[3];
				inputPointers[4][x0] = prop[4];
			}
		}

		x0 += YBLOCKS*nx;
		__syncthreads();
	}

}


/* The Cfreeze array is round_up(NX / blockDim.x) x NY x NZ and is reduced in the X direction after
 * blockdim = [ xsize, 1, 1]
 * griddim  = [ sup[nx/(xsize - 4)], ny, 1]
 */
__global__ void cukern_AUSM_firstorder_uniform(double *P, double *Qout, double lambdaQtr, int nx, int ny, int devArrayNumel)
{
	/* Declare variable arrays */
	double q_i[5], s_i[5];
	double Plocal, w, vx;
	double Csonic, Mach, Mabs, Mplus, Mminus, Pleftgoing, Prightgoing;
	__shared__ double fluxLeft[BLOCKLENP4], fluxRight[BLOCKLENP4], pressAux[BLOCKLENP4];

	/* My x index: thread + blocksize block, wrapped circularly */
	int thisThreadPonders  = (threadIdx.x > 0) && (threadIdx.x < blockDim.x-1);
	int thisThreadDelivers = (threadIdx.x > 1) && (threadIdx.x < blockDim.x-2);

	int x0 = threadIdx.x + (blockDim.x-4)*blockIdx.x - 2;
	if(x0 < 0) x0 += nx; // left wraps to right edge
	if(x0 > (nx+1)) return; // More than 2 past right returns
	if(x0 > (nx-1)) { x0 -= nx; thisThreadDelivers = 0; } // past right must wrap around to left


	/* Do some index calculations */
	x0 += nx*ny*blockIdx.y; /* This block is now positioned to start at its given (x,z) coordinate */

	int i, j;
	for(j = 0; j < ny; j++) {
		/* Calculate this x segment's update: */
		/* Load local variables */
		q_i[0] = inputPointers[0][x0]; /* rho */
		q_i[1] = inputPointers[1][x0]; /* E */
		q_i[2] = inputPointers[2][x0]; /* px */
		q_i[3] = inputPointers[3][x0]; /* py */
		q_i[4] = inputPointers[4][x0]; /* pz */
		Plocal = P[x0];

		Csonic = sqrt(FLUID_GAMMA * Plocal / q_i[0]); // adiabatic c_s = gamma P / rho

		vx = q_i[2] / q_i[0]; /* This is used repeatedly. */

		Mach = vx / Csonic;
		Mabs = abs(Mach);

		if(Mabs < 1.0) {
			Mplus = .25*(Mach+1)*(Mach+1);
			Mminus = -.25*(Mach-1)*(Mach-1);
			Pleftgoing = .5*(1-Mach)*Plocal;
			Prightgoing = .5*(1+Mach)*Plocal;
		} else {
			Mplus = .5*(Mach+Mabs);
			Mminus= .5*(Mach-Mabs);
			Pleftgoing = .5*(Mach-Mabs)*Plocal/Mach;
			Prightgoing= .5*(Mach+Mabs)*Plocal/Mach;
		}

		fluxLeft[threadIdx.x] = Mplus; fluxRight[threadIdx.x] = Mminus;
		__syncthreads();
		/* generate agreed upon values of M_i-1/2 in Mminus and M_i+1/2 in Mplus */
		if(thisThreadPonders) {
			Mplus += fluxRight[threadIdx.x+1]; /* mach on right side */
			Mminus+= fluxLeft[threadIdx.x-1];  /* mach on left side  */
		}
		__syncthreads();

		/* Iterating over each variable, */
		for(i = 0; i < 5; i++) {
			/* Share values of advective flux */
			switch(i) {
			case 0: fluxLeft[threadIdx.x] = Csonic * q_i[0]; break;
			case 1: fluxLeft[threadIdx.x] = Csonic * (Plocal + q_i[1]); break;
			case 2: fluxLeft[threadIdx.x] = Csonic * q_i[2];            fluxRight[threadIdx.x] = Pleftgoing; pressAux[threadIdx.x] = Prightgoing;  break;
			case 3: fluxLeft[threadIdx.x] = Csonic * q_i[3];            break;
			case 4: fluxLeft[threadIdx.x] = Csonic * q_i[4];            break;
			}

			//			fluxLeft[threadIdx.x]  = Cfreeze_loc * q_i[i] - w;
			//			fluxRight[threadIdx.x] = Cfreeze_loc * q_i[i] + w;

			/* Calculate timestep: Make sure all fluxes are visible and difference it */
			__syncthreads();

			if(thisThreadDelivers) {
				/* left side flux */
				Mach = Mminus*fluxLeft[threadIdx.x-1]*(Mminus >= 0) + Mminus*fluxLeft[threadIdx.x]*(Mminus < 0);
				/* right side flux */
				Mabs = Mplus*fluxLeft[threadIdx.x]*(Mplus >= 0)    + Mplus*fluxLeft[threadIdx.x+1]*(Mplus < 0);

				/* Difference */
				s_i[i] = q_i[i] + lambdaQtr*(Mach-Mabs);

				if(i == 2) { /* Momentum equation: Difference pressure term as well */
					s_i[i] += lambdaQtr*(-fluxRight[threadIdx.x+1] + fluxRight[threadIdx.x] + pressAux[threadIdx.x-1] - pressAux[threadIdx.x]);
				}
			}
			__syncthreads();

		}

		__syncthreads(); /* Prevent anyone from reaching Cfreeze step and overwriting flux array too soon */

		/* Run sanity checks, and compute predicted pressure + freezing speed */
		if(thisThreadDelivers) {
			w = .5*(s_i[2]*s_i[2]+s_i[3]*s_i[3]+s_i[4]*s_i[4])/s_i[0]; /* Kinetic energy density */

			if((s_i[1] - w) < s_i[0] * FLUID_MINEINT) { /* if( (E-T) < minimum e_int density, */
				s_i[1] = w + s_i[0] * FLUID_MINEINT; /* Assert smallest possible pressure */
			}

			Plocal = FLUID_GM1 * (s_i[1] - w); /* Final decision on pressure */
			Qout[x0                  ] = s_i[0];
			Qout[x0 + devArrayNumel  ] = s_i[1];
			Qout[x0 + 2*devArrayNumel] = s_i[2];
			Qout[x0 + 3*devArrayNumel] = s_i[3];
			Qout[x0 + 4*devArrayNumel] = s_i[4];
			Qout[x0 + 5*devArrayNumel] = Plocal;
		}
		__syncthreads();

		/* Move in the Y direction */
		x0 += nx;
	}
}


/* Read Qstore and calculate pressure in it */
/* invoke with nx threads and nb blocks, whatever's best for the arch */
/* Note, state and gasPressure are not necessarily separate allocations
 * they will, in fact, usually be the first 5 slabs of the fluid state & the sixth, respectively
 * However all reads/writes are safely nonoverlapping
 */
__global__ void cukern_PressureSolverHydro(double *state, double *gasPressure, int devArrayNumel)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;

	double rho, E, z, momsq, P;

	int hx = blockDim.x*gridDim.x;
	int DAN = devArrayNumel;

	while(x < DAN) {
		rho = state[x      ];
		E   = state[x + DAN];
		z   = state[x+2*DAN];
		momsq = z*z;
		z   = state[x+3*DAN];
		momsq += z*z;
		z   = state[x+4*DAN];
		momsq += z*z;
		P = FLUID_GM1 * (E - .5*momsq/rho);
		gasPressure[x] = P;
		x += hx;
	}

}


/* Invoke with [64 x N] threads and [ny/N nz 1] blocks */
__global__ void cukern_PressureFreezeSolverHydro(double *state, double *gasPressure, double *Cfreeze, int nx, int ny, int devArrayNumel)
{
	__shared__ double Cfshared[64*FREEZE_NY];

	state += threadIdx.x + nx*(threadIdx.y + blockIdx.x*FREEZE_NY + ny*blockIdx.y);
	gasPressure += threadIdx.x + nx*(threadIdx.y + blockIdx.x*FREEZE_NY + ny*blockIdx.y);

	int x = threadIdx.x;
	int i = threadIdx.x + 64*threadIdx.y;

	double invrho, px, psq, P, locCf, cmax;

	Cfshared[i] = 0.0;
	cmax = 0.0;

	for(; x < nx; x += 64) {
		invrho = 1.0/state[0]; // load inverse of density
		psq = state[3*devArrayNumel]; // accumulate p^2 and end with px
		px =  state[4*devArrayNumel];
		psq = psq*psq + px*px;
		px = state[2*devArrayNumel];
		psq += px*px;

		// Store pressure
		*gasPressure = P = (state[devArrayNumel] - .5*psq*invrho)*FLUID_GM1;

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
	x = (threadIdx.y + FREEZE_NY * blockIdx.x) + ny*(blockIdx.y);

	Cfreeze[x] = cmax;
}


/* Invoke with [64 x N] threads and [ny/N nz 1] blocks */\
/* Reads magnetic field from inputPointers[5, 6, 7][x] */
__global__ void cukern_PressureFreezeSolverMHD(double *Qstore, double *Cfreeze, int nx, int ny, int devArrayNumel)
{
	__shared__ double Cfshared[64*FREEZE_NY];

	int delta = threadIdx.x + nx*(threadIdx.y + blockIdx.x*FREEZE_NY + ny*blockIdx.y);

	Qstore += delta;

	int x = threadIdx.x;
	int i = threadIdx.x + 64*threadIdx.y;

	double invrho, px, psq, locCf, cmax;
	double b, bsq;

	Cfshared[i] = 0.0;
	cmax = 0.0;

	for(; x < nx; x += 64) {
		invrho = 1.0/Qstore[0]; // load inverse of density
		psq = Qstore[3*devArrayNumel]; // accumulate p^2 and end with px
		px =  Qstore[4*devArrayNumel];
		psq = psq*psq + px*px;
		px = Qstore[2*devArrayNumel];
		psq += px*px;

		b = inputPointers[5][delta];
		bsq = inputPointers[6][delta];
		bsq = bsq*bsq + b*b;
		b = inputPointers[7][delta];
		bsq = bsq + b*b;

		b = Qstore[devArrayNumel] - .5*psq*invrho;

		// Store pressure
		Qstore[5*devArrayNumel] = FLUID_GM1 *b + MHD_PRESS_B*bsq;

		// Find the maximal fast wavespeed
		locCf = fabs(px)*invrho + sqrt((FLUID_GG1*b + MHD_CS_B*bsq)*invrho);

		cmax = (locCf > cmax) ? locCf : cmax; // As we hop down the X direction, track the max C_f encountered
		Qstore += 64;
		delta += 64;
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
	x = (threadIdx.y + FREEZE_NY * blockIdx.x) + ny*(blockIdx.y);

	Cfreeze[x] = cmax;
}

