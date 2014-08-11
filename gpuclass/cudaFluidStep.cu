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

#include "mpi.h"
#include "parallel_halo_arrays.h"
//#include "mpi_common.h"

#define RK_PREDICT 0
#define RK_CORRECT 1

//void __syncthreads(void);

#define FLUID_METHOD_HLLC 1
//#define FLUID_METHOD_HLL 1

/* THIS FUNCTION

This function calculates a first order accurate upwind step of the conserved transport part of the 
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

In general thermal pressure is an arbitrary positive function of e, however the ideal gas
law is built into Imogen in multiple locations and significant re-checking would be needed
if it were to be generalized.

The hydro functions solve the same equations with B set to <0,0,0> which simplifies
and considerably speeds up the process.

*/

//__device__ void __syncthreads(void);

/*__device__ double slopeLimiter_Osher(double A, double B)
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
}*/

pParallelTopology topoStructureToC(const mxArray *prhs);
void cfSync(double *cfArray, int cfNumel, const mxArray *topo);

__global__ void cukern_Wstep_mhd_uniform(double *P, double *Cfreeze, double *Qout, double lambdaqtr, int nx);
__global__ void cukern_TVDStep_mhd_uniform(double *P, double *Cfreeze, double *Qin, double halflambda, int nx);

__global__ void replicateFreezeArray(double *freezeIn, double *freezeOut, int ncopies, int ny, int nz);
__global__ void reduceFreezeArray(double *freezeClone, double *freeze, int nx, int ny, int nz);

__global__ void cukern_AUSM_firstorder_uniform(double *P, double *Qout, double lambdaQtr, int nx, int ny);

template <unsigned int PCswitch>
__global__ void cukern_AUSM_step(double *Qstore, double lambda, int nx, int ny);

template <unsigned int PCswitch>
__global__ void cukern_HLL_step(double *Qstore, double lambda, int nx, int ny);

template <unsigned int PCswitch>
__global__ void cukern_HLLC_step(double *Qstore, double lambda, int nx, int ny);

/* Stopgap until I manage to stuff pressure solvers into all the predictors... */
__global__ void cukern_PressureSolver(double *Qstore);

#define BLOCKLEN 12
#define BLOCKLENP2 14
#define BLOCKLENP4 16

#define YBLOCKS 4

__constant__ __device__ double *inputPointers[9];
__constant__ __device__ double fluidQtys[8];
__constant__ __device__ int devArrayNumel;

#define LIMITERFUNC fluxLimiter_Zero
//#define LIMITERFUNC fluxLimiter_minmod
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
#define FLUID_GOVERGM1 fluidQtys[7]

#define MHD_PRESS_B   fluidQtys[5]
#define MHD_CS_B      fluidQtys[6]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// Input and result
	if ((nrhs!=14) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaFluidStep(rho, E, px, py, pz, bx, by, bz, Ptot, c_f, lambda, purehydro?, fluid gamma)\n");

	CHECK_CUDA_ERROR("entering cudaFluidStep");

	ArrayMetadata amd;
	double **srcs = getGPUSourcePointers(prhs, &amd, 0, 9);

	// Establish launch dimensions & a few other parameters
	int fluxDirection = 1;
	double lambda     = *mxGetPr(prhs[10]);

	dim3 arraySize;
	arraySize.x = amd.dim[0];
	arraySize.y = amd.dim[1];
	arraySize.z = amd.dim[2];

	dim3 blocksize, gridsize;

	// This bit is actually redundant now since arrays are always rotated so the fluid step is finite-differenced in the X direction
	blocksize.x = BLOCKLENP4; blocksize.y = YBLOCKS; blocksize.z = 1;
	switch(fluxDirection) {
	case 1: // X direction flux: x = sup(nx/blocksize.x), y = nz
		gridsize.x = (arraySize.x/BLOCKLEN); gridsize.x += 1*(gridsize.x*BLOCKLEN < arraySize.x);
		gridsize.y = arraySize.z;
		break;
	case 2: // Y direction flux: u = y, v = x, w = z
		gridsize.x = arraySize.x;
		gridsize.y = arraySize.z;
		break;
	case 3: // Z direction flux: u = z, v = x, w = y;
		gridsize.x = arraySize.x;
		gridsize.y = arraySize.y;
		break;
	}
	double *thermo = mxGetPr(prhs[12]);
	double gamma = thermo[0];
	double rhomin= thermo[1];
	double gamHost[8];
	gamHost[0] = gamma;
	gamHost[1] = gamma-1.0;
	gamHost[2] = gamma*(gamma-1.0);
	gamHost[3] = rhomin;
	// assert     cs > cs_min
	//     g P / rho > g rho_min^(g-1)
	// (g-1) e / rho > rho_min^(g-1)
	//             e > rho rho_min^(g-1)/(g-1)
	gamHost[4] = powl(rhomin, gamma-1.0)/(gamma-1.0);
	gamHost[5] = 1.0 - .5*gamma;
	gamHost[6] = ALFVEN_CSQ_FACTOR - .5*(gamma-1.0)*gamma;
	gamHost[7] = gamma/(gamma-1.0); // pressure to energy flux conversion for ideal gas adiabatic EoS

	// Even for gamma=5/3, soundspeed is very weakly dependent on density (cube root) for adiabatic fluid

	cudaMemcpyToSymbol(fluidQtys, &gamHost[0], 8*sizeof(double), 0, cudaMemcpyHostToDevice);

	int arrayNumel =  amd.dim[0]*amd.dim[1]*amd.dim[2];
	double *wStepValues;
	cudaMalloc((void **)&wStepValues,arrayNumel*sizeof(double)*6); // [rho px py pz E P]_.5dt
	CHECK_CUDA_ERROR("In cudaFluidStep: halfstep malloc");
	cudaMemcpyToSymbol(devArrayNumel, &arrayNumel, sizeof(int), 0, cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR("In cudaFluidStep: halfstep devArrayNumel memcpy");

	/* Replicate copies of the freezing speed array for the fluid routines to edit */
	dim3 repBlock; repBlock.x = 8; repBlock.y = 1; repBlock.z = 1;
	dim3 repGrid;
	repGrid.x = amd.dim[1];
	repGrid.y = amd.dim[2];

	dim3 reduceGrid;
	reduceGrid.x = amd.dim[1]; reduceGrid.y = amd.dim[2]; reduceGrid.z = 1;

	double *freezeClone;
	cudaMalloc((void **)&freezeClone, amd.dim[1]*amd.dim[2]*sizeof(double)*gridsize.x);
	CHECK_CUDA_ERROR("In cudaFluidStep: c_freeze replicated array alloc");

	int hydroOnly;
	hydroOnly = (int)*mxGetPr(prhs[11]);

	if(hydroOnly == 1) {
//		cfSync(srcs[9], amd.dim[1]*amd.dim[2], prhs[13]);
		CHECK_CUDA_ERROR("In cudaFluidStep: first hd c_f sync");
//		replicateFreezeArray<<<repGrid, repBlock>>>(srcs[9], freezeClone, gridsize.x, amd.dim[1], amd.dim[2]);
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, -666, "In cudaFluidStep: Freeze array cloning");

		// Copies [rho, px, py, pz, E, bx, by, bz, P] array pointers to inputPointers
		cudaMemcpyToSymbol(inputPointers,  srcs, 9*sizeof(double *), 0, cudaMemcpyHostToDevice);

//		cukern_AUSM_firstorder_uniform<<<gridsize, blocksize>>>(srcs[8], wStepValues, .5*lambda, arraySize.x, arraySize.y);
#ifdef FLUID_METHOD_HLL
		cukern_HLL_step<RK_PREDICT><<<gridsize, blocksize>>>(wStepValues, .5*lambda, arraySize.x, arraySize.y);
#warning NOTE: Using HLL flux method
#endif
#ifdef FLUID_METHOD_HLLC
		cukern_HLLC_step<RK_PREDICT><<<gridsize, blocksize>>>(wStepValues, .5*lambda, arraySize.x, arraySize.y);
#warning NOTE: Using HLLC flux method
#endif
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, hydroOnly, "In cudaFluidStep: hydro prediction step");

		cukern_PressureSolver<<<256, 32>>>(wStepValues);
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, hydroOnly, "In cudaFluidStep: hydro pressure solver");

/*
// This copies the first-order step's result to the output arrays
// Use to test first-order kernels
int NB = arrayNumel*sizeof(double);
cudaMemcpy(srcs[0], wStepValues,              NB, cudaMemcpyDeviceToDevice);
cudaMemcpy(srcs[1], wStepValues+  arrayNumel, NB, cudaMemcpyDeviceToDevice);
cudaMemcpy(srcs[2], wStepValues+2*arrayNumel, NB, cudaMemcpyDeviceToDevice);
cudaMemcpy(srcs[3], wStepValues+3*arrayNumel, NB, cudaMemcpyDeviceToDevice);
cudaMemcpy(srcs[4], wStepValues+4*arrayNumel, NB, cudaMemcpyDeviceToDevice);
cudaMemcpy(srcs[8], wStepValues+5*arrayNumel, NB, cudaMemcpyDeviceToDevice); */

		//
//		cudaMemcpy(srcs[8], wStepValues+5*arrayNumel, arrayNumel*sizeof(double), cudaMemcpyDeviceToDevice);
		//reduceFreezeArray<<<reduceGrid, 16>>>(freezeClone, srcs[9], gridsize.x, amd.dim[1], amd.dim[2]);

		//CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, hydroOnly, "In cudaFluidStep: Freeze array reduce");

		//cfSync(srcs[9], amd.dim[1]*amd.dim[2], prhs[13]);
		//CHECK_CUDA_ERROR("In cudaFluidStep: second hd c_f sync");

 /*This dumps the input/output arrays to the "half-step" arrays; Use to test TVD step */
/*cudaMemcpy(wStepValues,              srcs[0], NB, cudaMemcpyDeviceToDevice);
cudaMemcpy(wStepValues+  arrayNumel, srcs[1], NB, cudaMemcpyDeviceToDevice);
cudaMemcpy(wStepValues+2*arrayNumel, srcs[2], NB, cudaMemcpyDeviceToDevice);
cudaMemcpy(wStepValues+3*arrayNumel, srcs[3], NB, cudaMemcpyDeviceToDevice);
cudaMemcpy(wStepValues+4*arrayNumel, srcs[4], NB, cudaMemcpyDeviceToDevice);
cudaMemcpy(wStepValues+5*arrayNumel, srcs[8], NB, cudaMemcpyDeviceToDevice);*/

#ifdef FLUID_METHOD_HLL
		cukern_HLL_step<RK_CORRECT><<<gridsize, blocksize>>>(wStepValues, lambda, arraySize.x, arraySize.y);
#warning NOTE: Using HLL flux method
#endif
#ifdef FLUID_METHOD_HLLC
		cukern_HLLC_step<RK_CORRECT><<<gridsize, blocksize>>>(wStepValues, lambda, arraySize.x, arraySize.y);
#warning NOTE: Using HLLC flux method
#endif
//		cukern_AUSM_step<RK_CORRECT><<<gridsize, blocksize>>>(wStepValues, lambda, arraySize.x, arraySize.y);
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, hydroOnly, "In cudaFluidStep: hydro TVD step");
	} else {
		cfSync(srcs[9], amd.dim[1]*amd.dim[2], prhs[13]);
		CHECK_CUDA_ERROR("In cudaFluidStep: first mhd c_f sync");
		cudaMemcpyToSymbol(inputPointers,  srcs, 8*sizeof(double *), 0, cudaMemcpyHostToDevice);

		cukern_Wstep_mhd_uniform<<<gridsize, blocksize>>>(srcs[8], srcs[9], wStepValues, .25*lambda, arraySize.x);
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, hydroOnly, "In cudaFluidStep: mhd W step");

		cfSync(srcs[9], amd.dim[1]*amd.dim[2], prhs[13]);
		CHECK_CUDA_ERROR("In cudaFluidStep: second mhd c_f sync");
		cukern_TVDStep_mhd_uniform  <<<gridsize, blocksize>>>(wStepValues + 5*arrayNumel, srcs[9], wStepValues, .5*lambda, arraySize.x);
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, hydroOnly, "In cudaFluidStep: mhd TVD step");
	}

cudaFree(freezeClone);
cudaFree(wStepValues);


}

void cfSync(double *cfArray, int cfNumel, const mxArray *topo)
{
pParallelTopology topology = topoStructureToC(topo);

/* Reversed for silly Fortran memory ordering */
int d = 0;
int dmax = topology->nproc[d];

MPI_Comm commune = MPI_Comm_f2c(topology->comm);
int r0; MPI_Comm_rank(commune, &r0);

double *storeA = (double*)malloc(sizeof(double)*cfNumel);
double *storeB = (double*)malloc(sizeof(double)*cfNumel);

cudaMemcpy(storeA, cfArray, cfNumel*sizeof(double), cudaMemcpyDeviceToHost);

MPI_Status amigoingtodie;

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

pParallelTopology topoStructureToC(const mxArray *prhs)
{
mxArray *a;

pParallelTopology pt = (pParallelTopology)malloc(sizeof(ParallelTopology));

a = mxGetFieldByNumber(prhs,0,0);
pt->ndim = (int)*mxGetPr(a);
a = mxGetFieldByNumber(prhs,0,1);
pt->comm = (int)*mxGetPr(a);

int *val;
int i;

val = (int *)mxGetData(mxGetFieldByNumber(prhs,0,2));
for(i = 0; i < pt->ndim; i++) pt->coord[i] = val[i];

val = (int *)mxGetData(mxGetFieldByNumber(prhs,0,3));
for(i = 0; i < pt->ndim; i++) pt->neighbor_left[i] = val[i];

val = (int *)mxGetData(mxGetFieldByNumber(prhs,0,4));
for(i = 0; i < pt->ndim; i++) pt->neighbor_right[i] = val[i];

val = (int *)mxGetData(mxGetFieldByNumber(prhs,0,5));
for(i = 0; i < pt->ndim; i++) pt->nproc[i] = val[i];

for(i = pt->ndim; i < 4; i++) {
  pt->coord[i] = 0;
  pt->nproc[i] = 1;
  }

return pt;

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

__global__ void cukern_Wstep_mhd_uniform(double *P, double *Cfreeze, double *Qout, double lambdaqtr, int nx)
{

}

// These tell the HLL and HLLC solvers how to dereference their shmem block
// 16 threads, 8 circshifted shmem arrays
#define N_SHMEM_BLOCKS 8
#define BOS0 0
#define BOS1 16
#define BOS2 32
#define BOS3 48
#define BOS4 64
#define BOS5 80
#define BOS6 96
#define BOS7 112

#define HLL_LEFT 0
#define HLL_HLL  1
#define HLL_RIGHT 2

template <unsigned int PCswitch>
__global__ void cukern_HLLC_step(double *Qstore, double lambda, int nx, int ny)
{
	// Create center, look-left and look-right indexes
	int IC = threadIdx.x;
	int IL = threadIdx.x - 1;
	IL += 1*(IL < 0);
	int IR = threadIdx.x + 1;
	IR -= 16*(IL > 15);

	// Advance by the Y index
	IC += N_SHMEM_BLOCKS*16*threadIdx.y;
	IL += N_SHMEM_BLOCKS*16*threadIdx.y;
	IR += N_SHMEM_BLOCKS*16*threadIdx.y;

	/* Declare shared variable array */
	__shared__ double shblk[YBLOCKS*N_SHMEM_BLOCKS*BLOCKLENP4];

	/* Declare tons of doubles and hope optimizer can sort this out */
	double Ale, Ble, Cle;
	double Are, Bre, Cre;
	double Sleft, Sright, Utilde, Atilde;
	double Fa, Fb; /* temp vars */
        double Beta, Sstar;

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

	for(; j < ny; j += blockDim.y) {

		if(PCswitch == RK_PREDICT) {
			/* If making prediction use simple 0th order "reconstruction." */
			Ale = Are = inputPointers[0][x0]; /* load rho */
			Bre = inputPointers[2][x0]; /* load px */
			Cle = Cre = inputPointers[8][x0]; /* load pressure */
			Ble = Bre / Ale; /* Calculate vx */
			Bre = Ble;
		} else {
			/* If making correction, perform 1st order MUSCL reconstruction */
			Ale = Qstore[x0 + 0*devArrayNumel]; /* load rho */
			Bre = Qstore[x0 + 2*devArrayNumel]; /* load px */
			Cle = Qstore[x0 + 5*devArrayNumel]; /* load pressure */
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

		Sstar = (Cre - Cle + Ale*Ble*(Sleft-Ble) - Are*Bre*(Sright-Bre))/(Ale*(Sleft-Ble)-Are*(Sright-Bre));

		shblk[IC + BOS6] = Ale*Ble*Ble; // Accumulate kinetic energy density to use in energy fluxing
		shblk[IC + BOS7] = Are*Bre*Bre;

		if(Sstar > 0) { // Either left approximate wave or supersonic rightbound upwind
			if(Sleft < 0) { // approximate left wave
				Beta = (Sleft - Ble)/(Sleft-Sstar);
				shblk[IC + BOS1] = Ale * (Ble + Sleft*(Beta - 1.0));
				shblk[IC + BOS3] = Ale * (Ble * (Ble - Sleft) + Sleft * Sstar * Beta) + Cle;
			} else { // supersonic rightbound advection
				shblk[IC + BOS1] = Ale*Ble;
				shblk[IC + BOS3] = Ale*Ble*Ble + Cle;
			}
		} else { // Either rightgoing approximate wave or supersonic leftbound upwind
			if(Sright > 0) { // approximate right wave
				Beta = (Sright - Bre)/(Sright-Sstar);
				shblk[IC + BOS1] = Are * (Bre + Sright*(Beta - 1.0));
				shblk[IC + BOS3] = Are * (Bre * (Bre - Sright) + Sright*Sstar*Beta) + Cre;
			} else { // supersonic leftbound advection
				shblk[IC + BOS1] = Are*Bre;
				shblk[IC + BOS3] = Are*Bre*Bre + Cre;
			}
		}

		/* Transfer Atilde to shmem, freeing Utilde/Atilde as Ule/Ure pair */
		shblk[IC + BOS5] = Atilde;

		__syncthreads();

		// Calculate conservative differences for mass and x momentum fluxes
		shblk[IC + BOS2] = (shblk[IL + BOS1]-shblk[IC + BOS1]);
		shblk[IC + BOS4] = (shblk[IL + BOS3]-shblk[IC + BOS3]);

		/* Flux density and momentum... for prediction we explicitly did not use MUSCL and
                   therefore Ale = Acentered. For correction, we load/modify/store the centered value */
		if(thisThreadDelivers) {
			if(PCswitch == RK_PREDICT) {
				Qstore[x0                  ] = Ale + lambda * shblk[IC + BOS2];
				Qstore[x0 + 2*devArrayNumel] = Ale*Ble + lambda * shblk[IC + BOS4];
			} else {
				inputPointers[0][x0] += lambda * shblk[IC + BOS2];
				inputPointers[2][x0] += lambda * shblk[IC + BOS4];
			}
		}

		__syncthreads();
// 55 registers

		// Now go back and do the whole thing again for y momentum, z momentum and energy
		if(PCswitch == RK_PREDICT) {
			/* If making prediction use simple 0th order "reconstruction." */
			Fa =  inputPointers[3][x0]; /* load py */
			Utilde = inputPointers[4][x0]; /* load pz */
		} else {
			/* If making correction, perform 1st order MUSCL reconstruction */
			Fa = Qstore[x0 + 3*devArrayNumel]; /* load py */
			Utilde = Qstore[x0 + 4*devArrayNumel]; /* load pz */

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

// 55 registers

		// Compute the kinetic energy density, T
		shblk[IC + BOS6] = .5*(shblk[IC + BOS6] + (Fa*Fa + Utilde*Utilde)/Ale);
		shblk[IC + BOS7] = .5*(shblk[IC + BOS7] + (Fb*Fb + Atilde*Atilde)/Are);
// 57 registers

		/* Note that energy flux is computed without ever loading the E array */
		/* Either E or P provides the necessary state and consistency requires */
		/* Use of exactly one of them */
		if(Sstar > 0) {
			if(Sleft < 0) {
				shblk[IC + BOS2] = Fa * (Ble + Sleft*(Beta - 1.0));
				shblk[IC + BOS3] = Utilde * (Ble + Sleft*(Beta- 1.0));
				shblk[IC + BOS4] = (Ble + Sleft*(Beta - 1.0))*shblk[IC + BOS6] + \
						   (Ble * FLUID_GOVERGM1 + Sleft*( (Beta-1.0)/FLUID_GM1 + (Sstar - Ble)/(Sleft-Sstar)))*Cle + \
						   Sleft*(Sstar - Ble)*Beta*Ale*Sstar;
			} else {
				shblk[IC + BOS2] = Fa * Ble;
				shblk[IC + BOS3] = Utilde * Ble;
				shblk[IC + BOS4] = Ble * (shblk[IC + BOS6] + FLUID_GOVERGM1*Cle);
			}
		} else {
			if(Sright > 0) {
				shblk[IC + BOS2] = Fb * (Bre + Sright*(Beta - 1.0));
				shblk[IC + BOS3] = Atilde * (Bre + Sright*(Beta - 1.0));
				shblk[IC + BOS4] = (Bre + Sright*(Beta - 1.0))*shblk[IC + BOS7] + \
                                                   (Bre * FLUID_GOVERGM1 + Sright*( (Beta-1.0)/FLUID_GM1 + (Sstar - Bre)/(Sright-Sstar)))*Cre + \
						   Sright*(Sstar - Bre)*Beta*Are*Sstar;

			} else {
				shblk[IC + BOS2] = Fb * Bre;
				shblk[IC + BOS3] = Atilde * Bre;
				shblk[IC + BOS4] = Bre * (shblk[IC + BOS7] + FLUID_GOVERGM1*Cre);
			}
		}

// 63 registers
		__syncthreads(); /* shmem 2: py flux, shmem3: pz flux, shmem 4: E flux */

		if(thisThreadDelivers) {
			if(PCswitch == RK_PREDICT) {
				Qstore[x0 +   devArrayNumel] = inputPointers[1][x0] + lambda*(shblk[IL + BOS4]-shblk[IC + BOS4]);
				Qstore[x0 + 3*devArrayNumel] = Fa                   + lambda*(shblk[IL + BOS2]-shblk[IC + BOS2]);
				Qstore[x0 + 4*devArrayNumel] = Utilde               + lambda*(shblk[IL + BOS3]-shblk[IC + BOS3]);
			} else {
				inputPointers[1][x0] += lambda*(shblk[IL + BOS4]-shblk[IC + BOS4]);
				inputPointers[3][x0] += lambda*(shblk[IL + BOS2]-shblk[IC + BOS2]);
				inputPointers[4][x0] += lambda*(shblk[IL + BOS3]-shblk[IC + BOS3]);
			}
		}
		x0 += blockDim.y*nx;
		__syncthreads();


	}

}

template <unsigned int PCswitch>
__global__ void cukern_HLL_step(double *Qstore, double lambda, int nx, int ny)
{
	// Create center, rotate-left and rotate-right indexes
	int IC = threadIdx.x;
	int IL = threadIdx.x - 1;
	IL += 1*(IL < 0);
	int IR = threadIdx.x + 1;
	IR -= 16*(IL > 15);

	IC += N_SHMEM_BLOCKS*16*threadIdx.y;
	IL += N_SHMEM_BLOCKS*16*threadIdx.y;
	IR += N_SHMEM_BLOCKS*16*threadIdx.y;

	/* Declare variable arrays */
	__shared__ double shblk[YBLOCKS*N_SHMEM_BLOCKS*BLOCKLENP4];
	double Ale, Ble, Cle;
	double Are, Bre, Cre;
	int HLL_FluxMode;
	double Sleft, Sright, Utilde, Atilde;
	double Fa, Fb; /* temp vars */

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

	for(; j < ny; j += blockDim.y) {

		if(PCswitch == RK_PREDICT) {
			/* If making prediction use simple 0th order "reconstruction." */
			Ale = Are = inputPointers[0][x0]; /* load rho */
			Bre = inputPointers[2][x0]; /* load px */
			Cle = Cre = inputPointers[8][x0]; /* load pressure */
			Ble = Bre / Ale; /* Calculate vx */
			Bre = Ble;
		} else {
			/* If making correction, perform linear MUSCL reconstruction */
			Ale = Qstore[x0 + 0*devArrayNumel]; /* load rho */
			Bre = Qstore[x0 + 2*devArrayNumel]; /* load px */
			Cle = Qstore[x0 + 5*devArrayNumel]; /* load pressure */
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
				Qstore[x0                  ] = Ale + lambda * shblk[IC + BOS2];
				Qstore[x0 + 2*devArrayNumel] = Fa  + lambda * shblk[IC + BOS4];
			} else {
				inputPointers[0][x0] += lambda * shblk[IC + BOS2];
				inputPointers[2][x0] += lambda * shblk[IC + BOS4];
			}
		}

		__syncthreads();
// 55 registers
		if(PCswitch == RK_PREDICT) {
			/* If making prediction use simple 0th order "reconstruction." */
			Fa =  inputPointers[3][x0]; /* load py */
			Utilde = inputPointers[4][x0]; /* load pz */
		} else {
			/* If making correction, perform 1st order MUSCL reconstruction */
			Fa = Qstore[x0 + 3*devArrayNumel]; /* load py */
			Utilde = Qstore[x0 + 4*devArrayNumel]; /* load pz */

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

// 55 registers

		shblk[IC + BOS6] = .5*(shblk[IC + BOS6] + (Fa*Fa + Utilde*Utilde)/Ale);
		shblk[IC + BOS7] = .5*(shblk[IC + BOS7] + (Fb*Fb + Atilde*Atilde)/Are);
// 57 registers
#if 1
		// FATAL: This code section causes register utilization to jump from 57 to 63.
		// FUCK THAT NOISE, this must be stopped ;-(
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
// 63 registers
		__syncthreads(); /* shmem 2: py flux, shmem3: pz flux, shmem 4: E flux */

		if(thisThreadDelivers) {
			if(PCswitch == RK_PREDICT) {
				Qstore[x0 +   devArrayNumel] = inputPointers[1][x0] + lambda*(shblk[IL + BOS4]-shblk[IC + BOS4]);
				Qstore[x0 + 3*devArrayNumel] = Fa                   + lambda*(shblk[IL + BOS2]-shblk[IC + BOS2]);
				Qstore[x0 + 4*devArrayNumel] = Utilde               + lambda*(shblk[IL + BOS3]-shblk[IC + BOS3]);
			} else {
				inputPointers[1][x0] += lambda*(shblk[IL + BOS4]-shblk[IC + BOS4]);
				inputPointers[3][x0] += lambda*(shblk[IL + BOS2]-shblk[IC + BOS2]);
				inputPointers[4][x0] += lambda*(shblk[IL + BOS3]-shblk[IC + BOS3]);
			}
		}
#endif
		x0 += blockDim.y*nx;
		__syncthreads();


	}

}

__global__ void cukern_TVDStep_mhd_uniform(double *P, double *Cfreeze, double *Qin, double halfLambda, int nx)
{

}



 /* The Cfreeze array is round_up(NX / blockDim.x) x NY x NZ and is reduced in the X direction after
 * blockdim = [ xsize, 1, 1]
 * griddim  = [ sup[nx/(xsize - 4)], ny, 1]
 */
__global__ void cukern_AUSM_firstorder_uniform(double *P, double *Qout, double lambdaQtr, int nx, int ny)
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
__global__ void cukern_PressureSolver(double *Qstore)
{
int x = threadIdx.x + blockDim.x*blockIdx.x;

double rho, E, z, momsq, P;

int hx = blockDim.x*gridDim.x;
int DAN = devArrayNumel;

while(x < DAN) {
	rho = Qstore[x      ];
	E   = Qstore[x + DAN];
	z   = Qstore[x+2*DAN];
	momsq = z*z;
	z   = Qstore[x+3*DAN];
	momsq += z*z;
	z   = Qstore[x+4*DAN];
	momsq += z*z;
	P = FLUID_GM1 * (E - .5*momsq/rho);
	Qstore[x + 5*DAN] = P;
	x += hx;
}

}
