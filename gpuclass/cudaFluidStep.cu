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

void __syncthreads(void);

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

pParallelTopology topoStructureToC(const mxArray *prhs);
void cfSync(double *cfArray, int cfNumel, const mxArray *topo);

__global__ void cukern_Wstep_mhd_uniform(double *P, double *Cfreeze, double *Qout, double lambdaqtr, int nx);
__global__ void cukern_TVDStep_mhd_uniform(double *P, double *Cfreeze, double *Qin, double halflambda, int nx);

__global__ void replicateFreezeArray(double *freezeIn, double *freezeOut, int ncopies, int ny, int nz);
__global__ void reduceFreezeArray(double *freezeClone, double *freeze, int nx, int ny, int nz);

template <unsigned int PCswitch>
__global__ void cukern_AUSM_step(double *Qstore, double lambda, int nx, int ny);

#define BLOCKLEN 92
#define BLOCKLENP2 94
#define BLOCKLENP4 96

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
	blocksize.x = BLOCKLENP4; blocksize.y = blocksize.z = 1;
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
		cfSync(srcs[9], amd.dim[1]*amd.dim[2], prhs[13]);
		CHECK_CUDA_ERROR("In cudaFluidStep: first hd c_f sync");
		replicateFreezeArray<<<repGrid, repBlock>>>(srcs[9], freezeClone, gridsize.x, amd.dim[1], amd.dim[2]);
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, -666, "In cudaFluidStep: Freeze array cloning");

		// Copies [rho, px, py, pz, E, bx, by, bz, P] array pointers to inputPointers
		cudaMemcpyToSymbol(inputPointers,  srcs, 9*sizeof(double *), 0, cudaMemcpyHostToDevice);

		//cukern_Wstep_hydro_uniform<<<gridsize, blocksize>>>(srcs[8], wStepValues, .5*lambda, arraySize.x, arraySize.y);
		blocksize.y = 2;
		cukern_AUSM_step<RK_PREDICT><<<gridsize, blocksize>>>(wStepValues, .5*lambda, arraySize.x, arraySize.y);
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, hydroOnly, "In cudaFluidStep: hydro W step");

	/* This copies the Wstep_hydro_uniform values to the output arrays. Use to disable 2nd order step for testing */
/*cudaMemcpy(srcs[0], wStepValues,              arrayNumel*sizeof(double), cudaMemcpyDeviceToDevice);
cudaMemcpy(srcs[1], wStepValues+  arrayNumel, arrayNumel*sizeof(double), cudaMemcpyDeviceToDevice);
cudaMemcpy(srcs[2], wStepValues+2*arrayNumel, arrayNumel*sizeof(double), cudaMemcpyDeviceToDevice);
cudaMemcpy(srcs[3], wStepValues+3*arrayNumel, arrayNumel*sizeof(double), cudaMemcpyDeviceToDevice);
cudaMemcpy(srcs[4], wStepValues+4*arrayNumel, arrayNumel*sizeof(double), cudaMemcpyDeviceToDevice);
cudaMemcpy(srcs[8], wStepValues+5*arrayNumel, arrayNumel*sizeof(double), cudaMemcpyDeviceToDevice); */

		//reduceFreezeArray<<<reduceGrid, 16>>>(freezeClone, srcs[9], gridsize.x, amd.dim[1], amd.dim[2]);

		//CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, hydroOnly, "In cudaFluidStep: Freeze array reduce");

		//cfSync(srcs[9], amd.dim[1]*amd.dim[2], prhs[13]);
		//CHECK_CUDA_ERROR("In cudaFluidStep: second hd c_f sync");

		 /*This dumps the input/output arrays to the "half-step" arrays; Use to test TVD step */
		/*cudaMemcpy(wStepValues,              srcs[0], arrayNumel*sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(wStepValues+  arrayNumel, srcs[1], arrayNumel*sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(wStepValues+2*arrayNumel, srcs[2], arrayNumel*sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(wStepValues+3*arrayNumel, srcs[3], arrayNumel*sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(wStepValues+4*arrayNumel, srcs[4], arrayNumel*sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(wStepValues+5*arrayNumel, srcs[8], arrayNumel*sizeof(double), cudaMemcpyDeviceToDevice);*/

		blocksize.y = 2;
		cukern_AUSM_step<RK_CORRECT><<<gridsize, blocksize>>>(wStepValues, lambda, arraySize.x, arraySize.y);
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

#define BLK0 (tix+1               )
#define BLK1 (tix+1+  (BLOCKLENP4))
#define BLK2 (tix+1+2*(BLOCKLENP4))
#define BLK3 (tix+1+3*(BLOCKLENP4))
#define BLK4 (tix+1+4*(BLOCKLENP4))
#define BLK5 (tix+1+5*(BLOCKLENP4))
#define BLK6 (tix+1+6*(BLOCKLENP4))
#define BLK7 (tix+1+7*(BLOCKLENP4))

/* cukern_AUSM_predict reads inputPointers[0 1 2 3 4 9][x] to read [rho E px py pz P]
 * It them performs MUSCL reconstruction of left/right edge states to acheive 2nd order accuracy
 * It then uses AUSM to calculate fluxes and stores Q - lambda*diff(F) -> Qout */

/* if PCswitch is RK_PREDICT (0), derivs are found using inputPointers[n][x0] and Qstore = i.p.'s + lambda qdot
 * if PCswitch is RK_CORRECT (1), derivs are calculated using Qstore and inputPointers[n][x0] updated according to 2nd
 * stage of Runge-Kutta
 */


template <unsigned int PCswitch>
__global__ void cukern_AUSM_step(double *Qstore, double lambda, int nx, int ny)
{
	int tix = threadIdx.x;
	/* Declare variable arrays */
	__shared__ double shblk[8*BLOCKLENP4+2];
	double Ale, Ble, Cle;
	double Are, Bre, Cre;

	double Fa, Fb;
	double Csle, Csre, Mleft, Mright;
	double Pleft, Pright;

	/* My x index: thread + blocksize block, wrapped circularly */
	//int thisThreadPonders  = (threadIdx.x > 0) && (threadIdx.x < blockDim.x-1);
	int thisThreadDelivers = (threadIdx.x > 1) && (threadIdx.x < blockDim.x-2);

	int x0 = threadIdx.x + (BLOCKLEN)*blockIdx.x - 2;
	if(x0 < 0) x0 += nx; // left wraps to right edge
	if(x0 > (nx+1)) return; // More than 2 past right returns
	if(x0 > (nx-1)) { x0 -= nx; thisThreadDelivers = 0; } // past right must wrap around to left

	/* Do some index calculations */
	x0 += nx*ny*blockIdx.y; /* This block is now positioned to start at its given (x,z) coordinate */
	int j = 0;
	for(j = 0; j < ny; j++) {

		if(threadIdx.y == 0) { /* Cord 0 */

			/*************** BEGIN SECTION 1 */
			if(PCswitch == RK_PREDICT) {
				Ale = inputPointers[0][x0]; /* load rho */
				Bre = inputPointers[2][x0]; /* load px */
				Cle = inputPointers[8][x0]; /* load pressure */
			} else {
				Ale = Qstore[x0 + 0*devArrayNumel]; /* load rho */
				Bre = Qstore[x0 + 2*devArrayNumel]; /* load px */
				Cle = Qstore[x0 + 5*devArrayNumel]; /* load pressure */
			}
			Ble = Bre / Ale; /* Calculate vx */
			shblk[BLK0] = Ale;
			shblk[BLK1] = Ble;
			shblk[BLK2] = Cle;
			__syncthreads();

			/*************** BEGIN SECTION 2 */
			Are = Ale - shblk[BLK0 - 1];
			Bre = Ble - shblk[BLK1 - 1];
			Cre = Cle - shblk[BLK2 - 1];
			__syncthreads();

			/*************** BEGIN SECTION 3 */
			shblk[BLK0] = Are;
			shblk[BLK1] = Bre;
			shblk[BLK2] = Cre;
			__syncthreads();

			/*************** BEGIN SECTION 4 */
			Fa = SLOPEFUNC(Are, shblk[BLK0+1]);
			Are = Ale + Fa;
			Ale -= Fa;
			Fa = SLOPEFUNC(Bre, shblk[BLK1+1]);
			Bre = Ble + Fa;
			Ble -= Fa;
			Fa = SLOPEFUNC(Cre, shblk[BLK2+1]);
			Cre = Cle + Fa;
			Cle -= Fa;
			__syncthreads();

			/*************** BEGIN SECTION 5 */
			shblk[BLK4] = Csle = sqrt(FLUID_GAMMA*Cle/Ale);
			shblk[BLK5] = Csre = sqrt(FLUID_GAMMA*Cre/Are);

			Mleft  = Ble / Csle;
			Mright = Bre / Csre;
			Fa = fabs(Mleft);
			Fb = fabs(Mright);

			if(Mright*Mright < 1.0) {
				Pright = .5*(1+Mright)*Cre;
				Mright = .25*(Mright+1)*(Mright+1);
			} else {
				Pright= .5*(Mright+Fb)*Cre/Mright;
				Mright = .5*(Mright+Fb);

			}

			if(Mleft*Mleft < 1.0) {
				Pleft = .5*(1-Mleft)*Cle;
				Mleft = -.25*(Mleft-1)*(Mleft-1);
			} else {
				Pleft = .5*(Mleft-Fa)*Cle/Mleft;
				Mleft= .5*(Mleft-Fa);
			}

			shblk[BLK2] = Mleft;
			shblk[BLK3] = Mright;

			__syncthreads();

			/*************** BEGIN SECTION 6 */
			shblk[BLK0] = FLUID_GOVERGM1*Cle + .5*Ale*Ble*Ble; /* export our part of the energy flux for read by cord 1*/
			shblk[BLK1] = FLUID_GOVERGM1*Cre + .5*Are*Bre*Bre;
			shblk[BLK6] = Ale;
			shblk[BLK7] = Are; /* we don't give a crap about efficiency until this WORKS */
			Mleft  += shblk[BLK3-1];
			Mright += shblk[BLK2+1];
			__syncthreads();

			/*************** BEGIN SECTION 7 */

			shblk[BLK4] = Ale*Csle;
			shblk[BLK5] = Are*Csre;
			__syncthreads();

			/*************** BEGIN SECTION 8 */
			if(Mright > 0) { Fb = shblk[BLK5]; } else { Fb = shblk[BLK4+1]; }
			if(Mleft  > 0) { Fa = shblk[BLK5-1]; } else { Fa = shblk[BLK4]; }
			if(thisThreadDelivers) {/* Density update */
				if(PCswitch == RK_PREDICT) {
					Qstore[x0] = Cle = .5*(Ale+Are) + lambda * (Mleft*Fa - Mright*Fb);
				} else {
					inputPointers[0][x0] = inputPointers[0][x0] + lambda* (Mleft*Fa - Mright*Fb);
				}
			}

			Fa = Ale * Csle * Ble;
			Fb = Are * Csre * Bre;
			__syncthreads();

			/*************** BEGIN SECTION 9*/
			shblk[BLK2] = Fa;
			shblk[BLK3] = Fb;
			shblk[BLK4] = Pleft;
			shblk[BLK5] = Pright;
			__syncthreads();

			/*************** BEGIN SECTION 10 */
			if(Mright > 0) { Fb = shblk[BLK3]; } else { Fb = shblk[BLK2+1]; }
			if(Mleft  > 0) { Fa = shblk[BLK3-1]; } else { Fa = shblk[BLK2]; }
			if(thisThreadDelivers) { /* X momentum update */
				if(PCswitch == RK_PREDICT) {
					Qstore[x0 + 2*devArrayNumel] = shblk[BLK7] = .25*(Ale+Are)*(Ble+Bre) + lambda * (Mleft*Fa - Mright*Fb + Pleft - Pright - shblk[BLK4+1] + shblk[BLK5-1]);
					shblk[BLK6] = Cle;
				} else {
					inputPointers[2][x0] = inputPointers[2][x0] + lambda* (Mleft*Fa - Mright*Fb + Pleft - Pright - shblk[BLK4+1] + shblk[BLK5-1]);
				}
			}

			/*************** BEGIN SECTION 11 */
			if(PCswitch == RK_PREDICT) __syncthreads();

		} else { /* Cord 1 */
			/*************** BEGIN SECTION 1 */
			if(PCswitch == RK_PREDICT) {
				Ale = inputPointers[1][x0];  // energy
				Ble = inputPointers[3][x0];// mom y
				Cle = inputPointers[4][x0];// mom z
			} else {
				Ale = Qstore[x0 + devArrayNumel];  // energy
				Ble = Qstore[x0 + 3*devArrayNumel];// mom y
				Cle = Qstore[x0 + 4*devArrayNumel];// mom z
			}
			shblk[BLK3] = Ale;
			shblk[BLK4] = Ble;
			shblk[BLK5] = Cle;
			__syncthreads();

			/*************** BEGIN SECTION 2 */
			Are = Ale - shblk[BLK3-1];
			Bre = Ble - shblk[BLK4-1];
			Cre = Cle - shblk[BLK5-1];
			__syncthreads();

			/*************** BEGIN SECTION 3 */
			shblk[BLK3] = Are;
			shblk[BLK4] = Bre;
			shblk[BLK5] = Cre;
			__syncthreads();

			/*************** BEGIN SECTION 4 */
			Fa = SLOPEFUNC(shblk[BLK3], shblk[BLK3+1]);
			Are = Ale + Fa;
			Ale -= Fa;

			Fa = SLOPEFUNC(shblk[BLK4], shblk[BLK4+1]);
			Bre = Ble + Fa;
			Ble -= Fa;

			Fa = SLOPEFUNC(shblk[BLK5], shblk[BLK5+1]);
			Cre = Cle + Fa;
			Cle -= Fa;
			__syncthreads();

			/*************** BEGIN SECTION 5 */

			__syncthreads();

			/*************** BEGIN SECTION 6 */
			Mleft  = shblk[BLK2] + shblk[BLK3-1];
			Mright = shblk[BLK3] + shblk[BLK2+1];
			Csle = shblk[BLK4];
			Csre = shblk[BLK5];
			__syncthreads();

			/*************** BEGIN SECTION 7 */
			shblk[BLK0] = Csle * (shblk[BLK0] + .5*(Ble*Ble+Cle*Cle)/shblk[BLK6]); /* Publicize energy fluxes */
			shblk[BLK1] = Csre * (shblk[BLK1] + .5*(Bre*Bre+Cre*Cre)/shblk[BLK7]);
			shblk[BLK6] = Cle*Csle; /* pz momentum fluxes */
			shblk[BLK7] = Cre*Csre;
			__syncthreads();

			/*************** BEGIN SECTION 8 */
			if(thisThreadDelivers) {
				if(Mleft > 0) { Fa = shblk[BLK1 - 1]; } else { Fa = shblk[BLK0]; } // Energy update
				if(Mright > 0){ Fb = shblk[BLK1]; } else { Fb = shblk[BLK0 + 1]; }
				if(PCswitch == RK_PREDICT) {
					Qstore[x0 + 1*devArrayNumel] = Ale = .5*(Ale+Are) + lambda * (Mleft*Fa - Mright*Fb);
					/* STORED E_half in Ale! */
				} else {
					inputPointers[1][x0] = inputPointers[1][x0] + lambda * (Mleft*Fa - Mright*Fb);
				}
				if(Mleft > 0) { Fa = shblk[BLK7 - 1]; } else { Fa = shblk[BLK6]; } // Pz update
				if(Mright > 0){ Fb = shblk[BLK7]; } else { Fb = shblk[BLK6 + 1]; }
				if(PCswitch == RK_PREDICT) {
					Qstore[x0 + 4*devArrayNumel] = Are = .5*(Cle+Cre) + lambda * (Mleft*Fa - Mright*Fb);
					Are *= .5*Are;
				} else {
					inputPointers[4][x0] = inputPointers[4][x0] + lambda * (Mleft*Fa - Mright*Fb);
				}
			}

			__syncthreads();

			/*************** BEGIN SECTION 9 */
			shblk[BLK0] = Ble*Csle;
			shblk[BLK1] = Bre*Csre;

			__syncthreads();

			/*************** BEGIN SECTION 10 */
			if(thisThreadDelivers) {
							if(Mleft > 0) { Fa = shblk[BLK1 - 1]; } else { Fa = shblk[BLK0]; } // Py update
							if(Mright > 0){ Fb = shblk[BLK1]; } else { Fb = shblk[BLK0 + 1]; }
							if(PCswitch == RK_PREDICT) {
								Qstore[x0 + 3*devArrayNumel] = Cre = .5*(Ble+Bre) + lambda * (Mleft*Fa - Mright*Fb);
								Are += .5*Cre*Cre; /* Store py^2+pz^2 */
							} else {
								inputPointers[3][x0] = inputPointers[3][x0] + .5*lambda * (Mleft*Fa - Mright*Fb);
							}

						}

			/***************** BEGIN SECTION 11: Now cord 1 must calculate EoS if we are in predictor step*/
			if(PCswitch == RK_PREDICT) {
				__syncthreads();
				//P = (g-1)*(Etot - T)
			    if(thisThreadDelivers) Qstore[x0 + 5*devArrayNumel] = FLUID_GM1 * (Ale- (.5*shblk[BLK7]*shblk[BLK7] + Are)/shblk[BLK6]);
			}
		}
		x0 += nx;
		__syncthreads();

	}
}

__global__ void cukern_TVDStep_mhd_uniform(double *P, double *Cfreeze, double *Qin, double halfLambda, int nx)
{

}



__device__ __inline__ double Mminus(double M) {
	if(M > 1.0) return 0.0;
	if(M*M < 1) return -.25*(M-1.0)*(M-1.0);
	return M;
}
__device__ __inline__ double Mplus(double M) {
	if(M < -1.0) return 0.0;
	if(M*M < 1) return .25*(M+1.0)*(M+1.0);
	return M;
}

/* This kernel reads from inputPointers[0...4][x] and writes Qout[(0...4)*numel + x]
 * The Cfreeze array is round_up(NX / blockDim.x) x NY x NZ and is reduced in the X direction after
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
