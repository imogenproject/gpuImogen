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
__global__ void cukern_Wstep_hydro_uniform(double *P, double *Cfreeze, double *Qout, double lambdaqtr, int nx);

__global__ void cukern_TVDStep_mhd_uniform(double *P, double *Cfreeze, double *Qin, double halflambda, int nx);
__global__ void cukern_TVDStep_hydro_uniform(double *P, double *Cfreeze, double *Qin, double halfLambda, int nx);

#define BLOCKLEN 92
#define BLOCKLENP2 94
#define BLOCKLENP4 96

__constant__ __device__ double *inputPointers[8];
__constant__ __device__ double fluidQtys[7];
__constant__ __device__ int devArrayNumel;

//#define LIMITERFUNC fluxLimiter_Osher
#define LIMITERFUNC fluxLimiter_minmod

#define FLUID_GAMMA   fluidQtys[0]
#define FLUID_GM1     fluidQtys[1]
#define FLUID_GG1     fluidQtys[2]
#define FLUID_MINMASS fluidQtys[3]
#define FLUID_MINEINT fluidQtys[4]

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
	case 1: // X direction flux: u = x, v = y, w = z;
		gridsize.x = arraySize.y;
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
	double gamHost[7];
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
	gamHost[6] = ALFVEN_FACTOR - .5*(gamma-1.0)*gamma;
	// Even for gamma=5/3, soundspeed is very weakly dependent on density (cube root) for adiabatic fluid

	cudaMemcpyToSymbol(fluidQtys, &gamHost[0], 7*sizeof(double), 0, cudaMemcpyHostToDevice);

	int arrayNumel =  amd.dim[0]*amd.dim[1]*amd.dim[2];
	double *wStepValues;
	cudaMalloc((void **)&wStepValues,arrayNumel*sizeof(double)*6); // [rho px py pz E P]_.5dt
	CHECK_CUDA_ERROR("In cudaFluidStep: halfstep malloc");
	cudaMemcpyToSymbol(devArrayNumel, &arrayNumel, sizeof(int), 0, cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR("In cudaFluidStep: halfstep devArrayNumel memcpy");


	int hydroOnly;
	hydroOnly = (int)*mxGetPr(prhs[11]);

	if(hydroOnly == 1) {
		cfSync(srcs[9], amd.dim[1]*amd.dim[2], prhs[13]);
		CHECK_CUDA_ERROR("In cudaFluidStep: first hd c_f sync");
		cudaMemcpyToSymbol(inputPointers,  srcs, 5*sizeof(double *), 0, cudaMemcpyHostToDevice);

		cukern_Wstep_hydro_uniform<<<gridsize, blocksize>>>(srcs[8], srcs[9], wStepValues, .25*lambda, arraySize.x);
		CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &amd, hydroOnly, "In cudaFluidStep: hydro W step");

		cfSync(srcs[9], amd.dim[1]*amd.dim[2], prhs[13]);
		CHECK_CUDA_ERROR("In cudaFluidStep: second hd c_f sync");
		cukern_TVDStep_hydro_uniform<<<gridsize, blocksize>>>(wStepValues + 5*arrayNumel, srcs[9], wStepValues, .5*lambda, arraySize.x);
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












#define FLUXLa_OFFSET 0
#define FLUXLb_OFFSET (BLOCKLENP4)
#define FLUXRa_OFFSET (2*(BLOCKLENP4))
#define FLUXRb_OFFSET (3*(BLOCKLEN+4))
    #define FLUXA_DECOUPLE(i) fluxArray[FLUXLa_OFFSET+threadIdx.x] = q_i[i]*C_f - w_i; fluxArray[FLUXRa_OFFSET+threadIdx.x] = q_i[i]*C_f + w_i;
    #define FLUXB_DECOUPLE(i) fluxArray[FLUXLb_OFFSET+threadIdx.x] = q_i[i]*C_f - w_i; fluxArray[FLUXRb_OFFSET+threadIdx.x] = q_i[i]*C_f + w_i;

    #define FLUXA_DELTA lambdaqtr*(fluxArray[FLUXLa_OFFSET+threadIdx.x] - fluxArray[FLUXLa_OFFSET+threadIdx.x+1] + fluxArray[FLUXRa_OFFSET+threadIdx.x] - fluxArray[FLUXRa_OFFSET+threadIdx.x-1])
    #define FLUXB_DELTA lambdaqtr*(fluxArray[FLUXLb_OFFSET+threadIdx.x] - fluxArray[FLUXLb_OFFSET+threadIdx.x+1] + fluxArray[FLUXRb_OFFSET+threadIdx.x] - fluxArray[FLUXRb_OFFSET+threadIdx.x-1])

#define momhalfsq momhalfsq

__global__ void cukern_Wstep_mhd_uniform(double *P, double *Cfreeze, double *Qout, double lambdaqtr, int nx)
{
double C_f, velocity;
double q_i[5];
double b_i[3];
double w_i;
double velocity_half;
double rho_half;
__shared__ double fluxArray[4*(BLOCKLENP4)];
__shared__ double freezeSpeed[BLOCKLENP4];
freezeSpeed[threadIdx.x] = 0;

/* Step 0 - obligatory annoying setup stuff (ASS) */
int I0 = nx*(blockIdx.x + gridDim.x * blockIdx.y);
int Xindex = (threadIdx.x-2);
int Xtrack = Xindex;
Xindex += nx*(threadIdx.x < 2);

int x; /* = Xindex % nx; */
bool doIflux = (threadIdx.x > 1) && (threadIdx.x < BLOCKLEN+2);

/* Step 1 - calculate W values */
C_f = Cfreeze[blockIdx.x + gridDim.x * blockIdx.y];
double locP, momhalfsq, momdotB, invrho0;
double Ehalf;

while(Xtrack < nx+2) {
    x = I0 + (Xindex % nx);
    doIflux &= (Xindex < nx);

    b_i[0] = inputPointers[5][x]; /* Load the magnetic field */
    b_i[1] = inputPointers[6][x];
    b_i[2] = inputPointers[7][x];

    q_i[0] = inputPointers[0][x]; // Load mass density
    q_i[1] = inputPointers[1][x]; /* load the energy denstiy */
    q_i[2] = inputPointers[2][x]; // load x momentum density
    q_i[3] = inputPointers[3][x]; // load y momentum density
    q_i[4] = inputPointers[4][x]; // load z momentum density

    locP = P[x];
    velocity = q_i[2] / q_i[0];
    invrho0 = 1.0 / q_i[0]; // for when we need rho_0 to compute <v|b> from <p|b>

    w_i = q_i[2]; // rho flux = px
    FLUXA_DECOUPLE(0)
    w_i = q_i[3]*velocity - b_i[0]*b_i[1]; // py flux = py*v - b by
    FLUXB_DECOUPLE(3)

    momdotB = b_i[0]*q_i[2] + b_i[1]*q_i[3] + b_i[2]*q_i[4];

    __syncthreads();
    if(doIflux) {
        rho_half = q_i[0] - FLUXA_DELTA;
        q_i[3] -= FLUXB_DELTA;
        momhalfsq = q_i[3]*q_i[3]; // store py_half^2
        Qout[x+2*devArrayNumel] = q_i[3];
        //outputPointers[3][x] = q_i[3]; // WROTE PY_HALF
        }
    __syncthreads();

    w_i = velocity*q_i[4] - b_i[0]*b_i[2]; // p_z flux
    FLUXA_DECOUPLE(4);
    w_i = (velocity*q_i[2] + locP - b_i[0]*b_i[0]); /* px flux = v*px + P - bx^2*/
    FLUXB_DECOUPLE(2);
    __syncthreads();

    if(doIflux) {
        q_i[4] -= FLUXA_DELTA; // momz_half
        momhalfsq += q_i[4]*q_i[4]; // now have (py^2 + pz^2)|_half
        Qout[x+3*devArrayNumel] = q_i[3];
        //outputPointers[4][x] = q_i[4]; // WROTE PZ_HALF

        q_i[2] -= FLUXB_DELTA;
        momhalfsq += q_i[2]*q_i[2]; // now have complete p^2 at halfstep.
        Qout[x+devArrayNumel] = q_i[3];
        //outputPointers[2][x] = q_i[2]; // WROTE PX_HALF
// q; P psq pdb le vhf = [pzhalf pxhalf E; P (momhalf^2) (<p|b>) 1/rho rhohalf]
        }
    __syncthreads();

    w_i = velocity*(q_i[1]+locP) - b_i[0]*momdotB*invrho0; /* E flux = v*(E+P) - bx(p dot B)/rho */
    FLUXA_DECOUPLE(1)
    __syncthreads();

    if(doIflux) {
        Ehalf = q_i[1] - FLUXA_DELTA; /* Calculate Ehalf and store a copy in locP */

//        outputPointers[0][x] = q_i[0] = (rho_half > FLUID_MINMASS) ? rho_half : FLUID_MINMASS; // enforce minimum mass density.
        Qout[x] = q_i[0] = (rho_half > FLUID_MINMASS) ? rho_half : FLUID_MINMASS; // enforce minimum mass density.

        momhalfsq = .5*momhalfsq/q_i[0]; // calculate kinetic energy density at halfstep

        q_i[4] = b_i[0]*b_i[0]+b_i[1]*b_i[1]+b_i[2]*b_i[2]; // calculate scalar part of magnetic pressure.

        velocity_half = q_i[2] / q_i[0]; // Calculate vx_half = px_half / rho_half 

        q_i[1] = Ehalf; // set to energy
        locP = Ehalf - momhalfsq; // magnetic + epsilon energy density

        // We must enforce a sane thermal energy density
        // Do this for the thermal sound speed even though the fluid is magnetized
        // assert   cs^2 > cs^2(rho minimum)
        //     g P / rho > g rho_min^(g-1) under polytropic EOS
        //g(g-1) e / rho > g rho_min^(g-1)
        //             e > rho rho_min^(g-1)/(g-1) = rho FLUID_MINEINT
/*        if((locP - q_i[4]) < q_i[0]*FLUID_MINEINT) {
          q_i[1] = momhalfsq + q_i[4] + q_i[0]*FLUID_MINEINT; // Assert minimum E = T + B^2/2 + epsilon_min
          locP = q_i[4] + q_i[0]*FLUID_MINEINT;
          } */
 /* Assert minimum temperature */

        //outputPointers[5][x] = FLUID_GM1*locP + MHD_PRESS_B*q_i[4]; /* Calculate P = (gamma-1)*(E-T) + .5*(2-gamma)*B^2 */
        Qout[x+5*devArrayNumel] = FLUID_GM1*locP + MHD_PRESS_B*q_i[4]; /* Calculate P = (gamma-1)*(E-T) + .5*(2-gamma)*B^2 */
        //outputPointers[1][x] = q_i[1]; /* store total energy: We need to correct this for negativity shortly */
        Qout[x+4*devArrayNumel] = q_i[1]; /* store total energy: We need to correct this for negativity shortly */

        /* calculate local freezing speed = |v_x| + sqrt( g(g-1)*Pgas/rho + B^2/rho) = sqrt(c_thermal^2 + c_alfven^2)*/
        locP = abs(velocity_half) + sqrt( abs(FLUID_GG1*locP + MHD_CS_B * q_i[4])/q_i[0]);
        if(locP > freezeSpeed[threadIdx.x]) {
          // Do not update C_f from the edgemost cells, they are wrong.
          if((Xtrack > 2) && (Xtrack < (nx-3))) freezeSpeed[threadIdx.x] = locP;
          }
        }


    Xindex += BLOCKLEN;
    Xtrack += BLOCKLEN;
    __syncthreads();
    }

/* We have a block of 64 threads. Fold this shit in */

if(threadIdx.x >= 32) return;

if(freezeSpeed[threadIdx.x+32] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+32];
if(freezeSpeed[threadIdx.x+64] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+64];

__syncthreads();
if(threadIdx.x > 16) return;

if(freezeSpeed[threadIdx.x+16] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+16];
__syncthreads();
if(threadIdx.x > 8) return;

if(freezeSpeed[threadIdx.x+8] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+8];
__syncthreads();
if(threadIdx.x > 4) return;

if(freezeSpeed[threadIdx.x+4] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+4];
__syncthreads();
if(threadIdx.x > 2) return;

if(freezeSpeed[threadIdx.x+2] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+2];
__syncthreads();
if(threadIdx.x > 1) return;
/*if(threadIdx.x > 0) return;
for(x = 0; x < BLOCKLENP4; x++) { if(freezeSpeed[x] > freezeSpeed[0]) freezeSpeed[0] = freezeSpeed[x]; }
Cfreeze[blockIdx.x + gridDim.x * blockIdx.y] = freezeSpeed[0];*/

Cfreeze[blockIdx.x + gridDim.x * blockIdx.y] = (freezeSpeed[1] > freezeSpeed[0]) ? freezeSpeed[1] : freezeSpeed[0];

}

__global__ void cukern_Wstep_hydro_uniform(double *P, double *Cfreeze, double *Qout, double lambdaqtr, int nx)
{
double C_f, velocity;
double q_i[3];
double w_i;
double velocity_half;
__shared__ double fluxArray[4*(BLOCKLENP4)];
__shared__ double freezeSpeed[BLOCKLENP4];
freezeSpeed[threadIdx.x] = 0;

/* Step 0 - obligatory annoying setup stuff (ASS) */
int I0 = nx*(blockIdx.x + gridDim.x * blockIdx.y);
int Xindex = (threadIdx.x-2);
int Xtrack = Xindex;
Xindex += nx*(threadIdx.x < 2);

int x; /* = Xindex % nx; */
bool doIflux = (threadIdx.x > 1) && (threadIdx.x < BLOCKLEN+2);

/* Step 1 - calculate W values */
C_f = Cfreeze[blockIdx.x + gridDim.x * blockIdx.y];
double locPsq;
double locE;

// int stopme = (blockIdx.x == 0) && (blockIdx.y == 0); // For cuda-gdb

while(Xtrack < nx+2) {
    x = I0 + (Xindex % nx);
    doIflux &= (Xindex < nx);

/* rho q_i[0] = inputPointers[0][x];  Preload these out here 
     E q_i[1] = inputPointers[1][x];  So we avoid multiple loops 
    px q_i[2] = inputPointers[2][x];  over them inside the flux loop 
    py q_i[3] = inputPointers[3][x];  
    pz q_i[4] = inputPointers[4][x];  */

    q_i[0] = inputPointers[0][x];
    q_i[1] = inputPointers[2][x];
    q_i[2] = inputPointers[1][x];
    locPsq   = P[x];

    velocity = q_i[1] / q_i[0];

    w_i = velocity*(q_i[2]+locPsq); /* E flux = v*(E+P) */
    FLUXA_DECOUPLE(2)
    w_i = (velocity*q_i[1] + locPsq); /* px flux = v*px + P */
    FLUXB_DECOUPLE(1)
    __syncthreads();

    if(doIflux) {
        locE = q_i[2] - FLUXA_DELTA; /* Calculate Ehalf */
        velocity_half = locPsq = q_i[1] - FLUXB_DELTA; /* Calculate Pxhalf */
        //outputPointers[2][x] = locPsq; /* store pxhalf */
        Qout[x+devArrayNumel] = locPsq; /* store pxhalf */
        }
    __syncthreads();

    locPsq *= locPsq; /* store p^2 in locPsq */

    q_i[0] = inputPointers[3][x];
    q_i[2] = inputPointers[4][x];
    w_i = velocity*q_i[0]; /* py flux = v*py */
    FLUXA_DECOUPLE(0)
    w_i = velocity*q_i[2]; /* pz flux = v pz */
    FLUXB_DECOUPLE(2)
    __syncthreads();
    if(doIflux) {
        q_i[0] -= FLUXA_DELTA;
        locPsq += q_i[0]*q_i[0];
        //outputPointers[3][x] = q_i[0]; // py out
        Qout[x+2*devArrayNumel] = q_i[0]; // py out
        q_i[2] -= FLUXB_DELTA;
        locPsq += q_i[2]*q_i[2]; /* Finished accumulating p^2 */
        //outputPointers[4][x] = q_i[2]; // pz out
        Qout[x+3*devArrayNumel] = q_i[2]; // pz out
        }
    __syncthreads();

    q_i[0] = inputPointers[0][x];
    w_i = q_i[1]; /* rho flux = px */
    FLUXA_DECOUPLE(0)
    __syncthreads();
    if(doIflux) {
        q_i[0] -= FLUXA_DELTA; /* Calculate rho_half */
//      outputPointers[0][x] = q_i[0];
        q_i[0] = (q_i[0] < FLUID_MINMASS) ? FLUID_MINMASS : q_i[0]; /* Enforce minimum mass density */
        //outputPointers[0][x] = q_i[0];
        Qout[x] = q_i[0];

        velocity_half /= q_i[0]; /* calculate velocity at the halfstep for doing C_freeze */

        
        locPsq = (locE - .5*(locPsq/q_i[0])); /* Calculate epsilon = E - T */
//      P[x] = FLUID_GM1*locPsq; /* Calculate P = (gamma-1) epsilon */

// For now we have to store the above before fixing them so the original freezeAndPtot runs unperturbed
// but assert the corrected P, C_f values below to see what we propose to do.
// it should match the freezeAndPtot very accurately.

// assert   cs^2 > cs^2(rho minimum)
//     g P / rho > g rho_min^(g-1) under polytropic EOS
//g(g-1) e / rho > g rho_min^(g-1)
//             e > rho rho_min^(g-1)/(g-1) = rho FLUID_MINEINT
        if(locPsq < q_i[0]*FLUID_MINEINT) {
          locE = locE - locPsq + q_i[0]*FLUID_MINEINT; // Assert minimum E = T + epsilon_min
          locPsq = q_i[0]*FLUID_MINEINT; // store minimum epsilon.
          } /* Assert minimum temperature */

        //outputPointers[5][x] = FLUID_GM1*locPsq; /* Calculate P = (gamma-1) epsilon */
        Qout[x+5*devArrayNumel] = FLUID_GM1*locPsq; /* Calculate P = (gamma-1) epsilon */
        //outputPointers[1][x] = locE; /* store total energy: We need to correct this for negativity shortly */
        Qout[x+4*devArrayNumel] = locE; /* store total energy: We need to correct this for negativity shortly */

        /* calculate local freezing speed */
        locPsq = abs(velocity_half) + sqrt(FLUID_GG1*locPsq/q_i[0]);
        if(locPsq > freezeSpeed[threadIdx.x]) {
          if((Xtrack > 2) && (Xtrack < (nx-3))) freezeSpeed[threadIdx.x] = locPsq;
          }
        }

    Xindex += BLOCKLEN;
    Xtrack += BLOCKLEN;
    __syncthreads();
    }

/* We have a block of 64 threads. Fold this shit in */

if(threadIdx.x > 32) return;

if(freezeSpeed[threadIdx.x+32] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+32];
__syncthreads();
if(threadIdx.x > 16) return;

if(freezeSpeed[threadIdx.x+16] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+16];
__syncthreads();
if(threadIdx.x > 8) return;

if(freezeSpeed[threadIdx.x+8] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+8];
__syncthreads();
if(threadIdx.x > 4) return;

if(freezeSpeed[threadIdx.x+4] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+4];
__syncthreads();
if(threadIdx.x > 2) return;

if(freezeSpeed[threadIdx.x+2] > freezeSpeed[threadIdx.x]) freezeSpeed[threadIdx.x] = freezeSpeed[threadIdx.x+2];
__syncthreads();
if(threadIdx.x > 1) return;
/*if(threadIdx.x > 0) return;
for(x = 0; x < BLOCKLENP4; x++) { if(freezeSpeed[x] > freezeSpeed[0]) freezeSpeed[0] = freezeSpeed[x]; }
Cfreeze[blockIdx.x + gridDim.x * blockIdx.y] = freezeSpeed[0];*/

Cfreeze[blockIdx.x + gridDim.x * blockIdx.y] = (freezeSpeed[1] > freezeSpeed[0]) ? freezeSpeed[1] : freezeSpeed[0];

}

#define LEFTMOST_FLAG 1
#define RIGHTMOST_FLAG 2
#define ENDINGRHS_FLAG 4
#define IAM_MAIN_BLOCK 8

#define IAMLEFTMOST (whoflags & LEFTMOST_FLAG)
#define IAMRIGHTMOST (whoflags & RIGHTMOST_FLAG)
#define IAMENDRHS   (whoflags & ENDINGRHS_FLAG)
#define IAMMAIN     (whoflags & IAM_MAIN_BLOCK)

/* blockidx.{xy} is our index in {yz}, and gridDim.{xy} gives the {yz} size */
/* Expect invocation with n+4 threads */
__global__ void cukern_TVDStep_mhd_uniform(double *P, double *Cfreeze, double *Qin, double halfLambda, int nx)
{
// Declare a bunch of crap, much more than needed.
// In NVCC -O2 and symbolic algebra transforms we trust
double c_f, velocity;
double q_i[5];
double prop_i[5]; // proposed q_i values
double b_i[3];
double w_i;
__shared__ double fluxLR[2][BLOCKLENP4];
__shared__ double fluxDerivA[BLOCKLENP4+1];
__shared__ double fluxDerivB[BLOCKLENP4+1];

// Precompute some information about "special" threads.
int whoflags = 0;
if(threadIdx.x < 2)           whoflags += LEFTMOST_FLAG; // Mark which threads form the left most of the block,
if(threadIdx.x >= BLOCKLENP2) whoflags += RIGHTMOST_FLAG; // The rightmost, and the two which will form the final RHS
if((threadIdx.x == ( (nx % BLOCKLEN) + 2)) || (threadIdx.x == ( (nx % BLOCKLEN) + 3)) ) whoflags += ENDINGRHS_FLAG;
if((threadIdx.x > 1) && (threadIdx.x < BLOCKLENP2)) whoflags += IAM_MAIN_BLOCK;

// Calculate the usual stupid indexing tricks
int I0 = nx*(blockIdx.x + gridDim.x * blockIdx.y);
int Xindex = (threadIdx.x-2);
int Xtrack = Xindex;
Xindex += nx*(threadIdx.x < 2);
int x;
int i;

unsigned int threadIndexL = (threadIdx.x-1)%BLOCKLENP4;

// Load the freezing speed once
c_f = Cfreeze[blockIdx.x + gridDim.x * blockIdx.y];

while(Xtrack < nx+2) {
    x = I0 + (Xindex % nx);

    q_i[0] = Qin[x]; // rho
    q_i[1] = Qin[x+4*devArrayNumel]; // Etot      /* So we avoid multiple loops */
    q_i[2] = Qin[x+1*devArrayNumel]; // Px     /* over them inside the flux loop */
    q_i[3] = Qin[x+2*devArrayNumel]; // Py
    q_i[4] = Qin[x+3*devArrayNumel]; // Pz
    b_i[0] = inputPointers[5][x]; // Bx
    b_i[1] = inputPointers[6][x]; // By
    b_i[2] = inputPointers[7][x]; // Bz

    velocity = q_i[2]/q_i[0];

    __syncthreads();

    /* rho, E, px, py, pz going down */
    /* Iterate over variables to flux */
    for(i = 0; i < 5; i++) {
        /* Calculate raw fluxes */
        switch(i) {
            case 0: w_i = q_i[2]; break;
            case 1: w_i = (velocity * (q_i[1] + P[x]) - b_i[0]*(q_i[2]*b_i[0]+q_i[3]*b_i[1]+q_i[4]*b_i[2])/q_i[0] ); break;
            case 2: w_i = (velocity*q_i[2] + P[x] - b_i[0]*b_i[0]); break;
            case 3: w_i = (velocity*q_i[3]        - b_i[0]*b_i[1]); break;
            case 4: w_i = (velocity*q_i[4]        - b_i[0]*b_i[2]); break;
            }

        /* Decouple to L/R flux. */
        fluxLR[0][threadIdx.x] = (q_i[i]*c_f - w_i); /* Left  going flux */
        fluxLR[1][threadIdx.x] = (q_i[i]*c_f + w_i); /* Right going flux */
        __syncthreads();

        /* Derivative of left flux, then right flux */
        fluxDerivA[threadIdx.x] = (fluxLR[0][threadIdx.x] - fluxLR[0][threadIndexL])/2.0;
        fluxDerivB[threadIdx.x] = (fluxLR[1][threadIdx.x] - fluxLR[1][threadIndexL])/2.0;
        __syncthreads();

        /* Apply limiter function to 2nd order corrections */
        fluxLR[0][threadIdx.x] -= LIMITERFUNC(fluxDerivA[threadIdx.x], fluxDerivA[threadIdx.x+1]); // A=bkwd(x), B=fwd(x)
        fluxLR[1][threadIdx.x] += LIMITERFUNC(fluxDerivB[threadIdx.x+1], fluxDerivB[threadIdx.x]); // A=fwd(x), B=bkwd(x)
        __syncthreads();

        /* Perform flux and propose output value */
       if( IAMMAIN && (Xindex < nx) ) {
            prop_i[i] = inputPointers[i][x] - halfLambda * ( -fluxLR[0][threadIdx.x+1] + fluxLR[0][threadIdx.x] + \
                                                              fluxLR[1][threadIdx.x] - fluxLR[1][threadIndexL]  );
          }

        __syncthreads();
        }

    if( IAMMAIN && (Xindex < nx) ) {
      prop_i[0] = (prop_i[0] < FLUID_MINMASS) ? FLUID_MINMASS : prop_i[0]; // enforce min density

      w_i = .5*(prop_i[2]*prop_i[2] + prop_i[3]*prop_i[3] + prop_i[4]*prop_i[4])/prop_i[0] + .5*(b_i[0]*b_i[0] + b_i[1]*b_i[1] + b_i[2]*b_i[2]);

      if((prop_i[1] - w_i) < prop_i[0]*FLUID_MINEINT) {
        prop_i[1] = prop_i[0]*FLUID_MINEINT + w_i;
        }

      inputPointers[0][x] = prop_i[0];
      inputPointers[1][x] = prop_i[1];
      inputPointers[2][x] = prop_i[2];
      inputPointers[3][x] = prop_i[3];
      inputPointers[4][x] = prop_i[4];
      }

    Xindex += BLOCKLEN;
    Xtrack += BLOCKLEN;
    }

}

__global__ void cukern_TVDStep_hydro_uniform(double *P, double *Cfreeze, double *Qin, double halfLambda, int nx)
{
double C_f, velocity;
double q_i[5];
double w_i;
__shared__ double fluxLR[2][BLOCKLENP4];
__shared__ double fluxDerivA[BLOCKLENP4+1];
__shared__ double fluxDerivB[BLOCKLENP4+1];

/* Step 0 - obligatory annoying setup stuff (ASS) */
int I0 = nx*(blockIdx.x + gridDim.x * blockIdx.y);
int Xindex = (threadIdx.x-2);
int Xtrack = Xindex;
Xindex += nx*(threadIdx.x < 2);

int x; /* = Xindex % nx; */
int i;
bool doIflux = (threadIdx.x > 1) && (threadIdx.x < BLOCKLENP2);
double prop_i[5];

unsigned int threadIndexL = (threadIdx.x-1+BLOCKLENP4)%BLOCKLENP4;

/* Step 1 - calculate W values */
C_f = Cfreeze[blockIdx.x + gridDim.x * blockIdx.y];

while(Xtrack < nx+2) {
    x = I0 + (Xindex % nx);

    q_i[0] = Qin[x+0*devArrayNumel]; /* Preload these out here */
    q_i[1] = Qin[x+4*devArrayNumel]; /* So we avoid multiple loops */
    q_i[2] = Qin[x+1*devArrayNumel]; /* over them inside the flux loop */
    q_i[3] = Qin[x+2*devArrayNumel];
    q_i[4] = Qin[x+3*devArrayNumel];
    velocity = q_i[2] / q_i[0];

    /* rho, E, px, py, pz going down */
    /* Iterate over variables to flux */
    for(i = 0; i < 5; i++) {
        /* Calculate raw fluxes */
        switch(i) {
            case 0: w_i = q_i[2]; break;
            case 1: w_i = (velocity * (q_i[1] + P[x]) ) ; break;
            case 2: w_i = (velocity * q_i[2] + P[x]); break;
            case 3: w_i = (velocity * q_i[3]); break;
            case 4: w_i = (velocity * q_i[4]); break;
            }

        /* Decouple to L/R flux */
        fluxLR[0][threadIdx.x] = (C_f*q_i[i] - w_i); /* Left  going flux */
        fluxLR[1][threadIdx.x] = (C_f*q_i[i] + w_i); /* Right going flux */
        __syncthreads();

        /* Calculate proposed flux corrections */
        fluxDerivA[threadIdx.x] = (fluxLR[0][threadIndexL] - fluxLR[0][threadIdx.x]) / 2.0; /* Deriv of leftgoing flux */
        fluxDerivB[threadIdx.x] = (fluxLR[1][threadIdx.x] - fluxLR[1][threadIndexL]) / 2.0; /* Deriv of rightgoing flux */
        __syncthreads();

        /* Impose TVD limiter */
        fluxLR[0][threadIdx.x] += LIMITERFUNC(fluxDerivA[threadIdx.x], fluxDerivA[threadIdx.x+1]);
        fluxLR[1][threadIdx.x] += LIMITERFUNC(fluxDerivB[threadIdx.x+1], fluxDerivB[threadIdx.x]); // A=fwd(x), B=bkwd(x)
        __syncthreads();

        /* Perform flux and write to output array */
       if( doIflux && (Xindex < nx) ) {
            prop_i[i] = inputPointers[i][x] - halfLambda * ( fluxLR[1][threadIdx.x] - fluxLR[1][threadIndexL] +
                                                              -fluxLR[0][threadIdx.x+1] + fluxLR[0][threadIdx.x]);//
            }

        __syncthreads();
        }

    if( doIflux && (Xindex < nx) ) {
        prop_i[0] = (prop_i[0] < FLUID_MINMASS) ? FLUID_MINMASS : prop_i[0];
        w_i = .5*(prop_i[2]*prop_i[2] + prop_i[3]*prop_i[3] + prop_i[4]*prop_i[4])/prop_i[0];

        if((prop_i[1] - w_i) < prop_i[0]*FLUID_MINEINT) {
            prop_i[1] = w_i + prop_i[0]*FLUID_MINEINT;
            }

        inputPointers[0][x] = prop_i[0];
        inputPointers[1][x] = prop_i[1];
        inputPointers[2][x] = prop_i[2];
        inputPointers[3][x] = prop_i[3];
        inputPointers[4][x] = prop_i[4];
        }

    __syncthreads();

    Xindex += BLOCKLEN;
    Xtrack += BLOCKLEN;
    }

}
