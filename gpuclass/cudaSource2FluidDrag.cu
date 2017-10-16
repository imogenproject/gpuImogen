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
#include "cudaSource2FluidDrag.h"

__constant__ int devFluidParams[4];
#define FLUID_NX devFluidParams[0]
#define FLUID_NY devFluidParams[1]
#define FLUID_NZ devFluidParams[2]
#define FLUID_SLABPITCH devFluidParams[3]

__constant__ double dragparams[16];
__constant__ double devLambda[16]; // for gradient calculator kernels

#define PI 3.141592653589793

#define EXPO_TRAPEZOID

// This will vomit out data to console every single call
// designed for use with 32x1x1 test simulation
//#define DBGPRINT

int sourcefunction_2FluidDrag(MGArray *fluidA, MGArray *fluidB, GeometryParams geo, double gam, double sigmaGas, double muGas, double sigmaDust, double muDust, double dt, int method);

int solveDragEMP(MGArray *gas, MGArray *dust, double dt);
int solveDragRK4(MGArray *gas, MGArray *dust, double dt);
int solveDragExponentialMidpt(MGArray *gas, MGArray *dust, GeometryParams geo, double fluidGamma, double dt);

int prepareForERK2(MGArray *gas, MGArray *dust, MGArray *tempMem, GeometryParams geom, int spaceOrder, double scalingParameter);
int findMidGradP2(MGArray *gas, MGArray *tempMem, GeometryParams geom, int spaceOrder, double scalingParameter);

template <bool ONLY_DV_INI>
__global__ void cukern_GasDustDrag_full(double *gas, double *dust, double *tmpmem, int srcBlock, int dstBlock, int N);
__global__ void cukern_GasDustDrag_Epstein(double *gas, double *dust, double *vrel, int N);
template <bool resetAccumulator>
__global__ void cukern_GasDustDrag_linearTime(double *gas, double *dust, double *tmpmem, int srcBlock, int kBlock, int N);

__global__ void cukern_findInitialDeltaV(double *g, double *d, double *dv, unsigned long partNumel);

// Functions to evaluate explicit Butcher tableaus
template <bool resetAccumulator>
__global__ void cukern_SolveRK_single(double *tmpmem, int d, double A, int i, double B, unsigned long partNumel);
template <bool resetAccumulator>
__global__ void cukern_SolveRK_double(double *tmpmem, int d, double F[2], int i[2], double B, unsigned long partNumel);
template <bool resetAccumulator>
__global__ void cukern_SolveRK_triple(double *tmpmem, int d, double F[3], int i[3], double B, unsigned long partNumel);
__global__ void cukern_SolveRK_final(double *tmpmem, int i, double B, double W, unsigned long partNumel);

__global__ void cukern_applyFinalDeltaV(double *g, double *d, double *dv_final, unsigned long partNumel);

__global__ void cukern_ExponentialEulerHalf(double *gas, double *dust, double *tmpmem, double t, unsigned long partNumel);
__global__ void cukern_exponentialMidpoint(double *gas, double *dust, double t, double *tmpmem);
__global__ void cukern_exponentialTrapezoid(double *gas, double *dust, double t, double *tmpmem);

// Accept the following drag models:
// (1) full      : Use full Epstein+Stokes calculation with interpolation between all 4 quadrants
// (2) Epstein   : Use only Epstein force calculation, valid for any speed but only small particles
// (3) Linear    : Compute Epstein+Stokes in low-velocity limit, valid only for |delta-v/c| << 1 (and strictly, Re < 1)

// PARITY CONVENTIONS ARE AS FOLLOWS:
// delta-V is defined as GAS VELOCITY MINUS DUST VELOCITY
// Drag force is positive in the direction of delta-V,
// i.e. d/dt(dust momentum) = F_drag and d/dt(gas momentum) = -F_drag
// ergo d/dt(delta_V) ~ -F_drag / mass

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if ((nrhs!=3) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaSource2FluidDrag(FluidManager, geometry, [sigma_gas, mu_gas, dia_dust, mass_dust, dt, solverMethod])\n");

	if(CHECK_CUDA_ERROR("entering cudaSource2FluidDrag") != SUCCESSFUL) { DROP_MEX_ERROR("Failed upon entry to cudaSource2FLuidDrag."); }

	MGArray fluidA[5];
	int status = MGA_accessFluidCanister(prhs[0], 0, &fluidA[0]);
	if(status != SUCCESSFUL) {
		PRINT_FAULT_HEADER;
		printf("Unable to access first FluidManager.\n");
		PRINT_FAULT_FOOTER;
		DROP_MEX_ERROR("crashing.");
	}

	MGArray fluidB[5];
	status = MGA_accessFluidCanister(prhs[0], 1, &fluidB[0]);
	if(status != SUCCESSFUL) {
		PRINT_FAULT_HEADER;
		printf("Unable to access second FluidManager.\n");
		PRINT_FAULT_FOOTER;
		DROP_MEX_ERROR("crashing.");
	}

	GeometryParams geo = accessMatlabGeometryClass(prhs[1]);

	double *params = mxGetPr(prhs[2]);

	size_t ne = mxGetNumberOfElements(prhs[2]);
	if(ne != 6) {
		PRINT_FAULT_HEADER;
		printf("2nd argument to cudaSource2FluidDrag must have 6 elements:\n[sigmaGas muGas sigmaDust muDust dt (method: 0=midpt, 1=rk4, 2=exponential)]\nGiven argument has %i instead.\n", ne);
		PRINT_FAULT_FOOTER;
		DROP_MEX_ERROR("Crashing.");
	}	

	double fluidGamma = derefXdotAdotB_scalar(prhs[0], "gamma", NULL);
	double dt         = params[4];

	// fixme... get this from the params directly?...
	double sigmaGas   = params[0];
	double muGas      = params[1];
	double sigmaDust  = params[2];
	double muDust     = params[3];
	int solverMethod  = (int)params[5];

	//1nm iron sphere, 300K -> 56m/s thermal velocity
	//10nm iron ball, 300K -> 1.79m/s thermal velocity
	//100nm iron ball, 300K -> 56mm/s thermal velocity
	
	status = sourcefunction_2FluidDrag(&fluidA[0], &fluidB[0], geo, fluidGamma, sigmaGas, muGas, sigmaDust, muDust, dt, solverMethod);

	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) {
		DROP_MEX_ERROR("2-fluid drag code crashed!");
	}

	return;

}

int sourcefunction_2FluidDrag(MGArray *fluidA, MGArray *fluidB, GeometryParams geo, double gam, double sigmaGas, double muGas, double sigmaDust, double muDust, double dt, int method)
{
	int i;
	int sub[6];
	int hostFluidParams[4];

	int statusCode = SUCCESSFUL;

	double hostDrag[16];
	hostDrag[0] = 128.0 * sigmaGas * sqrt(sigmaDust) / (5 * muGas * PI * sqrt(gam - 1)); // alpha
	hostDrag[1] = 128*(gam-1.0)/(PI*9.0); // beta
	hostDrag[2] = 1/gam;
	hostDrag[3] = 5*PI*sqrt(PI/2.0)*muGas / (144.0 * sigmaGas);
	hostDrag[4] = muDust; // FIXME: this should be an array perhaps?
	hostDrag[5] = sigmaDust;
	hostDrag[6] = 1.0 / (gam-1.0);
	hostDrag[7] = dt;
	hostDrag[8] = (gam-1.0);
	
	for(i = 0; i < fluidA->nGPUs; i++) {
		cudaSetDevice(fluidA->deviceID[i]);
		statusCode = CHECK_CUDA_ERROR("cudaSetDevice");
		if(statusCode != SUCCESSFUL) break;

		calcPartitionExtent(fluidA, i, &sub[0]);
		hostFluidParams[0] = sub[3];
		hostFluidParams[1] = sub[4];
		hostFluidParams[2] = sub[5];
		hostFluidParams[3] = fluidA->slabPitch[i] / sizeof(double); // This is important, due to padding, is isn't just .partNumel
		cudaMemcpyToSymbol((const void *)devFluidParams, &hostFluidParams[0], 4*sizeof(int), 0, cudaMemcpyHostToDevice);
		statusCode = CHECK_CUDA_ERROR("memcpyToSymbol");
		if(statusCode != SUCCESSFUL) break;
		cudaMemcpyToSymbol((const void *)dragparams, &hostDrag[0], 9*sizeof(double), 0, cudaMemcpyHostToDevice);
		statusCode = CHECK_CUDA_ERROR("memcpyToSymbol");
		if(statusCode != SUCCESSFUL) break;
	}

	if(statusCode != SUCCESSFUL) return statusCode;

	// FIXME pick a numeric method here dynamically?
	switch(method) {
	case 0: // EMP
		statusCode = CHECK_IMOGEN_ERROR(solveDragEMP(fluidA, fluidB, dt));
		break;
	case 1: // RK4
		statusCode = CHECK_IMOGEN_ERROR(solveDragRK4(fluidA, fluidB, dt));
		break;
	case 2: // ERK2
		statusCode = CHECK_IMOGEN_ERROR(solveDragExponentialMidpt(fluidA, fluidB, geo, gam, dt));
		break;
	}
	
	return statusCode;
}

/* Solves the action of gas-dust drag for one dust using the explicit midpoint method
 * 2nd order in time, not A-stable (dt <~ t_stop) */
int solveDragEMP(MGArray *gas, MGArray *dust, double dt)
{

int n = gas->nGPUs;

double *tmpmem[n];
double *g; double *d;
double *vrel;

int statusCode = SUCCESSFUL;

int i;
for(i = 0; i < n; i++) {
	cudaSetDevice(gas->deviceID[i]);
	statusCode = CHECK_CUDA_ERROR("cudaSetDevice");
	if(statusCode != SUCCESSFUL) break;
	// allocate temp storage per gpu
	cudaMalloc((void **)(&tmpmem[i]), 5*gas->slabPitch[i]);
	statusCode = CHECK_CUDA_ERROR("cudaMalloc tmpmem for solveDragEMP");
	if(statusCode != SUCCESSFUL) break;
	// store initial v_relative, current v_relative, ini_uint, acceleration in slabs 1, 2, 3 and 4
}

if(statusCode != SUCCESSFUL) {
	printf("Unable to grab temporary memory: Crashing.\n");
	PRINT_FAULT_FOOTER;
	return statusCode;
}

int BS = 96;

dim3 blocksize(BS, 1, 1);
dim3 gridsize(32, 1, 1);

for(i = 0; i < n; i++) {
	long NE = gas->partNumel[i];

	// avoid launching tons of threads for small problems
	gridsize.x = 32;
	if(ROUNDUPTO(NE, BS)/BS < 32) {
		gridsize.x = ROUNDUPTO(NE, BS)/BS;
	}

	cudaSetDevice(gas->deviceID[i]);
	g = gas->devicePtr[i];
	d = dust->devicePtr[i];
	vrel = tmpmem[i] + 0;

	// compute initial delta-v
	cukern_findInitialDeltaV<<<gridsize, blocksize>>>(g, d, vrel, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_findInitialDeltaV");
	if(statusCode != SUCCESSFUL) break;
	// solve gas drag on y0, store in block 3: use only ini dv for u_specific
	cukern_GasDustDrag_full<true><<<gridsize, blocksize>>>(g, d, vrel, 0, 3, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_GasDustDrag_full<false>");
	if(statusCode != SUCCESSFUL) break;
	// compute delta-v at t=1/2; store stage at block 4
	cukern_SolveRK_single<true><<<gridsize, blocksize>>>(vrel, 4, .5*dt, 3, 0, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_SolveRK_single<true>");
	if(statusCode != SUCCESSFUL) break;
	// solve gas drag at t=1/2 using half stage, store in block 3
	cukern_GasDustDrag_full<false><<<gridsize, blocksize>>>(g, d, vrel, 4, 3, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_GasDustDrag_full<true>");
	if(statusCode != SUCCESSFUL) break;
	// Apply final stage derivative to compute y(t)
	cukern_SolveRK_final<<<gridsize, blocksize>>>(vrel, 3, 1.0, dt, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_SolveRK_final");
	if(statusCode != SUCCESSFUL) break;
	// compute new gas/dust momentum and temperature arrays using analytic forms
	cukern_applyFinalDeltaV<<<gridsize, blocksize>>>(g, d, vrel, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_applyFinalDeltaV");
	if(statusCode != SUCCESSFUL) break;
}

for(i = 0; i < n; i++) {
	cudaSetDevice(gas->deviceID[i]);
	cudaFree((void *)tmpmem[i]);
}

return SUCCESSFUL; // FIXME: check this once its working

}

/* Solves the action of the gas-dust drag for one dust using the 4th order RK method of Kutta (1903)
 * 4th order in time, conditionally stable (dt <~ 3t_stop) */
int solveDragRK4(MGArray *gas, MGArray *dust, double dt)
{

int n = gas->nGPUs;

double *tmpmem[n];
double *g; double *d;
double *vrel;

int statusCode = SUCCESSFUL;

int i;
for(i = 0; i < n; i++) {
	cudaSetDevice(gas->deviceID[i]);
	statusCode = CHECK_CUDA_ERROR("cudaSetDevice");
	if(statusCode != SUCCESSFUL) break;
	// allocate temp storage per gpu
	cudaMalloc((void **)(&tmpmem[i]), 5*gas->slabPitch[i]);
	statusCode = CHECK_CUDA_ERROR("cudaMalloc tmpmem for solveDragEMP");
	if(statusCode != SUCCESSFUL) break;
	// store initial v_relative, current v_relative, ini_uint, acceleration in slabs 1, 2, 3 and 4
}

if(statusCode != SUCCESSFUL) {
	printf("Unable to grab temporary memory: Crashing.\n");
	PRINT_FAULT_FOOTER;
	return statusCode;
}

int BS = 96;

dim3 blocksize(BS, 1, 1);
dim3 gridsize(32, 1, 1);
dim3 smallgrid(1,1,1);

double bWeights[4] = { 1.0, 2.0, 2.0, 1.0 };
double bRescale = dt / 6.0;

for(i = 0; i < n; i++) {
	long NE = gas->partNumel[i];

	// avoid launching tons of threads for small problems
	gridsize.x = 32;
	if(ROUNDUPTO(NE, BS)/BS < 32) {
		gridsize.x = ROUNDUPTO(NE, BS)/BS;
	}

	cudaSetDevice(gas->deviceID[i]);
	g = gas->devicePtr[i];
	d = dust->devicePtr[i];
	vrel = tmpmem[i] + 0;

	// compute initial delta-v
	cukern_findInitialDeltaV<<<gridsize, blocksize>>>(g, d, vrel, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_findInitialDeltaV");
	if(statusCode != SUCCESSFUL) break;
	// solve gas drag on y0, store in block 3
	cukern_GasDustDrag_full<true><<<gridsize, blocksize>>>(g, d, vrel, 0, 3, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_GasDustDrag_full");
	if(statusCode != SUCCESSFUL) break;
	// compute delta-v at t=1/2; store stage at block 4
	cukern_SolveRK_single<true><<<gridsize, blocksize>>>(vrel, 4, 0.5*dt, 3, bWeights[0], NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_SolveRK_single<true>");
	if(statusCode != SUCCESSFUL) break;
	// solve gas drag on k2, store in block 3
	cukern_GasDustDrag_full<false><<<gridsize, blocksize>>>(g, d, vrel, 4, 3, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_GasDustDrag_full<true>");
	if(statusCode != SUCCESSFUL) break;
	// compute delta-v at t=1/2; store stage at block 4
	cukern_SolveRK_single<false><<<gridsize, blocksize>>>(vrel, 4, 0.5*dt, 3, bWeights[1], NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_SolveRK_single<true>");
	if(statusCode != SUCCESSFUL) break;
	// solve gas drag on k3, store in block 3
	cukern_GasDustDrag_full<false><<<gridsize, blocksize>>>(g, d, vrel, 4, 3, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_GasDustDrag_full<true>");
	if(statusCode != SUCCESSFUL) break;
	// compute delta-v at t=1/2; store stage at block 4
	cukern_SolveRK_single<false><<<gridsize, blocksize>>>(vrel, 4, 1.0*dt, 3, bWeights[2], NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_SolveRK_single<true>");
	if(statusCode != SUCCESSFUL) break;
	// solve gas drag on k4, store in block 3
	cukern_GasDustDrag_full<false><<<gridsize, blocksize>>>(g, d, vrel, 4, 3, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_GasDustDrag_full<true>");
	if(statusCode != SUCCESSFUL) break;
	// add block 3 to accumulator, rescale by dt / 6.0 and add y0 to find final dv.
	cukern_SolveRK_final<<<gridsize, blocksize>>>(vrel, 3, bWeights[3], bRescale, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_SolveRK_final");
	if(statusCode != SUCCESSFUL) break;
	// compute new gas/dust momentum and temperature arrays
	cukern_applyFinalDeltaV<<<gridsize, blocksize>>>(g, d, vrel, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_applyFinalDeltaV");
	if(statusCode != SUCCESSFUL) break;
}

if(statusCode != SUCCESSFUL) {
	printf("Freeing temp memory and returning crash condition.\n");
	PRINT_FAULT_FOOTER;
}

for(i = 0; i < n; i++) {
	cudaSetDevice(gas->deviceID[i]);
	cudaFree((void *)tmpmem[i]);
}

return statusCode;
}

/* Solves the gas-dust drag equations using the 2nd order Explicit Exponential Runge-Kutta method
 * aka the exponential midpoint method:
 * u_stiff(hf) = u_stiff(0) exp(M_stiff(0) t/2)
 * u_soft(hf)  = u_0 + M_soft(0)*t/2
 * u_stiff(t)  = u_stiff(0) exp(M_stiff(hf) t)
 * u_soft(t)   = u_0 + M_soft(hf)*t/2
 * where the stiff term (gas-dust drag) is solved by directly exponentiating its characteristic matrix
 * and the nonstiff terms are handled by simple explicit RK2
 *
 * We are advantaged here that to an excellent approximation the stiff terms are truly linear
 * (i.e. the effect of drag heating in altering pressure gradients is neglectable) if drag is strong
 * enough to require calling this method.
 * formally order 2, stiff order 2, L-stable */
int solveDragExponentialMidpt(MGArray *gas, MGArray *dust, GeometryParams geo, double fluidGamma, double dt)
{
int n = gas->nGPUs;

double *g; double *d;
double *tempPtr;

int statusCode = SUCCESSFUL;

MGArray tmpMem;
MGArray *gs = &tmpMem;

statusCode = MGA_allocSlab(gas, gs, 6);
if(CHECK_IMOGEN_ERROR(statusCode) != SUCCESSFUL) return statusCode;

int i;
int BS = 96;

// for kernels not requiring finite differencing
dim3 linblock(BS, 1, 1);
dim3 lingrid(32, 1, 1);

// for kernels that do need to do FD
dim3 fdgrid(4, 4, 1);
dim3 fdblock(16, 16, 1);

// Emits [|dv_tr|, u_0, P_x, P_y, P_z] into temp memory at gs
statusCode = prepareForERK2(gas, dust, gs, geo, 2, fluidGamma - 1);

#ifdef DBGPRINT
	double *hstcpy = (double *)malloc(gas->slabPitch[0]*5);
	cudaMemcpy((void *)hstcpy, (const void *)gas->devicePtr[0], gas->slabPitch[0]*5, cudaMemcpyDeviceToHost);
	printf("Gas input state: [%e %e %e %e %e]\n", hstcpy[0], hstcpy[32], hstcpy[64], hstcpy[96], hstcpy[128]);
	cudaMemcpy((void *)hstcpy, (const void *)dust->devicePtr[0], gas->slabPitch[0]*5, cudaMemcpyDeviceToHost);
	printf("Dust input state: [%e %e %e %e %e]\n", hstcpy[0], hstcpy[32], hstcpy[64], hstcpy[96], hstcpy[128]);
	cudaMemcpy((void *)hstcpy, (const void *)gs->devicePtr[0], gas->slabPitch[0]*5, cudaMemcpyDeviceToHost);
	printf("post-prep tmp st: [%e %e %e %e %e]\n", hstcpy[0], hstcpy[32], hstcpy[64], hstcpy[96], hstcpy[128]);
#endif

int velblock = 0;
int kblock = 0;

double tPredict, tFull;
#ifdef EXPO_TRAPEZOID
tPredict = dt;
tFull = 0.5*dt; // this may look bass ackward but it's correct
#else
tPredict = .5*dt;
tFull = dt;
#endif

int doExpEuler = 0;

for(i = 0; i < n; i++) {
	long NE = gas->partNumel[i];

	// avoid launching tons of threads for small problems
	lingrid.x = 32;
	if(ROUNDUPTO(NE, BS)/BS < 32) {
		lingrid.x = ROUNDUPTO(NE, BS)/BS;
	}

	cudaSetDevice(gas->deviceID[i]);
	g = gas->devicePtr[i];
	d = dust->devicePtr[i];
	tempPtr = tmpMem.devicePtr[i];

	// This part is accomplished by the prepareForERK2 call above
	// Compute the initial specific internal energy, scalar |dv_timereversed|, & pressure gradient
	//cukern_exponentialDragSetup<<<fdgrid, fdblock>>>(g, d, tempPtr, gas->partNumel[i]);
	//statusCode = CHECK_CUDA_LAUNCH_ERROR(fdblock, fdgrid, gas, i, "cukern_exponentialDragSetup");
	//if(statusCode != SUCCESSFUL) break;

	// Use u_0 and dv_tr to compute the drag eigenvalue at t=0
	// overwrite the |dv_tr| value (block 0) with K
	cukern_GasDustDrag_linearTime<true><<<lingrid, linblock>>>(g, d, tempPtr, velblock, kblock, gas->partNumel[i]);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(linblock, lingrid, gas, i, "cukern_GasDustDrag_linearTime");
	if(statusCode != SUCCESSFUL) break;
#ifdef DBGPRINT
	cudaMemcpy((void *)hstcpy, (const void *)gs->devicePtr[0], gas->slabPitch[0]*5, cudaMemcpyDeviceToHost);
	printf("post-lint tmp st: [%e %e %e %e %e]\n", hstcpy[0], hstcpy[32], hstcpy[64], hstcpy[96], hstcpy[128]);
#endif

	// Use the eigenvalue from t=0 to advance to t=1/2
	//   Output only new uint & dv values from this stage,
	//   We do this only do re-evaluate the pressure gradient & eigenvalue at the midpoint
	//   This reads K from register 0 and overwrites it with dv_half

	if(doExpEuler) {
		cukern_exponentialMidpoint<<<lingrid, linblock>>>(g, d, dt, tempPtr);
	} else {
		cukern_ExponentialEulerHalf<<<lingrid, linblock>>>(g, d, tempPtr, tPredict, gas->partNumel[i]);
	}
	statusCode = CHECK_CUDA_LAUNCH_ERROR(linblock, lingrid, gas, i, "cukern_ExponentialEulerHalf");
	if(statusCode != SUCCESSFUL) break;
#ifdef DBGPRINT
	cudaMemcpy((void *)hstcpy, (const void *)gs->devicePtr[0], gas->slabPitch[0]*5, cudaMemcpyDeviceToHost);
	printf("post-eehf tmp st: [%e %e %e %e %e]\n", hstcpy[0], hstcpy[32], hstcpy[64], hstcpy[96], hstcpy[128]);
#endif
}

if(doExpEuler == 0) {
// call grad-p resolver
statusCode = findMidGradP2(gas, gs, geo, 2, fluidGamma - 1);
#ifdef DBGPRINT
	cudaMemcpy((void *)hstcpy, (const void *)gs->devicePtr[0], gas->slabPitch[0]*5, cudaMemcpyDeviceToHost);
	printf("post-gradp_hf st: [%e %e %e %e %e]\n", hstcpy[0], hstcpy[32], hstcpy[64], hstcpy[96], hstcpy[128]);
#endif

#ifdef EXPO_TRAPEZOID
velblock = 5; // exp euler wrote it here instead if doing trapezoid to avoid overwriting original k
#endif

for(i = 0; i < n; i++) {
	long NE = gas->partNumel[i];

	// avoid launching tons of threads for small problems
	lingrid.x = 32;
	if(ROUNDUPTO(NE, BS)/BS < 32) {
		lingrid.x = ROUNDUPTO(NE, BS)/BS;
	}

	cudaSetDevice(gas->deviceID[i]);
	g = gas->devicePtr[i];
	d = dust->devicePtr[i];
	tempPtr = tmpMem.devicePtr[i];

	// accumulates new k onto original k, such that block 0 is now (k0 + k1)...
#ifdef EXPO_TRAPEZOID
	cukern_GasDustDrag_linearTime<false><<<lingrid, linblock>>>(g, d, tempPtr, velblock, kblock, gas->partNumel[i]);
#else
	cukern_GasDustDrag_linearTime<true><<<lingrid, linblock>>>(g, d, tempPtr, velblock, kblock, gas->partNumel[i]);
#endif
	statusCode = CHECK_CUDA_LAUNCH_ERROR(linblock, lingrid, gas, i, "cukern_GasDustDrag_linearTime");
	if(statusCode != SUCCESSFUL) break;
#ifdef DBGPRINT
	cudaMemcpy((void *)hstcpy, (const void *)gs->devicePtr[0], gas->slabPitch[0]*5, cudaMemcpyDeviceToHost);
	printf("second t0 st: [%e %e %e %e %e]\n", hstcpy[0], hstcpy[32], hstcpy[64], hstcpy[96], hstcpy[128]);
#endif
	// Use averaged pressure gradient and k value to compute timestep.
	// we divide t by 2 since we simply summed the k values previously
	cukern_exponentialMidpoint<<<lingrid, linblock>>>(g, d, tFull, tempPtr);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(linblock, lingrid, gas, i, "cukern_exponentialMidpoint");
	if(statusCode != SUCCESSFUL) break;
#ifdef DBGPRINT
	cudaMemcpy((void *)hstcpy, (const void *)gas->devicePtr[0], gas->slabPitch[0]*5, cudaMemcpyDeviceToHost);
	printf("Gas input state: [%e %e %e %e %e]\n", hstcpy[0], hstcpy[32], hstcpy[64], hstcpy[96], hstcpy[128]);
	cudaMemcpy((void *)hstcpy, (const void *)dust->devicePtr[0], gas->slabPitch[0]*5, cudaMemcpyDeviceToHost);
	printf("Dust input state: [%e %e %e %e %e]\n", hstcpy[0], hstcpy[32], hstcpy[64], hstcpy[96], hstcpy[128]);
	free(hstcpy);
#endif
}
}
// Make sure node's internal boundaries are consistent
MGA_exchangeLocalHalos(gas  + 1, 4);
MGA_exchangeLocalHalos(dust + 1, 4);

MGA_delete(gs);

return SUCCESSFUL; // FIXME: check this once its working

}

/* This function returns the Stokes coefficient, scaled by 1/2
 * This parameter is experimentally measured except for the low-Re regime */
__device__ double drag_coeff(double Re)
{
	if(Re < 1) {
		// 24 / Re
		return 12 / Re;
	}
	if(Re > 7.845084191866316e+02) {
		// .44
		return 0.22;
	}
	// 24 Re^-.6
	return 12.0*pow(Re,-0.6);
}

/* Compute drag between gas and dust particles, utilizing precomputed global
 * factors in dragparams[]:
 *
 * Note that we make the approximation that the dust volume fraction is zero,
 * which is highly valid in astrophysical circumstances
 *
 * Then we have the following general equations:
 * d(pgas)/dt = -Fdrag ndust
 * d(pdust)/dt= Fdrag ndust
 * d(Etotal,gas)/dt = -Fdrag . (vdust) + Qtherm
 * d(Etotal,dust)/dt = Fdrag . vdust - Qtherm
 *
 * psi = sqrt(gamma/2) * dv / cs
 * cs = sqrt(gamma Pgas / rho)
 *
 * F_epstein = - (4 pi / 3) rho_g s^2 sqrt(8 / gamma pi)
 *
 * where Fdrag may be computed, given
		dv == Vgas - Vdust is the velocity differential,
		nu == ( 5 mu c_s / 64 sigma rho ) sqrt(pi/gamma) is the kinematic viscosity
		Rd == 2 s |dv| / nu is the local Reynolds number for a particle, and
		C_d = { 12 pi/Rd	|     Rd <  1     }
			  { 12 pi/Rd^.6 | 1 < Rd <  784.5 }
			  { .22 pi      |     Rd >= 784.5 } is the experimentally known Stokes drag coefficient
		MFP = sqrt(gamma pi / 2) 5 m_gas / (64 sigma_gas rho_gas)
 * We may extract common factors from both drag regimes,
 * multiply by the dust number density to get the volume force density, and
 * asymptotically interpolate between them to find that
 *							n_dust
 * Fdrag = s^2 rho_gas [ s0^2 K_epstein + s^2 K_stokes ] -------- \vec{dv}
 *							   s0^2+s^2
 * Where we find the particledynamic and viscous-fluid drag coefficients as
 *	  K_epstein = (4 pi /3) sqrt(8 / gamma pi) sqrt(c_s^2 + 9 pi |dv|^2 / 128)
 *		= sqrt(dragparams[0]*uint + dragparams[1]* dv.dv)
 *	  K_stokes  = C_d |dv|
 * And the interpolation cutover s_0 is chosen s_0 = (4/9) MFP
 *
 * Then relative acceleration = dv_relative/dt is
 * a_r = s^2 (rho_gas+rho_dust) [ s0^2 K_epstein + s^2 K_stokes ] \vec{dv}
 *							          M_dust(s0^2+s^2)
 */


/*
 * From dimensional analysis, by choosing L and T we can rescale...
 */
#define ALPHA      dragparams[0]
#define BETA	   dragparams[1]
#define EPSILON    dragparams[2]
#define THETA      dragparams[3]
#define DUSTMASS   dragparams[4]
#define SIGMA_DUST dragparams[5]
#define GAMMAM1    dragparams[8]

/* This function directly computes the gas-dust drag force in the full (stokes+epstein) regime
 * This is suited for weaker drag or strange regimes, but unnecessary and time-consuming for
 * small particles which will never exit the low-speed Epstein regime.
 * Uses stage value stored at srcBlock, writes acceleration into dstBlock */
template <bool ONLY_DV_INI>
__global__ void cukern_GasDustDrag_full(double *gas, double *dust, double *tmpmem, int srcBlock, int dstBlock, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	double rhoA, rhoB;   // gas and dust densities respectively 
	double magdv;	// magnitude velocity difference
	double uspecific;	// specific internal energy density
	double Re;	   // Spherical particle Reynolds number
	double accel;	// Relative acceleration (d/dt of vrel)
	double sigma0;
	double kEpstein, kStokes;

	gas  += i;
	dust += i;
	tmpmem += i;

	for(; i < N; i+= blockDim.x*gridDim.x) {
		magdv = tmpmem[srcBlock*FLUID_SLABPITCH];

		rhoA = gas[0];
		rhoB = dust[0];
	
		if(ONLY_DV_INI) {
			uspecific = tmpmem[FLUID_SLABPITCH];
		} else {
			// make sure computation includes gas heating term!
			uspecific = tmpmem[FLUID_SLABPITCH] + .5 * rhoB * (tmpmem[0]*tmpmem[0] - magdv*magdv) / (rhoA + rhoB);
		}

		kEpstein = sqrt(BETA*uspecific + EPSILON*magdv*magdv);

		// FIXME this implementation is poorly conditioned (re ~ 1/v for v << v0)
		Re = ALPHA*magdv*rhoA/sqrt(uspecific);
		kStokes = drag_coeff(Re) * magdv;
	
		sigma0 = THETA / rhoA; // sqrt(pi)*(4 l_mfp / 9) = sqrt(pi) * s0
		sigma0 *= sigma0; // = pi s0^2 = epstein/stokes cutover crossection

		 //a_rel = ( sigma0 * sqrt(beta*uinternal(dv) + epsilon*dv^2) + D_dust^2 * C_hat(dv) * dv) * dv * D_dust^2 * (rho_g + rho_d) / (m_dust * (d0^2 + D_dust^2));
		accel = ( sigma0 * kEpstein + SIGMA_DUST * kStokes) * magdv * SIGMA_DUST * (rhoA + rhoB) / (DUSTMASS * (sigma0 + SIGMA_DUST));
	
		tmpmem[dstBlock*FLUID_SLABPITCH] = -accel;
	
		gas += blockDim.x*gridDim.x;
		dust += blockDim.x*gridDim.x;
		tmpmem += blockDim.x*gridDim.x;
	}

}

/* This function computes particle drag in the Epstein regime (particles much smaller than gas MFP)
 * but is unsuited to large particles or dense gas
 */
__global__ void cukern_GasDustDrag_Epstein(double *gas, double *dust, double *vrel, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	double rhoA, rhoB;   // gas and dust densities respectively
	double magdv;	// magnitude velocity difference
	double uinternal;	// specific internal energy density
	double accel;	// Relative acceleration (d/dt of vrel)
	double kEpstein;

	gas  += i;
	dust += i;
	vrel += i;

	for(; i < N; i+= blockDim.x*gridDim.x) {
		magdv = vrel[FLUID_SLABPITCH];
		rhoA = gas[0];
		rhoB = dust[0];

		// make sure computation includes gas heating term!
		uinternal = vrel[2*FLUID_SLABPITCH] + rhoB * (vrel[0]*vrel[0] - magdv*magdv) / (rhoA + rhoB);
		kEpstein = sqrt(BETA*uinternal + EPSILON*magdv*magdv);
		accel = kEpstein * magdv * SIGMA_DUST * (rhoA + rhoB) / DUSTMASS;

		vrel[3*FLUID_SLABPITCH] = accel;

		gas += blockDim.x*gridDim.x;
		dust += blockDim.x*gridDim.x;
		vrel += blockDim.x*gridDim.x;
	}

}

/* This function returns the drag rate K = (dv/dt) / v which is useful for e.g. exponential methods
 * for very stiff drag
 *
 * If motion is acted on exclusively by drag, a simple formula is available to determine heating
 * as a result of drag friction exactly. In this case, the original and current velocities are used
 * If it is not, the result is not as trivial and ONLY_DV_INI = true just uses a given input specific
 * internal energy.
 */
template <bool resetAccumulator>
__global__ void cukern_GasDustDrag_linearTime(double *gas, double *dust, double *tmpmem, int srcBlock, int kBlock, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	double rhoA, rhoB;   // gas and dust densities respectively
	double magdv;	// magnitude velocity difference
	double uSpecific;	// specific internal energy density
	double k0;	// Relative acceleration (d/dt of vrel) divided by vrel
	double sigma0;
	double kEpstein, kStokes, Re;

	gas  += i;
	dust += i;
	tmpmem += i;

	for(; i < N; i+= blockDim.x*gridDim.x) {
		magdv = tmpmem[srcBlock*FLUID_SLABPITCH];

		rhoA = gas[0];
		rhoB = dust[0];

		uSpecific = tmpmem[FLUID_SLABPITCH];
		kEpstein = sqrt(BETA*uSpecific + EPSILON*magdv*magdv);

		// FIXME this implementation is poorly conditioned (re ~ 1/v for Re << 1 then gets Re*v for drag...)
		Re = ALPHA*magdv*rhoA/sqrt(uSpecific);
		kStokes = drag_coeff(Re) * magdv;

		sigma0 = THETA / rhoA; // sqrt(pi)*(4 l_mfp / 9) = sqrt(pi) * s0
		sigma0 *= sigma0; // = pi s0^2 = epstein/stokes cutover crossection

		//a_rel = ( sigma0 * sqrt(beta*uinternal(dv) + epsilon*dv^2) + D_dust^2 * C_hat(dv) * dv) * dv * D_dust^2 * (rho_g + rho_d) / (m_dust * (d0^2 + D_dust^2));
		k0 = ( sigma0 * kEpstein + SIGMA_DUST * kStokes) * SIGMA_DUST * (rhoA + rhoB) / (DUSTMASS * (sigma0 + SIGMA_DUST));

		if(resetAccumulator) {
			tmpmem[kBlock*FLUID_SLABPITCH] = k0;
		} else {
			tmpmem[kBlock*FLUID_SLABPITCH] += k0;
		}

		gas += blockDim.x*gridDim.x;
		dust += blockDim.x*gridDim.x;
		tmpmem += blockDim.x*gridDim.x;
	}
}

/* Computes initial magnitude velocity ("w") into dv[0] and u_internal initial into dv[slabPitch]
 * and computes Uint_ini (e_internal / rho evaluated at original |w|) into dv[2*slabNumel] */
__global__ void cukern_findInitialDeltaV(double *g, double *d, double *dv, unsigned long partNumel)
{
int x = threadIdx.x + blockIdx.x*blockDim.x;
g += x;
d += x;
dv+= x;

double u, q, dvsq, rhoginv, rhodinv;
double momsq;

while(x < partNumel) {
	rhoginv = 1/g[0];
	rhodinv = 1/d[0];
	
	q = g[2*FLUID_SLABPITCH];
	u = q*rhoginv - d[2*FLUID_SLABPITCH]*rhodinv;
	momsq = q*q;
	dvsq = u*u;
	q = g[3*FLUID_SLABPITCH];
	u = q*rhoginv - d[3*FLUID_SLABPITCH]*rhodinv;
	momsq += q*q;
	dvsq += u*u;
	q = g[4*FLUID_SLABPITCH];
	u = q*rhoginv - d[4*FLUID_SLABPITCH]*rhodinv;
	momsq += q*q;
	dvsq += u*u;
	
	// Store magnitude delta-v and initial specific internal energy for use by gas drag routine
	dv[0]               = sqrt(dvsq);
	dv[FLUID_SLABPITCH] = (g[FLUID_SLABPITCH] - .5*momsq * rhoginv)*rhoginv;

	x += blockDim.x*gridDim.x;
	g += blockDim.x*gridDim.x;
	d += blockDim.x*gridDim.x;
	dv+= blockDim.x*gridDim.x;
}

}

/* This function completes evaluation of an explicit Butcher tableau.
 * the final y' stored at i gets added with weight B to the accumulator
 * The accumulator is rescaled by W, added to block 0, and overwritten
  *     tmpmem[2] += B * tmpmem[i]
  *     tmpmem[d] = tmpmem[0] + W*tmpmem[2];
  */
__global__ void cukern_SolveRK_final(double *tmpmem, int i, double B, double W, unsigned long partNumel)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	tmpmem += x;

	while(x < partNumel) {
		/* compute Y1 value */
		tmpmem[2*FLUID_SLABPITCH] = tmpmem[0] + W*(tmpmem[2*FLUID_SLABPITCH] + B*tmpmem[i*FLUID_SLABPITCH]);

		x += blockDim.x*gridDim.x;
		tmpmem += blockDim.x*gridDim.x;
	}
}

/* This function computes an explicit RK intermediate that takes one F eval
 * the new stage is computed using
 *     tmpmem[d] = tmpmem[0] + (A*tmpmem[i]
 * and the accumulator goes as
 *     tmpmem[2] += B * tmpmem[i1] */
template <bool resetAccumulator>
__global__ void cukern_SolveRK_single(double *tmpmem, int d, double A, int i, double B, unsigned long partNumel)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	tmpmem += x;

	while(x < partNumel) {
		/* compute stage value */
		tmpmem[d*FLUID_SLABPITCH]	   = tmpmem[0] + A*tmpmem[i*FLUID_SLABPITCH];
		/* compute accumulator */
		if(resetAccumulator) {
			tmpmem[2*FLUID_SLABPITCH]  = B * tmpmem[i*FLUID_SLABPITCH];
		} else {
			tmpmem[2*FLUID_SLABPITCH] += B * tmpmem[i*FLUID_SLABPITCH];
		}
		x += blockDim.x*gridDim.x;
		tmpmem += blockDim.x*gridDim.x;
	}
}

/* This function computes an explicit RK intermediate that takes two F evals
 * the new stage is computed using
 *     tmpmem[d] = tmpmem[0] + (F0 * tmpmem[i0] + F1 * tmpmem[i1]);
 * and the accumulator goes as
 *     tmpmem[2] += B * tmpmem[i1]
 * (Implicitly, F1 at i1 is the new F eval to be accumulated) */
template <bool resetAccumulator>
 __global__ void cukern_SolveRK_double(double *tmpmem, int d, double F[2], int i[2], double B, unsigned long partNumel)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	tmpmem += x;

	while(x < partNumel) {
		/* compute stage value */
		tmpmem[d*FLUID_SLABPITCH] = tmpmem[0] + (F[0]*tmpmem[i[0]*FLUID_SLABPITCH] + F[1]*tmpmem[i[1]*FLUID_SLABPITCH]);
		/* compute accumulator */
		if(resetAccumulator) {
			tmpmem[2*FLUID_SLABPITCH]  = B * tmpmem[i[1]*FLUID_SLABPITCH];
		} else {
			tmpmem[2*FLUID_SLABPITCH] += B * tmpmem[i[1]*FLUID_SLABPITCH];
		}
		x += blockDim.x*gridDim.x;
		tmpmem += blockDim.x*gridDim.x;
	}
}

/* This function computes an explicit RK intermediate that takes two F evals
 * the new stage is computed using
 *     tmpmem[d] = tmpmem[0] + sum_{i=0}^{i=2} (F[i] * tmpmem[idx[i]]);
 * and the accumulator goes as
 *     tmpmem[2] += B * tmpmem[i[2]]
 * (Implicitly, F1 at i[2] is the new F eval to be accumulated)
 */
template <bool resetAccumulator>
__global__ void cukern_SolveRK_triple(double *tmpmem, int d, double F[3], int i[3], double B, unsigned long partNumel)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	tmpmem += x;

	while(x < partNumel) {
		/* compute stage value */
		tmpmem[d*FLUID_SLABPITCH]	   = tmpmem[0] + (F[0]*tmpmem[i[0]*FLUID_SLABPITCH] +
		                                              F[1]*tmpmem[i[1]*FLUID_SLABPITCH] +
		                                              F[2]*tmpmem[i[2]*FLUID_SLABPITCH]);
		/* compute accumulator */
		if(resetAccumulator) {
			tmpmem[2*FLUID_SLABPITCH]  = B * tmpmem[i[2]*FLUID_SLABPITCH];
		} else {
			tmpmem[2*FLUID_SLABPITCH] += B * tmpmem[i[2]*FLUID_SLABPITCH];
		}
		x += blockDim.x*gridDim.x;
		tmpmem += blockDim.x*gridDim.x;
	}
}

/* From the initial momentum difference from *gas and *dust, computes the change in their momentum
 * densities to reach momentum difference *dp, given the relative fraction of acceleration
 * experienced by the gas and dust particles, and applies total energy conservation to solve
 * the gas/dust energy densities */
__global__ void cukern_applyFinalDeltaV(double *g, double *d, double *dv_final, unsigned long partNumel)
{
int x = threadIdx.x + blockIdx.x*blockDim.x;
g  += x;
d += x;
dv_final += x;

double vstick[3]; double dvhat[3]; 
double rhog, rhod;

double a, b, c, p1, p2;
double dustmom, dustmomfin;

while(x < partNumel) {
	rhog = g[0];
	rhod = d[0];

	// convert rho & momentum into CoM velocity & differential velocity
	p1 = g[2*FLUID_SLABPITCH];
	p2 = d[2*FLUID_SLABPITCH];
	vstick[0] = (p1+p2)/(rhog+rhod);
	dvhat[0] = p1/rhog - p2/rhod;
	
	p1 = g[3*FLUID_SLABPITCH];
	p2 = d[3*FLUID_SLABPITCH];
	vstick[1] = (p1+p2)/(rhog+rhod);
	dvhat[1] = p1/rhog - p2/rhod;
	
	p1 = g[4*FLUID_SLABPITCH];
	p2 = d[4*FLUID_SLABPITCH];
	vstick[2] = (p1+p2)/(rhog+rhod);
	dvhat[2] = p1/rhog - p2/rhod;

	// Compute differential velocity unit vector
	a = dv_final[2*FLUID_SLABPITCH] / sqrt(dvhat[0]*dvhat[0] + dvhat[1]*dvhat[1]+dvhat[2]*dvhat[2]);
	dvhat[0] *= a;
	dvhat[1] *= a;
	dvhat[2] *= a;
	
	// Reduced mass proves useful
	b = rhog*rhod/(rhog+rhod);

	// Accumulate initial & final dust momenta for exact energy conservation;
	// Convert CoM and decayed differential velocities back to momenta densities
	dustmom = d[2*FLUID_SLABPITCH]*d[2*FLUID_SLABPITCH];
	g[2*FLUID_SLABPITCH] = rhog*vstick[0] + dvhat[0]*b;
	d[2*FLUID_SLABPITCH] = c = rhod*vstick[0] - dvhat[0]*b;
	dustmomfin = c*c;
	
	dustmom += d[3*FLUID_SLABPITCH]*d[3*FLUID_SLABPITCH];
	g[3*FLUID_SLABPITCH] = rhog*vstick[1] + dvhat[1]*b;
	d[3*FLUID_SLABPITCH] = c = rhod*vstick[1] - dvhat[1]*b;
	dustmomfin += c*c;
	
	dustmom += d[4*FLUID_SLABPITCH]*d[4*FLUID_SLABPITCH];
	g[4*FLUID_SLABPITCH] = rhog*vstick[2] + dvhat[2]*b;
	d[4*FLUID_SLABPITCH] = c = rhod*vstick[2] - dvhat[2]*b;
	dustmomfin += c*c;
	
	// Conserve total energy to machine precision
	// d/dt (KE_gas + Eint_gas + KE_dust) = 0
	// d/dt (KE_gas + Eint_gas) = -d/dt(KE_dust)
	// Etot_gas(after) - Etot_gas(before) = -(KE_dust(after)-KE_dust(before))
	// -> Etot_gas += KE_dust(ini) - KE_dust(fin)
	g[FLUID_SLABPITCH] += .5*(dustmom - dustmomfin)/d[0];

	// FIXME - this is a hack to preserve dust "pressure" because I lack an inviscid
	// FIXME - Burgers solver or sticky-dust Godunov routine. So simply set it to a
	// FIXME - uniform low temperature
	d[FLUID_SLABPITCH] = .5*dustmomfin/d[0] + 1e-4 * d[0];

	x +=  blockDim.x*gridDim.x;
	g += blockDim.x*gridDim.x;
	d += blockDim.x*gridDim.x;
	dv_final += blockDim.x*gridDim.x;
}

}

/*
(2) [u_hf, |dv_hf|] = exponentialEulerHalf(gas_state, dust_state, k_0, P_x, P_y, P_z)
* compute time-reversed elements of dv again (memory & memory BW precious, v_i = (p_i - 2 P_i t)/rho cheap as dirt)
* solve y_i' = -k_0 y_i + a_i, a_i = - P_i / rho_gas per vector element
    * y(t) = a_i / k_0 + (y_i - a_i/k_0) exp(-k_0 t)
    * this is an L-stable method for the drag equation
* Our only interest in solving this is to re-evaluate the linear operation matrix at t_half
    * Linear matrix is diag([k_n k_n k_n]) -> require only |dv_half| to re-call gasDustDrag */
__global__ void cukern_ExponentialEulerHalf(double *gas, double *dust, double *tmpmem, double t, unsigned long partNumel)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	gas    += x;
	dust   += x;
	tmpmem += x;

	double rhoginv; // 1/rho_gas
	double dv_i;    // element of delta-v
	double k;       // drag eigenvalue
	double a0;      // element of accel = gradient(P)/rho_gas
	double dv_t;    // updated delta v. not sure if needed independently...
	double dvsq;    // accumulated (delta-v)^2
	double duint;   // accumulated drag heating

	while(x < partNumel) {
		// load k, solve driven linear system
		k      = tmpmem[0];

		// load & compute time-reversed delta-vx
		rhoginv= 1.0 / gas[0];
		a0     = tmpmem[2*FLUID_SLABPITCH];
		dv_i   = (gas[2*FLUID_SLABPITCH] + t*a0)*rhoginv - dust[2*FLUID_SLABPITCH]/dust[0];

		// compute decay of this value
		a0    *= rhoginv / k;
		dv_t   = a0 + (dv_i - a0)*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation

		// accumulate new delta-v^2
		dvsq   = dv_t*dv_t;
		// accumulate drag heating
		duint  = k*a0*a0*t - 2*a0*(dv_i - a0)*expm1(-k*t) - (dv_i - a0)*(dv_i - a0)*expm1(-2*k*t);

		// Repeat the above for the other two components
		a0     = tmpmem[3*FLUID_SLABPITCH];
		dv_i   = (gas[3*FLUID_SLABPITCH] + t*a0)*rhoginv - dust[3*FLUID_SLABPITCH]/dust[0];
		a0    *= rhoginv/k;
		dv_t   = a0 + (dv_i - a0)*exp(-t*k);
		dvsq  += dv_t*dv_t;
		duint += k*a0*a0*t - 2*a0*(dv_i - a0)*expm1(-k*t) - (dv_i - a0)*(dv_i - a0)*expm1(-2*k*t);

		a0     = tmpmem[4*FLUID_SLABPITCH];
		dv_i   = (gas[4*FLUID_SLABPITCH] + t*a0)*rhoginv - dust[3*FLUID_SLABPITCH]/dust[0];
		a0    *= rhoginv/k;
		dv_t   = a0 + (dv_i - a0)*exp(-t*k);
		dvsq  += dv_t*dv_t;
		duint += k*a0*a0*t - 2*a0*(dv_i - a0)*expm1(-k*t) - (dv_i - a0)*(dv_i - a0)*expm1(-2*k*t);


#ifdef EXPO_TRAPEZOID
		tmpmem[5*FLUID_SLABPITCH] = sqrt(dvsq); // generate outputs: store new dv in block 6, increment original temperature
#else
		tmpmem[0] = sqrt(dvsq); // overwrite in place
#endif
		tmpmem[FLUID_SLABPITCH]  += GAMMAM1 * duint * rhoginv;

		// advance ptrs
		x +=  blockDim.x*gridDim.x;
		gas += blockDim.x*gridDim.x;
		dust += blockDim.x*gridDim.x;
		tmpmem += blockDim.x*gridDim.x;
	}
}

/*(5) [(gas_state), (dust_state)] = exponentialMidpt(gas_state, dust_state, k_hf, P_x, P_y, P_z)
    * compute time-reversed elements of dv a 3rd time (memory & memory BW precious, v_i = (p_i - 2 P_i t)/rho cheap as dirt)
    * advance to drag-applied dv values dv_i <- -P_i/(k_hf rho) + (dv_i + P_i/(k_hf rho))*exp(-k_hf t)
    * compute new u_specific? or let d/dt(Etotal) = 0 do the job? does that still work?
    * overwrite gas_state/dust_state using updated values
        * ...
 */
__global__ void cukern_exponentialMidpoint(double *gas, double *dust, double t, double *tmpmem)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	gas    += x;
	dust   += x;
	tmpmem += x;

	double rhoginv; // 1/rho_gas
//	double rhodinv; // 1/rho_dust
	double dv_i;    // element of delta-v
	double k;       // drag eigenvalue
	double a0;      // element of accel = gradient(P)/rho_gas
	double dv_t;    // updated delta v. not sure if needed independently...
	double pdustsq; // use to track accumulated transfer of total energy
	double vstick;  // barycentric velocity of gas-dust system
	double mu;      // reduced mass
	double q;       // scratchpad variable

	while(x < FLUID_SLABPITCH) {
		// load & compute time-reversed delta-vx and stick velocity
		rhoginv = 1.0 / gas[0];
		mu      = gas[0]*dust[0]/(gas[0]+dust[0]);

		pdustsq = -dust[2*FLUID_SLABPITCH] * dust[2*FLUID_SLABPITCH];
		vstick  = (gas[2*FLUID_SLABPITCH]+dust[2*FLUID_SLABPITCH]) / (gas[0] + dust[0]);
		a0      = tmpmem[2*FLUID_SLABPITCH];
		dv_i    = (gas[2*FLUID_SLABPITCH] + t*a0)*rhoginv - dust[2*FLUID_SLABPITCH]/dust[0];
		// load k, solve driven linear system
		k       = tmpmem[0];

		a0     *= rhoginv;
		dv_t    = a0/k + (dv_i - a0/k)*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation

		// recalculate new differential velocities
		gas[2*FLUID_SLABPITCH] = gas[0]*vstick + dv_t*mu;
		dust[2*FLUID_SLABPITCH] = q = dust[0]*vstick - dv_t*mu;
		// accumulate change in dust kinetic energy
		pdustsq += q*q; //

		// do Y direction
		pdustsq -= dust[3*FLUID_SLABPITCH]*dust[3*FLUID_SLABPITCH];
		vstick  = (gas[3*FLUID_SLABPITCH]+dust[3*FLUID_SLABPITCH]) / (gas[0] + dust[0]);
		a0      = tmpmem[3*FLUID_SLABPITCH];
		dv_i    = (gas[3*FLUID_SLABPITCH] + t*a0)*rhoginv - dust[3*FLUID_SLABPITCH]/dust[0];
		a0     *= rhoginv;
		dv_t    = a0/k + (dv_i - a0/k)*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation
		gas[3*FLUID_SLABPITCH]     = gas[0]*vstick + dv_t*mu;
		dust[3*FLUID_SLABPITCH]= q = dust[0]*vstick - dv_t*mu;
		pdustsq += q*q;

		// do Z direction
		pdustsq -= dust[4*FLUID_SLABPITCH]*dust[4*FLUID_SLABPITCH];
		vstick  = (gas[4*FLUID_SLABPITCH]+dust[4*FLUID_SLABPITCH]) / (gas[0] + dust[0]);
		a0      = tmpmem[4*FLUID_SLABPITCH];
		dv_i    = (gas[4*FLUID_SLABPITCH] + t*a0)*rhoginv - dust[4*FLUID_SLABPITCH]/dust[0];
		a0     *= rhoginv;
		dv_t    = a0/k + (dv_i - a0/k)*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation
		gas[4*FLUID_SLABPITCH]  = gas[0]*vstick + dv_t*mu;
		dust[4*FLUID_SLABPITCH] = q = dust[0]*vstick - dv_t*mu;
		pdustsq += q*q;

		// From conservation of total energy we have that the gas total energy decreases by whatever
		// amount the dust kinetic energy rises; Under (Mdust >> M_atoms) the gas gets ~100% of heating
		gas[FLUID_SLABPITCH] -= .5*pdustsq / dust[0];

		// advance ptrs
		x +=  blockDim.x*gridDim.x;
		gas += blockDim.x*gridDim.x;
		dust += blockDim.x*gridDim.x;
		tmpmem += blockDim.x*gridDim.x;
	}
}








// This awful wad of mutated copypasta from the cudaGradientKernels.cu file provides the
// initial conditions for the ERK2 integrator to run; It computes five output values from nine
// input values

__global__ void writeScalarToVector(double *x, long numel, double f);

// compute grad(phi) in XYZ or R-Theta-Z with 2nd or 4th order accuracy
template <geometryType_t coords>
__global__ void  cukern_prepareForERK3D_h2(double *gas, double *dust, double *em, int3 arraysize);
template <geometryType_t coords>
__global__ void  cukern_prepareForERK3D_h4_partone(double *phi, double *fx, double *fy, int3 arraysize);
__global__ void  cukern_prepareForERK3D_h4_parttwo(double *phi, double *fz, int3 arraysize);

// compute grad(phi) in X-Y or R-Theta with 2nd or 4th order accuracy
template <geometryType_t coords>
__global__ void  cukern_prepareForERK2D_h2(double *gas, double *dust, double *em, int3 arraysize);
template <geometryType_t coords>
__global__ void  cukern_prepareForERK2D_h4(double *phi, double *fx, double *fy, int3 arraysize);

// Compute grad(phi) in X-Z or R-Z with 2nd or 4th order accuracy
__global__ void  cukern_prepareForERKRZ_h2(double *gas, double *dust, double *em, int3 arraysize);
__global__ void  cukern_prepareForERKRZ_h4(double *phi, double *fx, double *fz, int3 arraysize);

#define GRADBLOCKX 18
#define GRADBLOCKY 18

// scalingParameter / 2h or /12h depending on spatial order of scheme
#define LAMX devLambda[0]
#define LAMY devLambda[1]
#define LAMZ devLambda[2]

#define RINNER devLambda[7]
#define DELTAR devLambda[8]


/* Given the gas (5xMGArray), dust (5xMGArray), and temporary memory (5 regs) pointers, along with
 * geometry information, computes five outputs into the 5 temp memory slabs: [|dv_timereversed|, uinternal, dP/dx, dP/dy, dP/dz]
 * for this call, spaceOrder must be 2 (or error) and scalingParameter should be 1 (or the math is wrong).
 */
int prepareForERK2(MGArray *gas, MGArray *dust, MGArray *tempMem, GeometryParams geom, int spaceOrder, double scalingParameter)
{
	dim3 gridsize, blocksize;

	double lambda[11];

	int i;
	int worked;
	int sub[6];

	double *dx = &geom.h[0];
	if(spaceOrder == 4) {
		lambda[0] = scalingParameter/(12.0*dx[0]);
		lambda[1] = scalingParameter/(12.0*dx[1]);
		lambda[2] = scalingParameter/(12.0*dx[2]);
	} else {
		lambda[0] = scalingParameter/(2.0*dx[0]);
		lambda[1] = scalingParameter/(2.0*dx[1]);
		lambda[2] = scalingParameter/(2.0*dx[2]);
	}

	lambda[7] = geom.Rinner; // This is actually overwritten per partition below
	lambda[8] = dx[1];

	int isThreeD = (gas->dim[2] > 1);
	int isRZ = (gas->dim[2] > 1) & (gas->dim[1] == 1);

	for(i = 0; i < gas->nGPUs; i++) {
		cudaSetDevice(gas->deviceID[i]);
		calcPartitionExtent(gas, i, &sub[0]);

		lambda[7] = geom.Rinner + dx[0] * sub[0]; // Innermost cell coord may change per-partition

		cudaMemcpyToSymbol((const void *)devLambda, lambda, 11*sizeof(double), 0, cudaMemcpyHostToDevice);
		worked = CHECK_CUDA_ERROR("cudaMemcpyToSymbol");
		if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;

		//cudaMemcpyToSymbol((const void *)devIntParams, &sub[3], 3*sizeof(int), 0, cudaMemcpyHostToDevice);
		//worked = CHECK_CUDA_ERROR("memcpy to symbol");
		//if(worked != SUCCESSFUL) break;
	}

	if(worked != SUCCESSFUL) return worked;

	double *gasPtr;
	double *dustPtr;
	double *tmpPtr;
	long slabsize;

	// Iterate over all partitions, and here we GO!
	for(i = 0; i < gas->nGPUs; i++) {
		cudaSetDevice(gas->deviceID[i]);
		worked = CHECK_CUDA_ERROR("cudaSetDevice");
		if(worked != SUCCESSFUL) break;

		calcPartitionExtent(gas, i, sub);

		int3 arraysize; arraysize.x = sub[3]; arraysize.y = sub[4]; arraysize.z = sub[5];
		dim3 blocksize(GRADBLOCKX, GRADBLOCKY, 1);
		gridsize.x = arraysize.x / (blocksize.x - spaceOrder);
		gridsize.x += ((blocksize.x-spaceOrder) * gridsize.x < arraysize.x);
		if(isRZ) {
			gridsize.y = arraysize.z / (blocksize.y - spaceOrder); gridsize.y += ((blocksize.y-spaceOrder) * gridsize.y < arraysize.z);
		} else {
			gridsize.y = arraysize.y / (blocksize.y - spaceOrder); gridsize.y += ((blocksize.y-spaceOrder) * gridsize.y < arraysize.y);
		}
		gridsize.z = 1;

		gasPtr = gas->devicePtr[i]; // WARNING: this could be garbage if spaceOrder == 0 and we rx'd no potential array
		dustPtr = dust->devicePtr[i];

		tmpPtr = tempMem->devicePtr[i];
		slabsize = gas->slabPitch[i] / 8;

		switch(spaceOrder) {
		/*case 0:
			// dump zeros so as to have a technically-valid result and not cause reads of uninitialized memory
			writeScalarToVector<<<32, 256>>>(tmpPtr + 0 * slabsize, gas->partNumel[i], 0.0);
			writeScalarToVector<<<32, 256>>>(tmpPtr + 1 * slabsize, gas->partNumel[i], 0.0);
			writeScalarToVector<<<32, 256>>>(tmpPtr + 2 * slabsize, gas->partNumel[i], 0.0);
			break;*/
		case 2:
			if(isThreeD) {
				if(isRZ) {
					cukern_prepareForERKRZ_h2<<<gridsize, blocksize>>>(gasPtr, tmpPtr, tmpPtr+2*slabsize, arraysize);
				} else {
					if(geom.shape == SQUARE) {
						cukern_prepareForERK3D_h2<SQUARE><<<gridsize, blocksize>>> (gasPtr, dustPtr, tmpPtr, arraysize); }
					if(geom.shape == CYLINDRICAL) {
						cukern_prepareForERK3D_h2<CYLINDRICAL><<<gridsize, blocksize>>> (gasPtr, dustPtr, tmpPtr, arraysize); }
				}
			} else {
				if(geom.shape == SQUARE) {
					cukern_prepareForERK2D_h2<SQUARE><<<gridsize, blocksize>>>(gasPtr, dustPtr, tmpPtr, arraysize); }
				if(geom.shape == CYLINDRICAL) {
					cukern_prepareForERK2D_h2<CYLINDRICAL><<<gridsize, blocksize>>>(gasPtr, dustPtr, tmpPtr, arraysize); }
			}
			break;
		/*case 4:
			if(isThreeD) {
				if(isRZ) {
					cukern_prepareForERKRZ_h4<<<gridsize, blocksize>>>(gasPtr, tmpPtr, tmpPtr + 2*gas->partNumel[i],  arraysize);
					writeScalarToVector<<<32, 256>>>(tmpPtr + slabsize, gas->partNumel[i], 0.0);
				} else {
					if(geom.shape == SQUARE) {
						cukern_prepareForERK3D_h4_partone<SQUARE><<<gridsize, blocksize>>>(gasPtr, tmpPtr, tmpPtr+ slabsize, arraysize);
						cukern_prepareForERK3D_h4_parttwo<<<gridsize, blocksize>>>(gasPtr, tmpPtr+ slabsize*2, arraysize);
					}
					if(geom.shape == CYLINDRICAL) {
						cukern_prepareForERK3D_h4_partone<CYLINDRICAL><<<gridsize, blocksize>>>(gasPtr, tmpPtr, tmpPtr+ slabsize, arraysize);
						cukern_prepareForERK3D_h4_parttwo<<<gridsize, blocksize>>>(gasPtr, tmpPtr+ slabsize*2, arraysize);
					}
				}
			} else {
				if(geom.shape == SQUARE) {
					cukern_prepareForERK2D_h4<SQUARE><<<gridsize, blocksize>>>(gasPtr, tmpPtr, tmpPtr+ slabsize, arraysize); }
				if(geom.shape == CYLINDRICAL) {
					cukern_prepareForERK2D_h4<CYLINDRICAL><<<gridsize, blocksize>>>(gasPtr, tmpPtr, tmpPtr+ slabsize, arraysize); }

				writeScalarToVector<<<32, 256>>>(tmpPtr+2*gas->partNumel[i], gas->partNumel[i], 0.0);

			}

			break;*/
		default:
			PRINT_FAULT_HEADER;
			printf("Was passed spatial order parameter of %i, must be passed 2 (2nd order)\n", spaceOrder);
			PRINT_FAULT_FOOTER;
			return ERROR_INVALID_ARGS;
		}

		worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_prepareForERK");
		if(worked != SUCCESSFUL) break;
	}

	if(worked != SUCCESSFUL) return worked;

	// FIXME this needs to either understand slabs, or we need to fetch 3 slab ptrs into an array & pass it instead
	//    worked = MGA_exchangeLocalHalos(gradient, 5); // need to?
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) return worked;

	return CHECK_IMOGEN_ERROR(worked);

}

// Needed with the gradient calculators in 2D because they leave the empty directions uninitialized
// Vomits the value f into array x, from x[0] to x[numel-1]
__global__ void writeScalarToVector(double *x, long numel, double f)
{
	long a = threadIdx.x + blockDim.x*blockIdx.x;

	for(; a < numel; a+= blockDim.x*gridDim.x) {
		x[a] = f;

	}

}

/* todo: fantasy - pull this into the copypasta & eliminate global reads */
__device__ double gas2press(double *g)
{
	return (g[FLUID_SLABPITCH]-.5*(g[2*FLUID_SLABPITCH]*g[2*FLUID_SLABPITCH]+g[3*FLUID_SLABPITCH]*g[3*FLUID_SLABPITCH]+g[4*FLUID_SLABPITCH]*g[4*FLUID_SLABPITCH])/g[0]);
}

/* Algorithm:
 *     [|dv_tr|, u_0, P_x, P_y, P_z] = exponentialSetup(gas_state, dust_state)
 * 5 output registers
 * may need slope limiter on gradient calculation? */
template <geometryType_t coords>
__global__ void  cukern_prepareForERK3D_h2(double *gas, double *dust, double *em, int3 arraysize)
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

	double deltaP; // Store derivative of phi in one direction
	double dv, dvsq;

	__shared__ double phiA[GRADBLOCKX*GRADBLOCKY];
	__shared__ double phiB[GRADBLOCKX*GRADBLOCKY];
	__shared__ double phiC[GRADBLOCKX*GRADBLOCKY];

	double *U; double *V; double *W;
	double *temp;

	U = phiA; V = phiB; W = phiC;

	// compute P on lower plane
	U[myLocAddr] = gas2press(gas + (globAddr + arraysize.x*arraysize.y*(arraysize.z-1)));
	V[myLocAddr] = gas2press(gas + globAddr);

	__syncthreads();

	int z;
	int deltaz = arraysize.x*arraysize.y;
	for(z = 0; z < arraysize.z; z++) {
		if(z >= arraysize.z - 1) deltaz = - arraysize.x*arraysize.y*(arraysize.z-1);

		if(IWrite) {
			deltaP         = LAMX*(V[myLocAddr+1]-V[myLocAddr-1]);
			em[globAddr + 2*FLUID_SLABPITCH] = deltaP;

			// need time-reversed dv = (vgas - vdust) + t*(deltaP / rho)
			//                       = ((pgas + t deltaP)/rhogas - pdust/rhodust
			dv = (gas[globAddr+2*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+2*FLUID_SLABPITCH]/dust[globAddr];
			dvsq = dv*dv; // accumulate |dv_tr|
		}

		if(IWrite) {
			if(coords == SQUARE) {
				deltaP         = LAMY*(V[myLocAddr+GRADBLOCKX]-V[myLocAddr-GRADBLOCKX]);
			}
			if(coords == CYLINDRICAL) {
				// In cylindrical coords, use dt/dphi * (delta-phi) / r to get d/dy
				deltaP         = LAMY*(V[myLocAddr+GRADBLOCKX]-V[myLocAddr-GRADBLOCKX]) / (RINNER + DELTAR*myX);
			}
			em[globAddr + 3*FLUID_SLABPITCH] = deltaP;
			dv = (gas[globAddr+3*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+3*FLUID_SLABPITCH]/dust[globAddr];
			dvsq += dv*dv;
		}

		W[myLocAddr]       = gas2press(gas + (globAddr + deltaz));

		__syncthreads();

		if(IWrite) {
			deltaP           = LAMZ*(W[myLocAddr] - U[myLocAddr]);
			em[globAddr + 4*FLUID_SLABPITCH] = deltaP;
			dv = (gas[globAddr+4*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+4*FLUID_SLABPITCH]/dust[globAddr];
			dvsq += dv*dv;
		}

		em[globAddr] = sqrt(dvsq); // output initial delta-v
		em[globAddr + FLUID_SLABPITCH] = V[myLocAddr] / gas[globAddr]; // specific internal energy for

		temp = U; U = V; V = W; W = temp; // cyclically shift them back
		globAddr += arraysize.x * arraysize.y;

	}

}

/* Computes the gradient of 3d array phi using the 4-point centered derivative and
 * stores phi_x in fx, phi_y in fy, phi_z in fz.
 * All arrays (rho, phi, fx, fy, fz) must be of size arraysize.
 * In cylindrical geometry, f_x -> f_r,
 *                          f_y -> f_phi
 * This call must be invoked in two parts:
 * cukern_prepareForERK3D_h4_partone computes the X and Y (or r/theta) derivatives,
 * cukern_prepareForERK3D_h4_parttwo computes the Z derivative.
 */
template <geometryType_t coords>
__global__ void  cukern_prepareForERK3D_h4_partone(double *phi, double *fx, double *fy, int3 arraysize)
{
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	int myX = threadIdx.x + (GRADBLOCKX-4)*blockIdx.x - 2;
	int myY = threadIdx.y + (GRADBLOCKY-4)*blockIdx.y - 2;

	if((myX > (arraysize.x+1)) || (myY > (arraysize.y+1))) return;

	bool IWrite = (threadIdx.x > 1) && (threadIdx.x < (GRADBLOCKX-2)) && (threadIdx.y > 1) && (threadIdx.y < (GRADBLOCKY-2));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.y);

	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.y) % arraysize.y;

	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction

	__shared__ double phishm[GRADBLOCKX*GRADBLOCKY];

	__syncthreads();

	int z;
	int deltaz = arraysize.x*arraysize.y;
	for(z = 0; z < arraysize.z; z++) {
		phishm[myLocAddr] = phi[globAddr];

		__syncthreads();

		if(IWrite) {
			deltaphi         = LAMX*(-phishm[myLocAddr+2]+8.0*phishm[myLocAddr+1]-8.0*phishm[myLocAddr-1]+phishm[myLocAddr-2]);
			fx[globAddr]     = deltaphi; // store px <- px - dt * rho dphi/dx;

			if(coords == SQUARE) {
				deltaphi         = LAMY*(-phishm[myLocAddr+2*GRADBLOCKX]+8*phishm[myLocAddr+GRADBLOCKX]-8*phishm[myLocAddr-GRADBLOCKX]+phishm[myLocAddr-2*GRADBLOCKX]);
			}
			if(coords == CYLINDRICAL) {
				// In cylindrical coords, use dt/dphi * (delta-phi) / r to get d/dy
				deltaphi         = LAMY*(-phishm[myLocAddr+2*GRADBLOCKX]+8*phishm[myLocAddr+GRADBLOCKX]-8*phishm[myLocAddr-GRADBLOCKX]+phishm[myLocAddr-2*GRADBLOCKX]) / (RINNER + DELTAR*myX);
			}
			fy[globAddr]     = deltaphi;
		}

		globAddr += deltaz;
	}
}

/* 2nd part of 4th order 3D spatial gradient computes d/dz (same in cart & cyl coords so no template */
__global__ void  cukern_prepareForERK3D_h4_parttwo(double *phi, double *fz, int3 arraysize)
{
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	int myX = threadIdx.x + (GRADBLOCKX-4)*blockIdx.x - 2;
	int myZ = threadIdx.y + (GRADBLOCKY-4)*blockIdx.y - 2;

	if((myX > (arraysize.x+1)) || (myZ > (arraysize.z+1))) return;

	bool IWrite = (threadIdx.x > 1) && (threadIdx.x < (GRADBLOCKX-2)) && (threadIdx.y > 1) && (threadIdx.y < (GRADBLOCKY-2));
	IWrite = IWrite && (myX < arraysize.x) && (myZ < arraysize.z);

	myX = (myX + arraysize.x) % arraysize.x;
	myZ = (myZ + arraysize.z) % arraysize.z;

	int delta = arraysize.x*arraysize.y;

	int globAddr = myX + delta*myZ;

	double deltaphi; // Store derivative of phi in one direction

	__shared__ double phishm[GRADBLOCKX*GRADBLOCKY];

	__syncthreads();

	int y;
	for(y = 0; y < arraysize.y; y++) {
		phishm[myLocAddr] = phi[globAddr];

		if(IWrite) {
			deltaphi         = LAMZ*(-phishm[myLocAddr+2*GRADBLOCKX]+8*phishm[myLocAddr+GRADBLOCKX]-8*phishm[myLocAddr-GRADBLOCKX]+phishm[myLocAddr-2*GRADBLOCKX]);
			fz[globAddr]     = deltaphi;
		}
		globAddr += arraysize.x;
	}
}

/* Compute the gradient of 2d array phi with 2nd order accuracy; store the results in f_x, f_y
 *    In cylindrical geometry, f_x -> f_r,
 *                             f_y -> f_phi
 */
template <geometryType_t coords>
__global__ void  cukern_prepareForERK2D_h2(double *gas, double *dust, double *em, int3 arraysize)
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

	double deltaP; // Store derivative of phi in one direction
	double dv, dvsq;
	__shared__ double locPress[GRADBLOCKX*GRADBLOCKY];

	locPress[myLocAddr] = gas2press(gas+globAddr);

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		// compute dt * (dphi/dx)
		deltaP         = LAMX*(locPress[myLocAddr+1]-locPress[myLocAddr-1]);
		em[globAddr+2*FLUID_SLABPITCH] = deltaP;

		dv = (gas[globAddr+4*FLUID_SLABPITCH])/gas[globAddr] - dust[globAddr+4*FLUID_SLABPITCH]/dust[globAddr];
		dvsq = dv*dv;

		dv = (gas[globAddr+2*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+2*FLUID_SLABPITCH]/dust[globAddr];
		dvsq += dv*dv;
		// Calculate dt*(dphi/dy)
		if(coords == SQUARE) {
			deltaP         = LAMY*(locPress[myLocAddr+GRADBLOCKX]-locPress[myLocAddr-GRADBLOCKX]);
		}
		if(coords == CYLINDRICAL) {
			// Converts d/dphi into physical distance based on R
			deltaP         = LAMY*(locPress[myLocAddr+GRADBLOCKX]-locPress[myLocAddr-GRADBLOCKX]) / (RINNER + myX*DELTAR);
		}
		em[globAddr+3*FLUID_SLABPITCH] = deltaP;
		em[globAddr+4*FLUID_SLABPITCH] = 0.0; // FIXME is this needed?... yes it is I think, solver blindly reads all 3 dmis.

		dv = (gas[globAddr+3*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+3*FLUID_SLABPITCH]/dust[globAddr];
		dvsq += dv*dv;

		em[globAddr] = sqrt(dvsq);
		em[globAddr + FLUID_SLABPITCH] = locPress[myLocAddr] / gas[globAddr]; // specific internal energy for

	}

}

/* Compute the gradient of 2d array phi with 4th order accuracy; store the results in f_x, f_y
 *    In cylindrical geometry, f_x -> f_r,
 *                             f_y -> f_phi
 */
template <geometryType_t coords>
__global__ void  cukern_prepareForERK2D_h4(double *phi, double *fx, double *fy, int3 arraysize)
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

/* Compute the gradient of R-Z array phi with 2nd order accuracy; store the results in f_x, f_z
 *    In cylindrical geometry, f_x -> f_r
 */
__global__ void  cukern_prepareForERKRZ_h2(double *gas, double *dust, double *em, int3 arraysize)
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

	double deltaP, dv, dvsq; // Store derivative of phi in one direction
	__shared__ double pressLoc[GRADBLOCKX*GRADBLOCKY];

	pressLoc[myLocAddr] = gas2press(gas + globAddr);

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		em[globAddr + 3*FLUID_SLABPITCH] = 0.0; // zero phi gradient
		// compute v_phi contribution to |dv|^2 for 2.5-D
		dv = (gas[globAddr+3*FLUID_SLABPITCH])/gas[globAddr] - dust[globAddr+3*FLUID_SLABPITCH]/dust[globAddr];
		dvsq = dv*dv;

		// compute dt * (dphi/dx)
		deltaP         = LAMX*(pressLoc[myLocAddr+1]-pressLoc[myLocAddr-1]);
		em[globAddr + 2*FLUID_SLABPITCH] = deltaP;

		dv = (gas[globAddr+2*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+2*FLUID_SLABPITCH]/dust[globAddr];
		dvsq += dv*dv;

		// Calculate dt*(dphi/dz)
		deltaP         = LAMZ*(pressLoc[myLocAddr+GRADBLOCKX]-pressLoc[myLocAddr-GRADBLOCKX]);
		em[globAddr + 4*FLUID_SLABPITCH] = deltaP;
		dv = (gas[globAddr+4*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+4*FLUID_SLABPITCH]/dust[globAddr];
		dvsq += dv*dv;

		em[globAddr] = sqrt(dvsq); // magnitude delta-v with time reversed pressure gradient
		em[globAddr + FLUID_SLABPITCH] =  pressLoc[myLocAddr] / gas[globAddr]; // specific internal energy for
	}

}

/* Compute the gradient of RZ array phi with 4th order accuracy; store the results in f_x, f_y
 *    In cylindrical geometry, f_x -> f_r,
 */
__global__ void  cukern_prepareForERKRZ_h4(double *phi, double *fx, double *fz, int3 arraysize)
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
		fx[globAddr]     = deltaphi;

		// Calculate dt*(dphi/dz)
		deltaphi         = LAMZ*(-phiLoc[myLocAddr+2*GRADBLOCKX] + 8*phiLoc[myLocAddr+1*GRADBLOCKX] - 8*phiLoc[myLocAddr-1*GRADBLOCKX] + phiLoc[myLocAddr-2*GRADBLOCKX]);
		fz[globAddr]     = deltaphi;
	}

}























// This awful wad of mutated copypasta from the cudaGradientKernels.cu file turns the
// midpoint internal energy density and the mass density into pressure & computes the
// gradient for the ERK solver's second stage

// compute grad(phi) in XYZ or R-Theta-Z with 2nd or 4th order accuracy
template <geometryType_t coords>
__global__ void  cukern_findMidGradP3D_h2(double *gas, double *em, int3 arraysize);
template <geometryType_t coords>
__global__ void  cukern_findMidGradP3D_h4_partone(double *phi, double *fx, double *fy, int3 arraysize);
__global__ void  cukern_findMidGradP3D_h4_parttwo(double *phi, double *fz, int3 arraysize);

// compute grad(phi) in X-Y or R-Theta with 2nd or 4th order accuracy
template <geometryType_t coords>
__global__ void  cukern_findMidGradP2D_h2(double *gas, double *em, int3 arraysize);
template <geometryType_t coords>
__global__ void  cukern_findMidGradP2D_h4(double *phi, double *fx, double *fy, int3 arraysize);

// Compute grad(phi) in X-Z or R-Z with 2nd or 4th order accuracy
__global__ void  cukern_findMidGradPRZ_h2(double *gas, double *em, int3 arraysize);
__global__ void  cukern_findMidGradPRZ_h4(double *phi, double *fx, double *fz, int3 arraysize);

#define GRADBLOCKX 18
#define GRADBLOCKY 18

// scalingParameter / 2h or /12h depending on spatial order of scheme
#define LAMX devLambda[0]
#define LAMY devLambda[1]
#define LAMZ devLambda[2]

#define RINNER devLambda[7]
#define DELTAR devLambda[8]


/* Given the gas pointer, temp memory and geometry, uses the midpoint specific internal energy density from tempMem
 * and the gas mass density to compute the pressure gradient into tempMem slabs 2 through 4. scalingParameter needs
 * to be (gamma-1) to convert rho * u_specific = e_internal = P / (gamma-1) to P. */
int findMidGradP2(MGArray *gas, MGArray *tempMem, GeometryParams geom, int spaceOrder, double scalingParameter)
{
	dim3 gridsize, blocksize;

	double lambda[11];

	int i;
	int worked;
	int sub[6];

	double *dx = &geom.h[0];
	if(spaceOrder == 4) {
		lambda[0] = scalingParameter/(12.0*dx[0]);
		lambda[1] = scalingParameter/(12.0*dx[1]);
		lambda[2] = scalingParameter/(12.0*dx[2]);
	} else {
		lambda[0] = scalingParameter/(2.0*dx[0]);
		lambda[1] = scalingParameter/(2.0*dx[1]);
		lambda[2] = scalingParameter/(2.0*dx[2]);
	}

	lambda[7] = geom.Rinner; // This is actually overwritten per partition below
	lambda[8] = dx[1];

	int isThreeD = (gas->dim[2] > 1);
	int isRZ = (gas->dim[2] > 1) & (gas->dim[1] == 1);

	for(i = 0; i < gas->nGPUs; i++) {
		cudaSetDevice(gas->deviceID[i]);
		calcPartitionExtent(gas, i, &sub[0]);

		lambda[7] = geom.Rinner + dx[0] * sub[0]; // Innermost cell coord may change per-partition

		cudaMemcpyToSymbol((const void *)devLambda, lambda, 11*sizeof(double), 0, cudaMemcpyHostToDevice);
		worked = CHECK_CUDA_ERROR("cudaMemcpyToSymbol");
		if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;

		//cudaMemcpyToSymbol((const void *)devIntParams, &sub[3], 3*sizeof(int), 0, cudaMemcpyHostToDevice);
		//worked = CHECK_CUDA_ERROR("memcpy to symbol");
		//if(worked != SUCCESSFUL) break;
	}

	if(worked != SUCCESSFUL) return worked;

	double *gasPtr;
	double *tmpPtr;

	// Iterate over all partitions, and here we GO!
	for(i = 0; i < gas->nGPUs; i++) {
		cudaSetDevice(gas->deviceID[i]);
		worked = CHECK_CUDA_ERROR("cudaSetDevice");
		if(worked != SUCCESSFUL) break;

		calcPartitionExtent(gas, i, sub);

		int3 arraysize; arraysize.x = sub[3]; arraysize.y = sub[4]; arraysize.z = sub[5];
		dim3 blocksize(GRADBLOCKX, GRADBLOCKY, 1);
		gridsize.x = arraysize.x / (blocksize.x - spaceOrder);
		gridsize.x += ((blocksize.x-spaceOrder) * gridsize.x < arraysize.x);
		if(isRZ) {
			gridsize.y = arraysize.z / (blocksize.y - spaceOrder); gridsize.y += ((blocksize.y-spaceOrder) * gridsize.y < arraysize.z);
		} else {
			gridsize.y = arraysize.y / (blocksize.y - spaceOrder); gridsize.y += ((blocksize.y-spaceOrder) * gridsize.y < arraysize.y);
		}
		gridsize.z = 1;

		gasPtr = gas->devicePtr[i]; // WARNING: this could be garbage if spaceOrder == 0 and we rx'd no potential array

		tmpPtr = tempMem->devicePtr[i];

		switch(spaceOrder) {
		/*case 0:
			// dump zeros so as to have a technically-valid result and not cause reads of uninitialized memory
			writeScalarToVector<<<32, 256>>>(tmpPtr + 0 * slabsize, gas->partNumel[i], 0.0);
			writeScalarToVector<<<32, 256>>>(tmpPtr + 1 * slabsize, gas->partNumel[i], 0.0);
			writeScalarToVector<<<32, 256>>>(tmpPtr + 2 * slabsize, gas->partNumel[i], 0.0);
			break;*/
		case 2:
			if(isThreeD) {
				if(isRZ) {
					cukern_findMidGradPRZ_h2<<<gridsize, blocksize>>>(gasPtr, tmpPtr, arraysize);
				} else {
					if(geom.shape == SQUARE) {
						cukern_findMidGradP3D_h2<SQUARE><<<gridsize, blocksize>>> (gasPtr, tmpPtr, arraysize); }
					if(geom.shape == CYLINDRICAL) {
						cukern_findMidGradP3D_h2<CYLINDRICAL><<<gridsize, blocksize>>> (gasPtr, tmpPtr, arraysize); }
				}
			} else {
				if(geom.shape == SQUARE) {
					cukern_findMidGradP2D_h2<SQUARE><<<gridsize, blocksize>>>(gasPtr, tmpPtr, arraysize); }
				if(geom.shape == CYLINDRICAL) {
					cukern_findMidGradP2D_h2<CYLINDRICAL><<<gridsize, blocksize>>>(gasPtr, tmpPtr, arraysize); }
			}
			break;
		/*case 4:
			if(isThreeD) {
				if(isRZ) {
					cukern_findMidGradPRZ_h4<<<gridsize, blocksize>>>(gasPtr, tmpPtr, tmpPtr + 2*gas->partNumel[i],  arraysize);
					writeScalarToVector<<<32, 256>>>(tmpPtr + slabsize, gas->partNumel[i], 0.0);
				} else {
					if(geom.shape == SQUARE) {
						cukern_findMidGradP3D_h4_partone<SQUARE><<<gridsize, blocksize>>>(gasPtr, tmpPtr, tmpPtr+ slabsize, arraysize);
						cukern_findMidGradP3D_h4_parttwo<<<gridsize, blocksize>>>(gasPtr, tmpPtr+ slabsize*2, arraysize);
					}
					if(geom.shape == CYLINDRICAL) {
						cukern_findMidGradP3D_h4_partone<CYLINDRICAL><<<gridsize, blocksize>>>(gasPtr, tmpPtr, tmpPtr+ slabsize, arraysize);
						cukern_findMidGradP3D_h4_parttwo<<<gridsize, blocksize>>>(gasPtr, tmpPtr+ slabsize*2, arraysize);
					}
				}
			} else {
				if(geom.shape == SQUARE) {
					cukern_findMidGradP2D_h4<SQUARE><<<gridsize, blocksize>>>(gasPtr, tmpPtr, tmpPtr+ slabsize, arraysize); }
				if(geom.shape == CYLINDRICAL) {
					cukern_findMidGradP2D_h4<CYLINDRICAL><<<gridsize, blocksize>>>(gasPtr, tmpPtr, tmpPtr+ slabsize, arraysize); }

				writeScalarToVector<<<32, 256>>>(tmpPtr+2*gas->partNumel[i], gas->partNumel[i], 0.0);

			}

			break;*/
		default:
			PRINT_FAULT_HEADER;
			printf("Was passed spatial order parameter of %i, must be passed 2 (2nd order)\n", spaceOrder);
			PRINT_FAULT_FOOTER;
			return ERROR_INVALID_ARGS;
		}

		worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_findMidGradP");
		if(worked != SUCCESSFUL) break;
	}

	if(worked != SUCCESSFUL) return worked;

	// FIXME this needs to either understand slabs, or we need to fetch 3 slab ptrs into an array & pass it instead
	//    worked = MGA_exchangeLocalHalos(gradient, 5); // need to?
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) return worked;

	return CHECK_IMOGEN_ERROR(worked);

}


/* Algorithm:
 *     [|dv_tr|, u_0, P_x, P_y, P_z] = exponentialSetup(gas_state, dust_state)
 * 5 output registers
 * may need slope limiter on gradient calculation? */
template <geometryType_t coords>
__global__ void  cukern_findMidGradP3D_h2(double *gas, double *em, int3 arraysize)
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

	double deltaP; // Store derivative of phi in one direction

	__shared__ double phiA[GRADBLOCKX*GRADBLOCKY];
	__shared__ double phiB[GRADBLOCKX*GRADBLOCKY];
	__shared__ double phiC[GRADBLOCKX*GRADBLOCKY];

	double *U; double *V; double *W;
	double *temp;

	U = phiA; V = phiB; W = phiC;

	// compute epsilon_internal on lower & current planes
	U[myLocAddr] = gas[globAddr + arraysize.x*arraysize.y*(arraysize.z-1)] * em[globAddr + arraysize.x*arraysize.y*(arraysize.z-1) + FLUID_SLABPITCH];
	V[myLocAddr] = gas[globAddr] * em[globAddr + FLUID_SLABPITCH];

	__syncthreads();

	int z;
	int deltaz = arraysize.x*arraysize.y;
	for(z = 0; z < arraysize.z; z++) {
		if(z >= arraysize.z - 1) deltaz = - arraysize.x*arraysize.y*(arraysize.z-1);

		if(IWrite) {
			deltaP         = LAMX*(V[myLocAddr+1]-V[myLocAddr-1]);
#ifdef EXPO_TRAPEZOID
			em[globAddr + 2*FLUID_SLABPITCH] = .5*(em[globAddr + 2*FLUID_SLABPITCH] + deltaP);
#else
			em[globAddr + 2*FLUID_SLABPITCH] = deltaP;
#endif
		}

		if(IWrite) {
			if(coords == SQUARE) {
				deltaP         = LAMY*(V[myLocAddr+GRADBLOCKX]-V[myLocAddr-GRADBLOCKX]);
			}
			if(coords == CYLINDRICAL) {
				// In cylindrical coords, use dt/dphi * (delta-phi) / r to get d/dy
				deltaP         = LAMY*(V[myLocAddr+GRADBLOCKX]-V[myLocAddr-GRADBLOCKX]) / (RINNER + DELTAR*myX);
			}
#ifdef EXPO_TRAPEZOID
			em[globAddr + 3*FLUID_SLABPITCH] = .5*(em[globAddr + 3*FLUID_SLABPITCH] + deltaP);
#else
			em[globAddr + 3*FLUID_SLABPITCH] = deltaP;
#endif
		}

		W[myLocAddr]       = gas[globAddr + deltaz] * em[globAddr + deltaz + FLUID_SLABPITCH];

		__syncthreads();

		if(IWrite) {
			deltaP           = LAMZ*(W[myLocAddr] - U[myLocAddr]);
#ifdef EXPO_TRAPEZOID
			em[globAddr + 4*FLUID_SLABPITCH] = .5*(em[globAddr + 4*FLUID_SLABPITCH] + deltaP);
#else
			em[globAddr + 4*FLUID_SLABPITCH] = deltaP;
#endif
		}

		temp = U; U = V; V = W; W = temp; // cyclically shift them back
		globAddr += arraysize.x * arraysize.y;

	}

}

/* Computes the gradient of 3d array phi using the 4-point centered derivative and
 * stores phi_x in fx, phi_y in fy, phi_z in fz.
 * All arrays (rho, phi, fx, fy, fz) must be of size arraysize.
 * In cylindrical geometry, f_x -> f_r,
 *                          f_y -> f_phi
 * This call must be invoked in two parts:
 * cukern_findMidGradP3D_h4_partone computes the X and Y (or r/theta) derivatives,
 * cukern_findMidGradP3D_h4_parttwo computes the Z derivative.
 */
template <geometryType_t coords>
__global__ void  cukern_findMidGradP3D_h4_partone(double *phi, double *fx, double *fy, int3 arraysize)
{
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	int myX = threadIdx.x + (GRADBLOCKX-4)*blockIdx.x - 2;
	int myY = threadIdx.y + (GRADBLOCKY-4)*blockIdx.y - 2;

	if((myX > (arraysize.x+1)) || (myY > (arraysize.y+1))) return;

	bool IWrite = (threadIdx.x > 1) && (threadIdx.x < (GRADBLOCKX-2)) && (threadIdx.y > 1) && (threadIdx.y < (GRADBLOCKY-2));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.y);

	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.y) % arraysize.y;

	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction

	__shared__ double phishm[GRADBLOCKX*GRADBLOCKY];

	__syncthreads();

	int z;
	int deltaz = arraysize.x*arraysize.y;
	for(z = 0; z < arraysize.z; z++) {
		phishm[myLocAddr] = phi[globAddr];

		__syncthreads();

		if(IWrite) {
			deltaphi         = LAMX*(-phishm[myLocAddr+2]+8.0*phishm[myLocAddr+1]-8.0*phishm[myLocAddr-1]+phishm[myLocAddr-2]);
			fx[globAddr]     = deltaphi; // store px <- px - dt * rho dphi/dx;

			if(coords == SQUARE) {
				deltaphi         = LAMY*(-phishm[myLocAddr+2*GRADBLOCKX]+8*phishm[myLocAddr+GRADBLOCKX]-8*phishm[myLocAddr-GRADBLOCKX]+phishm[myLocAddr-2*GRADBLOCKX]);
			}
			if(coords == CYLINDRICAL) {
				// In cylindrical coords, use dt/dphi * (delta-phi) / r to get d/dy
				deltaphi         = LAMY*(-phishm[myLocAddr+2*GRADBLOCKX]+8*phishm[myLocAddr+GRADBLOCKX]-8*phishm[myLocAddr-GRADBLOCKX]+phishm[myLocAddr-2*GRADBLOCKX]) / (RINNER + DELTAR*myX);
			}
			fy[globAddr]     = deltaphi;
		}

		globAddr += deltaz;
	}
}

/* 2nd part of 4th order 3D spatial gradient computes d/dz (same in cart & cyl coords so no template */
__global__ void  cukern_findMidGradP3D_h4_parttwo(double *phi, double *fz, int3 arraysize)
{
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	int myX = threadIdx.x + (GRADBLOCKX-4)*blockIdx.x - 2;
	int myZ = threadIdx.y + (GRADBLOCKY-4)*blockIdx.y - 2;

	if((myX > (arraysize.x+1)) || (myZ > (arraysize.z+1))) return;

	bool IWrite = (threadIdx.x > 1) && (threadIdx.x < (GRADBLOCKX-2)) && (threadIdx.y > 1) && (threadIdx.y < (GRADBLOCKY-2));
	IWrite = IWrite && (myX < arraysize.x) && (myZ < arraysize.z);

	myX = (myX + arraysize.x) % arraysize.x;
	myZ = (myZ + arraysize.z) % arraysize.z;

	int delta = arraysize.x*arraysize.y;

	int globAddr = myX + delta*myZ;

	double deltaphi; // Store derivative of phi in one direction

	__shared__ double phishm[GRADBLOCKX*GRADBLOCKY];

	__syncthreads();

	int y;
	for(y = 0; y < arraysize.y; y++) {
		phishm[myLocAddr] = phi[globAddr];

		if(IWrite) {
			deltaphi         = LAMZ*(-phishm[myLocAddr+2*GRADBLOCKX]+8*phishm[myLocAddr+GRADBLOCKX]-8*phishm[myLocAddr-GRADBLOCKX]+phishm[myLocAddr-2*GRADBLOCKX]);
			fz[globAddr]     = deltaphi;
		}
		globAddr += arraysize.x;
	}
}

/* Compute the gradient of 2d array phi with 2nd order accuracy; store the results in f_x, f_y
 *    In cylindrical geometry, f_x -> f_r,
 *                             f_y -> f_phi
 */
template <geometryType_t coords>
__global__ void  cukern_findMidGradP2D_h2(double *gas, double *em, int3 arraysize)
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

	double deltaP; // Store derivative of phi in one direction
	__shared__ double locPress[GRADBLOCKX*GRADBLOCKY];

	locPress[myLocAddr] = gas[globAddr] * em[globAddr + FLUID_SLABPITCH];

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		// compute dt * (dphi/dx)
		deltaP         = LAMX*(locPress[myLocAddr+1]-locPress[myLocAddr-1]);
#ifdef EXPO_TRAPEZOID
		em[globAddr+2*FLUID_SLABPITCH] = .5*(em[globAddr+2*FLUID_SLABPITCH] + deltaP);
#else
		em[globAddr+2*FLUID_SLABPITCH] = deltaP;
#endif

		if(coords == SQUARE) {
			deltaP         = LAMY*(locPress[myLocAddr+GRADBLOCKX]-locPress[myLocAddr-GRADBLOCKX]);
		}
		if(coords == CYLINDRICAL) {
			// Converts d/dphi into physical distance based on R
			deltaP         = LAMY*(locPress[myLocAddr+GRADBLOCKX]-locPress[myLocAddr-GRADBLOCKX]) / (RINNER + myX*DELTAR);
		}
#ifdef EXPO_TRAPEZOID
		em[globAddr+3*FLUID_SLABPITCH] = .5*(em[globAddr+3*FLUID_SLABPITCH] + deltaP);
#else
		em[globAddr+3*FLUID_SLABPITCH] = deltaP;
#endif

		em[globAddr+4*FLUID_SLABPITCH] = 0.0;
	}

}

/* Compute the gradient of 2d array phi with 4th order accuracy; store the results in f_x, f_y
 *    In cylindrical geometry, f_x -> f_r,
 *                             f_y -> f_phi
 */
template <geometryType_t coords>
__global__ void  cukern_findMidGradP2D_h4(double *phi, double *fx, double *fy, int3 arraysize)
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

/* Compute the gradient of R-Z array phi with 2nd order accuracy; store the results in f_x, f_z
 *    In cylindrical geometry, f_x -> f_r
 */
__global__ void  cukern_findMidGradPRZ_h2(double *gas, double *em, int3 arraysize)
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

	double deltaP; // Store derivative of phi in one direction
	__shared__ double pressLoc[GRADBLOCKX*GRADBLOCKY];

	pressLoc[myLocAddr] = gas[globAddr] * em[globAddr + FLUID_SLABPITCH];

	__syncthreads(); // Make sure loaded P is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		em[globAddr + 3*FLUID_SLABPITCH] = 0.0; // zero phi gradient
		// compute dP/dr
		deltaP         = LAMX*(pressLoc[myLocAddr+1]-pressLoc[myLocAddr-1]);
#ifdef EXPO_TRAPEZOID
		em[globAddr+2*FLUID_SLABPITCH] = .5*(em[globAddr+2*FLUID_SLABPITCH] + deltaP);
#else
		em[globAddr+2*FLUID_SLABPITCH] = deltaP;
#endif

		// Calculate dP/dz
		deltaP         = LAMZ*(pressLoc[myLocAddr+GRADBLOCKX]-pressLoc[myLocAddr-GRADBLOCKX]);
#ifdef EXPO_TRAPEZOID
		em[globAddr+4*FLUID_SLABPITCH] = .5*(em[globAddr+4*FLUID_SLABPITCH] + deltaP);
#else
		em[globAddr+4*FLUID_SLABPITCH] = deltaP;
#endif
	}

}

/* Compute the gradient of RZ array phi with 4th order accuracy; store the results in f_x, f_y
 *    In cylindrical geometry, f_x -> f_r,
 */
__global__ void  cukern_findMidGradPRZ_h4(double *phi, double *fx, double *fz, int3 arraysize)
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
		fx[globAddr]     = deltaphi;

		// Calculate dt*(dphi/dz)
		deltaphi         = LAMZ*(-phiLoc[myLocAddr+2*GRADBLOCKX] + 8*phiLoc[myLocAddr+1*GRADBLOCKX] - 8*phiLoc[myLocAddr-1*GRADBLOCKX] + phiLoc[myLocAddr-2*GRADBLOCKX]);
		fz[globAddr]     = deltaphi;
	}

}











