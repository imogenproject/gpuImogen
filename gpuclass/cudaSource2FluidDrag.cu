#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif

// CUDA
#include "cuda.h"


__constant__ int devFluidParams[4];
#define FLUID_NX devFluidParams[0]
#define FLUID_NY devFluidParams[1]
#define FLUID_NZ devFluidParams[2]
#define FLUID_SLABPITCH devFluidParams[3]

__constant__ double dragparams[16];
__constant__ double devLambda[16]; // for gradient calculator kernels

#define PI 3.141592653589793

//#define THREAD0_PRINTS_DBG
/* NOTE NOTE NOTE IMPORTANT:
   If this is turned on to make them print, then the functions
     allregimeCdrag
     cukern_GasDustDrag_GeneralLinearCore
     cukern_LogTrapSolve
   which contain cuda kernel printf()s must all be moved UP HERE, ABOVE #include mex.h!!!
   mex.h wraps printf() and is fundamentally incompatible with cuda kernel printfs.
*/

/*
 * From dimensional analysis, by choosing L and T we can rescale...
 */
/* compute this on CPU and store in __constant__ double thevars[] */
#define VISC0      dragparams[0]
#define VISCPOW	   dragparams[1]
#define LAMPOW     dragparams[2]
#define ALPHA      dragparams[3]
#define BETA	   dragparams[4]
#define DELTA	   dragparams[5]
#define EPSILON    dragparams[6]
#define GAMMAM1    dragparams[8]


#include "mex.h"


#include "cudaCommon.h"
#include "cudaSource2FluidDrag.h"

// If defined in concert with ACCOUNT_GRADP, exponential methods will attempt to run the
// action of the pressure gradient backwards in time to solve v' = -k(v) v + a on an
// interval [-.5 .5] instead of [0 1]. This does not work and yields wrong dispersion
// relations entirely.
//#define EXPO_DOTR

// This will account for the pressure gradient and solve v' = -k(v) v + a
//#define ACCOUNT_GRADP

// If the viscous temperature exponent is found to be 0.5 and the cross section exponent
// is zero, the viscosity is hard spheres and some function calls can be simplified
// for a speedup.
typedef enum ViscosityModel { HARD_SPHERES, PCOF } ViscosityModel;

//int sourcefunction_2FluidDrag(MGArray *fluidA, MGArray *fluidB, GeometryParams geo, double gam, double sigmaGas, double muGas, double sigmaDust, double muDust, double dt, int method);
int sourcefunction_2FluidDrag(MGArray *fluidA, MGArray *fluidB, GeometryParams *geo, ThermoDetails *thermogas, ThermoDetails *thermodust, double dt, int method);

int solveDragEMP(MGArray *gas, MGArray *dust, double dt);
int solveDragRK4(MGArray *gas, MGArray *dust, double dt);
int solveDragETDRK1(MGArray *gas, MGArray *dust, GeometryParams *geo, double fluidGamma, double dt);
int solveDragETDRK2(MGArray *gas, MGArray *dust, GeometryParams *geo, double fluidGamma, double dt);
int solveDragLogTrapezoid(MGArray *gas, MGArray *dust, GeometryParams *geo, double fluidGamma, double dt);

int prepareForExpMethod(MGArray *gas, MGArray *dust, MGArray *tempMem, GeometryParams geom, int spaceOrder, double scalingParameter);
int findMidGradP2(MGArray *gas, MGArray *tempMem, GeometryParams geom, int spaceOrder, double scalingParameter);

void dbgPrint(MGArray *gas, MGArray *dust, MGArray *t, int who, int idx);

template <bool ONLY_DV_INI>
__global__ void cukern_GasDustDrag_GeneralAccel(double *gas, double *dust, double *tmpmem, int srcBlock, int dstBlock, int N);
__global__ void cukern_GasDustDrag_EpsteinAccel(double *gas, double *dust, double *vrel, int N);
template <bool resetAccumulator>
__global__ void cukern_GasDustDrag_GeneralLinearTime(double *gas, double *dust, double *tmpmem, int srcBlock, int kBlock, int N);

// shell call for inner loop of above kernel
template <bool resetAccumulator>
__device__ void cukern_GasDustDrag_GeneralLinearCore(double *gas, double *dust, double *tmpmem, int srcBlock, int kBlock, int N);


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

__global__ void cukern_ExpMidpoint_partA(double *gas, double *dust, double *tmpmem, double t, unsigned long partNumel);
__global__ void cukern_ExpMidpoint_partB(double *gas, double *dust, double t, double *tmpmem);
__global__ void cukern_ETDRK1(double *gas, double *dust, double t, double *tmpmem);

__global__ void cukern_LogTrapSolve(double *gas, double *dust, double t, double *tmpmem, int partNumel);

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
	if ((nrhs!=3) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaSource2FluidDrag(FluidManager[2], geometry, [dt, solverMethod])\n");

	if(CHECK_CUDA_ERROR("entering cudaSource2FluidDrag") != SUCCESSFUL) { DROP_MEX_ERROR("Failed upon entry to cudaSource2FLuidDrag."); }

	MGArray fluidA[5];
	int status = MGA_accessFluidCanister(prhs[0], 0, &fluidA[0]);
	if(status != SUCCESSFUL) {
		PRINT_FAULT_HEADER;
		printf("Unable to access first FluidManager.\n");
		PRINT_FAULT_FOOTER;
		DROP_MEX_ERROR("crashing.");
	}
	const mxArray *thermostruct = derefXatNdotAdotB(prhs[0], 0, "thermoDetails", NULL);
	ThermoDetails thermA = accessMatlabThermoDetails(thermostruct);

	MGArray fluidB[5];
	status = MGA_accessFluidCanister(prhs[0], 1, &fluidB[0]);
	if(status != SUCCESSFUL) {
		PRINT_FAULT_HEADER;
		printf("Unable to access second FluidManager.\n");
		PRINT_FAULT_FOOTER;
		DROP_MEX_ERROR("crashing.");
	}
	thermostruct = derefXatNdotAdotB(prhs[0], 1, "thermoDetails", NULL);
	ThermoDetails thermB = accessMatlabThermoDetails(thermostruct);

	GeometryParams geo  = accessMatlabGeometryClass(prhs[1]);

	double *params = mxGetPr(prhs[2]);

	size_t ne = mxGetNumberOfElements(prhs[2]);
	if(ne != 2) {
		PRINT_FAULT_HEADER;
		printf("3rd argument to cudaSource2FluidDrag must have 2 elements:\n[ dt (method: 0=midpt, 1=rk4, 2=exponential)]\nGiven argument has %i instead.\n", (int)ne);
		PRINT_FAULT_FOOTER;
		DROP_MEX_ERROR("Crashing.");
	}	

	double dt         = params[0];
	int solverMethod  = (int)params[1];

	// For reference:
	//1nm iron sphere, 300K -> 56m/s thermal velocity
	//10nm iron ball, 300K -> 1.79m/s thermal velocity
	//100nm iron ball, 300K -> 56mm/s thermal velocity
	
	status = sourcefunction_2FluidDrag(&fluidA[0], &fluidB[0], &geo, &thermA, &thermB, dt, solverMethod);

	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) {
		DROP_MEX_ERROR("2-fluid drag code crashed!");
	}

	return;
}

/* Calculates the drag between fluids A and B where B is presumed to be dust.
 * geo describes the physical geometry of the grids, to which fluidA an fluidB must conform.
 * thermogas and thermodust provide the necessary fluid microphysics constants.
 * dt is the time to integrate and method selects the numeric integration scheme to employ.
 */
int sourcefunction_2FluidDrag(MGArray *fluidA, MGArray *fluidB, GeometryParams *geo, ThermoDetails *thermogas, ThermoDetails *thermodust, double dt, int method)
{
	int i;
	int sub[6];
	int hostFluidParams[4];

	int statusCode = SUCCESSFUL;

	double hostDrag[16];

	double gam = thermogas -> gamma;

	// Reference viscosity & viscosity temperature dependence (0.5 for hard spheres)
	double nu0    = thermogas->mu0;
	double nupow  = thermogas->muTindex;
	//
	double lampow = thermogas->sigmaTindex;
	double ddust  = sqrt(thermodust->sigma0 / 3.141592653589793); // based on sigma being a kinetic cross section = pi (2r)^2, this is correct and needn't be divided by 4
	double mgas   = thermogas->m;
	double mdust  = thermodust->m;

	hostDrag[0] = nu0; // reference viscosity, fluidDetailModel.viscosity
	hostDrag[1] = -nupow; // FIXME viscosity temperature dependence, fluidDetailModel.visocityasdfasdf
	hostDrag[2] = lampow; // cross section temperature dependence, fluidDetailModel. ...
	hostDrag[3] = mgas *(gam-1.0) / (298.15*thermogas->kBolt); // alpha = mgas * (gamma-1) / (t_ref * k_b)
	hostDrag[4] = sqrt(2.0)*mgas/(thermogas->sigma0 * ddust); // beta =2 mgas / (sqrt(2) * sigmaGas * dustDiameter);
	hostDrag[5] = ddust / nu0; // delta= dustDiameter / (visc0)
	hostDrag[6] = thermodust->sigma0 / (1.0*mdust); // epsilon = sigmaDust / 8 mdust
	hostDrag[7] = dt;
	hostDrag[8] = (gam-1.0);
	hostDrag[9] = .25*thermodust->sigma0 / mdust;
	hostDrag[10]= 16*(gam-1.0)/3.0;
	
	#ifdef THREAD0_PRINTS_DBG
	printf("hostDrag[] in sourceFunction_2FluidDrag:\n");
	printf("VISC0 = %le\nVISCPOW = %le\nLAMPOW = %le\nALPHA=%le\nBETA=%le\nDELTA=%le\nEPSILON=%le\n", hostDrag[0], hostDrag[1], hostDrag[2], hostDrag[3], hostDrag[4], hostDrag[5], hostDrag[6]);
	#endif

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
		cudaMemcpyToSymbol((const void *)dragparams, &hostDrag[0], 11*sizeof(double), 0, cudaMemcpyHostToDevice);
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
	case 2: // ETDRK1 (exponential Euler)
		statusCode = CHECK_IMOGEN_ERROR(solveDragETDRK1(fluidA, fluidB, geo, gam, dt));
		break;
	case 3: // ETDRK2 (exponential midpoint)
		statusCode = CHECK_IMOGEN_ERROR(solveDragETDRK2(fluidA, fluidB, geo, gam, dt));
		break;
	case 4: // LogTrap method (cubic accuracy with time-variable drag coefficient)
		statusCode = CHECK_IMOGEN_ERROR(solveDragLogTrapezoid(fluidA, fluidB, geo, gam, dt));
		break;
	}
	
	cudaDeviceSynchronize(); // fixme remove this once we're done

	return statusCode;
}

/* Helps track the state of the integrator when debugging w/o needing cuda-gdb
 * i.e. a slightly more sophisticated printf()-debug
 * gas, dust, t are the five-MGArray pointers to gas, dust and tmp storage
 * who: bit 1 = print about gas, 2 = about dust, 4 = about t
 * idx: the linear index of the cell to print about (the test suite element generates a uniform
 * in space fluid state)
 */
void dbgPrint(MGArray *gas, MGArray *dust, MGArray *t, int who, int idx)
{
	double *hstcpy = (double *)malloc(gas->slabPitch[0]*5);

	int qq = gas->slabPitch[0]/8;
	if(who & 1) {
		cudaMemcpy((void *)hstcpy, (const void *)gas->devicePtr[0], gas->slabPitch[0]*5, cudaMemcpyDeviceToHost);
		printf("Gas input state: [%e %e %e %e %e]\n", hstcpy[idx+0*qq], hstcpy[idx+1*qq], hstcpy[idx+2*qq], hstcpy[idx+3*qq], hstcpy[idx+4*qq]);
	}

	if(who & 2) {
		cudaMemcpy((void *)hstcpy, (const void *)dust->devicePtr[0], gas->slabPitch[0]*5, cudaMemcpyDeviceToHost);
		printf("Dust input state: [%e %e %e %e %e]\n", hstcpy[idx+0*qq], hstcpy[idx+1*qq], hstcpy[idx+2*qq], hstcpy[idx+3*qq], hstcpy[idx+4*qq]);
	}

	if(who & 4) {
		cudaMemcpy((void *)hstcpy, (const void *)t->devicePtr[0], gas->slabPitch[0]*5, cudaMemcpyDeviceToHost);
		printf("tmp memory state: [%e %e %e %e %e]\n", hstcpy[idx+0*qq], hstcpy[idx+1*qq], hstcpy[idx+2*qq], hstcpy[idx+3*qq], hstcpy[idx+4*qq]);
	}

	free(hstcpy);

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
	cukern_GasDustDrag_GeneralAccel<true><<<gridsize, blocksize>>>(g, d, vrel, 0, 3, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_GasDustDrag_full<false>");
	if(statusCode != SUCCESSFUL) break;
	// compute delta-v at t=1/2; store stage at block 4
	cukern_SolveRK_single<true><<<gridsize, blocksize>>>(vrel, 4, .5*dt, 3, 0, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_SolveRK_single<true>");
	if(statusCode != SUCCESSFUL) break;
	// solve gas drag at t=1/2 using half stage, store in block 3
	cukern_GasDustDrag_GeneralAccel<false><<<gridsize, blocksize>>>(g, d, vrel, 4, 3, NE);
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

return CHECK_IMOGEN_ERROR(statusCode);
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

// FIXME this should determine an appropriate blocksize at runtime perhaps?
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
	cukern_GasDustDrag_GeneralAccel<true><<<gridsize, blocksize>>>(g, d, vrel, 0, 3, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_GasDustDrag_GeneralAccel<true>");
	if(statusCode != SUCCESSFUL) break;
	// compute delta-v at t=1/2; store stage at block 4
	cukern_SolveRK_single<true><<<gridsize, blocksize>>>(vrel, 4, 0.5*dt, 3, bWeights[0], NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_SolveRK_single<true>");
	if(statusCode != SUCCESSFUL) break;
	// solve gas drag on k2, store in block 3
	cukern_GasDustDrag_GeneralAccel<false><<<gridsize, blocksize>>>(g, d, vrel, 4, 3, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_GasDustDrag_GeneralAccel<false>");
	if(statusCode != SUCCESSFUL) break;
	// compute delta-v at t=1/2; store stage at block 4
	cukern_SolveRK_single<false><<<gridsize, blocksize>>>(vrel, 4, 0.5*dt, 3, bWeights[1], NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_SolveRK_single<true>");
	if(statusCode != SUCCESSFUL) break;
	// solve gas drag on k3, store in block 3
	cukern_GasDustDrag_GeneralAccel<false><<<gridsize, blocksize>>>(g, d, vrel, 4, 3, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_GasDustDrag_GeneralAccel<false>");
	if(statusCode != SUCCESSFUL) break;
	// compute delta-v at t=1/2; store stage at block 4
	cukern_SolveRK_single<false><<<gridsize, blocksize>>>(vrel, 4, 1.0*dt, 3, bWeights[2], NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_SolveRK_single<true>");
	if(statusCode != SUCCESSFUL) break;
	// solve gas drag on k4, store in block 3
	cukern_GasDustDrag_GeneralAccel<false><<<gridsize, blocksize>>>(g, d, vrel, 4, 3, NE);
	statusCode = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, gas, i, "cukern_GasDustDrag_GeneralAccel<false>");
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

return CHECK_IMOGEN_ERROR(statusCode);
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
 * formally order 2, stiff order 1, L-stable */
int solveDragETDRK1(MGArray *gas, MGArray *dust, GeometryParams *geo, double fluidGamma, double dt)
{
	int dbprint = 0;
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
	statusCode = prepareForExpMethod(gas, dust, gs, *geo, 2, fluidGamma - 1);
	if(CHECK_IMOGEN_ERROR(statusCode) != SUCCESSFUL) return statusCode;

	if(dbprint) { dbgPrint(gas, dust, gs, 7, 6); }

	int velblock = 0;
	int kblock = 0;

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

		// Use u_0 and dv_tr to compute the drag eigenvalue at t=0
		// overwrite the |dv_tr| value (block 0) with K
		cukern_GasDustDrag_GeneralLinearTime<true><<<lingrid, linblock>>>(g, d, tempPtr, velblock, kblock, gas->partNumel[i]);
		statusCode = CHECK_CUDA_LAUNCH_ERROR(linblock, lingrid, gas, i, "cukern_GasDustDrag_linearTime");
		if(statusCode != SUCCESSFUL) break;

		if(dbprint) { dbgPrint(gas, dust, gs, 4, 6); }

		// Use 1st order exponential time differencing (exponential euler)
		cukern_ETDRK1<<<lingrid, linblock>>>(g, d, dt, tempPtr);

		statusCode = CHECK_CUDA_LAUNCH_ERROR(linblock, lingrid, gas, i, "cukern_ExponentialEulerHalf");
		if(statusCode != SUCCESSFUL) break;
		if(dbprint) { dbgPrint(gas, dust, gs, 7, 6); }

	}

	// Make sure node's internal boundaries are consistent
	if(CHECK_IMOGEN_ERROR(statusCode) == SUCCESSFUL) MGA_exchangeLocalHalos(gas  + 1, 4);
	if(CHECK_IMOGEN_ERROR(statusCode) == SUCCESSFUL) MGA_exchangeLocalHalos(dust + 1, 4);

	if(CHECK_IMOGEN_ERROR(statusCode) == SUCCESSFUL) MGA_delete(gs);

	return statusCode;
}

// TODO
/* Implement Exponential Time Differencing, 2nd order RK:
 * y_1   = exp(h L) y_0 + h phi_1(h L) f(t=0)
 * y_n+1 = exp(h L) y_0 + h (phi_1(h L) - phi_2(h L)) f(t=0) + h phi_2(h L) f(t=1)
 *
 *L = -k
 *-> y_1 = exp(-k t) y_0 + t (exp(-k t)-1) / (-k t) f_0
 *-> y_1 = exp(-k t) y_0 + f_0 (1 - exp(-k t)) / k
 *-> y_1 = f_0 / k + (y_0 - f_0/k) exp(-k t)
 *
 *y_n+1 = exp(-k t) y_0 + f_0 (-(exp(-k t)-1)/k - (exp(-k t)-1-kt)/(k^2 t)) + f_1 (exp(-kt)-1-k t)/k^2t
 *y_n+1 = exp(-k t) y_0 + f_0/k + (f_0-f_1)/k^2t + f_0/k - f_1/k + (-f_0/k - f_0/k^2t + f_1/k^2t) exp(-k t)
 *y_n+1 = exp(-k t) y_0 + (2f_0-f_1)/k -f_0/k exp(-kt) + (f_0-f_1)/k^2t - (f_0 - f_1) exp(-k t)/k^2t
 *y_n+1 = (2f_0-f_1)/k + (y_0-f_0/k) exp(-kt) + (f_0-f_1)(1-exp(-kt))/k^2t
 *y_n+1 = y_0 exp(-kt) + f_0(2/k - exp(-kt)/k + 1/k^2t -exp(-kt)/k^2t) + f_1(-1/k + exp(-kt)/k^2t)
 */
int solveDragETDRK2(MGArray *gas, MGArray *dust, GeometryParams *geo, double fluidGamma, double dt)
{
	int n = gas->nGPUs;
	int dbprint = 0;

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
	statusCode = prepareForExpMethod(gas, dust, gs, *geo, 2, fluidGamma - 1);
	if(CHECK_IMOGEN_ERROR(statusCode) != SUCCESSFUL) return statusCode;

	if(dbprint) { dbgPrint(gas, dust, gs, 7, 6); }

	int velblock = 0;
	int kblock = 0;

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

		// Use u_0 and dv_tr to compute the drag eigenvalue at t=0
		// overwrites the |dv_tr| value (block 0) with K
		// replace [|dv_tr|, u_0, P_x, P_y, P_z] into temp memory at gs
		// with    [K      , u_0, P_x, P_y, P_z] into temp memory at gs
		cukern_GasDustDrag_GeneralLinearTime<true><<<lingrid, linblock>>>(g, d, tempPtr, velblock, kblock, gas->partNumel[i]);
		statusCode = CHECK_CUDA_LAUNCH_ERROR(linblock, lingrid, gas, i, "cukern_GasDustDrag_linearTime");
		if(statusCode != SUCCESSFUL) break;
		if(dbprint) { dbgPrint(gas, dust, gs, 4, 6); }

		// Use the eigenvalue from t=0 to advance to t=1/2
		//   Output only new uint & dv values from this stage,
		//   We do this only do re-evaluate the pressure gradient & eigenvalue at the midpoint
		//   This reads K from register 0 and overwrites it with dv_half
		// overwrite [K      , u_0,   P_x, P_y, P_z] into temp memory at gs
		// with      [dv_new , u_new, P_x, P_y, P_z] into temp memory at gs
		cukern_ExpMidpoint_partA<<<lingrid, linblock>>>(g, d, tempPtr, dt, gas->partNumel[i]);
		statusCode = CHECK_CUDA_LAUNCH_ERROR(linblock, lingrid, gas, i, "doing cukern_ExponentialEulerIntermediate");
		if(statusCode != SUCCESSFUL) break;
		if(dbprint) { dbgPrint(gas, dust, gs, 4, 6); }
	}


	// Solve gradient-P again
	statusCode = findMidGradP2(gas, gs, *geo, 2, fluidGamma - 1);
	if(dbprint) { dbgPrint(gas, dust, gs, 4, 6); }

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
		cukern_GasDustDrag_GeneralLinearTime<true><<<lingrid, linblock>>>(g, d, tempPtr, velblock, kblock, gas->partNumel[i]);
		statusCode = CHECK_CUDA_LAUNCH_ERROR(linblock, lingrid, gas, i, "cukern_GasDustDrag_linearTime");
		if(statusCode != SUCCESSFUL) break;
		if(dbprint) { dbgPrint(gas, dust, gs, 4, 6); }

		// Use averaged pressure gradient and k value to compute timestep.
		// we divide t by 2 since we simply summed the k values previously
		cukern_ExpMidpoint_partB<<<lingrid, linblock>>>(g, d, dt, tempPtr);
		statusCode = CHECK_CUDA_LAUNCH_ERROR(linblock, lingrid, gas, i, "cukern_exponentialMidpoint");
		if(statusCode != SUCCESSFUL) break;
		if(dbprint) { dbgPrint(gas, dust, gs, 3, 6); }
	}

	// Make sure node's internal boundaries are consistent
	if(CHECK_IMOGEN_ERROR(statusCode) == SUCCESSFUL) statusCode = MGA_exchangeLocalHalos(gas  + 1, 4);
	if(CHECK_IMOGEN_ERROR(statusCode) == SUCCESSFUL) statusCode = MGA_exchangeLocalHalos(dust + 1, 4);

	if(CHECK_IMOGEN_ERROR(statusCode) == SUCCESSFUL) statusCode = MGA_delete(gs);

	return statusCode;
}

/* Second or third order method that handles variable drag coefficients
 * 2nd order (trapezoid):
 * u_0        = P(y_0)
 * k_0        = compute_kdrag(y_0, u_0)
 * (y_1, u_1) = y_0 exp(-k_0 t)
 * k_1        = compute_kdrag(y_1, u_1)
 * y_n+1      = y_0 exp(-0.5(k_0 + k_1)t)
 *
 * 3rd order: (Richardson extrapolated trapezoid)
 * u_0          = P(y_0)
 * k_0          = compute_kdrag(y_0, u_0)
 * (y_1, u_1)   = y_0 exp(-k_0 t)
 * k_1          = compute_kdrag(y_1, u_1)
 * (y_nhf,u_nhf)= y_0 exp(-0.5 * 0.5(k_0 + k_1)t)
 * k_nhf        = compute_kdrag(y_nhf, u_nhf)
 * k_integral   = richardson_extrap(.25 k_0 + .5 k_nhf + .25 k1, .5k_0 + .5k_1)
 * y_1          = y_0 exp(-k_integral t)
 */
int solveDragLogTrapezoid(MGArray *gas, MGArray *dust, GeometryParams *geo, double fluidGamma, double dt)
{
	int n = gas->nGPUs;
	int dbprint = 0;

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
	statusCode = prepareForExpMethod(gas, dust, gs, *geo, 2, fluidGamma - 1);
	if(CHECK_IMOGEN_ERROR(statusCode) != SUCCESSFUL) return statusCode;

	int fuckoff = dbgfcn_CheckArrayVals(gs, 5, 1);
		if(CHECK_IMOGEN_ERROR(fuckoff) != SUCCESSFUL) return fuckoff;

	if(dbprint) { dbgPrint(gas, dust, gs, 7, 6); }

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

		cukern_LogTrapSolve<<<lingrid, linblock>>>(g, d, dt, tempPtr, gas->partNumel[i]);
		statusCode = CHECK_CUDA_LAUNCH_ERROR(linblock, lingrid, gas, i, "doing cukern_LogTrapSolve");
		if(statusCode != SUCCESSFUL) break;
		if(dbprint) { dbgPrint(gas, dust, gs, 7, 6); }
	}

	cudaDeviceSynchronize();

	fuckoff = dbgfcn_CheckFluidVals(gas, 1);
	if(CHECK_IMOGEN_ERROR(fuckoff) != SUCCESSFUL) return fuckoff;
	fuckoff = dbgfcn_CheckFluidVals(dust, 1);
	if(CHECK_IMOGEN_ERROR(fuckoff) != SUCCESSFUL) return fuckoff;

// Make extra sure node's internal boundaries are consistent
if(CHECK_IMOGEN_ERROR(statusCode) == SUCCESSFUL) statusCode = MGA_exchangeLocalHalos(gas  + 1, 5);
if(CHECK_IMOGEN_ERROR(statusCode) == SUCCESSFUL) statusCode = MGA_exchangeLocalHalos(dust + 1, 4);

if(CHECK_IMOGEN_ERROR(statusCode) == SUCCESSFUL) statusCode = MGA_delete(gs);

return statusCode;
}


/* This function returns the Stokes coefficient, scaled by 1/2
 * This parameter is experimentally measured except for the low-Re regime */
__device__ double drag_coeff(double Re)
{
	if(Re < 1) {
		// 24 / Re
		return 12 / (Re+1e-15);
	}
	if(Re > 7.845084191866316e+02) {
		// .44
		return 0.22;
	}
	// 24 Re^-.6
	return 12.0*pow(Re,-0.6);
}

/* Computes the drag coefficient for all Reynolds and Knudsen numbers with an accuracy of <1% for
 * speeds less than approximately Mach 0.1.
 * The coefficients are divided by 8 per a factor that appears in the drag time formula
 * The Cunninghand correction coefficients of Allen & Raabe (1.142, .558, .998) are used.
 */
__device__ double allregimeCdrag(double Re, double Kn)
{
	// Prevent 1/0 errors which may occur when a simulation is initialized with dv = 0
	// The only physical way to acheive Re = 0 is if dv = 0, and if dv =0 then skip wasting time

	double cunningham = 1 + Kn*(1.142 + 1*0.558*exp(-0.999/Kn));
	double C_drag = (3 / Re + .5*pow(Re, -1.0/3.0) + .055*Re/(12000+Re)) / cunningham;
	#ifdef THREAD0_PRINTS_DBG
	if(threadIdx.x == 0) { printf("b=%i,t=%i: Cdrag reporting: Cd0 = %.12lf, Cu = %.12lf\n, Cd = %.12lf\n", blockIdx.x, threadIdx.x, C_drag, cunningham, C_drag / cunningham); }
	#endif
	return C_drag;
}

/* The general linear core is called upon by the LogTrap solver as well so it is here separated out */
template <bool resetAccumulator>
__device__ void cukern_GasDustDrag_GeneralLinearCore(double *gas, double *dust, double *tmpmem, int srcBlock, int kBlock, int N)
{
double rhoA, rhoB;    // gas and dust densities respectively
	double magdv;	  // magnitude velocity difference
	double uspecific; // specific internal energy density = e_{int} / rho
	double Tnormalized; // Temperature normalized by the reference temp for the viscosity
	double Re, Kn; // Reynolds number and Knudsen number
	double kdrag, Cd_hat; // drag time constant & drag coefficient

	magdv = tmpmem[srcBlock*FLUID_SLABPITCH];

	if(magdv < 1e-9) {
		if(resetAccumulator) { tmpmem[kBlock*FLUID_SLABPITCH] = 0; }
		#ifdef THREAD0_PRINTS_DBG
		if(threadIdx.x == 0) { printf("b=%i,t=%i: general linear core reporting: |dv| < 1e-9, returning no drag\n", blockIdx.x, threadIdx.x); }
		#endif
		return;
	}

	rhoA = gas[0];
	rhoB = dust[0];

	// make sure computation includes gas heating term!
	// fixme double check this calculation I think it may be in error
	if(srcBlock != 0) {
		// If srcblock != zero, we're evaluating a different dv than originally used to give uinternal:
		// must add dissipated relative KE to gas internal energy.
		uspecific = tmpmem[FLUID_SLABPITCH] + .5 * rhoB * (tmpmem[0]*tmpmem[0] - magdv*magdv) / (rhoA + rhoB);
	} else {
		// If srcBlock is zero, we're reading the original dv for which uinternal was computed: No change
		uspecific = tmpmem[FLUID_SLABPITCH];
	}

	Tnormalized = ALPHA * uspecific;
	Re = DELTA * rhoA * magdv * pow(Tnormalized, VISCPOW);
    Kn = BETA  * pow(Tnormalized, LAMPOW) / rhoA;

    Cd_hat = allregimeCdrag(Re, Kn);
	kdrag = Cd_hat * magdv * (rhoA + rhoB) * EPSILON;

	#ifdef THREAD0_PRINTS_DBG
	if(threadIdx.x == 0) { printf("b=%i,t=%i: general linear core reporting: uspecific=%le, T/T0=%le, Re=%le, Kn=%le, Cd=%le, k=%le, a=k*v=%le\n", blockIdx.x, threadIdx.x, uspecific, Tnormalized, Re, Kn, 8*Cd_hat, kdrag, kdrag*magdv); }
	#endif

	if(resetAccumulator) {
		tmpmem[kBlock*FLUID_SLABPITCH] = kdrag;
	} else {
		tmpmem[kBlock*FLUID_SLABPITCH] += kdrag;
	}

	tmpmem[2*FLUID_SLABPITCH] = Re;
	tmpmem[3*FLUID_SLABPITCH] = Kn;
	tmpmem[4*FLUID_SLABPITCH] = Cd_hat * 8;
}


/* This function directly computes the gas-dust drag force in the full (stokes+epstein) regime
 * This is suited for weaker drag or strange regimes, but unnecessary and time-consuming for
 * small particles which will never exit the low-speed Epstein regime.
 * - Uses staged dv value stored at srcBlock, writes acceleration into dstBlock
 * - template saves on evaluating drag heating if true */
template <bool ONLY_DV_INI>
__global__ void cukern_GasDustDrag_GeneralAccel(double *gas, double *dust, double *tmpmem, int srcBlock, int dstBlock, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	double rhoA, rhoB;   // gas and dust densities respectively 
	double magdv;	// magnitude velocity difference
	double uspecific;	// specific internal energy density
	double Tnormalized;
	double Re, Kn;	   // Spherical particle Reynolds number
	double Cd_hat, accel;

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

		Tnormalized = ALPHA * uspecific;
		Re = DELTA * rhoA * magdv * pow(Tnormalized, VISCPOW);
	    Kn = BETA  * pow(Tnormalized, LAMPOW) / rhoA;

	    Cd_hat = allregimeCdrag(Re, Kn);
		accel = Cd_hat * magdv * magdv * (rhoA + rhoB) * EPSILON;

		tmpmem[dstBlock*FLUID_SLABPITCH] = -accel;
	
		gas += blockDim.x*gridDim.x;
		dust += blockDim.x*gridDim.x;
		tmpmem += blockDim.x*gridDim.x;
	}

}

#define EPSTEIN_ALPHA dragparams[9]
#define EPSTEIN_BETA dragparams[10]
/* This function computes particle drag in the Epstein regime (particles much smaller than gas MFP)
 * but is unsuited to large particles or dense gas
 */
__global__ void cukern_GasDustDrag_EpsteinAccel(double *gas, double *dust, double *vrel, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	double rhoA, rhoB;   // gas and dust densities respectively
	double magdv;	// magnitude velocity difference
	double uinternal;	// specific internal energy density
	double accel;	// Relative acceleration (d/dt of vrel)

	gas  += i;
	dust += i;
	vrel += i;

	for(; i < N; i+= blockDim.x*gridDim.x) {
		magdv = vrel[FLUID_SLABPITCH];
		rhoA = gas[0];
		rhoB = dust[0];

		// make sure computation includes gas heating term!
		uinternal = vrel[2*FLUID_SLABPITCH] + rhoB * (vrel[0]*vrel[0] - magdv*magdv) / (rhoA + rhoB);

		// compute f(single particle) = sqrt(f_slow^2 + f_fast^2)
		// where f_slow = (4/3) A_dust cbar rho_g dv
		//       f_fast = A_dust rho_g dv^2
		// and accel = f(single particle) * (rho_dust / m_dust) * (rho_g + rho_d)/(rhog rhod)
		//           = f(single particle) * ndust / reduced mass
		accel = EPSTEIN_ALPHA * magdv * rhoA * sqrt(magdv*magdv + EPSTEIN_BETA*uinternal) * (1.0+rhoB/rhoA);

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
__global__ void cukern_GasDustDrag_GeneralLinearTime(double *gas, double *dust, double *tmpmem, int srcBlock, int kBlock, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	gas  += i;
	dust += i;
	tmpmem += i;

	for(; i < N; i+= blockDim.x*gridDim.x) {
		cukern_GasDustDrag_GeneralLinearCore<resetAccumulator>(gas, dust, tmpmem, srcBlock, kBlock, N);
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

/* This set of functions implement evaluation of the rows in RK Butcher tableaux containing
 * from 1 to 3 nonzero entries */

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
(2) [u_hf, |dv_hf|] = cukern_ExpMidpoint_partA(gas_state, dust_state, k_0, P_x, P_y, P_z)
* compute time-reversed elements of dv again (memory & memory BW precious, v_i = (p_i - 2 P_i t)/rho cheap as dirt)
* solve y_i' = -k_0 y_i + a_i, a_i = - P_i / rho_gas per vector element
    * y(t) = a_i / k_0 + (y_i - a_i/k_0) exp(-k_0 t)
    * this is an L-stable method for the drag equation
* Our only interest in solving this is to re-evaluate the linear operation matrix at t_half
    * Linear matrix is diag([k_n k_n k_n]) -> require only |dv_half| to re-call gasDustDrag */
__global__ void cukern_ExpMidpoint_partA(double *gas, double *dust, double *tmpmem, double t, unsigned long partNumel)
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
#ifdef EXPO_DOTR
		dv_i   = (gas[2*FLUID_SLABPITCH] + t*a0)*rhoginv - dust[2*FLUID_SLABPITCH]/dust[0];
#else
		dv_i   = (gas[2*FLUID_SLABPITCH] + t*a0)*rhoginv - dust[2*FLUID_SLABPITCH]/dust[0];
#endif

		// compute decay of this value
		a0    *= rhoginv / k;
		dv_t   = a0 + (dv_i - a0)*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation

		// accumulate new delta-v^2
		dvsq   = dv_t*dv_t;
		// accumulate drag heating
		duint  = k*a0*a0*t - 2*a0*(dv_i - a0)*expm1(-k*t) - (dv_i - a0)*(dv_i - a0)*expm1(-2*k*t);

		// Repeat the above for the other two components
		a0     = tmpmem[3*FLUID_SLABPITCH];
#ifdef EXPO_DOTR
		dv_i   = (gas[3*FLUID_SLABPITCH] + t*a0)*rhoginv - dust[3*FLUID_SLABPITCH]/dust[0];
#else
		dv_i   = (gas[3*FLUID_SLABPITCH])*rhoginv - dust[3*FLUID_SLABPITCH]/dust[0];
#endif
		a0    *= rhoginv/k;
		dv_t   = a0 + (dv_i - a0)*exp(-t*k);
		dvsq  += dv_t*dv_t;
		duint += k*a0*a0*t - 2*a0*(dv_i - a0)*expm1(-k*t) - (dv_i - a0)*(dv_i - a0)*expm1(-2*k*t);

		a0     = tmpmem[4*FLUID_SLABPITCH];
#ifdef EXPO_DOTR
		dv_i   = (gas[4*FLUID_SLABPITCH] + t*a0)*rhoginv - dust[3*FLUID_SLABPITCH]/dust[0];
#else
		dv_i   = (gas[4*FLUID_SLABPITCH])*rhoginv - dust[3*FLUID_SLABPITCH]/dust[0];
#endif
		a0    *= rhoginv/k;
		dv_t   = a0 + (dv_i - a0)*exp(-t*k);
		dvsq  += dv_t*dv_t;
		duint += k*a0*a0*t - 2*a0*(dv_i - a0)*expm1(-k*t) - (dv_i - a0)*(dv_i - a0)*expm1(-2*k*t);

		tmpmem[0] = sqrt(dvsq); // overwrite in place
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
__global__ void cukern_ExpMidpoint_partB(double *gas, double *dust, double t, double *tmpmem)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	gas    += x;
	dust   += x;
	tmpmem += x;

	double rhoginv; // 1/rho_gas
//	double rhodinv; // 1/rho_dust
	double dv_i;    // element of delta-v
	double k;       // drag eigenvalue
	double dpdt;      // element of accel = gradient(P)/rho_gas
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
#ifdef ACCOUNT_GRADP
		dpdt      = -tmpmem[2*FLUID_SLABPITCH];
#else
		dpdt = 0;
#endif
#ifdef EXPO_DOTR
		dv_i    = (gas[2*FLUID_SLABPITCH] - t*dpdt)*rhoginv - dust[2*FLUID_SLABPITCH]/dust[0];
#else
		dv_i    = (gas[2*FLUID_SLABPITCH])*rhoginv - dust[2*FLUID_SLABPITCH]/dust[0];
#endif
		// load k, solve driven linear system
		k       = tmpmem[0];

		dpdt     *= mu*rhoginv;
		dv_t    = dpdt/k + (dv_i - dpdt/k)*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation

		// recalculate new differential velocities
		gas[2*FLUID_SLABPITCH] = gas[0]*vstick + dv_t*mu;
		dust[2*FLUID_SLABPITCH] = q = dust[0]*vstick - dv_t*mu;
		// accumulate change in dust kinetic energy
		pdustsq += q*q; //

		// do Y direction
		pdustsq -= dust[3*FLUID_SLABPITCH]*dust[3*FLUID_SLABPITCH];
		vstick  = (gas[3*FLUID_SLABPITCH]+dust[3*FLUID_SLABPITCH]) / (gas[0] + dust[0]);
#ifdef ACCOUNT_GRADP
		dpdt      = -tmpmem[3*FLUID_SLABPITCH];
#endif
#ifdef EXPO_DOTR
		dv_i    = (gas[3*FLUID_SLABPITCH] - t*dpdt)*rhoginv - dust[3*FLUID_SLABPITCH]/dust[0];
#else
		dv_i    = (gas[3*FLUID_SLABPITCH])*rhoginv - dust[3*FLUID_SLABPITCH]/dust[0];
#endif
		dpdt     *= mu*rhoginv;
		dv_t    = dpdt/k + (dv_i - dpdt/k)*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation
		gas[3*FLUID_SLABPITCH]     = gas[0]*vstick + dv_t*mu;
		dust[3*FLUID_SLABPITCH]= q = dust[0]*vstick - dv_t*mu;
		pdustsq += q*q;

		// do Z direction
		pdustsq -= dust[4*FLUID_SLABPITCH]*dust[4*FLUID_SLABPITCH];
		vstick  = (gas[4*FLUID_SLABPITCH]+dust[4*FLUID_SLABPITCH]) / (gas[0] + dust[0]);
#ifdef ACCOUNT_GRADP
		dpdt      = -tmpmem[4*FLUID_SLABPITCH];
#endif
#ifdef EXPO_DOTR
		dv_i    = (gas[4*FLUID_SLABPITCH] - t*dpdt)*rhoginv - dust[4*FLUID_SLABPITCH]/dust[0];
#else
		dv_i    = (gas[4*FLUID_SLABPITCH])*rhoginv - dust[4*FLUID_SLABPITCH]/dust[0];
#endif
		dpdt     *= mu*rhoginv;
		dv_t    = dpdt/k + (dv_i - dpdt/k)*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation
		gas[4*FLUID_SLABPITCH]  = gas[0]*vstick + dv_t*mu;
		dust[4*FLUID_SLABPITCH] = q = dust[0]*vstick - dv_t*mu;
		pdustsq += q*q;

		// From conservation of total energy we have that the gas total energy decreases by whatever
		// amount the dust kinetic energy rises; Under (M_dust >> M_atom) the gas gets ~100% of heating
		gas[FLUID_SLABPITCH] -= .5*pdustsq / dust[0];

		// advance ptrs
		x +=  blockDim.x*gridDim.x;
		gas += blockDim.x*gridDim.x;
		dust += blockDim.x*gridDim.x;
		tmpmem += blockDim.x*gridDim.x;
	}
}


/*(5) [(gas_state), (dust_state)] = cukern_ETD1RK(gas_state, dust_state, k_hf, P_x, P_y, P_z)
    * compute time-reversed elements of dv a 3rd time (memory & memory BW precious, v_i = (p_i - 2 P_i t)/rho cheap as dirt)
    * advance to drag-applied dv values dv_i <- -P_i/(k_hf rho) + (dv_i + P_i/(k_hf rho))*exp(-k_hf t)
    * compute new u_specific? or let d/dt(Etotal) = 0 do the job? does that still work?
    * overwrite gas_state/dust_state using updated values
        * ...
 */
__global__ void cukern_ETDRK1(double *gas, double *dust, double t, double *tmpmem)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	gas    += x;
	dust   += x;
	tmpmem += x;

	double rhoginv; // 1/rho_gas
//	double rhodinv; // 1/rho_dust
	double dv_i;    // element of delta-v
	double k;       // drag eigenvalue
	double dpdt;      // element of accel = gradient(P)/rho_gas
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
#ifdef ACCOUNT_GRADP
		dpdt      = -tmpmem[2*FLUID_SLABPITCH];
#else
		dpdt = 0;
#endif
#ifdef EXPO_DOTR
		dv_i    = (gas[2*FLUID_SLABPITCH] - t*dpdt)*rhoginv - dust[2*FLUID_SLABPITCH]/dust[0];
#else
		dv_i    = (gas[2*FLUID_SLABPITCH])*rhoginv - dust[2*FLUID_SLABPITCH]/dust[0];
#endif
		// load k, solve driven linear system
		k       = tmpmem[0];

		dpdt     *= mu*rhoginv;
		dv_t    = dpdt/k + (dv_i - dpdt/k)*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation

		// recalculate new differential velocities
		gas[2*FLUID_SLABPITCH] = gas[0]*vstick + dv_t*mu;
		dust[2*FLUID_SLABPITCH] = q = dust[0]*vstick - dv_t*mu;
		// accumulate change in dust kinetic energy
		pdustsq += q*q; //

		// do Y direction
		pdustsq -= dust[3*FLUID_SLABPITCH]*dust[3*FLUID_SLABPITCH];
		vstick  = (gas[3*FLUID_SLABPITCH]+dust[3*FLUID_SLABPITCH]) / (gas[0] + dust[0]);
#ifdef ACCOUNT_GRADP
		dpdt      = -tmpmem[3*FLUID_SLABPITCH];
#endif
#ifdef EXPO_DOTR
		dv_i    = (gas[3*FLUID_SLABPITCH] - t*dpdt)*rhoginv - dust[3*FLUID_SLABPITCH]/dust[0];
#else
		dv_i    = (gas[3*FLUID_SLABPITCH])*rhoginv - dust[3*FLUID_SLABPITCH]/dust[0];
#endif
		dpdt     *= mu*rhoginv;
		dv_t    = dpdt/k + (dv_i - dpdt/k)*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation
		gas[3*FLUID_SLABPITCH]     = gas[0]*vstick + dv_t*mu;
		dust[3*FLUID_SLABPITCH]= q = dust[0]*vstick - dv_t*mu;
		pdustsq += q*q;

		// do Z direction
		pdustsq -= dust[4*FLUID_SLABPITCH]*dust[4*FLUID_SLABPITCH];
		vstick  = (gas[4*FLUID_SLABPITCH]+dust[4*FLUID_SLABPITCH]) / (gas[0] + dust[0]);
#ifdef ACCOUNT_GRADP
		dpdt      = -tmpmem[4*FLUID_SLABPITCH];
#endif
#ifdef EXPO_DOTR
		dv_i    = (gas[4*FLUID_SLABPITCH] - t*dpdt)*rhoginv - dust[4*FLUID_SLABPITCH]/dust[0];
#else
		dv_i    = (gas[4*FLUID_SLABPITCH])*rhoginv - dust[4*FLUID_SLABPITCH]/dust[0];
#endif
		dpdt     *= mu*rhoginv;
		dv_t    = dpdt/k + (dv_i - dpdt/k)*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation
		gas[4*FLUID_SLABPITCH]  = gas[0]*vstick + dv_t*mu;
		dust[4*FLUID_SLABPITCH] = q = dust[0]*vstick - dv_t*mu;
		pdustsq += q*q;

		// From conservation of total energy we have that the gas total energy decreases by whatever
		// amount the dust kinetic energy rises; Under (M_dust >> M_atom) the gas gets ~100% of heating
		gas[FLUID_SLABPITCH] -= .5*pdustsq / dust[0];

		// advance ptrs
		x +=  blockDim.x*gridDim.x;
		gas += blockDim.x*gridDim.x;
		dust += blockDim.x*gridDim.x;
		tmpmem += blockDim.x*gridDim.x;
	}
}



/* Assuming the temp registers are preloaded with
 * [dv_0 u_0 Px Py Pz]
 * First call drag solve to get
 * [dv_0 u_0 Px Py Pz k_0]
 * Then solve ETD1RK to get dv_1, u_1:
 * [dv_1 u_1 Px Py Pz k_0]
 * call drag solve with accumulate=yes set to get
 * [dv_1 u_1 Px Py Pz (k_0+k_1)]
 * solve the log integral to find y_n+1
 */
__global__ void cukern_LogTrapSolve(double *gas, double *dust, double t, double *tmpmem, int partNumel)
{
	double rhoginv; // 1/rho_gas
//	double rhodinv; // 1/rho_dust
	double dv_i;    // element of delta-v
	double k;       // drag eigenvalue
	double pdustsq; // element of accel = gradient(P)/rho_gas
	double dv_t;    // updated delta v. not sure if needed independently...
	double dvsq; // use to track accumulated transfer of total energy
	double vstick;  // barycentric velocity of gas-dust system
	double mu;      // reduced mass
	double q;       // scratchpad variable
	double duint;

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	gas    += x;
	dust   += x;
	tmpmem += x;

	// Use ETDRK1 to approximate y_1 to first order
	while(x < FLUID_SLABPITCH) {
		mu      = dust[0]/(gas[0]+dust[0]); // reduced density is needed more or less immediately

		/* Assuming the temp registers are preloaded with
		 * [dv_0 u_0 Px Py Pz] */
		// call drag eigenvalue solver.
		cukern_GasDustDrag_GeneralLinearCore<true>(gas, dust, tmpmem, 0, 5, partNumel);

		/* temp contents:
		 * [dv_0 u_0 Px Py Pz k_0]
		 */
		k      = tmpmem[5*FLUID_SLABPITCH];

		dv_i   = tmpmem[0];
		dv_t   = dv_i*exp(-t*k);

		#ifdef THREAD0_PRINTS_DBG
		if(threadIdx.x == 0) { printf("b=%i,t=%i: first point: initial dv=%le, k = %le, t=%le, new dv=%le\n", blockIdx.x, threadIdx.x, dv_i, k, t, dv_t); }
		#endif

		// accumulate new delta-v^2 and drag heating effect
		dvsq   = dv_t*dv_t;
		duint  = -.5*dv_i*dv_i*mu*expm1(-2*k*t);

		tmpmem[0] = sqrt(dvsq);
		// Add the dissipated relative KE before reassessing the drag coefficient
		tmpmem[FLUID_SLABPITCH] += duint;

		/* temp contents:
		 * [dv_1 u_1 Px Py Pz k_0]
		 */
		// Solve drag eigenvalue k1 and accumulate in register 5
		cukern_GasDustDrag_GeneralLinearCore<false>(gas, dust, tmpmem, 0, 5, partNumel);

		#ifdef THREAD0_PRINTS_DBG
		if(threadIdx.x == 0) { printf("b=%i,t=%i: Second k solve: k = %le\n", blockIdx.x, threadIdx.x, tmpmem[5*FLUID_SLABPITCH]); }
		#endif
		/* temp contents:
		 * [dv_1 u_1 Px Py Pz (k_0+k_1)]
		 */

		// Now the cutesy tricksy bit:
		// Cleverly reverse our way to t=1/2 and compute k just once more...

		// If this is set to zero, our calculation is exponential trapezoid
		// and has stiff time order two
		if(1) {
			// If one, we perform a cubic algorithmic fit and acheive third stiff order
			// with outrageous accuracy

			// Step one, back half way up
			//dv_i = dv_t*exp(t*k);
			tmpmem[FLUID_SLABPITCH] -= duint;

			// apply halfstep of new (k_0+k_1):
			k = .25*(tmpmem[5*FLUID_SLABPITCH]);

			dv_t   = dv_i*exp(-t*k);

			// accumulate new delta-v^2 and drag heating effect
			dvsq   = dv_t*dv_t;
			duint  = -0.5*dv_i*dv_i*mu*expm1(-2*k*t); //

			#ifdef THREAD0_PRINTS_DBG
			if(threadIdx.x == 0) { printf("b=%i,t=%i: halfstep: k=%le, dv_t = %le\n", blockIdx.x, threadIdx.x, k, dv_t); }
			#endif

			tmpmem[0] = sqrt(dvsq);
			tmpmem[FLUID_SLABPITCH] += duint;

			// store k_half in register 0: This is needed separately from k0 and k1
			cukern_GasDustDrag_GeneralLinearCore<true>(gas, dust, tmpmem, 0, 0, partNumel);

			// Richardson extrapolation formula for trapezoid method yields this formula
			// Note formula is convex combination of stable values and therefore unconditionally stable
			// This experimentally results in 3rd order convergence
			k = (0.16666666666666666667 *tmpmem[5*FLUID_SLABPITCH] +  0.66666666666666666667*tmpmem[0]);
			// if 3rd order
			//k = (0.21428571428571428571*tmpmem[5*FLUID_SLABPITCH] + 0.57142857142857142857*tmpmem[0]);
			// if 1st order
			//k = tmpmem[0];
		} else {
			k = .5*tmpmem[5*FLUID_SLABPITCH];
		}

		// The clever weighting of k above yields in it a value which will take us to the point that the
		// complex actual drag ends up at, to third order, when we do exp(-k t)
		// horray for path independent work integrals!

		mu      = mu * gas[0]; // mu was abused to compute the heating integral above

		// load & compute time-reversed delta-vx and stick velocity
		rhoginv = 1.0 / gas[0];

		pdustsq = -dust[2*FLUID_SLABPITCH] * dust[2*FLUID_SLABPITCH];
		vstick  = (gas[2*FLUID_SLABPITCH]+dust[2*FLUID_SLABPITCH]) / (gas[0] + dust[0]);
		dv_i    = (gas[2*FLUID_SLABPITCH])*rhoginv - dust[2*FLUID_SLABPITCH]/dust[0];

		dv_t    = dv_i*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation

		#ifdef THREAD0_PRINTS_DBG
		if(threadIdx.x == 0) { printf("final solve reporting: dv_i = %le, dt=%le, k = %le, dv_f = %le\n", dv_i, t, k, dv_t); }
		#endif

		// recalculate new differential velocities
		gas[2*FLUID_SLABPITCH] = gas[0]*vstick + dv_t*mu;
		dust[2*FLUID_SLABPITCH] = q = dust[0]*vstick - dv_t*mu;
		// accumulate change in dust kinetic energy
		pdustsq += q*q; //

		// do Y direction
		pdustsq -= dust[3*FLUID_SLABPITCH]*dust[3*FLUID_SLABPITCH];
		vstick  = (gas[3*FLUID_SLABPITCH]+dust[3*FLUID_SLABPITCH]) / (gas[0] + dust[0]);
		dv_i    = (gas[3*FLUID_SLABPITCH])*rhoginv - dust[3*FLUID_SLABPITCH]/dust[0];

		dv_t    = dv_i*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation
		gas[3*FLUID_SLABPITCH]     = gas[0]*vstick + dv_t*mu;
		dust[3*FLUID_SLABPITCH]= q = dust[0]*vstick - dv_t*mu;
		pdustsq += q*q;

		// do Z direction
		pdustsq -= dust[4*FLUID_SLABPITCH]*dust[4*FLUID_SLABPITCH];
		vstick  = (gas[4*FLUID_SLABPITCH]+dust[4*FLUID_SLABPITCH]) / (gas[0] + dust[0]);
		dv_i    = (gas[4*FLUID_SLABPITCH])*rhoginv - dust[4*FLUID_SLABPITCH]/dust[0];

		dv_t    = dv_i*exp(-t*k); // I assume it will auto-optimize this into one transcendental evaluation
		gas[4*FLUID_SLABPITCH]  = gas[0]*vstick + dv_t*mu;
		dust[4*FLUID_SLABPITCH] = q = dust[0]*vstick - dv_t*mu;
		pdustsq += q*q;

		// From conservation of total energy we have that the gas total energy decreases by whatever
		// amount the dust kinetic energy rises; Under (M_dust >> M_atom) the gas gets ~100% of heating
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

#define GRADBLOCKX 16
#define GRADBLOCKY 16

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
int prepareForExpMethod(MGArray *gas, MGArray *dust, MGArray *tempMem, GeometryParams geom, int spaceOrder, double scalingParameter)
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
	//long slabsize;

	// Iterate over all partitions, and here we GO!
	for(i = 0; i < gas->nGPUs; i++) {
		cudaSetDevice(gas->deviceID[i]);
		worked = CHECK_CUDA_ERROR("cudaSetDevice");
		if(worked != SUCCESSFUL) break;

		calcPartitionExtent(gas, i, sub);

		int3 arraysize; arraysize.x = sub[3]; arraysize.y = sub[4]; arraysize.z = sub[5];
		dim3 blocksize(GRADBLOCKX, GRADBLOCKY, 1);
		gridsize.x = arraysize.x / (blocksize.x - spaceOrder);
		gridsize.x += ((blocksize.x-spaceOrder) * gridsize.x < arraysize.x) * 1 ;
		if(isRZ) {
			gridsize.y = arraysize.z / (blocksize.y - spaceOrder); gridsize.y += ((blocksize.y-spaceOrder) * gridsize.y < arraysize.z);
		} else {
			gridsize.y = arraysize.y / (blocksize.y - spaceOrder); gridsize.y += ((blocksize.y-spaceOrder) * gridsize.y < arraysize.y) * 1;
		}
		gridsize.z = 1;

		gasPtr = gas->devicePtr[i]; // WARNING: this could be garbage if spaceOrder == 0 and we rx'd no potential array
		dustPtr = dust->devicePtr[i];

		tmpPtr = tempMem->devicePtr[i];
		//slabsize = gas->slabPitch[i] / 8;

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
					cukern_prepareForERKRZ_h2<<<gridsize, blocksize>>>(gasPtr, dustPtr, tmpPtr, arraysize);
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
#ifdef EXPO_DOTR
			dv = (gas[globAddr+2*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+2*FLUID_SLABPITCH]/dust[globAddr];
#else
			dv = (gas[globAddr+2*FLUID_SLABPITCH])/gas[globAddr] - dust[globAddr+2*FLUID_SLABPITCH]/dust[globAddr];
#endif
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
#ifdef EXPO_DOTR
			dv = (gas[globAddr+3*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+3*FLUID_SLABPITCH]/dust[globAddr];
#else
			dv = (gas[globAddr+3*FLUID_SLABPITCH])/gas[globAddr] - dust[globAddr+3*FLUID_SLABPITCH]/dust[globAddr];
#endif
			dvsq += dv*dv;
		}

		/* we must protect on both sides of this
		 * tl;dr: with only barrier B, warps 1 and 2 depart at the same time
		 * But if, suppose, warp 1 gets delayed in the slow sqrt() calculation while
		 * warp 2 goes ahead and runs all the way back to where barrier A is.
		 *
		 * Without barrier A, warp 2 will overwrite W (which for warp 1 is still U)
		 * and the calculation will be corrupted.
		 */
		__syncthreads();

		W[myLocAddr]       = gas2press(gas + (globAddr + deltaz));

		__syncthreads(); // barrier B

		if(IWrite) {
			deltaP           = LAMZ*(W[myLocAddr] - U[myLocAddr]);
			em[globAddr + 4*FLUID_SLABPITCH] = deltaP;
#ifdef EXPO_DOTR
			dv = (gas[globAddr+4*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+4*FLUID_SLABPITCH]/dust[globAddr];
#else
			dv = (gas[globAddr+4*FLUID_SLABPITCH])/gas[globAddr] - dust[globAddr+4*FLUID_SLABPITCH]/dust[globAddr];
#endif
			dvsq += dv*dv;


		em[globAddr] = sqrt(dvsq); // output initial delta-v
		em[globAddr + FLUID_SLABPITCH] = V[myLocAddr] / gas[globAddr]; // = P/rho = specific internal energy density
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

#ifdef EXPO_DOTR
		dv = (gas[globAddr+2*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+2*FLUID_SLABPITCH]/dust[globAddr];
#else
		dv = (gas[globAddr+2*FLUID_SLABPITCH])/gas[globAddr] - dust[globAddr+2*FLUID_SLABPITCH]/dust[globAddr];
#endif
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
		em[globAddr+4*FLUID_SLABPITCH] = 0.0;
#ifdef EXPO_DOTR
		dv = (gas[globAddr+3*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+3*FLUID_SLABPITCH]/dust[globAddr];
#else
		dv = (gas[globAddr+3*FLUID_SLABPITCH])/gas[globAddr] - dust[globAddr+3*FLUID_SLABPITCH]/dust[globAddr];
#endif
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

#ifdef EXPO_DOTR
		dv = (gas[globAddr+2*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+2*FLUID_SLABPITCH]/dust[globAddr];
#else
		dv = (gas[globAddr+2*FLUID_SLABPITCH])/gas[globAddr] - dust[globAddr+2*FLUID_SLABPITCH]/dust[globAddr];
#endif
		dvsq += dv*dv;

		// Calculate dt*(dphi/dz)
		deltaP         = LAMZ*(pressLoc[myLocAddr+GRADBLOCKX]-pressLoc[myLocAddr-GRADBLOCKX]);
		em[globAddr + 4*FLUID_SLABPITCH] = deltaP;
#ifdef EXPO_DOTR
		dv = (gas[globAddr+4*FLUID_SLABPITCH]+dragparams[7]*deltaP)/gas[globAddr] - dust[globAddr+4*FLUID_SLABPITCH]/dust[globAddr];
#else
		dv = (gas[globAddr+4*FLUID_SLABPITCH])/gas[globAddr] - dust[globAddr+4*FLUID_SLABPITCH]/dust[globAddr];
#endif
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

#undef GRADBLOCKX
#undef GRADBLOCKY





















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











