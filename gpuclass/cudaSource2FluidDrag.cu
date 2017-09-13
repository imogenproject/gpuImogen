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

#define PI 3.141592653589793

int sourcefunction_2FluidDrag(MGArray *fluidA, MGArray *fluidB, double gam, double sigmaGas, double muGas, double sigmaDust, double Mdust, double dt);

int solveDragEMP(MGArray *gas, MGArray *dust, double dt);

__global__ void cukern_GasDustDrag_full(double *gas, double *dust, double *tmpmem, int srcBlock, int dstBlock, int N);
__global__ void cukern_GasDustDrag_Epstein(double *gas, double *dust, double *vrel, int N);
__global__ void cukern_GasDustDrag_linearTime(double *gas, double *dust, double *vrel, int N);

__global__ void cukern_findInitialDeltaV(double *g, double *d, double *dv, unsigned long partNumel);

template <bool resetAccumulator>
__global__ void cukern_SolveRK_single(double *tmpmem, int d, double A, int i, double B, unsigned long partNumel);

template <bool resetAccumulator>
__global__ void cukern_SolveRK_double(double *tmpmem, int d, double F[2], int i[2], double B, unsigned long partNumel);

template <bool resetAccumulator>
__global__ void cukern_SolveRK_triple(double *tmpmem, int d, double F[3], int i[3], double B, unsigned long partNumel);

__global__ void cukern_SolveRK_final(double *tmpmem, int i, double B, double W, unsigned long partNumel);

__global__ void cukern_applyFinalDeltaV(double *g, double *d, double *dv_final, unsigned long partNumel);
__global__ void cukern_cvtToGasDust(double *g, double *d, unsigned long slabNumel, unsigned long partNumel);
__global__ void cukern_cvtToBarycentric(double *g, double *d, unsigned long slabNumel, unsigned long partNumel);

// Accept the following drag models:
// (1) full      : Use full Epstein+Stokes calculation with interpolation between all 4 quadrants
// (2) Epstein   : Use only Epstein force calculation, valid for any speed but only small particles
// (3) Linear    : Compute Epstein+Stokes in low-velocity limit, valid only for |delta v/c| << 1 (and strictly, Re < 1)

// PARITY CONVENTIONS ARE AS FOLLOWS:
// delta-V is defined as GAS VELOCITY MINUS DUST VELOCITY
// Drag force is positive in the direction of delta-V,
// i.e. d/dt(dust momentum) = F_drag and d/dt(gas momentum) = -F_drag
// ergo d/dt(delta_V) ~ -F_drag / mass

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if ((nrhs!=2) || (nlhs != 0)) mexErrMsgTxt("Wrong number of arguments: need cudaSource2FluidDrag(FluidManager, [sigma_gas, mu_gas, dia_dust, mass_dust, dt])\n");

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

	double *params = mxGetPr(prhs[1]);

	size_t ne = mxGetNumberOfElements(prhs[1]);
	if(ne < 5) {
		PRINT_FAULT_HEADER;
		printf("2nd argument to cudaSource2FluidDrag must have 5 elements.\nGiven argument has %i instead.\n", ne);
		PRINT_FAULT_FOOTER;
		DROP_MEX_ERROR("Crashing.");
	}	

	double fluidGamma = derefXdotAdotB_scalar(prhs[0], "gamma", NULL);
	double dt         = params[4];

	double sigmaGas   = params[0];
	double muGas      = params[1];
	double sigmaDust  = params[2];
	double muDust     = params[3];
	
	//1nm iron sphere, 300K -> 56m/s thermal velocity
	//10nm iron ball, 300K -> 1.79m/s thermal velocity
	//100nm iron ball, 300K -> 56mm/s thermal velocity
	
	status = CHECK_IMOGEN_ERROR(sourcefunction_2FluidDrag(&fluidA[0], &fluidB[0], fluidGamma, sigmaGas, muGas, sigmaDust, muDust, dt));

	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) {
		DROP_MEX_ERROR("2-fluid drag code crashed.");
	}

	return;

}

int sourcefunction_2FluidDrag(MGArray *fluidA, MGArray *fluidB, double gam, double sigmaGas, double muGas, double sigmaDust, double muDust, double dt)
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
		cudaMemcpyToSymbol((const void *)dragparams, &hostDrag[0], 6*sizeof(double), 0, cudaMemcpyHostToDevice);
		statusCode = CHECK_CUDA_ERROR("memcpyToSymbol");
		if(statusCode != SUCCESSFUL) break;
	}
	
	if(statusCode != SUCCESSFUL) {
		printf("Unsuccessful attempt to setup fluid drag parameters.\n");
		PRINT_FAULT_FOOTER;
	}

	statusCode = CHECK_IMOGEN_ERROR(solveDragEMP(fluidA, fluidB, dt));
	
	return statusCode;
}

/* Solves the action of gas-dust drag for one dust using the explicit midpoint method
 * 2nd order in time, not A-stable (dt < t_stop) */
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
dim3 smallgrid(1,1,1);

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

	// solve gas drag on y0, store in block 3
	cukern_GasDustDrag_full<<<gridsize, blocksize>>>(g, d, vrel, 0, 3, NE);

	// compute delta-v at t=1/2; store stage at block 4
	cukern_SolveRK_single<true><<<gridsize, blocksize>>>(vrel, 4, .5*dt, 3, 0, NE);

	// solve gas drag at t=1/2 using half stage, store in block 3
	cukern_GasDustDrag_full<<<gridsize, blocksize>>>(g, d, vrel, 4, 3, NE);

	cukern_SolveRK_final<<<gridsize, blocksize>>>(vrel, 3, 1.0, dt, NE);

	// compute new gas/dust momentum and temperature arrays
	cukern_applyFinalDeltaV<<<gridsize, blocksize>>>(g, d, vrel, NE);
}

for(i = 0; i < n; i++) {
	cudaSetDevice(gas->deviceID[i]);
	cudaFree((void *)tmpmem[i]);
}

return SUCCESSFUL; // FIXME: check this once its working

}

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

	// solve gas drag on y0, store in block 3
	cukern_GasDustDrag_full<<<gridsize, blocksize>>>(g, d, vrel, 0, 3, NE);

	// compute delta-v at t=1/2; store stage at block 4
	cukern_SolveRK_single<true><<<gridsize, blocksize>>>(vrel, 4, 0.5*dt, 3, bWeights[0], NE);

	// solve gas drag on k2, store in block 3
	cukern_GasDustDrag_full<<<gridsize, blocksize>>>(g, d, vrel, 4, 3, NE);

	// compute delta-v at t=1/2; store stage at block 4
	cukern_SolveRK_single<true><<<gridsize, blocksize>>>(vrel, 4, 0.5*dt, 3, bWeights[1], NE);

	// solve gas drag on k3, store in block 3
	cukern_GasDustDrag_full<<<gridsize, blocksize>>>(g, d, vrel, 4, 3, NE);

	// compute delta-v at t=1/2; store stage at block 4
	cukern_SolveRK_single<true><<<gridsize, blocksize>>>(vrel, 4, 1.0*dt, 3, bWeights[2], NE);

	// solve gas drag on k4, store in block 3
	cukern_GasDustDrag_full<<<gridsize, blocksize>>>(g, d, vrel, 4, 3, NE);

	// add block 3 to accumulator, rescale by dt / 6.0 and add y0 to find final dv.
	cukern_SolveRK_final<<<gridsize, blocksize>>>(vrel, 3, bWeights[3], bRescale, NE);

	// compute new gas/dust momentum and temperature arrays
	cukern_applyFinalDeltaV<<<gridsize, blocksize>>>(g, d, vrel, NE);
}

for(i = 0; i < n; i++) {
	cudaSetDevice(gas->deviceID[i]);
	cudaFree((void *)tmpmem[i]);
}

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
	if(Re > 800) {
		// .44
		return 0.22;
	}
	// 24 Re^-.6
	return 12.0*pow(Re,-0.6);
}

/* Computes drag between gas and dust particles, utilizing precomputed factors  in
 * dragparams[]:
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
		C_d = { 24/Rd	 |     Rd < 1    }
			  { 24/Rd^-.6 | 1 < Rd < 800  }
			  { .44       |     Rd >= 800 } is the experimentally known Stokes drag coefficient
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
 *	  K_stokes  = .5 C_d pi |dv|
 * And the interpolation cutover s_0 is chosen s_0 = (4/9) MFP
 */

/* ALGORITHM
 * 0. STORE
 * 		alpha   = dragparams[0] = 128*sigma_gas * D_dust / (5 * mu_gas * sqrt(pi))
 * 		beta	= dragparams[1] = 128*pi*(gamma-1) / 9
 *		epsilon = dragparams[2] = pi^2 / gamma
 *		theta   = dragparams[3] = 5*sqrt(gamma*pi/2)*mu_gas/(144*sigma_gas)
 *	 in __constant__ memory before launching kernel
 *
 * 1. load magnitude dv 
 * 3. compute uinternal = Ugas(t)/rhogas	 // Specific internal energy
 * 4. compute d0 = theta / rho_gas			  // 4/9 of gas mean free path - used to interp between epstein & stokes drag
 * 5. compute Re = alpha * sqrt(|dv^2|/uint) * rho_gas  // Reynolds number for stokes gas drag
 * 6. compute C_hat = C_hat(Re)			 // Drag coeff, from model or (TODO) table?
 * 7. compute f_drag = rho_gas * D_dust^2 * (d0^2 * sqrt(beta*uinternal + epsilon*|dv|^2) + D_dust^2 * C_hat * sqrt(|dv|^2) ) * rho_dust * vector{dv} / (m_dust * (d0^2+D_dust^2));
 * 7. compute f_drag = rho_gas * D_dust^2 * (d0^2 * sqrt(beta*uinternal(dv) + epsilon*dv^2) + D_dust^2 * C_hat * dv ) * rho_dust * dv / (m_dust * (d0^2+D_dust^2)); 
 * -> a_rel = -f_d / rho_g - f_d / rho_d = - f_d (1/rho_g + 1/rho_d) = -f_d (rho_g + rho_d) / (rho_g rho_d) = -f_d / reduced mass
 * 7. compute a_rel  = (d0^2 * sqrt(beta*uinternal(dv) + epsilon*dv^2) + D_dust^2 * C_hat * dv ) * dv * D_dust^2 * (rho_g + rho_d) / (m_dust * (d0^2+D_dust^2));
 *
 * 7. compute a_rel = [ d0^2 * sqrt(beta*uinternal(dv) + epsilon*dv^2) + D_dust^2 * C_hat(dv) * dv] * dv * D_dust^2 * (rho_g + rho_d) / (m_dust * (d0^2 + D_dust^2));

 *	W/
 *	y1 = d0^2 * D_dust^2 * (rho_gas + rho_dust) / (m_dust * (d0^2+D_dust^2))
 *	y2 = D_dust^2  D_dust^2 * rho_dust / (m_dust * (d0^2+D_dust^2))
 */

#define ALPHA   dragparams[0]
#define BETA	dragparams[1]
#define EPSILON dragparams[2]
#define THETA   dragparams[3]
#define DUSTMASS dragparams[4]
#define SIGMA_DUST dragparams[5]

/* This function directly computes the gas-dust drag force in the full (stokes+epstein) regime
 * This is suited for weaker drag or strange regimes, but unnecessary and time-consuming for
 * small particles which will never exit the low-speed Epstein regime.
 * Uses stage value stored at srcBlock, writes acceleration into dstBlock */
__global__ void cukern_GasDustDrag_full(double *gas, double *dust, double *tmpmem, int srcBlock, int dstBlock, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	double rhoA, rhoB;   // gas and dust densities respectively 
	double magdv;	// magnitude velocity difference
	double uinternal;	// specific internal energy density
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
	
		// make sure computation includes gas heating term!
		uinternal = tmpmem[FLUID_SLABPITCH] + rhoB * (tmpmem[0]*tmpmem[0] - magdv*magdv) / (rhoA + rhoB);

		kEpstein = sqrt(BETA*uinternal + EPSILON*magdv*magdv);

		// FIXME this implementation is poorly conditioned (re ~ 1/v for v << v0)
		Re = ALPHA*magdv*rhoA/sqrt(uinternal);
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
 */
__global__ void cukern_GasDustDrag_linearTime(double *gas, double *dust, double *vrel, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	double rhoA, rhoB;   // gas and dust densities respectively
	double magdv;	// magnitude velocity difference
	double uinternal;	// specific internal energy density
	double tau;	// Relative acceleration (d/dt of vrel) divided by vrel
	double sigma0;
	double kEpstein, kStokes, Re;

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

		// FIXME this implementation is poorly conditioned (re ~ 1/v for v << v0)
		Re = ALPHA*magdv*rhoA/sqrt(uinternal);
		kStokes = drag_coeff(Re) * magdv;

		sigma0 = THETA / rhoA; // sqrt(pi)*(4 l_mfp / 9) = sqrt(pi) * s0
		sigma0 *= sigma0; // = pi s0^2 = epstein/stokes cutover crossection

		//a_rel = ( sigma0 * sqrt(beta*uinternal(dv) + epsilon*dv^2) + D_dust^2 * C_hat(dv) * dv) * dv * D_dust^2 * (rho_g + rho_d) / (m_dust * (d0^2 + D_dust^2));
		tau = ( sigma0 * kEpstein + SIGMA_DUST * kStokes) * SIGMA_DUST * (rhoA + rhoB) / (DUSTMASS * (sigma0 + SIGMA_DUST));

		vrel[3*FLUID_SLABPITCH] = tau;

		gas += blockDim.x*gridDim.x;
		dust += blockDim.x*gridDim.x;
		vrel += blockDim.x*gridDim.x;
	}

}


/* Computes initial magnitude velocity ("w") into dv[0] and dv[slabPitch]
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
	// FIXME - low uniform low temperature
	d[FLUID_SLABPITCH] = .5*dustmomfin/d[0] + 1e-4 * d[0];

	x +=  blockDim.x*gridDim.x;
	g += blockDim.x*gridDim.x;
	d += blockDim.x*gridDim.x;
	dv_final += blockDim.x*gridDim.x;
}

}

/* Converts a gas fluid *g and dust fluid *d, in place, to barycenter / difference
 * velocities:
 * [rho Etot   p1 p2 p3]_gas	 [rho_g e_int  sigma_1 sigma_2 sigma_3]
 * [rho T_dust p1 p2 p3]_dust -> [rho_d T_dust delta_1 delta_2 delta_3]
 * . */
__global__ void cukern_cvtToGasDust(double *g, double *d, unsigned long slabNumel, unsigned long partNumel)
{
int x = threadIdx.x + blockIdx.x*blockDim.x;
g += x;
d += x;

double rho1, rho2, pa, pb, sigma, delta;

while(x < partNumel) {
	rho1 = g[0];
	rho2 = d[0];

	sigma = g[2*slabNumel];
	delta = d[2*slabNumel];
	//sigma = (rho1 v1 + rho2 v2)/(rho1 + rho2) = (p1 + p2) / (rho1 + rho2);
	sigma = (pb + pa)/(rho1 + rho2);
	delta = pa/rho1 - pb/rho2;
	g[2*slabNumel] = sigma;
	d[2*slabNumel] = delta;

	pa = g[3*slabNumel];
	pb = d[3*slabNumel];
	//sigma = (rho1 v1 + rho2 v2)/(rho1 + rho2) = (p1 + p2) / (rho1 + rho2);
	sigma = (pb + pa)/(rho1 + rho2);
	delta = pa/rho1 - pb/rho2;
	g[3*slabNumel] = sigma;
	d[3*slabNumel] = delta;

	pa = g[4*slabNumel];
	pb = d[4*slabNumel];
	//sigma = (rho1 v1 + rho2 v2)/(rho1 + rho2) = (p1 + p2) / (rho1 + rho2);
	sigma = (pb + pa)/(rho1 + rho2);
	delta = pa/rho1 - pb/rho2;
	g[4*slabNumel] = sigma;
	d[4*slabNumel] = delta;

	x += blockDim.x*gridDim.x;
	g += blockDim.x*gridDim.x;
	d += blockDim.x*gridDim.x;
}

}

/* Converts...
 * [rho_g e_int  sigma_1 sigma_2 sigma_3]  -> [rho Etotal p1 p2 p3]_gas
 * [rho_d T_dust delta_1 delta_2 delta_3]  -> [rho T_dust p1 p2 p3]_dust
 */
__global__ void cukern_cvtToBarycentric(double *g, double *d, unsigned long slabNumel, unsigned long partNumel)
{
int x = threadIdx.x + blockIdx.x*blockDim.x;
g += x;
d += x;

double rho1, rho2, pa, pb, sigma, delta;

while(x < partNumel) {
	rho1 = g[0];
	rho2 = d[0];

	pa = g[2*slabNumel];
	pb = d[2*slabNumel];
	//sigma = (rho1 v1 + rho2 v2)/(rho1 + rho2) = (p1 + p2) / (rho1 + rho2);
	sigma = (pb + pa)/(rho1 + rho2);
	delta = pa/rho1 - pb/rho2;
	g[2*slabNumel] = sigma;
	d[2*slabNumel] = delta;

	pa = g[3*slabNumel];
	pb = d[3*slabNumel];
	//sigma = (rho1 v1 + rho2 v2)/(rho1 + rho2) = (p1 + p2) / (rho1 + rho2);
	sigma = (pb + pa)/(rho1 + rho2);
	delta = pa/rho1 - pb/rho2;
	g[3*slabNumel] = sigma;
	d[3*slabNumel] = delta;

	pa = g[4*slabNumel];
	pb = d[4*slabNumel];
	//sigma = (rho1 v1 + rho2 v2)/(rho1 + rho2) = (p1 + p2) / (rho1 + rho2);
	sigma = (pb + pa)/(rho1 + rho2);
	delta = pa/rho1 - pb/rho2;
	g[4*slabNumel] = sigma;
	d[4*slabNumel] = delta;

	x += blockDim.x*gridDim.x;
	g += blockDim.x*gridDim.x;
	d += blockDim.x*gridDim.x;
}

}

