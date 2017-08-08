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

int sourcefunction_2FluidDrag(MGArray *fluidA, MGArray *fluidB, double gam, double sigma, double mu, double Ddust, double Mdust, double dt);

int solveDragEMP(MGArray *gas, MGArray *dust, double dt);

__global__ void cukern_GasDustDrag(double *gas, double *dust, double *vrel, int N);
__global__ void cukern_findInitialDeltaV(double *g, double *d, double *dv, unsigned long slabNumel, unsigned long partNumel);
__global__ void cukern_SolveDvDt(double *tmparray, double dt, unsigned long slabNumel, unsigned long partNumel);
__global__ void cukern_applyFinalDeltaV(double *g, double *d, double *dv_final, unsigned long slabNumel, unsigned long partNumel);
__global__ void cukern_cvtToGasDust(double *g, double *d, unsigned long slabNumel, unsigned long partNumel);
__global__ void cukern_cvtToBarycentric(double *g, double *d, unsigned long slabNumel, unsigned long partNumel);

// Accept the following drag models:
// (1) Allregimes: Use full Epstein+Stokes calculation, notionally valid for any particle size & velocity differential
// (2) Epstein   : Use only Epstein force calculation, valid for any speed but only small particles
// (3) Stokes    : Use only Stokes force calculation, valid for any speed (incl supersonic?) but only large particles
// (4) Slow      : Compute Epstein+Stokes in low-velocity limit, valid for any size but only |delta v/c| << 1

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

	// FIXME check numel in 2nd parameter
	double sigma = params[0];
	double fluidGamma = derefXdotAdotB_scalar(prhs[0], "gamma", NULL);
	double gasMu = params[1];
	double dustDiameter = params[2];
	double dustMass = params[3];
	double dt    = params[4];
    
	//1nm iron sphere, 300K -> 56m/s thermal velocity?
	//10nm iron ball, 300K -> 1.79m/s thermal velocity?
	//100nm iron ball, 300K -> 56mm/s thermal velocity?
    
	status = CHECK_IMOGEN_ERROR(sourcefunction_2FluidDrag(&fluidA[0], &fluidB[0], fluidGamma, sigma, gasMu, dustDiameter, dustMass, dt));

	if(status != SUCCESSFUL) {
		DROP_MEX_ERROR("2-fluid drag code crashed.");
	}

	return;

}

int sourcefunction_2FluidDrag(MGArray *fluidA, MGArray *fluidB, double gam, double sigma, double mu, double Ddust, double Mdust, double dt)
{
	int i;
	int sub[6];
	int hostFluidParams[4];

	int statusCode = SUCCESSFUL;

    double hostDrag[16];
    hostDrag[0] = 128.0 * sigma * Ddust / (5 * mu * sqrt(PI));
    hostDrag[1] = 128*PI*(gam-1.0)/9.0;
    hostDrag[2] = PI*PI/gam;
    hostDrag[3] = 5*sqrt(gam*PI/2.0)*mu / (144.0 * sigma);
    hostDrag[4] = Mdust; // FIXME: this should be an array perhaps?
    hostDrag[5] = Ddust*Ddust;
    
	for(i = 0; i < fluidA->nGPUs; i++) {
		cudaSetDevice(fluidA->deviceID[i]);
		statusCode = CHECK_CUDA_ERROR("cudaSetDevice");
		if(statusCode != SUCCESSFUL) break;

		calcPartitionExtent(fluidA, i, &sub[0]);
		hostFluidParams[0] = sub[3];
		hostFluidParams[1] = sub[4];
		hostFluidParams[2] = sub[5];
		hostFluidParams[3] = fluidA->slabPitch[i] / sizeof(double); // This is important, due to padding, is isn't just .partNumel
		cudaMemcpyToSymbol(devFluidParams, &hostFluidParams[0], 4*sizeof(int), 0, cudaMemcpyHostToDevice);
		statusCode = CHECK_CUDA_ERROR("memcpyToSymbol");
		if(statusCode != SUCCESSFUL) break;
		cudaMemcpyToSymbol(dragparams, &hostDrag[0], 6*sizeof(double), 0, cudaMemcpyHostToDevice);
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
 * 2nd order in time, not A-stable/L-stable */
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
	cudaMalloc((void **)(&tmpmem[i]), 4*gas->slabPitch[i]);
	statusCode = CHECK_CUDA_ERROR("cudaMalloc tmpmem for solveDragEMP");
	if(statusCode != SUCCESSFUL) break;
    // store initial v_relative, current v_relative, ini_uint, acceleration in slabs 1, 2, 3 and 4
}

if(statusCode != SUCCESSFUL) {
	printf("Unable to grab temporary memory: Crashing.\n");
	PRINT_FAULT_FOOTER;
	return statusCode;
}

dim3 blocksize(128, 1, 1);
dim3 gridsize(32, 1, 1);

for(i = 0; i < n; i++) {
	// avoid launching tons of threads for small problems
	gridsize.x = 32;
	if(ROUNDUPTO(gas->partNumel[i], 128)/128 < 32) {
		gridsize.x = ROUNDUPTO(gas->partNumel[i], 128)/128;
	}

	cudaSetDevice(gas->deviceID[i]);

	g = gas->devicePtr[i];
	d = dust->devicePtr[i];
	vrel = tmpmem[i] + 0;
	// compute initial delta-v
	cukern_findInitialDeltaV<<<gridsize, gridsize>>>(g, d, vrel, gas->slabPitch[i]/8, gas->partNumel[i]);

	// solve gas drag at t=0
	cukern_GasDustDrag<<<gridsize, blocksize>>>(g, d, vrel, gas->partNumel[i]);

	// compute delta-v at t=1/2
	cukern_SolveDvDt<<<gridsize, blocksize>>>(vrel, dt, gas->slabPitch[i]/8, gas->partNumel[i]);

	// solve gas drag at t=1/2
	cukern_GasDustDrag<<<gridsize, blocksize>>>(g, d, vrel, gas->partNumel[i]);

	// compute delta-v at t=1
	cukern_SolveDvDt<<<gridsize, blocksize>>>(vrel, dt, gas->slabPitch[i]/8, gas->partNumel[i]);

	// compute new gas/dust momentum and temperature arrays
	cukern_applyFinalDeltaV<<<gridsize, blocksize>>>(g, d, vrel, gas->slabPitch[i]/8, gas->partNumel[i]);
}

for(i = 0; i < n; i++) {
	cudaSetDevice(gas->deviceID[i]);
	cudaFree((void *)tmpmem[i]);
}

return SUCCESSFUL; // FIXME: check this once its working

}

/* This function returns the normal Stokes coefficients, scaled by pi/2
 * This parameter is experimentally measured except for the low-Re regime */
__device__ double drag_coeff(double Re)
{
	if(Re < 1) {
		// 24 / Re
		return PI * 12 / Re;
	}
	if(Re > 800) {
		// .44
		return PI * 0.22;
	}
	// 24 Re^-.6
	return PI*12.0*pow(Re,-0.6);
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
 * where Fdrag may be computed, given
		dv == Vgas - Vdust is the velocity differential,
		nu == ( 5 mu c_s / 64 sigma rho ) sqrt(pi/gamma) is the kinematic viscosity
		Rd == 2 s |dv| / nu is the local Reynolds number for a particle, and
		C_d = { 24/Rd     |     Rd < 1    }
		      { 24/Rd^-.6 | 1 < Rd < 800  }
		      { .44       |     Rd >= 800 } is the experimentally known Stokes drag coefficient
		MFP = sqrt(gamma pi / 2) 5 m_gas / (64 sigma_gas rho_gas)
 * We may extract common factors from both drag regimes,
 * multiply by the dust number density to get the volume force density, and
 * asymptotically interpolate between them to find that
 *							n_dust
 * Fdrag = s^2 rho_gas [ s0^2 K_epstein + s^2 K_stokes ] -------- \vec{dv}
 *						       s0^2+s^2
 * Where we find the particledynamic and viscous-fluid drag coefficients as
 *      K_epstein = (4 pi /3) sqrt(8 / gamma pi) sqrt(c_s^2 + 9 pi |dv|^2 / 128)
 *		= sqrt(dragparams[0]*uint + dragparams[1]* dv.dv)
 *      K_stokes  = .5 C_d pi |dv|
 * And the interpolation cutover s_0 is chosen s_0 = (4/9) MFP
 */

/* ALGORITHM
 * 0. STORE
 * 		alpha   = dragparams[0] = 128*sigma_gas * D_dust / (5 * mu_gas * sqrt(pi))
 * 		beta    = dragparams[1] = 128*pi*(gamma-1) / 9
 *		epsilon = dragparams[2] = pi^2 / gamma
 *		theta   = dragparams[3] = 5*sqrt(gamma*pi/2)*mu_gas/(144*sigma_gas)
 *     in __constant__ memory before launching kernel
 *
 * 1. load magnitude dv 
 * 3. compute uinternal = Ugas(t)/rhogas     // Specific internal energy
 * 4. compute d0 = theta / rho_gas		      // 4/9 of gas mean free path - used to interp between epstein & stokes drag
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
#define BETA    dragparams[1]
#define EPSILON dragparams[2]
#define THETA   dragparams[3]
#define DUSTMASS dragparams[4]
#define DDUSTSQR dragparams[5]

/* This function directly computes the gas-dust drag force in the full (stokes+epstein) regime
 * This is suited for weaker drag or strange regimes, but unnecessary and time-consuming for
 * small particles which will never exit the low-speed Epstein regime.
 */
__global__ void cukern_GasDustDrag(double *gas, double *dust, double *vrel, int N)
{
	int i = threadIdx.x + blockIdx.x*gridDim.x;

	double rhoA, rhoB;   // gas and dust densities respectively 
	double magdv;	// magnitude velocity difference
	double uinternal;    // specific internal energy density
	double Re;	   // Spherical particle Reynolds number
	double accel;	// Relative acceleration (d/dt of vrel)
	double d0squared;
	double kEpstein, kStokes;

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
	
		d0squared = THETA / rhoA;
		d0squared *= d0squared;

		 //a_rel = ( d0squared * sqrt(beta*uinternal(dv) + epsilon*dv^2) + D_dust^2 * C_hat(dv) * dv) * dv * D_dust^2 * (rho_g + rho_d) / (m_dust * (d0^2 + D_dust^2));
		accel = ( d0squared * kEpstein + DDUSTSQR * kStokes) * magdv * DDUSTSQR * (rhoA + rhoB) / (DUSTMASS * (d0squared + DDUSTSQR));
	
		vrel[3*FLUID_SLABPITCH] = accel;
	
		gas += blockDim.x*gridDim.x;
		dust += blockDim.x*gridDim.x;
		vrel += blockDim.x*gridDim.x;
	}

}

/* Computes initial magnitude velocity ("w") into dv[0]
 * and computes Uint_ini (e_internal / rho evaluated at original |w|) into dv[slabNumel] */
__global__ void cukern_findInitialDeltaV(double *g, double *d, double *dv, unsigned long slabNumel, unsigned long partNumel)
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
    
    q = g[2*slabNumel];
    u = q*rhoginv - d[2*slabNumel]*rhodinv;
    momsq = q*q;
    dvsq = u*u;
    q = g[3*slabNumel];
    u = q*rhoginv - d[3*slabNumel]*rhodinv;
    momsq += q*q;
    dvsq += u*u;
    q = g[3*slabNumel];
    u = q*rhoginv - d[4*slabNumel]*rhodinv;
    momsq += q*q;
    dvsq += u*u;
    
    // Store magnitude delta-v and initial specific internal energy for use by gas drag routine
	dv[0] = dv[slabNumel] = sqrt(dvsq);
    dv[2*slabNumel]       = (g[slabNumel] - .5*momsq * rhoginv)*rhoginv;

	x += blockDim.x*gridDim.x;
	g += blockDim.x*gridDim.x;
	d += blockDim.x*gridDim.x;
	dv+= blockDim.x*gridDim.x;
}

}

/* dv_rel/dt = dv_gas/dt - dv_dust/dt
 *	   = -F / rho_g - (F / rho_d)
 *	   = -F (1/rho_g + 1/rho_d)
 *	   = -F (rho_g + rho_d) / (rho_g rho_d) = -F/rho_reduced
 */
__global__ void cukern_SolveDvDt(double *tmparray, double dt, unsigned long slabNumel, unsigned long partNumel)
{
int x = threadIdx.x + blockIdx.x*blockDim.x;
tmparray += x;

while(x < partNumel) {
	// solve pdot
	tmparray[slabNumel]	   -= tmparray[3*slabNumel] * dt;

	x += blockDim.x*gridDim.x;
	tmparray += blockDim.x*gridDim.x;
}

}

/* From the initial momentum difference from *gas and *dust, computes the change in their momentum
 * densities to reach momentum difference *dp, given the relative fraction of acceleration
 * experienced by the gas and dust particles, and applies total energy conservation to solve
 * the gas/dust energy densities */
__global__ void cukern_applyFinalDeltaV(double *g, double *d, double *dv_final, unsigned long slabNumel, unsigned long partNumel)
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
    
    p1 = g[2*slabNumel];
    p2 = d[2*slabNumel];
    vstick[0] = (p1+p2)/(rhog+rhod);
    dvhat[0] = p1/rhog - p2/rhod;
    
    p1 = g[3*slabNumel];
    p2 = d[3*slabNumel];
    vstick[1] = (p1+p2)/(rhog+rhod);
    dvhat[1] = p1/rhog - p2/rhod;
    
    p1 = g[4*slabNumel];
    p2 = d[4*slabNumel];
    vstick[2] = (p1+p2)/(rhog+rhod);
    dvhat[2] = p1/rhog - p2/rhod;
    
    a = dv_final[slabNumel] / sqrt(dvhat[0]*dvhat[0] + dvhat[1]*dvhat[1]+dvhat[2]*dvhat[2]);
    dvhat[0] *= a;
    dvhat[1] *= a;
    dvhat[2] *= a;
    
    b = rhog*rhod/(rhog+rhod);

    dustmom = d[2*slabNumel]*d[2*slabNumel];
    g[2*slabNumel] = rhog*vstick[0] + dvhat[0]*b;
    d[2*slabNumel] = c = rhod*vstick[0] - dvhat[0]*b;
    dustmomfin = c*c;
    
    dustmom += d[3*slabNumel]*d[3*slabNumel];
    g[3*slabNumel] = rhog*vstick[1] + dvhat[1]*b;
    d[3*slabNumel] = c = rhod*vstick[1] - dvhat[1]*b;
    dustmomfin += c*c;
    
    dustmom += d[4*slabNumel]*d[4*slabNumel];
    g[4*slabNumel] = rhog*vstick[2] + dvhat[2]*b;
    d[4*slabNumel] = c = rhod*vstick[2] - dvhat[2]*b;
    dustmomfin += c*c;
    
	// From conservation of total energy
	// d/dt (KE_gas + Eint_gas + KE_dust) = 0
	// d/dt (KE_gas + Eint_gas) = -d/dt(KE_dust)
	// Etot_gas(after) - Etot_gas(before) = -(KE_dust(after)-KE_dust(before))
	// -> Etot_gas += KE_dust(ini) - KE_dust(fin)
	g[slabNumel] += .5*(dustmom - dustmomfin)/d[0];

	x +=  blockDim.x*gridDim.x;
	g += blockDim.x*gridDim.x;
	d += blockDim.x*gridDim.x;
	dv_final += blockDim.x*gridDim.x;
}

}

/* Converts a gas fluid *g and dust fluid *d, in place, to barycenter / difference
 * velocities:
 * [rho Etot   p1 p2 p3]_gas     [rho_g e_int  sigma_1 sigma_2 sigma_3]
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

