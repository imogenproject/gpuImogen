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
#include "cudaCommon.h"

#include "cudaFreeRadiation.h"

/* THIS FUNCTION
   cudaFreeRadiation solves the operator equation
   dE = -beta rho^2 T^theta dt

   typically written with radiation rate Lambda,
   Lambda = beta rho^(2-theta) Pgas^(theta)

   where E is the total energy density, dt the time to pass, beta the radiation strength scale
   factor, rho the mass density, Pgas the thermal pressure, and theta parameterizes the
   radiation (nonrelativistic bremsstrahlung is theta = 0.5)

   It implements a temperature floor (Lambda = 0 for T < T_critical) and checks for negative
   energy density both before (safety) and after (time accuracy truncation) the physics.

   FIXME: This function should have a "cell encountered complete cooling" flag so we can warn the CFL controller...
*/

__global__ void cukern_FreeHydroRadiationRate(double *rho, double *px, double *py, double *pz, double *E, double *radrate, int numel);
__global__ void cukern_FreeMHDRadiationRate(double *rho, double *px, double *py, double *pz, double *E, double *bx, double *by, double *bz, double *radrate, int numel);

template <unsigned int radiationAlgorithm>
__global__ void cukern_FreeHydroRadiation(double *rho, double *px, double *py, double *pz, double *E, int numel);
template <unsigned int radiationAlgorithm>
__global__ void cukern_FreeMHDRadiation(double *rho, double *px, double *py, double *pz, double *E, double *bx, double *by, double *bz, int numel);

// Define prelaunch-knowable scalars
__constant__ __device__ double radparam[8];
#define GAMMA_M1 radparam[0]
#define STRENGTH radparam[1]
#define EXPONENT radparam[2]
#define TWO_MEXPONENT radparam[3]
#define TFLOOR radparam[4]
#define ANALYTIC_SCALE radparam[5] // Analytic power law uses (theta-1)*strength*(gamma-1)
#define ONE_MINUS_THETA radparam[6]
#define INVERSE_ONE_M_THETA radparam[7]

#define BLOCKDIM 256
#define GRIDDIM 64

// These and the freeHydroRadiation templating are because of the different
// integral outcomes when theta is exactly zero (P-independent) or one (logarithm outcome)
#define ALGO_THETAZERO 0
#define ALGO_THETAONE 1
#define ALGO_GENERAL_ANALYTIC 2
#define ALGO_GENERAL_IMPLICIT 3

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	if ((nrhs != 5) || (nlhs > 1))
		mexErrMsgTxt("Wrong number of arguments. Expected forms: rate = cudaFreeRadiation(FluidManager, bx, by, bz, [gamma theta beta*dt Tmin isPureHydro]) or cudaFreeRadiation(FluidManager, bx, by, bz, [gamma theta beta*dt Tmin isPureHydro]\n");

	CHECK_CUDA_ERROR("Entering cudaFreeRadiation");

	double *inputParams = mxGetPr(prhs[4]);
	int ne = mxGetNumberOfElements(prhs[4]);
	if(ne != 5) {
		printf("Parameter vector (arg 5) to cudaFreeRadiation has %i arguments and not 5. Error.\n", ne);
		DROP_MEX_ERROR("Aborting.");
	}

	double gam      = inputParams[0];
	double exponent = inputParams[1];
	double strength = inputParams[2];
	double minTemp  = inputParams[3];
	int isHydro     = (int)inputParams[4] != 0;

	MGArray f[8]; // 0..4 = fluid, 5..7 = B
	int worked;

	worked = MGA_accessFluidCanister(prhs[0], 0, &f[0]);
        // This returns [rho E px py pz] in arrays [0...4],
	// We need [rho px py pz E] to avoid reordering BS in the kernel caller
	MGArray s;
	s = f[1];
	f[1] = f[2];
	f[2] = f[3];
	f[3] = f[4];
	f[4] = s;


	if( isHydro == false ) {
		worked = MGA_accessMatlabArrays(prhs, 1, 3, &f[5]);
		if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) {
			DROP_MEX_ERROR("failed to access GPU arrays entering cudaFreeRadiation.\n");
		}
	}

	MGArray *dest = NULL;
	/* If NULL, radiation is applied to fluid using given timestep
	 * It not NULL, radiation RATE is written to dest */
	if(nlhs == 1) {
		dest = MGA_createReturnedArrays(plhs, 1, &f[0]);
	}

	worked = sourcefunction_OpticallyThinPowerLawRadiation(&f[0], dest, isHydro, gam, exponent, strength, minTemp);
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) { DROP_MEX_ERROR("Calculation of radiation failed!"); }

	if(nlhs == 1) free(dest);

	return;
}


int sourcefunction_OpticallyThinPowerLawRadiation(MGArray *fluid, MGArray *radRate, int isHydro, double gamma, double exponent, double prefactor, double minimumTemperature)
{

	int returnCode = SUCCESSFUL;
	double hostRP[8];
	hostRP[0] = gamma-1.0;
	hostRP[1] = prefactor;
	hostRP[2] = exponent;
	hostRP[3] = 2.0 - exponent;
	hostRP[4] = minimumTemperature;
	hostRP[5] = (exponent-1.0)*prefactor*(gamma-1);
	hostRP[6] = 1.0-exponent;
	hostRP[7] = 1.0/(1.0-exponent);

	int j, k;
	for(j = 0; j < fluid->nGPUs; j++) {
		cudaSetDevice(fluid->deviceID[j]);
		cudaMemcpyToSymbol((const void *)radparam, hostRP, 8*sizeof(double), 0, cudaMemcpyHostToDevice);
		returnCode = CHECK_CUDA_ERROR("cudaMemcpyToSymbol");
		if(returnCode != SUCCESSFUL) break;
	}
	if(returnCode != SUCCESSFUL) return returnCode;


	int sub[6];
	int kernNumber;

	for(j = 0; j < fluid->nGPUs; j++) {
		calcPartitionExtent(fluid, j, sub);
		cudaSetDevice(fluid->deviceID[j]);
		CHECK_CUDA_ERROR("cudaSetDevice");

		double *ptrs[8];
		for(k = 0; k < 8; k++) { ptrs[k] = fluid[k].devicePtr[j]; }

		int nx = fluid->partNumel[j];

		// Run through the conditionals that pick the algorithm applicable to the current case
		switch(1*(isHydro == 1) + 2*(radRate != NULL)) {
		case 0: { // Apply radiation to MHD gas
			if(exponent == 0.0) {
				cukern_FreeMHDRadiation<ALGO_THETAZERO><<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], nx);
				kernNumber = 1;
			} else if(exponent == 1.0) {
				cukern_FreeMHDRadiation<ALGO_THETAONE><<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], nx);
				kernNumber = 2;
			} else {
				cukern_FreeMHDRadiation<ALGO_GENERAL_ANALYTIC><<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], nx);
				kernNumber = 3;
			}

			break; }
		case 1: { // Apply radiation to hydrodynamic gas
			if(exponent == 0.0) {
				cukern_FreeHydroRadiation<ALGO_THETAZERO><<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], nx);
				kernNumber = 4;
			} else if(exponent == 1.0) {
				cukern_FreeHydroRadiation<ALGO_THETAONE><<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], nx);
				kernNumber = 5;
			} else {
				cukern_FreeHydroRadiation<ALGO_GENERAL_ANALYTIC><<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], nx);
				kernNumber = 6;
			}
			break; }
		case 2: { // Calculate radiation rate under MHD
			cukern_FreeMHDRadiationRate<<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], radRate->devicePtr[j], nx);
			kernNumber = 7;
			break; }
		case 3: { // Calculate radiation rate under hydrodynamics
			cukern_FreeHydroRadiationRate<<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], radRate->devicePtr[j], nx);
			kernNumber = 8;
			break; }
		}
		returnCode = CHECK_CUDA_LAUNCH_ERROR(BLOCKDIM, GRIDDIM, fluid, kernNumber, "cudaFreeRadiation, int=kernel number attempted (see source)");
		if(returnCode != SUCCESSFUL) break;
	}

	return returnCode;

}

/* NOTE: This uses an explicit algorithm to perform radiation,
 * i.e. E[t+dt] = E[t] - Lambda[t] dt with radiation rate Lambda
 * This is conditionally stable with a CFL set by Lambda dt < E
 * 
 * Normally E / Lambda >> [dx / (c+max(Vx)) ], i.e. cooling time much
 * longer than advection time, but an implicit algorithm would be
 * wise as it is unconditionally stable. */

/* ALGORITHMIC NOTES:
 * In all these cases it is important to be cognizant of the critical
 * changes in behavior depending on theta. The exact solution of the
 * cooling operator has the form
 * Tfin = [Tini^q + phi]^(1/q)
 * in MOST cases.
 *
 * The breakdown by the cooling exponent theta is as follows:
 *
 * CASE theta >= 1: cooling of a uniform flow never finishes: phi is
 * positive, q is negative, and Tfin > 0 if Tini > 0; The solution
 * is a fractional power.
 * 
 * CASE theta == 1: Temperature decays exponentially.
 *
 * FOR ANY theta < 1, q is positive and phi is negative. Therefore the
 * ODE contains a critical point where positive feedback causes runaway
 * cooling and all internal energy is radiated in finite time.
 * 
 * CASE 0 < theta < 1: q > 1; Temperature approaches 0 smoothly (parabolic up near singularity)
 * CASE theta == 0: Temperature linearly goes to 0 (r.r. independent of T)
 * CASE theta < 0:  Temperature approaches 0 nonsmoothly (parabolic down near singularity)
 * 
 * For all cases with theta < 1, the solver notes if we have exceeded the
 * critical time and returns the presumptive post-singularity (T = 0) solution
 */

/* NOTE:
 * Throughout this code page, the Boltzmann constant Kb and mean molecular
 * mass mu which are present in the ideal gas equation written as,
 * P = rho Kb T / mu,
 * Are dimensionally shuffled onto the radiation prefactor lambda0
 * here called STRENGTH (after multiplying by elapsed time)
 */

/* This is the original thermal radiation formulation, given in
 * terms of P and rho, which requires an additional transcendental
 * function evaluation rho^(2-theta) */
template <unsigned int radiationAlgorithm>
__device__ double cufunc_ThermalRadiation(double Pini, double rho)
{
double Pf, beta;

switch(radiationAlgorithm) {
/* The first two cases deal with special cases of the integral over internal energy */
	case ALGO_THETAZERO:
		// Special case: dE = -strength*rho^2 dt; Linear-time outcome
		Pf = Pini - GAMMA_M1*STRENGTH*rho*rho; break;
	case ALGO_THETAONE:
		// Special case: dE = -strength rho P dt; Logarithmic outcome
		Pf = Pini * exp(-GAMMA_M1*STRENGTH*rho); break;
	case ALGO_GENERAL_ANALYTIC:
		// General case: dE = -strength rho^(2-theta) P^theta dt
                              // ANALYTIC_SCALE = (theta-1) (gamma-1) strength
		beta = ANALYTIC_SCALE*pow(rho,TWO_MEXPONENT);
		Pf = pow(Pini, ONE_MINUS_THETA) + beta;
		/* For nearly all values of theta, the analytic expression yields a finite cooling
		 * time at which both temperature and Pf vanish. Beyond this, the Pf above will be
		 * negative; The conditional check here will fall through to the temperature floor
		 * check below if that is the case.
		 * 
		 * In almost all cases, continuing naively will result in a complex value (pow() returns NAN)
		 * In an infinitesmal number (e.g. theta = 1/2), even worse, the return value will be real-valued nonsense.
		 */
		if(Pf > 0) Pf = pow(Pf, INVERSE_ONE_M_THETA);
		break;
	case ALGO_GENERAL_IMPLICIT:
		int i;
		beta = .5*STRENGTH*pow(rho, TWO_MEXPONENT)*GAMMA_M1;
		// Explicit prediction
		Pf = Pini - 2*beta*pow(Pini, EXPONENT);

		// Some newton-raphson to finish it off
		for(i = 0; i < 4; i++) {
			Pf -= (Pf - Pini + beta*(pow(Pf,EXPONENT) + pow(Pini, EXPONENT)))/(1+beta*EXPONENT*pow(Pf,EXPONENT-1.0));
		}
		break;
}

return Pf;
}

/* This is the new temperature-based formulation which adds one
 * divide instruction in the caller (to find T = P / rho) but removes
 * one transcendental (pow()) from here */
template<unsigned int radAlgorithm>
__device__ double cufunc_TempRadiation(double Tini, double rho)
{
double Tfin;

switch(radAlgorithm) {
    /* dT = - lambda0 rho (gamma-1) dt
     * Tfin = Tini - lambda0 rho (gamma-1) delta-t
     */
    case ALGO_THETAZERO: 
        Tfin = Tini - STRENGTH * rho * GAMMA_M1;
        Tfin = (Tfin < 0.0) ? 0.0 : Tfin;
        break;

    /* dT = -T lambda0 rho (gamma-1) dt
       ln(Tfin/Tini) = -lambda0 rho (gamma-1) delta-t
       Tfin = Tini exp(-lambda0 rho (gamma-1) delta-t
     */
    case ALGO_THETAONE:
        Tfin = Tini * exp(-STRENGTH*rho*GAMMA_M1);
        break;

    /* dT T^-theta = - lambda0 rho (gamma-1) dt
     * Tfin^(1-theta) - Tini^(1-theta) = (theta-1)(gamma-1)STRENGTH rho delta-t
     * Tfin = [Tini^q + phi]^(1/phi),
     *        q   = 1-theta
     *        phi = (theta-1) * (gamma-1) * STRENGTH * rho * delta-t
     */
    case ALGO_GENERAL_ANALYTIC:
        Tfin = pow(Tini,ONE_MINUS_THETA) + ANALYTIC_SCALE * rho;
        if(Tfin > 0) {
            Tfin = pow(Tfin, INVERSE_ONE_M_THETA);
        } else {
            Tfin = 0;
        }
        break;
    }

return Tfin;
}

#define PSQUARED px[x]*px[x]+py[x]*py[x]+pz[x]*pz[x]
#define BSQUARED bx[x]*bx[x]+by[x]*by[x]+bz[x]*bz[x]
template <unsigned int radiationAlgorithm>
__global__ void cukern_FreeHydroRadiation(double *rho, double *px, double *py, double *pz, double *E, int numel)
{
	int x = threadIdx.x + BLOCKDIM*blockIdx.x;

	double Tini, Tf, invden, KE;

	while(x < numel) {
		invden = 1.0/rho[x];
		KE = (PSQUARED)*.5*invden;
		Tini = GAMMA_M1*(E[x] - KE)*invden; // gas temperature

		if(Tini > TFLOOR) {
			invden = 1.0/invden; // now actually density

			// Compute final pressure due to radiation operator
//			Pf = cufunc_ThermalRadiation<radiationAlgorithm>(Pini, den);
			Tf = cufunc_TempRadiation<radiationAlgorithm>(Tini, invden);

			// Apply temperature floor
			Tf = (Tf > TFLOOR) ? Tf : TFLOOR;
			E[x] = KE + invden*Tf / GAMMA_M1;
//E[x] = Tf;
		}

		x += BLOCKDIM*GRIDDIM;
	}

}

// STRENGTH = beta*dt
template <unsigned int radiationAlgorithm>
__global__ void cukern_FreeMHDRadiation(double *rho, double *px, double *py, double *pz, double *E, double *bx, double *by, double *bz, int numel)
{
	int x = threadIdx.x + BLOCKDIM*blockIdx.x;

	double Pini, Pf, den, KB;

	while(x < numel) {
		den = rho[x];
		KB = ((PSQUARED)/den + (BSQUARED))/2.0; // KE + B energy densities

		Pini = GAMMA_M1*(E[x] - KB); // gas pressure

		if(Pini > den*TFLOOR) {
			// Compute final gas pressure due to radiation operator
			Pf = cufunc_ThermalRadiation<radiationAlgorithm>(Pini, den);

			// Apply temperature floor
			Pf = (Pf > den*TFLOOR) ? Pf : den*TFLOOR;
			E[x] = KB + Pf / GAMMA_M1;
		}

		x += BLOCKDIM*GRIDDIM;
	}

}

/* These functions return the instantaneous rate, strictly the first derivative e_t */
__global__ void cukern_FreeHydroRadiationRate(double *rho, double *px, double *py, double *pz, double *E, double *radrate, int numel)
{
	int x = threadIdx.x + BLOCKDIM*blockIdx.x;

	double P;
	while(x < numel) {
		P = GAMMA_M1*(E[x] - (PSQUARED)/(2*rho[x])); // gas pressure
		radrate[x] = pow(rho[x], TWO_MEXPONENT)*pow(P, EXPONENT);

		x += BLOCKDIM*GRIDDIM;
	}

}

__global__ void cukern_FreeMHDRadiationRate(double *rho, double *px, double *py, double *pz, double *E, double *bx, double *by, double *bz, double *radrate, int numel)
{
	int x = threadIdx.x + BLOCKDIM*blockIdx.x;

	double P;
	while(x < numel) {
		P = GAMMA_M1*(E[x] - (  (PSQUARED)/rho[x] + (BSQUARED))/2.0); // gas pressure
		radrate[x] = pow(rho[x], TWO_MEXPONENT)*pow(P, EXPONENT);

		x += BLOCKDIM*GRIDDIM;
	}

}

