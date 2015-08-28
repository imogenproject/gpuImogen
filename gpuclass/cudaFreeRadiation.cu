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

/* THIS FUNCTION
   cudaFreeRadiation solves the operator equation
   d/dt E = -beta rho^2 T^theta


   Lambda = beta rho^(2-theta) Pgas^(theta)

   where E is the total energy density, dt the time to pass, beta the radiation strength scale
   factor, rho the mass density, Pgas the thermal pressure, and theta parameterizes the
   radiation (nonrelativistic bremsstrahlung is theta = 0.5)

   It implements a temperature floor (Lambda = 0 for T < T_critical) and checks for negative
   energy density both before (safety) and after (time accuracy truncation) the physics.
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
// Analytic power law uses (theta-1)*strength*(gamma-1)
#define ANALYTIC_SCALE radparam[5]
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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if ((nrhs != 9) || (nlhs > 1))
		mexErrMsgTxt("Wrong number of arguments. Expected forms: rate = cudaFreeRadiation(rho, px, py, pz, E, bx, by, bz, [gamma theta beta*dt Tmin isPureHydro]) or cudaFreeRadiation(rho, px, py, pz, E, bx, by , bz, [gamma theta beta*dt Tmin isPureHydro]\n");

	CHECK_CUDA_ERROR("Entering cudaFreeRadiation");

	double *inParams = mxGetPr(prhs[8]);

	double gam      = inParams[0];
	double exponent = inParams[1];
	double strength = inParams[2];
	double minTemp  = inParams[3];
	int isHydro     = (int)inParams[4] != 0;

	MGArray f[8];

	if( isHydro == false ) {
		MGA_accessMatlabArrays(prhs, 0, 7, &f[0]);
	} else {
		MGA_accessMatlabArrays(prhs, 0, 4, &f[0]);
	}

	MGArray *dest;
	if(nlhs == 1) {
		dest = MGA_createReturnedArrays(plhs, 1, &f[0]);
	}

	double hostRP[8];
	hostRP[0] = gam-1.0;
	hostRP[1] = strength;
	hostRP[2] = exponent;
	hostRP[3] = 2.0 - exponent;
	hostRP[4] = minTemp;
	hostRP[5] = (exponent-1.0)*strength*(gam-1);
	hostRP[6] = 1.0-exponent;
	hostRP[7] = 1.0/(1.0-exponent);

	int j, k;
	for(j = 0; j < f->nGPUs; j++) {
		cudaSetDevice(f->deviceID[j]);
		CHECK_CUDA_ERROR("cudaSetDevice");
		cudaMemcpyToSymbol(radparam, hostRP, 8*sizeof(double), 0, cudaMemcpyHostToDevice);
		CHECK_CUDA_ERROR("cudaMemcpyToSymbol");
	}

	int sub[6];
	int kernNumber;

	for(j = 0; j < f[0].nGPUs; j++) {
		calcPartitionExtent(&f[0], j, sub);
		cudaSetDevice(f[0].deviceID[j]);
		CHECK_CUDA_ERROR("cudaSetDevice");

		double *ptrs[8];
		for(k = 0; k < 8; k++) { ptrs[k] = f[k].devicePtr[j]; }

		// Run through the conditionals that pick the algorithm applicable to the current case
		switch(isHydro + 2*nlhs) {
		case 0: { // Apply radiation to MHD gas
			if(exponent == 0.0) {
				cukern_FreeMHDRadiation<ALGO_THETAZERO><<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], f[0].partNumel[j]);
				kernNumber = 1;
			} else if(exponent == 1.0) {
				cukern_FreeMHDRadiation<ALGO_THETAONE><<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], f[0].partNumel[j]);
				kernNumber = 2;
			} else {
				cukern_FreeMHDRadiation<ALGO_GENERAL_ANALYTIC><<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], f[0].partNumel[j]);
				kernNumber = 3;
			}

			break; }
		case 1: { // Apply radiation to hydro gas
			if(exponent == 0.0) {
				cukern_FreeHydroRadiation<ALGO_THETAZERO><<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], f[0].partNumel[j]);
				kernNumber = 4;
			} else if(exponent == 1.0) {
				cukern_FreeHydroRadiation<ALGO_THETAONE><<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], f[0].partNumel[j]);
				kernNumber = 5;
			} else {
				cukern_FreeHydroRadiation<ALGO_GENERAL_ANALYTIC><<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], f[0].partNumel[j]);
				kernNumber = 6;
			}
			break; }
		case 2: { // Calculate radiation rate under MHD
			cukern_FreeMHDRadiationRate<<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], dest->devicePtr[j], f[0].partNumel[j]);
			kernNumber = 7;
			break; }
		case 3: { // Calculate radiation rate under hydrodynamics
			cukern_FreeHydroRadiationRate<<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], dest->devicePtr[j], f[0].partNumel[j]);
			kernNumber = 8;
			break; }
		}
		CHECK_CUDA_LAUNCH_ERROR(BLOCKDIM, GRIDDIM, &f[0], kernNumber, "cudaFreeRadiation, int=kernel number attempted (see source)");
	}

	if(nlhs == 1) free(dest);



}

/* NOTE: This uses an explicit algorithm to perform radiation,
 * i.e. E[t+dt] = E[t] - Lambda[t] dt with radiation rate Lambda
 * This is conditionally stable with a CFL set by Lambda dt < E
 * 
 * Normally E / Lambda >> [dx / (c+max(Vx)) ], i.e. cooling time much
 * longer than advection time, but an implicit algorithm would be
 * wise as it is unconditionally stable. */

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
		Pf = exp(log(Pini) - GAMMA_M1*STRENGTH*rho); break;
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
		 * In almost all cases, continuing naively will result in a complex value (NAN from real-valued pow)
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

/* FIXME: This would be improved if it were rewritten do avoid the Einternal <-> Pressure conversion */
#define PSQUARED px[x]*px[x]+py[x]*py[x]+pz[x]*pz[x]
#define BSQUARED bx[x]*bx[x]+by[x]*by[x]+bz[x]*bz[x]
template <unsigned int radiationAlgorithm>
__global__ void cukern_FreeHydroRadiation(double *rho, double *px, double *py, double *pz, double *E, int numel)
{
	int x = threadIdx.x + BLOCKDIM*blockIdx.x;

	double Pini, Pf, den, KE;

	while(x < numel) {
		den = rho[x];
		KE = (PSQUARED)/(2*den);
		Pini = GAMMA_M1*(E[x] - KE); // gas pressure

		if(Pini > den*TFLOOR) {
			// Compute final pressure due to radiation operator
			Pf = cufunc_ThermalRadiation<radiationAlgorithm>(Pini, den);

			// Apply temperature floor
			Pf = (Pf > den*TFLOOR) ? Pf : den*TFLOOR;
			E[x] = KE + Pf / GAMMA_M1;
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

