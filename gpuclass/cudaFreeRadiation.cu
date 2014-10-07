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
   //cudaFreeRadiation performs a purely local update to energy density of the form

   E = E - dt * beta rho^(2-theta) Pgas^(theta),

   i.e. a sink on the energy term of

   Lambda = beta rho^(2-theta) Pgas^(theta)

   where E is the total energy density, dt the time to pass, beta the radiation strength scale
   factor, rho the mass density, Pgas the thermal pressure, and theta parameterizes the
   radiation (nonrelativistic bremsstrahlung is theta = 0.5)

   It implements a temperature floor (Lambda = 0 for T < T_critical) and checks for negative
   energy density both before (safety) and after (time accuracy truncation) the physics.
*/

__global__ void cukern_FreeHydroRadiationRate(double *rho, double *px, double *py, double *pz, double *E, double *radrate, int numel);
__global__ void cukern_FreeMHDRadiationRate(double *rho, double *px, double *py, double *pz, double *E, double *bx, double *by, double *bz, double *radrate, int numel);

__global__ void cukern_FreeHydroRadiation(double *rho, double *px, double *py, double *pz, double *E, int numel);
__global__ void cukern_FreeMHDRadiation(double *rho, double *px, double *py, double *pz, double *E, double *bx, double *by, double *bz, int numel);

__constant__ __device__ double radparam[5];
#define GAMMA_M1 radparam[0]
#define STRENGTH radparam[1]
#define EXPONENT radparam[2]
#define TWO_MEXPONENT radparam[3]
#define TFLOOR radparam[4]

#define BLOCKDIM 256
#define GRIDDIM 64

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if ((nrhs != 9) || (nlhs > 1))
		mexErrMsgTxt("Wrong number of arguments. Expected forms: rate = cudaFreeRadiation(rho, px, py, pz, E, bx, by, bz, [gamma theta beta*dt Tmin isPureHydro]) or cudaFreeRadiation(rho, px, py, pz, E, bx, by , bz, [gamma theta beta*dt Tmin isPureHydro]\n");

	double *inParams = mxGetPr(prhs[8]);

	double gam      = inParams[0];
	double exponent = inParams[1];
	double strength = inParams[2];
	double minTemp  = inParams[3];
	int isHydro     = (int)inParams[4] != 0;

	MGArray f[8];

	if( isHydro == false ) {
		accessMGArrays(prhs, 0, 7, &f[0]);
	} else {
		accessMGArrays(prhs, 0, 4, &f[0]);
	}

	MGArray *dest;
	if(nlhs == 1) {
		dest = createMGArrays(plhs, 1, &f[0]);
	}

	double hostRP[5];
	hostRP[0] = gam-1.0;
	hostRP[1] = strength;
	hostRP[2] = exponent;
	hostRP[3] = 2.0 - exponent;
	hostRP[4] = minTemp;

	int j, k;
	for(j = 0; j < f->nGPUs; j++) {
		cudaSetDevice(f->deviceID[j]);
		cudaMemcpyToSymbol(radparam, hostRP, 5*sizeof(double), 0, cudaMemcpyHostToDevice);
	}

	int sub[6];
	for(j = 0; j < f[0].nGPUs; j++) {
		calcPartitionExtent(&f[0], j, sub);
		cudaSetDevice(f[0].deviceID[j]);

		double *ptrs[8];
		for(k = 0; k < 8; k++) { ptrs[k] = f[k].devicePtr[j]; }
		// Save some readability below...

		switch(isHydro + 2*nlhs) {
		case 0: {
			cukern_FreeMHDRadiation<<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], f[0].partNumel[j]);
			break; }
		case 1: {
			cukern_FreeHydroRadiation<<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], f[0].partNumel[j]);
			break; }
		case 2: {
			cukern_FreeMHDRadiationRate<<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], dest->devicePtr[j], f[0].partNumel[j]);
			break; }
		case 3: {
			cukern_FreeHydroRadiationRate<<<GRIDDIM, BLOCKDIM>>>(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], dest->devicePtr[j], f[0].partNumel[j]);
			break; }
		}
	}

	if(nlhs == 1) free(dest);

	//CHECK_CUDA_LAUNCH_ERROR(BLOCKDIM, GRIDDIM, &amd, 666, "cudaFreeGasRadiation");

}

/* NOTE: This uses an explicit algorithm to perform radiation,
 * i.e. E[t+dt] = E[t] - Lambda[t] dt with radiation rate Lambda
 * This is conditionally stable with a CFL set by Lambda dt < E
 * 
 * Normally E / Lambda >> [dx / (c+max(Vx)) ], i.e. cooling time much
 * longer than advection time, but an implicit algorithm would be
 * wise as it is unconditionally stable. */
#define PSQUARED px[x]*px[x]+py[x]*py[x]+pz[x]*pz[x]
#define BSQUARED bx[x]*bx[x]+by[x]*by[x]+bz[x]*bz[x]

__global__ void cukern_FreeHydroRadiation(double *rho, double *px, double *py, double *pz, double *E, int numel)
{
	int x = threadIdx.x + BLOCKDIM*blockIdx.x;

	double P; double dE; double den;

	while(x < numel) {
		den = rho[x];
		P = GAMMA_M1*(E[x] - (PSQUARED)/(2*den)); // gas pressure

		dE = STRENGTH*pow(den, TWO_MEXPONENT)*pow(P, EXPONENT); // amount to be lost
		if(P > den*TFLOOR) {
			if(P - (GAMMA_M1*dE) < den*TFLOOR) { E[x] -= (P-den*TFLOOR)/GAMMA_M1; } else { E[x] -= dE; } }

		x += BLOCKDIM*GRIDDIM;
	}

}

// STRENGTH = beta*dt
__global__ void cukern_FreeMHDRadiation(double *rho, double *px, double *py, double *pz, double *E, double *bx, double *by, double *bz, int numel)
{
	int x = threadIdx.x + BLOCKDIM*blockIdx.x;

	double P, dE, den;

	while(x < numel) {
		den = rho[x];
		P = GAMMA_M1*(E[x] - (  (PSQUARED)/den + (BSQUARED))/2.0); // gas pressure
		dE = STRENGTH*pow(den, TWO_MEXPONENT)*pow(P, EXPONENT);
		if(P > den*TFLOOR) {
			if(P - (GAMMA_M1 * dE) < den*TFLOOR) { E[x] -= (P-den*TFLOOR)/GAMMA_M1; } else { E[x] -= dE; } }

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

