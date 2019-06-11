#ifndef CUDAFLUIDSTEP_H
#define CUDAFLUIDSTEP_H

#include "cudaCommon.h"

typedef enum FluidMethods {
	METHOD_HLL = 1, METHOD_HLLC = 2, METHOD_XINJIN = 3
} FluidMethods;

typedef struct __FluidStepParams {
	double dt;      // = dt / dx;
	int onlyHydro;   // true if B == zero
	double thermoGamma; // Gas adiabatic index
	double minimumRho; // Smallest density, enforced for non-positivity-preserving methods
	double Cisothermal; // isothermal soundspeed, if not equal to -1; C << 1 -> dust-like (pressureless)
	
	FluidMethods stepMethod;
	int stepDirection;

	GeometryParams geometry;
} FluidStepParams;

#include "fluidMethod.h"

// For kernels templated on upwind/corrector step
#define RK_PREDICT 0
#define RK_CORRECT 1

// For kernels that template based on direction of fluxing
#define FLUX_X 1
#define FLUX_Y 2
#define FLUX_Z 3
#define FLUX_RADIAL 4
#define FLUX_THETA_213 5
#define FLUX_THETA_231 6

#define TIMESCHEME_RK2 0
#define TIMESCHEME_SSPRK_A 8
#define TIMESCHEME_SSPRK_B 16

// SSPRK_A timestep is a full forward euler
#define TIMESCHEME_EULER TIMESCHEME_SSPRK_A

// TIMESCHEME_RK3_A coincides with TIMESCHEME_SSPRK_A
#define TIMESCHEME_RK3_A TIMESCHEME_SSPRK_A
#define TIMESCHEME_RK3_B 24
#define TIMESCHEME_RK3_C 32


#ifdef DEBUGMODE
int performFluidUpdate_1D(MGArray *fluid, FluidStepParams params, ParallelTopology * topo, MGArray *tmpst, mxArray **dbOutput);
#else
int performFluidUpdate_1D(MGArray *fluid, FluidStepParams params, ParallelTopology * topo, MGArray *tmpst);
#endif

int grabTemporaryMemory(double **m, MGArray *ref, int nCopies);
int releaseTemporaryMemory(double **m, MGArray *ref);

#endif
