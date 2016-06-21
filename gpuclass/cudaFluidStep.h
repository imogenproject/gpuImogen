#ifndef CUDAFLUIDSTEP_H
#define CUDAFLUIDSTEP_H

typedef struct __FluidStepParams {
	double lambda;      // = dt / dx;
	int onlyHydro;   // true if B == zero
	double thermoGamma; // Gas adiabatic index

	double minimumRho; // Smallest density, enforced for non-positivity-preserving methods
	int stepMethod;
	int stepDirection;
} FluidStepParams;

// To pick among them
#define METHOD_HLL 1
#define METHOD_HLLC 2
#define METHOD_XINJIN 3

// For kernels templated on upwind/corrector step
#define RK_PREDICT 0
#define RK_CORRECT 1

// For kernels that template based on direction of fluxing
#define FLUX_X 1
#define FLUX_Y 2
#define FLUX_Z 3

#ifdef DEBUGMODE
int performFluidUpdate_1D(MGArray *fluid, FluidStepParams params, ParallelTopology * topo, mxArray **dbOutput);
#else
int performFluidUpdate_1D(MGArray *fluid, FluidStepParams params, ParallelTopology * topo);
#endif

int grabTemporaryMemory(double **m, MGArray *ref, int nCopies);
int releaseTemporaryMemory(double **m, MGArray *ref);

#endif
