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

ParallelTopology * topoStructureToC(const mxArray *prhs);
void cfSync(double *cfArray, int cfNumel, ParallelTopology * topology);

__global__ void replicateFreezeArray(double *freezeIn, double *freezeOut, int ncopies, int ny, int nz);
__global__ void reduceFreezeArray(double *freezeClone, double *freeze, int nx, int ny, int nz);

__global__ void cukern_AUSM_firstorder_uniform(double *P, double *Qout, double lambdaQtr, int nx, int ny, int devArrayNumel);

template <unsigned int PCswitch>
__global__ void cukern_AUSM_step(double *Qstore, double lambda, int nx, int ny);

template <unsigned int PCswitch>
__global__ void cukern_HLL_step(double *Qstore, double lambda, int nx, int ny, int devArrayNumel);

template <unsigned int fluxDirection>
__global__ void cukern_HLLC_1storder(double *Qin, double *Qout, double lambda);
template <unsigned int fluxDirection>
__global__ void cukern_HLLC_2ndorder(double *Qin, double *Qout, double lambda);

template <unsigned int PCswitch>
__global__ void cukern_XinJinMHD_step(double *Qstore, double *Cfreeze, double lambda, int nx, int ny, int devArrayNumel);

template <unsigned int PCswitch>
__global__ void cukern_XinJinHydro_step(double *Qbase, double *Qstore, double *Cfreeze, double lambda, int nx, int ny);


/* Stopgap until I manage to stuff pressure solvers into all the predictors... */
__global__ void cukern_PressureSolverHydro(double *state, double *gasPressure, int devArrayNumel);
__global__ void cukern_PressureFreezeSolverHydro(double *state, double *gasPressure, double *Cfreeze, int nx, int ny, int devArrayNumel);
__global__ void cukern_PressureFreezeSolverMHD(double *Qstore, double *Cfreeze, int nx, int ny, int devArrayNumel);

#endif
