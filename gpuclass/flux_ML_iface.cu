/*
 * flux_ML_iface.c
 *
 *  Created on: Nov 25, 2015
 *      Author: erik
 */

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif
#include "mex.h"

#include "cuda.h"
#include "mpi.h"

#include "cudaCommon.h"
#include "cudaFluidStep.h"
#include "flux.h"

// Only uncomment this if you plan to debug this file.
//#define DEBUGMODE

FluidMethods mlmethodToEnum(int mlmethod);

#ifdef DEBUGMODE
    #include "debug_inserts.h"
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	int wanted_nlhs = 0;
#ifdef DEBUGMODE
	wanted_nlhs = 1;
#endif
	if ((nrhs!= 6) || (nlhs != wanted_nlhs)) mexErrMsgTxt("Wrong number of arguments: need flux_ML_iface(fluid, bx, by, bz, [dt, purehydro?, fluid gamma, order, step #, step method], run.geometry)\n");

	MGArray fluid[5];

	/* Access bx/by/bz cell-centered arrays if magnetic!!!! */
	/* ... */

    int idxpost = 4; // 8 for the old way

	double *scalars = mxGetPr(prhs[idxpost]);

	if(mxGetNumberOfElements(prhs[idxpost]) != 6) {
		DROP_MEX_ERROR("Must rx 8 parameters in params vector: [dt, purehydro?, fluid gamma, rhomin, order, step #, step method, geomtype, Rin]");
	}

	double dt     = scalars[0]; /* Access lambda (dt / dx) */
	int ishydro   = scalars[1]; /* determine if purely hydrodynamic */
	double gamma  = scalars[2]; /* Adiabatic index of fluid */

	int sweepDirect = (int)scalars[3]; /* Identify if forwards (sweepDirect = 1) or backwards (-1) */
	int stepNum     = (int)scalars[4]; /* step number (used to pick the permutation of the fluid propagators) */
	int stepMethod  = (int)scalars[5]; /* 1=HLL, 2=HLLC, 3=Xin/Jin */

	/* Access topology structure */
	ParallelTopology topo;
	FluidStepParams fsp;

	fsp.geometry = accessMatlabGeometryClass(prhs[idxpost+1]);

	const mxArray *mxtopo = mxGetProperty(prhs[idxpost+1], 0, "topology");
	topoStructureToC(mxtopo, &topo);

	fsp.dt = dt;

	fsp.onlyHydro = ishydro;
	fsp.stepDirection = sweepDirect;
	fsp.stepMethod = mlmethodToEnum(stepMethod);
	fsp.thermoGamma = gamma;

	int numFluids = mxGetNumberOfElements(prhs[0]);
	int fluidct;
	CHECK_CUDA_ERROR("entering compiled fluid step");

	int status;

	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		status = MGA_accessFluidCanister(prhs[0], fluidct, &fluid[0]);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
		double rhoMin;
		mxArray *flprop = mxGetProperty(prhs[0], fluidct, "MINMASS");
		if(flprop != NULL) {
			rhoMin = *((double *)mxGetPr(flprop));
		} else {
			status = ERROR_NULL_POINTER;
		}
		fsp.minimumRho = rhoMin;
		performFluidUpdate_3D(&fluid[0], &topo, fsp, stepNum, sweepDirect);

		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
	}
	if(status != SUCCESSFUL) {
		DROP_MEX_ERROR("Fluid update code returned unsuccessfully!");
	}

}

FluidMethods mlmethodToEnum(int mlmethod)
{
	FluidMethods f;
	switch(mlmethod) {
	case 1: f = METHOD_HLL; break;
	case 2: f = METHOD_HLLC; break;
	case 3: f = METHOD_XINJIN; break;
	}

return f;
}
