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
	if ((nrhs!= 6) || (nlhs != wanted_nlhs)) mexErrMsgTxt("Wrong number of arguments: need flux_ML_iface(fluid, bx, by, bz, [dt, purehydro?, order, step #, step method], run.geometry)\n");

	MGArray fluid[5];

	/* Access bx/by/bz cell-centered arrays if magnetic!!!! */
	/* ... */

    int idxpost = 4; // 8 for the old way

	double *scalars = mxGetPr(prhs[idxpost]);

	if(mxGetNumberOfElements(prhs[idxpost]) != 5) {
		DROP_MEX_ERROR("Must rx 5 parameters in params vector: [dt, purehydro?, order, step #, step method]");
	}

	double dt     = scalars[0]; /* Access lambda (dt / dx) */
	int ishydro   = scalars[1]; /* determine if purely hydrodynamic */
	int sweepDirect = (int)scalars[2]; /* Identify if forwards (sweepDirect = 1) or backwards (-1) */
	int stepNum     = (int)scalars[3]; /* step number (used to pick the permutation of the fluid propagators) */
	int stepMethod  = (int)scalars[4]; /* 1=HLL, 2=HLLC, 3=Xin/Jin */

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
			PRINT_FAULT_HEADER;
			printf("Unable to access fluid(%i).MINMASS property.\n", fluidct);
			PRINT_FAULT_FOOTER;
			status = ERROR_NULL_POINTER;
			break;
		}

		flprop = mxGetProperty(prhs[0], fluidct, "gamma");
		if(flprop != NULL) {
			fsp.thermoGamma = *mxGetPr(flprop);
		} else {
			PRINT_FAULT_HEADER;
			printf("Unable to access fluid(%i).gamma!\n", fluidct);
			PRINT_FAULT_FOOTER;
			status = ERROR_NULL_POINTER;
			break;
		}
		fsp.minimumRho = rhoMin;

		MGArray tempStorage;
		tempStorage.nGPUs = -1; // not allocated

		status = performFluidUpdate_3D(&fluid[0], &topo, fsp, stepNum, sweepDirect, &tempStorage);

		// This was allocated & re-used many times in performFluidUpdate_3D
		MGA_delete(&tempStorage);

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
