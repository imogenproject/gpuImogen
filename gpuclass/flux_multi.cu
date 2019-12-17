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

#include "mpi.h"

#include "cuda.h"
#include "nvToolsExt.h"

#include "cudaCommon.h"
#include "cudaFluidStep.h"
#include "cudaFreeRadiation.h"
#include "flux.h"

// Only uncomment this if you plan to debug this file.
// This will cause it to require output arguments to return data in,
// and perturb code behavior by generating writes to the output debug arrays
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
	if ((nrhs!= 6) || (nlhs != wanted_nlhs)) mexErrMsgTxt("Wrong number of arguments: need flux_ML_iface(fluid, bx, by, bz, [dt, purehydro?, order, step #, step method, rad exponent, rad beta, mintemp], run.geometry)\n");

	MGArray fluid[5];

#ifdef USE_NVTX
	nvtxRangePush("flux_multi step");
#endif

	/* Access bx/by/bz cell-centered arrays if magnetic!!!! */
	/* ... */

    int idxpost = 4; // 8 for the old way

	double *scalars = mxGetPr(prhs[idxpost]);

	if(mxGetNumberOfElements(prhs[idxpost]) != 8) {
		DROP_MEX_ERROR("Must rx 8 parameters in params vector: [dt, purehydro?, order, step #, step method, radiation exponent, radiation beta, radiation min temperature]");
	}

	
	double dt       = scalars[0]; /* Access lambda (dt / dx) */
	int ishydro     = scalars[1]; /* determine if purely hydrodynamic */
	int sweepDirect = (int)scalars[2]; /* Identify if forwards (sweepDirect = 1) or backwards (-1) */
	int stepNum     = (int)scalars[3]; /* step number (used to pick the permutation of the fluid propagators) */
	int stepMethod  = (int)scalars[4]; /* 1=HLL, 2=HLLC, 3=Xin/Jin */
	
	double radExp   = scalars[5];
	double radBeta  = scalars[6];
	double radMintemp=scalars[7];

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

	MGArray tempStorage;
	tempStorage.nGPUs = -1; // not allocated
	int numarrays;
#ifdef DEBUGMODE
			numarrays = 6 + DBG_NUMARRAYS;
#else
#ifdef USE_RK3
			numarrays = 11;
#else
			numarrays = 6;
#endif
#endif

	// TAKE RADIATION HALF STEP 
	// FIXME - this only does bremsstrahlung from the gas, if that is a problem.
	status = MGA_accessFluidCanister(prhs[0], 0, &fluid[0]);
	MGArray fluidReorder[5];
	// the radiation function requires arrays in [rho px py pz E] order for some reason.
	fluidReorder[0] = fluid[0];
	fluidReorder[1] = fluid[2];
	fluidReorder[2] = fluid[3];
	fluidReorder[3] = fluid[4];
	fluidReorder[4] = fluid[1]; 
	
	ThermoDetails therm;
	therm = accessMatlabThermoDetails(mxGetProperty(prhs[0], 0, "thermoDetails"));
	
	status = sourcefunction_OpticallyThinPowerLawRadiation(&fluidReorder[0], NULL, fsp.onlyHydro, therm.gamma, radExp, .5*dt*radBeta, radMintemp);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { DROP_MEX_ERROR("Oh dear: merged step crashed after first radiation half-step!"); }
	
	
// INITIATE FORWARD-ORDERED TRANSPORT STEP 
	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		// If multiple fluids, only have to do this if after first since first was accessed above
		if(fluidct > 0) {
			therm = accessMatlabThermoDetails(mxGetProperty(prhs[0], fluidct, "thermoDetails"));
			status = MGA_accessFluidCanister(prhs[0], fluidct, &fluid[0]);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
		}

		if(tempStorage.nGPUs == -1) {
			nvtxMark("flux_ML_iface.cu:107 large malloc 6 arrays");
			status = MGA_allocSlab(fluid, &tempStorage, numarrays);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
		}

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

		fsp.thermoGamma = therm.gamma;
		fsp.Cisothermal = therm.Cisothermal;
		if(therm.Cisothermal != -1) {
			fsp.thermoGamma = 2;
			// This makes the hydro pressure solver return internal energy when it multiplies eint by (gamma-1)
		}

		fsp.minimumRho = rhoMin;

		status = performFluidUpdate_3D(&fluid[0], &topo, fsp, stepNum, sweepDirect, &tempStorage);

		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
	}
	
// TAKE FULL RADIATION STEP 
	if(numFluids > 1) { 
		therm = accessMatlabThermoDetails(mxGetProperty(prhs[0], 0, "thermoDetails"));
	}
	
	status = sourcefunction_OpticallyThinPowerLawRadiation(&fluidReorder[0], NULL, fsp.onlyHydro, therm.gamma, radExp, dt*radBeta, radMintemp);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { DROP_MEX_ERROR("Oh dear: merged step crashed after first radiation half-step!"); }
	
// INITIATE BACKWARD-ORDERED TRANSPORT STEP 
	sweepDirect = -1;
	fsp.stepDirection = sweepDirect;

	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		ThermoDetails therm = accessMatlabThermoDetails(mxGetProperty(prhs[0], fluidct, "thermoDetails"));

		status = MGA_accessFluidCanister(prhs[0], fluidct, &fluid[0]);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;

		if(tempStorage.nGPUs == -1) {
			nvtxMark("flux_ML_iface.cu:107 large malloc 6 arrays");
			status = MGA_allocSlab(fluid, &tempStorage, numarrays);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
		}

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

		fsp.thermoGamma = therm.gamma;
		fsp.Cisothermal = therm.Cisothermal;
		if(therm.Cisothermal != -1) {
			fsp.thermoGamma = 2;
			// This makes the hydro pressure solver return internal energy when it multiplies eint by (gamma-1)
		}

		fsp.minimumRho = rhoMin;

		status = performFluidUpdate_3D(&fluid[0], &topo, fsp, stepNum, sweepDirect, &tempStorage);


		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
	}

	// TAKE FINAL RADIATION HALF-STEP
	if(numFluids > 1) { 
		therm = accessMatlabThermoDetails(mxGetProperty(prhs[0], fluidct, "thermoDetails"));
	}
	
	status = sourcefunction_OpticallyThinPowerLawRadiation(&fluidReorder[0], NULL, fsp.onlyHydro, therm.gamma, radExp, .5*dt*radBeta, radMintemp);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { DROP_MEX_ERROR("Oh dear: merged step crashed after first radiation half-step!"); }


	// CLEANUP
	// This was allocated & re-used many times in performFluidUpdate_3D
	if((tempStorage.nGPUs != -1) && (status == SUCCESSFUL)) {
		#ifdef USE_NVTX
		nvtxMark("Large free flux_ML_iface.cu:144");
		#endif
		status = MGA_delete(&tempStorage);
	}

	if(status != SUCCESSFUL) {
		DROP_MEX_ERROR("Fluid update code returned unsuccessfully!");
	}

	#ifdef SYNCMEX
		MGA_sledgehammerSequentialize(&fluid[0]);
	#endif

#ifdef USE_NVTX
		nvtxRangePop();
#endif
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
