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
#include "cudaSoundspeed.h"
#include "cflTimestep.h"

#include "sourceStep.h"
#include "flux.h"


// Only uncomment this if you plan to debug this file.
// This will cause it to require output arguments to return data in,
// and perturb code behavior by generating writes to the output debug arrays
//#define DEBUGMODE

FluidMethods mlmethodToEnum(int mlmethod);
int fetchMinDensity(mxArray *mxFluids, int fluidNum, double *rhoMin);

int calculateMaxTimestep(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParallelTopology *topo, MGArray *tempStorage, double *timestep);
int performCompleteTimestep(GridFluid *fluids, int numFluids, FluidStepParams fsp, ParallelTopology topo, GravityData *gravdata, ParametricRadiation *rad, int srcType);

#ifdef DEBUGMODE
    #include "debug_inserts.h"
#endif


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	int wanted_nlhs = 0;
#ifdef DEBUGMODE
	wanted_nlhs = 1;
#endif
	if ((nrhs!= 7) || (nlhs != wanted_nlhs)) mexErrMsgTxt("Wrong number of arguments: need sourceStep(run, fluid, bx, by, bz, xyvector, [order, step #, step method, tFraction])\n");

	//MGArray fluid[5];

#ifdef USE_NVTX
	nvtxRangePush("flux_multi step");
#endif

	double *scalars = mxGetPr(prhs[6]);

	if(mxGetNumberOfElements(prhs[6]) != 4) {
		DROP_MEX_ERROR("Must rx 4 parameters in params vector: [ order, step #, step method, tFraction]");
	}

	int worked = SUCCESSFUL;

	const mxArray* theImogenManager = prhs[0];

	double dt = derefXdotAdotB_scalar(theImogenManager, "time", "dTime");
	int ishydro = (int)derefXdotAdotB_scalar(theImogenManager, "pureHydro", (const char *)NULL);

	// Load up the FluidStepParameters structure
	int sweepDirect = (int)scalars[0]; /* Identify if forwards (sweepDirect = 1) or backwards (-1) */
	int stepNum     = (int)scalars[1]; /* step number (used to pick the permutation of the fluid propagators) */
	int stepMethod  = (int)scalars[2]; /* 1=HLL, 2=HLLC, 3=Xin/Jin */

	FluidStepParams fsp;
	fsp.dt            = dt;
	fsp.onlyHydro     = ishydro;
	fsp.stepDirection = sweepDirect;
	fsp.stepMethod    = mlmethodToEnum(stepMethod);
	fsp.stepNumber    = stepNum;
	fsp.cflPrefactor  = derefXdotAdotB_scalar(theImogenManager, "time","CFL");

	// Load up the radiation structure
	ParametricRadiation prad;
	int isRadiating = derefXdotAdotB_scalar(theImogenManager, "radiation", "type");

	if(isRadiating) {
		prad.exponent = derefXdotAdotB_scalar(theImogenManager, "radiation", "exponent");
		// FIXME NASTY HACK min temperature set by hard code (this copied from ./fluid/Radiation.m:124)
		prad.minTemperature = 1.05;
		prad.prefactor = derefXdotAdotB_scalar(theImogenManager, "radiation", "strength") * dt;
	} else {
		prad.prefactor = 0;
	}

	const mxArray *geo = mxGetProperty(prhs[0], 0, "geometry");

	// Load up the topology structure
	ParallelTopology topo;
	const mxArray *mxtopo = mxGetProperty(geo, 0, "topology");
	topoStructureToC(mxtopo, &topo);

	// Load up the geometry structure inside the FluidStepParams
	fsp.geometry = accessMatlabGeometryClass(geo);
	int numFluids     = mxGetNumberOfElements(prhs[1]);

	if(numFluids > 1) {
		fsp.multifluidDragMethod = (int)derefXdotAdotB_scalar(theImogenManager, "multifluidDragMethod", (const char *)NULL);
	} else {
		fsp.multifluidDragMethod = 0;
	}

	// Access the potential field, if relevant
	GravityData gravdat;
	MGArray gravphi;
	int haveg = derefXdotAdotB_scalar(theImogenManager, "potentialField", "ACTIVE");
	if(haveg) {
		const mxArray *gravfield;
		gravfield =  derefXdotAdotB(theImogenManager, "potentialField", "field");
		worked = MGA_accessMatlabArrays(&gravfield, 0, 0, &gravphi);
		gravdat.phi = &gravphi;
		double orderstmp[2];
		derefXdotAdotB_vector(theImogenManager, "compositeSrcOrders", (const char *)NULL, &orderstmp[0], 2);
		gravdat.spaceOrder = (int)orderstmp[0];
		gravdat.timeOrder = (int)orderstmp[1];
	} else {
		gravdat.spaceOrder = 0;
		gravdat.timeOrder = 0;
	}

	if(worked != SUCCESSFUL) { DROP_MEX_ERROR("performCompleteTimestep crashing because of failure to fetch run.potentialField.field"); }

	// Access the fluids themselves
	GridFluid fluids[numFluids];
	int fluidct;
	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		MGA_accessFluidCanister(prhs[1], fluidct, &fluids[fluidct].data[0]);
		fluids[fluidct].thermo = accessMatlabThermoDetails(mxGetProperty(prhs[1], fluidct, "thermoDetails"));
		worked = fetchMinDensity((mxArray *)prhs[1], fluidct, &fluids[fluidct].rhoMin);
	}

	// Fetch the XY vectors
	MGArray XYvectors;
	worked = MGA_accessMatlabArrays(&prhs[5], 0, 0, &XYvectors);
	if(worked != SUCCESSFUL) { DROP_MEX_ERROR("performCompleteTimestep crashing because of failure to fetch input xyVectors (arg 6)"); }

	fsp.geometry.XYVector = &XYvectors;

	// Get the global domain rez for doing cfl
	double *globrez = mxGetPr(derefXdotAdotB(theImogenManager, "geometry", "globalDomainRez"));
	int i;
	for(i = 0; i < 3; i++) {
		fsp.geometry.globalRez[i] = globrez[i];
	}

	// Determine what kind of source type we're going to do
	int cylcoords = (fsp.geometry.shape == CYLINDRICAL) || (fsp.geometry.shape == RZCYLINDRICAL);
	// sourcerFunction = (fsp.geometry. useCyl + 2*useRF + 4*usePhi + 8*use2F;
	int srcType = 1*(cylcoords == 1) + 2*(fsp.geometry.frameOmega != 0) + 4*(haveg) + 8*(numFluids > 1);

	double resultingTimestep;

	worked = CHECK_IMOGEN_ERROR(performCompleteTimestep(fluids, numFluids, fsp, topo, &gravdat, &prad, srcType));
	if(worked != SUCCESSFUL) {
		DROP_MEX_ERROR("Big problem: performCompleteTimestep crashed! See compiled backtrace generated above.");
	}

}

int performCompleteTimestep(GridFluid *fluids, int numFluids, FluidStepParams fsp, ParallelTopology topo, GravityData *gravdata, ParametricRadiation *rad, int srcType)
{
	int status = SUCCESSFUL;
	int fluidct;

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

	if(tempStorage.nGPUs == -1) {
		nvtxMark("flux_multi.cu:131 large malloc 6 arrays");
		status = MGA_allocSlab(&fluids[0].data[0], &tempStorage, numarrays);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	}

	//double propdt;
	//status = calculateMaxTimestep(fluids, numFluids, &fsp, &topo, &tempStorage, &propdt);
	//printf("Input dt = %le, proposed dt = %le, diff = %le\n", fsp.dt, propdt, propdt - fsp.dt);

// TAKE SOURCE HALF STEP
	fsp.dt *= 0.5;
	status = performSourceFunctions(srcType, fluids, numFluids, fsp, &topo, gravdata, rad, &tempStorage);
	fsp.dt *= 2;

	//rad->prefactor *= .5;
	//status = sourcefunction_OpticallyThinPowerLawRadiation(&fluidReorder[0], NULL, fsp.onlyHydro, fluids[0].thermo.gamma, rad);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	
	fsp.stepDirection = 1;
// INITIATE FORWARD-ORDERED TRANSPORT STEP 
	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		fsp.minimumRho = fluids[fluidct].rhoMin;
		fsp.thermoGamma = fluids[fluidct].thermo.gamma;
		fsp.Cisothermal = fluids[fluidct].thermo.Cisothermal;
		if(fluids[fluidct].thermo.Cisothermal != -1) {
			fsp.thermoGamma = 2;
			// This makes the hydro pressure solver return internal energy when it multiplies eint by (gamma-1)
		}

		status = performFluidUpdate_3D(&fluids[fluidct].data[0], &topo, fsp, &tempStorage);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
	}
	if(status != SUCCESSFUL) { return status; }
	
// TAKE FULL SOURCE STEP
	status = performSourceFunctions(srcType, fluids, numFluids, fsp, &topo, gravdata, rad, &tempStorage);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	
// INITIATE BACKWARD-ORDERED TRANSPORT STEP 
	fsp.stepDirection = -1;

	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		fsp.minimumRho = fluids[fluidct].rhoMin;
		fsp.thermoGamma = fluids[fluidct].thermo.gamma;
		fsp.Cisothermal = fluids[fluidct].thermo.Cisothermal;
		if(fluids[fluidct].thermo.Cisothermal != -1) {
			fsp.thermoGamma = 2;
		}

		status = performFluidUpdate_3D(&fluids[fluidct].data[0], &topo, fsp, &tempStorage);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
	}
	if(status != SUCCESSFUL) { return status; }

	// TAKE FINAL SOURCE HALF STEP
	fsp.dt *= 0.5;
	status = performSourceFunctions(srcType, fluids, numFluids, fsp, &topo, gravdata, rad, &tempStorage);
	fsp.dt *= 2;
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	// CLEANUP
	// This was allocated & re-used many times in performFluidUpdate_3D
	if((tempStorage.nGPUs != -1) && (status == SUCCESSFUL)) {
		#ifdef USE_NVTX
		nvtxMark("Large free flux_ML_iface.cu:144");
		#endif
		status = CHECK_IMOGEN_ERROR(MGA_delete(&tempStorage));
		if(status != SUCCESSFUL) { return status; }
	}

#ifdef SYNCMEX
	MGA_sledgehammerSequentialize(&fluid[0]);
#endif

#ifdef USE_NVTX
	nvtxRangePop();
#endif

	return SUCCESSFUL;
}

/* Computes the maximum permitted timestep allowed by the CFL constraint on the fluid method */
int calculateMaxTimestep(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParallelTopology *topo, MGArray *tempStorage, double *timestep)
{
	int status = SUCCESSFUL;
    double dt = 1e38;

    double currentdt = dt;
    double tau;

    int i, j;
    // compute each fluid's min timestep on this node
    for(i = 0; i < nFluids; i++) {
    	status = CHECK_IMOGEN_ERROR(calculateSoundspeed(&fluids[i].data[0], (MGArray *)NULL, tempStorage, fluids[i].thermo.gamma));

    	double globrez[3];
    	for(j = 0; j < 3; j++) { globrez[j] = fsp->geometry.globalRez[j]; }

    	status = computeLocalCFLTimestep(&fluids[i].data[0], tempStorage, &fsp->geometry, fsp->stepMethod, &globrez[0], &tau);
    	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { break; }

    	// crash on invalid timesteps
    	if(isnan(tau)) {
    		PRINT_SIMPLE_FAULT("Calculated a timestep that is either infinity or NaN. Crashing\n!");
    		return ERROR_CRASH;
    	}

    	// This is computed globally by computeLocalCFLTimestep
    	currentdt = (currentdt < tau) ? currentdt : tau;
    }

    *timestep = currentdt * fsp->cflPrefactor;
	return SUCCESSFUL;
}

int fetchMinDensity(mxArray *mxFluids, int fluidNum, double *rhoMin)
{
	int status = SUCCESSFUL;
	mxArray *flprop = mxGetProperty(mxFluids, fluidNum, "MINMASS");
	if(flprop != NULL) {
		rhoMin[0] = *((double *)mxGetPr(flprop));
	} else {
		PRINT_FAULT_HEADER;
		printf("Unable to access fluid(%i).MINMASS property.\n", fluidNum);
		PRINT_FAULT_FOOTER;
		status = ERROR_NULL_POINTER;
	}

	return status;
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
