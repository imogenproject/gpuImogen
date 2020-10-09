
#ifndef NOMATLAB
#include "mex.h"
#else
#include "stdio.h"
#endif

#include "nvToolsExt.h"
#include "cudaCommon.h"

#include "cudaHaloExchange.h"
#include "cudaStatics.h"

#include "cudaFluidStep.h" // for FluidStepParameters
#include "cudaFreeRadiation.h"
#include "cudaArrayRotateB.h"
#include "cudaSourceScalarPotential.h"
#include "cudaSource2FluidDrag.h"
#include "cudaTestSourceComposite.h"
#include "cudaSourceRotatingFrame.h"
#include "cudaSourceCylindricalTerms.h"

#include "sourceStep.h"

// These have been extracted
int sourceRadiation(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad);
int setAllBoundaryConditions(GridFluid *fluids, int nFluids, GeometryParams *geo);

// The source function calls themselves. We keep these private in here 'cause we don't want randos calling them
int srcBlank(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad);
int src___D(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad);
int src__G_(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad, ParallelTopology *topo, GravityData *gravdata);
int src__GD(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad, ParallelTopology *topo, GravityData *gravdata);
int src_F__(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad);
int src_F_D(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad);
int srcC___(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad);
int srcC__D(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad);
int srcComp(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad, ParallelTopology *topo, GravityData *gravdata, MGArray *storage = NULL);
int src2f_cmp_2f(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad, ParallelTopology *topo, GravityData *gravdata, MGArray *storage = NULL);

//#define STANDALONE_MEX_FUNCTION
// The MEX gateway function that allows Matlab to call this independently
#ifdef STANDALONE_MEX_FUNCTION
// Copied these straight from flux_multi.cu FIXME dumpster fire...
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
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int wanted_nlhs = 0;
#ifdef DEBUGMODE
	wanted_nlhs = 1;
#endif
	if ((nrhs!= 7) || (nlhs != wanted_nlhs)) mexErrMsgTxt("Wrong number of arguments: need sourceStep(run, fluid, bx, by, bz, xyvector, [order, step #, step method, tFraction])\n");

#ifdef USE_NVTX
	nvtxRangePush("sourceStep.cu");
#endif

	/* Access bx/by/bz cell-centered arrays if magnetic!!!! */
	/* ... FIXME lol some day*/

	double *scalars = mxGetPr(prhs[6]);

	if(mxGetNumberOfElements(prhs[6]) != 4) {
		DROP_MEX_ERROR("Must rx 4 parameters in params vector: [ order, step #, step method, tFraction]");
	}

	int worked = SUCCESSFUL;

	const mxArray* theImogenManager = prhs[0];

	double dt = derefXdotAdotB_scalar(theImogenManager, "time", "dTime") * scalars[3];
	int ishydro = (int)derefXdotAdotB_scalar(theImogenManager, "pureHydro", NULL);

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
		fsp.multifluidDragMethod = (int)derefXdotAdotB_scalar(theImogenManager, "multifluidDragMethod", NULL);
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
		derefXdotAdotB_vector(theImogenManager, "compositeSrcOrders", NULL, &orderstmp[0], 2);
		gravdat.spaceOrder = (int)orderstmp[0];
		gravdat.timeOrder = (int)orderstmp[1];
	} else {
		gravdat.spaceOrder = 0;
		gravdat.timeOrder = 0;
	}

	if(worked != SUCCESSFUL) { DROP_MEX_ERROR("performSourceFunctions crashing because of failure to fetch run.potentialField.field"); }

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
	if(worked != SUCCESSFUL) { DROP_MEX_ERROR("performSourceFunctions crashing because of failure to fetch input xyVectors (arg 6)"); }

	fsp.geometry.XYVector = &XYvectors;

	// Determine what kind of source type we're going to do
	int cylcoords = (fsp.geometry.shape == CYLINDRICAL) || (fsp.geometry.shape == RZCYLINDRICAL);
	// sourcerFunction = (fsp.geometry. useCyl + 2*useRF + 4*usePhi + 8*use2F;
	int srcType = 1*(cylcoords == 1) + 2*(fsp.geometry.frameOmega != 0) + 4*(haveg) + 8*(numFluids > 1);

	worked = CHECK_IMOGEN_ERROR(performSourceFunctions(srcType, &fluids[0], numFluids, fsp, &topo, &gravdat, &prad, NULL));
	if(worked != SUCCESSFUL) {
		DROP_MEX_ERROR("Big problem: performSourceFunctions crashed! See compiled backtrace generated above.");
	}

#ifdef USE_NVTX
		nvtxRangePop();
#endif

}

#endif


/* performSourceFunctions computes all source terms that are currently in operation
 * We note that currently this has the "fatal flaw" that it requires access to the
 * Matlab ImogenManager class in order to be able to fetch any of a huge range of crap
 */
int performSourceFunctions(int srcType, GridFluid *fluids, int nFluids, FluidStepParams fsp, ParallelTopology *topo, GravityData *gravdata, ParametricRadiation *prad, MGArray *tmpStorage)
{
	int status = SUCCESSFUL;

	switch(srcType) {
	case 0: status = srcBlank(fluids, nFluids, &fsp, prad); break;
	case 1: status = srcC___(fluids, nFluids, &fsp, prad); break;
	case 2: status = src_F__(fluids, nFluids, &fsp, prad); break;
	case 3: status = srcComp(fluids, nFluids, &fsp, prad, topo, gravdata, tmpStorage); break;
	case 4: status = src__G_(fluids, nFluids, &fsp, prad, topo, gravdata); break;
	case 5: status = srcComp(fluids, nFluids, &fsp, prad, topo, gravdata, tmpStorage); break;
	case 6: status = srcComp(fluids, nFluids, &fsp, prad, topo, gravdata, tmpStorage); break;
	case 7: status = srcComp(fluids, nFluids, &fsp, prad, topo, gravdata, tmpStorage); break;
	case 8: status = src___D(fluids, nFluids, &fsp, prad); break;
	case 9: status = srcC__D(fluids, nFluids, &fsp, prad); break;
	case 10: status= src_F_D(fluids, nFluids, &fsp, prad); break;
	case 11: status= src2f_cmp_2f(fluids, nFluids, &fsp, prad, topo, gravdata, tmpStorage); break;
	case 12: status= src__GD(fluids, nFluids, &fsp, prad, topo, gravdata); break;
	case 13: status= src2f_cmp_2f(fluids, nFluids, &fsp, prad, topo, gravdata, tmpStorage); break;
	case 14: status= src2f_cmp_2f(fluids, nFluids, &fsp, prad, topo, gravdata, tmpStorage); break;
	case 15: status= src2f_cmp_2f(fluids, nFluids, &fsp, prad, topo, gravdata, tmpStorage); break;
	default: status = ERROR_INVALID_ARGS; break;
	}

return CHECK_IMOGEN_ERROR(status);

}

/*
% sourcerFunction = useCyl + 2*useRF + 4*usePhi + 8*use2F;

% radiation: 'c' -> commutes with the other operators
%            'N' -> does not commute & is symmetrized
%  cyl    rf      phi     2f      rad     | SOLUTION
%0  0      0       0       0       c       | blank fcn             CHECK
%1  1      0       0       0       c       | cyl only              CHECK
%2  0      1       0       0       c       | rot frame only        CHECK
%3  1      1       0       0       c       | call composite        CHECK
%4  0      0       1       0       c       | call scalarPot        CHECK
%5  1      0       1       0       c       | call composite        CHECK
%6  0      1       1       0       c       | call composite        CHECK
%7  1      1       1       0       c       | call composite        CHECK
%8  0      0       0       1       N       | 2f-drag               CHECK
%9  1      0       0       1       N       | cyl, 2f, cyl          CHECK
%A  0      1       0       1       N       | rf, 2f, rf            CHECK
%B  1      1       0       1       N       | 2f, cmp, 2f           CHECK
%C  0      0       1       1       N       | 2f, phi, 2f           CHECK
%D  1      0       1       1       N       | 2f, composite, 2f     CHECK
%E  0      1       1       1       N       | 2f, composite, 2f     CHECK
%F  1      1       1       1       N       | 2f, composite, 2f     CHECK
%---------------------------------------+-----------------
% We note that two sets of four calls are the same and roll those
% into srcComp and src2f_cmp_2f... */

int sourceRadiation(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad)
{
	int status = SUCCESSFUL;

	if(rad->prefactor == 0) return SUCCESSFUL;

	MGArray fluidReorder[5];
	fluidReorder[0] = fluids[0].data[0];
	fluidReorder[1] = fluids[0].data[2];
	fluidReorder[2] = fluids[0].data[3];
	fluidReorder[3] = fluids[0].data[4];
	fluidReorder[4] = fluids[0].data[1];

	// The radiation solver expects the radiation prefactor to also have the time elapsed multiplied on
	ParametricRadiation rad_time = *rad;
	rad_time.prefactor *= fsp->dt;

	//int sourcefunction_OpticallyThinPowerLawRadiation(MGArray *fluid, MGArray *radRate, int isHydro, double gamma, ParametricRadiation *rad)
	status = sourcefunction_OpticallyThinPowerLawRadiation(&fluidReorder[0], (MGArray *)NULL, fsp->onlyHydro, fluids[0].thermo.gamma, &rad_time);
	return CHECK_IMOGEN_ERROR(status);
}

int setAllBoundaryConditions(GridFluid *fluids, int nFluids, GeometryParams *geo)
{
	int i, status;

	for(i = 0; i < nFluids; i++) {
		status = setFluidBoundary(&fluids[0].data[0],geo, 1);
		if(status != SUCCESSFUL) break;
		status = setFluidBoundary(&fluids[0].data[0],geo, 2);
		if(status != SUCCESSFUL) break;
		status = setFluidBoundary(&fluids[0].data[0],geo, 3);
		if(status != SUCCESSFUL) break;
	}

	return CHECK_IMOGEN_ERROR(status);
}

// srcBlank = 0 0 0 0 = only (maybe) radiation
int srcBlank(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad)
{
	if(rad->prefactor > 0) return CHECK_IMOGEN_ERROR(sourceRadiation(fluids, nFluids, fsp, rad));
	return SUCCESSFUL;
}

// src1000 = 1 0 0 0 = computes 2-fluid drag only
int src___D(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad)
{
	int status = SUCCESSFUL;

	if(nFluids < 2) {
		PRINT_FAULT_HEADER; printf("Somehow, multifluid drag got called but nFluids = %i < 2\n", nFluids); PRINT_FAULT_FOOTER;
		return ERROR_INVALID_ARGS;
	}

	int dragMethod = fsp->multifluidDragMethod;

	if(rad->prefactor > 0) {
		//int sourcefunction_2FluidDrag(MGArray *fluidA, MGArray *fluidB, GeometryParams *geo, ThermoDetails *thermogas, ThermoDetails *thermodust, double dt, int method);
		status = sourcefunction_2FluidDrag(&fluids[0].data[0], &fluids[1].data[0], &(fsp->geometry), &fluids[0].thermo, &fluids[1].thermo, fsp->dt / 2, dragMethod);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

		status = sourceRadiation(fluids, nFluids, fsp, rad);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

		status = sourcefunction_2FluidDrag(&fluids[0].data[0], &fluids[1].data[0], &(fsp->geometry), &fluids[0].thermo, &fluids[1].thermo, fsp->dt / 2, dragMethod);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	} else {
		status = sourcefunction_2FluidDrag(&fluids[0].data[0], &fluids[1].data[0], &(fsp->geometry), &fluids[0].thermo, &fluids[1].thermo, fsp->dt, dragMethod);
		return CHECK_IMOGEN_ERROR(status);
	}

return SUCCESSFUL; // never reached but makes editor stop whining
}

// 0 1 0 0 = source gravity potential only
int src__G_(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad, ParallelTopology *topo, GravityData *gravdata)
{
	// This function sources gravity potential only
	MGArray *phi = gravdata->phi;
	int status;
	int i;

	for(i = 0; i < nFluids; i++) {
		// matlab call:
		//cudaSourceScalarPotential(fluids, run.potentialField.field, run.geometry, [dTime, run.fluid(1).MINMASS, run.fluid(1).MINMASS * 0]);
		status = sourcefunction_ScalarPotential(&fluids[i].data[0], phi, fsp->dt, fsp->geometry, fluids[i].rhoMin, 1*fluids[i].rhoMin);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;

		// Synchronize energy & momentum arrays within devices
		status = MGA_exchangeLocalHalos(&fluids[i].data[1], 4);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;

		// Synchronize across devices
		status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 1);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
		status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 2);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
		status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 3);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
	}

	// Emit radiation if applicable
	if((rad->prefactor != 0) && (status == SUCCESSFUL)) {
		status = CHECK_IMOGEN_ERROR(sourceRadiation(fluids, nFluids, fsp, rad));
	}

	// And set BCs
	if(status == SUCCESSFUL) {
		status = CHECK_IMOGEN_ERROR(setAllBoundaryConditions(fluids, nFluids, &(fsp->geometry)));
	}

	return status;
}

// This function sources 2-fluid drag in the presence of gravity
int src__GD(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad, ParallelTopology *topo, GravityData *gravdata)
{
	int status;
	int dragMethod = fsp->multifluidDragMethod;

	status = sourcefunction_2FluidDrag(&fluids[0].data[0], &fluids[1].data[0], &(fsp->geometry), &fluids[0].thermo, &fluids[1].thermo, fsp->dt / 2.0, dragMethod);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	status = sourceRadiation(fluids, nFluids, fsp, rad);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	MGArray *phi = gravdata->phi;

	int i;
	for(i = 0; i < nFluids; i++) {
			// matlab call:
			//cudaSourceScalarPotential(fluids, run.potentialField.field, run.geometry, [dTime, run.fluid(1).MINMASS, run.fluid(1).MINMASS * 0]);
			// FIXME this function internally uses a fixed 2nd order derivative on the scalar potential.
			status = sourcefunction_ScalarPotential(&fluids[i].data[0], phi, fsp->dt, fsp->geometry, fluids[i].rhoMin, 1*fluids[i].rhoMin);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;

			// Synchronize energy & momentum arrays within devices
			status = MGA_exchangeLocalHalos(&fluids[i].data[1], 4);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;

			// Synchronize across devices
			status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 1);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
			status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 2);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
			status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 3);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
		}

	if(status == SUCCESSFUL) {
		status = sourcefunction_2FluidDrag(&fluids[0].data[0], &fluids[1].data[0], &(fsp->geometry), &fluids[0].thermo, &fluids[1].thermo, fsp->dt / 2.0, dragMethod);
	}

	// And set BCs
	if(status == SUCCESSFUL) {
		status = CHECK_IMOGEN_ERROR(setAllBoundaryConditions(fluids, nFluids, &(fsp->geometry)));
	}

	return CHECK_IMOGEN_ERROR(status);

}

// This function sources rotating frame terms
int src_F__(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad)
{
//	[uv, vv, ~] = run.geometry.ndgridVecs('pos');
//	xyvector = GPU_Type([ (uv-run.geometry.frameRotationCenter(1)) (vv-run.geometry.RotationCenter(2)) ], 1);

	int status;
	int i;
	//MGArray * XYVectors; // Contained inside GeometryParams
	double omega = fsp->geometry.frameOmega;

	for(i = 0; i < nFluids; i++) {
		status = sourcefunction_RotatingFrame(&fluids[i].data[0], fsp->geometry.XYVector, omega, fsp->dt);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
	}

	// Emit radiation if applicable
	if((rad->prefactor != 0) && (status == SUCCESSFUL)) {
		status = CHECK_IMOGEN_ERROR(sourceRadiation(fluids, nFluids, fsp, rad));
	}

	// Assert boundary conditions
	if(status == SUCCESSFUL) {
		status = CHECK_IMOGEN_ERROR(setAllBoundaryConditions(fluids, nFluids, &(fsp->geometry)));
	}

	return status;

}

// This function sources 2-fluid drag in a rotating frame
int src_F_D(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad)
{

//[uv, vv, ~] = run.geometry.ndgridVecs('pos');
//xyvector = GPU_Type([ (uv-run.frameTracking.rotateCenter(1)) (vv-run.frameTracking.rotateCenter(2)) ], 1);

int status;
int i;
//MGArray * XYVectors; // contained in fsp->geometry.XYVector
double omega = fsp->geometry.frameOmega;
int dragMethod = fsp->multifluidDragMethod;

fsp->dt /= 2.0;

for(i = 0; i < nFluids; i++) {
	status = sourcefunction_RotatingFrame(&fluids[i].data[0], fsp->geometry.XYVector, omega, fsp->dt);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
}

// Emit radiation if applicable
if((rad->prefactor != 0) && (status == SUCCESSFUL)) {
	status = CHECK_IMOGEN_ERROR(sourceRadiation(fluids, nFluids, fsp, rad));
}

if(status == SUCCESSFUL) {
	status = sourcefunction_2FluidDrag(&fluids[0].data[0], &fluids[1].data[0], &(fsp->geometry), &fluids[0].thermo, &fluids[1].thermo, fsp->dt*2, dragMethod);
}

// Emit radiation if applicable
if((rad->prefactor != 0) && (status == SUCCESSFUL)) {
	status = CHECK_IMOGEN_ERROR(sourceRadiation(fluids, nFluids, fsp, rad));
}

// rotate frame
for(i = 0; i < nFluids; i++) {
	status = sourcefunction_RotatingFrame(&fluids[i].data[0], fsp->geometry.XYVector, omega, fsp->dt);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
}

fsp->dt *= 2.0;

// Assert boundary conditions
if(status == SUCCESSFUL) {
	status = CHECK_IMOGEN_ERROR(setAllBoundaryConditions(fluids, nFluids, &(fsp->geometry)));
}

return status;
}

// This function sources cylindrical geometry terms
int srcC___(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad)
{

	int status;
	int i;

	for(i = 0; i < nFluids; i++) {
		//cudaSourceCylindricalTerms(fluids, dTime, run.geometry);
		status = sourcefunction_CylindricalTerms(&fluids[i].data[0], fsp->dt, &fsp->geometry.h[0], fsp->geometry.Rinner);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
	}

	if(status == SUCCESSFUL) {
		status = sourceRadiation(fluids, nFluids, fsp, rad);
	}

	if(status == SUCCESSFUL) {
		status = CHECK_IMOGEN_ERROR(setAllBoundaryConditions(fluids, nFluids, &(fsp->geometry)));
	}

	return status;

}

// This function sources 2-fluid drag and cylindrical coordinates
int srcC__D(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad)
{
int status;
int i;

int dragMethod = fsp->multifluidDragMethod;

// Utilize standard (A/2)(B)(A/2) operator split to achieve 2nd order time accuracy in the splitting
for(i = 0; i < nFluids; i++) {
	//cudaSourceCylindricalTerms(fluids, dTime, run.geometry);
	status = sourcefunction_CylindricalTerms(&fluids[i].data[0], fsp->dt/2, &fsp->geometry.h[0], fsp->geometry.Rinner);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
}

if(status == SUCCESSFUL) {
		fsp->dt /= 2.0;
		status = CHECK_IMOGEN_ERROR(sourceRadiation(fluids, nFluids, fsp, rad));
		fsp->dt *= 2.0;
	}

if(status == SUCCESSFUL) {
		status = CHECK_IMOGEN_ERROR(sourcefunction_2FluidDrag(&fluids[0].data[0], &fluids[1].data[0], &(fsp->geometry), &fluids[0].thermo, &fluids[1].thermo, fsp->dt, dragMethod));
	}

if(status == SUCCESSFUL) {
		fsp->dt /= 2.0;
		status = CHECK_IMOGEN_ERROR(sourceRadiation(fluids, nFluids, fsp, rad));
		fsp->dt *= 2.0;
	}

for(i = 0; i < nFluids; i++) {
	//cudaSourceCylindricalTerms(fluids, dTime, run.geometry);
	status = sourcefunction_CylindricalTerms(&fluids[i].data[0], fsp->dt/2, &fsp->geometry.h[0], fsp->geometry.Rinner);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
}

// Enforce BCs & return
if(status == SUCCESSFUL) {
	status = CHECK_IMOGEN_ERROR(setAllBoundaryConditions(fluids, nFluids, &(fsp->geometry)));
}

return status;
}

// This function handles rotating frame, cylindrical coordinates, and gravity potentials.
int srcComp(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad, ParallelTopology *topo, GravityData *gravdata, MGArray *storage)
{

	int status;
	int i;

	//[uv, vv, ~] = run.geometry.ndgridVecs('pos');
	//xyvector = GPU_Type([ (uv-run.geometry.frameRotationCenter(1)) (vv-run.geometry.frameRotationCenter(2)) ], 1);

	// This call solves geometric source terms, frame rotation and gravity simultaneously
	// parameter vector:
	// [rho_no gravity, rho_full gravity, omega, dt, space order, time order]
	// It can be programmed to use either implicit midpoint (IMP), Runge-Kutta 4 (RK4), Gauss-Legendre 4 (GL4), or GL6

	MGArray *phi = gravdata->phi;

	for(i = 0; i < nFluids; i++) {
		status = sourcefunction_Composite(&fluids[i].data[0], phi, fsp->geometry.XYVector, fsp->geometry, fluids[i].rhoMin, fluids[i].rhoMin*2, fsp->dt, gravdata->spaceOrder, gravdata->timeOrder, storage);

		// Synchronize across devices
		status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 1);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
		status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 2);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
		status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 3);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
	}

	if(status == SUCCESSFUL) {
		status = CHECK_IMOGEN_ERROR(sourceRadiation(fluids, nFluids, fsp, rad));
	}

	if(status == SUCCESSFUL) {
		status = CHECK_IMOGEN_ERROR(setAllBoundaryConditions(fluids, nFluids, &(fsp->geometry)));
	}

return status;

}


// This function handles
// * 2-fluid drag
// * in a rotating frame in cylindrical coordinates with gravity
int src2f_cmp_2f(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParametricRadiation *rad, ParallelTopology *topo, GravityData *gravdata, MGArray *storage)
{

	//[uv, vv, ~] = run.geometry.ndgridVecs('pos');
	//xyvector = GPU_Type([ (uv-run.geometry.frameRotationCenter(1)) (vv-run.geometry.frameRotationCenter(2)) ], 1);


	int dragMethod = fsp->multifluidDragMethod;

	MGArray *phi = gravdata->phi;

	int i, status;

	// This call solves geometric source terms, frame rotation and gravity simultaneously
	// parameter vector:
	// [rho_no gravity, rho_full gravity, omega, dt, space order, time order]
	// It can be programmed to use either implicit midpoint (IMP), Runge-Kutta 4 (RK4), Gauss-Legendre 4 (GL4), or GL6

	status = CHECK_IMOGEN_ERROR(sourcefunction_2FluidDrag(&fluids[0].data[0], &fluids[1].data[0], &(fsp->geometry), &fluids[0].thermo, &fluids[1].thermo, fsp->dt / 2.0, dragMethod));

	if(status == SUCCESSFUL) {
		for(i = 0; i < nFluids; i++) {
			status = sourcefunction_Composite(&fluids[i].data[0], phi, fsp->geometry.XYVector, fsp->geometry, fluids[i].rhoMin, fluids[i].rhoMin*2, fsp->dt, gravdata->spaceOrder, gravdata->timeOrder, storage);

			// Synchronize across devices
			status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 1);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
			status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 2);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
			status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 3);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
		}
	}

	if(status == SUCCESSFUL) {
		fsp->dt /= 2.0;
		status = CHECK_IMOGEN_ERROR(sourceRadiation(fluids, nFluids, fsp, rad));
		fsp->dt *= 2.0;
	}

	if(status == SUCCESSFUL) {
		status = CHECK_IMOGEN_ERROR(sourcefunction_2FluidDrag(&fluids[0].data[0], &fluids[1].data[0], &(fsp->geometry), &fluids[0].thermo, &fluids[1].thermo, fsp->dt / 2.0, dragMethod));
	}

	if(gravdata->spaceOrder > 0) {
		for(i = 0; i < nFluids; i++) {
			status = sourcefunction_Composite(&fluids[i].data[0], phi, fsp->geometry.XYVector, fsp->geometry, fluids[i].rhoMin, fluids[i].rhoMin*2, fsp->dt, gravdata->spaceOrder, gravdata->timeOrder, storage);

			// Synchronize across devices
			status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 1);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
			status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 2);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
			status = exchange_MPI_Halos(&fluids[i].data[1], 4, topo, 3);
			if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
		}
	}

	if(status == SUCCESSFUL) {
		fsp->dt /= 2.0;
		status = CHECK_IMOGEN_ERROR(sourceRadiation(fluids, nFluids, fsp, rad));
		fsp->dt *= 2.0;
	}

	if(status == SUCCESSFUL) {
		status = CHECK_IMOGEN_ERROR(setAllBoundaryConditions(fluids, nFluids, &(fsp->geometry)));
	}

	return status;
}


