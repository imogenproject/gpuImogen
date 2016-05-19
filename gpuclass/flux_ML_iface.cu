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
#include "flux.h"

// Only uncomment this if you plan to debug this file.
//#define DEBUGMODE


#ifdef DEBUGMODE
    #include "debug_inserts.h"
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	int wanted_nlhs = 0;
#ifdef DEBUGMODE
	wanted_nlhs = 1;
#endif

// HACK HACK AHCK
// FIXME: This needs to accept FluidManager containers and unpack their rho/E/mom() arrays here

	// Input and result
	//if ((nrhs!= 11) || (nlhs != wanted_nlhs)) mexErrMsgTxt("Wrong number of arguments: need flux_ML_iface(rho, E, px, py, pz, bx, by, bz, [dt, purehydro?, fluid gamma, order, step #, step method], topology, {dx,dy,dz})\n");

	if ((nrhs!= 7) || (nlhs != wanted_nlhs)) mexErrMsgTxt("Wrong number of arguments: need flux_ML_iface(fluid, bx, by, bz, [dt, purehydro?, fluid gamma, order, step #, step method], topology, {dx,dy,dz})\n");

	/* Access the FluidManager canisters */
	const mxArray *fluidmgr = prhs[0];
	mxArray *fluidPtrs[5];
	fluidPtrs[0] = mxGetProperty(fluidmgr,0,(const char *)("mass"));
	fluidPtrs[1] = mxGetProperty(fluidmgr,0,(const char *)("ener"));
	fluidPtrs[2] = mxGetProperty(fluidmgr,0,(const char *)("mom"));

	/* Access rho/E/px/py/pz arrays */
	MGArray fluid[5];

	MGA_accessMatlabArrays((const mxArray **)&fluidPtrs[0], 0, 1, &fluid[0]);
    MGA_accessMatlabArrayVector(fluidPtrs[2], 0, 2, &fluid[2]);
	/* Access bx/by/bz cell-centered arrays */
	/* ... */

    int idxpost = 4; // 8 for the old way

	double *scalars = mxGetPr(prhs[idxpost]);

	double dt = scalars[0]; /* Access lambda (dt / dx) */
	int ishydro   = scalars[1]; /* determine if purely hydrodynamic */
	double gamma  = scalars[2]; /* Adiabatic index of fluid */

	int sweepDirect = (int)scalars[3]; /* Identify if forwards (sweepDirect = 1) or backwards (-1) */
	int stepNum     = (int)scalars[4]; /* step number (used to pick the permutation of the fluid propagators) */
	int stepMethod  = (int)scalars[5]; /* 1=HLL, 2=HLLC, 3=Xin/Jin */

	/* Access topology structure */
	ParallelTopology topo;
	topoStructureToC(prhs[idxpost+1], &topo);

	double lambda[3];

	// Getting cell spacing data:
	int gotcells = mxIsCell(prhs[idxpost+2]);

	if(gotcells) {
		mxArray *dxi;
		int q;
		for(q = 0; q < 3; q++) {
			dxi = mxGetCell(prhs[idxpost+2], q);
			if(dxi != NULL) {
				lambda[q] = dt / (*mxGetPr(dxi));
			} else {
				printf("Attempted to get %ith cell element", q+1);
				mexErrMsgTxt("Attempt to get array in {dx,dy,dz} failed!\n");
			}
		}
	} else {
		mexErrMsgTxt("Expected argument 11 to be {dx, dy, dz}, was not a cell array");
	}

	CHECK_CUDA_ERROR("entering compiled fluid step");
	int returnCode = performFluidUpdate_3D(&fluid[0], &topo, sweepDirect, stepNum, &lambda[0], gamma, stepMethod);
	CHECK_IMOGEN_ERROR(returnCode);

	if(returnCode != SUCCESSFUL) mexErrMsgTxt("Fluid update code returned unsuccessfully!");

}
