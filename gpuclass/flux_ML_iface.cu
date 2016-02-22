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

#include "cudaCommon.h"
#include "flux.h"

// Only uncomment this if you plan to debug this file.
//#define DEBUGMODE

#include "mpi.h"
#include "parallel_halo_arrays.h"

#ifdef DEBUGMODE
    #include "debug_inserts.h"
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	int wanted_nlhs = 0;
#ifdef DEBUGMODE
	wanted_nlhs = 1;
#endif

	// Input and result
	if ((nrhs!= 11) || (nlhs != wanted_nlhs)) mexErrMsgTxt("Wrong number of arguments: need flux_ML_iface(rho, E, px, py, pz, bx, by, bz, [dt, purehydro?, fluid gamma, order, step #, step method], topology, {dx,dy,dz})\n");

	/* Access rho/E/px/py/pz arrays */
	MGArray fluid[5];
	MGA_accessMatlabArrays(prhs, 0, 4, &fluid[0]);

	/* Access bx/by/bz cell-centered arrays */
	/* ... */

	double *scalars = mxGetPr(prhs[8]);

	double dt = scalars[0]; /* Access lambda (dt / dx) */
	int ishydro   = scalars[1]; /* determine if purely hydrodynamic */
	double gamma  = scalars[2]; /* Adiabatic index of fluid */

	int sweepDirect = (int)scalars[3]; /* Identify if forwards (sweepDirect = 1) or backwards (-1) */
	int stepNum     = (int)scalars[4]; /* step number (used to pick the permutation of the fluid propagators) */
	int stepMethod  = (int)scalars[5]; /* 1=HLL, 2=HLLC, 3=Xin/Jin */

	/* Access topology structure */
	pParallelTopology topo = topoStructureToC(prhs[9]);

	double lambda[3];

	// Getting cell spacing data:
	int gotcells = mxIsCell(prhs[10]);

	if(gotcells) {
		mxArray *dxi;
		int q;
		for(q = 0; q < 3; q++) {
			dxi = mxGetCell(prhs[10], q);
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
	int returnCode = performFluidUpdate_3D(&fluid[0], topo, sweepDirect, stepNum, &lambda[0], gamma, stepMethod);
	CHECK_IMOGEN_ERROR(returnCode);

	if(returnCode != SUCCESSFUL) mexErrMsgTxt("Fluid update code returned unsuccessfully!");

}
