
#include "mex.h"

#include "cudaCommon.h"

#include "cudaFluidStep.h"
#include "cudaArrayRotateB.h"
#include "cudaHaloExchange.h"
#include "cudaStatics.h"

#include "flux.h"

int setFluidBoundaries(MGArray *x, int nArrays, int dir);

//int performFluidUpdate_3D(MGArray *fluid, ParallelTopology* parallelTopo, int order, int stepNumber, double *lambda, double gamma, double minRho, double stepMethod, int geomType, double Rinner)
int performFluidUpdate_3D(MGArray *fluid, ParallelTopology* parallelTopo, FluidStepParams fsp, int stepNumber, int order)
{
int sweep, flag_1D = 0;

// Choose our sweep number depending on whether we are 1- or 2-dimensional
//if(fluid[0].dim[2] > 1) { // if nz > 1, three-dimensional
	sweep = (stepNumber + 3*(order > 0)) % 6;
//} else {
//	if(fluid[0].dim[1] > 3) { // if ny > 3, two dimensional
//		sweep = (stepNumber + (order < 0)) % 2;
//	} else {
//		flag_1D = 1;
//	}
//}

int preperm[6] = {0, 2, 0, 2, 3, 3};

int fluxcall[3][6] = {{1,2,1,2,3,3},{3,1,2,3,1,2},{2,3,3,1,2,1}};
int permcall[3][6] = {{3,2,2,3,3,2},{2,3,3,5,2,6},{6,3,5,0,2,0}};

int n;
int returnCode = SUCCESSFUL;
int nowDir;

FluidStepParams stepParameters = fsp;

// Just short-circuit for a one-D run, don't try to make the 2/3D loop reduce for it
if(flag_1D) {
	nowDir = 1;
	stepParameters.stepDirection = nowDir;

	returnCode = performFluidUpdate_1D(fluid, stepParameters, parallelTopo);
	if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
	returnCode = setFluidBoundary(fluid, fluid->matlabClassHandle, nowDir);
	if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
	returnCode = exchange_MPI_Halos(fluid, 5, parallelTopo, nowDir);
	return CHECK_IMOGEN_ERROR(returnCode);
}

if(order > 0) { /* If we are doing forward sweep */
	returnCode = (preperm[sweep] != 0 ? flipArrayIndices(fluid, NULL, 5, preperm[sweep]) : SUCCESSFUL);
	if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

	for(n = 0; n < 3; n++) {
		nowDir = fluxcall[n][sweep];
		if(fluid->dim[0] > 3) {
			stepParameters.stepDirection = nowDir;

			/* FIXME: This SHOULD NOT be necessary but I don't have time to track down
			 * FIXME: why the damn halos are O(t) dirty at this point */
			if(nowDir == fluid->partitionDir) {
				returnCode = MGA_exchangeLocalHalos(fluid, 5);
                		if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;
			}

			returnCode = performFluidUpdate_1D(fluid, stepParameters, parallelTopo);
			if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
			returnCode = setFluidBoundary(fluid, fluid->matlabClassHandle, nowDir);
			if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
			returnCode = exchange_MPI_Halos(fluid, 5, parallelTopo, nowDir);
			if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
		}
		/* FIXME: INSERT MAGNETIC FLUX ROUTINES HERE */

		returnCode = (permcall[n][sweep] != 0 ? flipArrayIndices(fluid, NULL, 5, permcall[n][sweep]) : SUCCESSFUL );
		if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
	}

} else { /* If we are doing backwards sweep */
	returnCode = (preperm[sweep] != 0 ? flipArrayIndices(fluid, NULL, 5, preperm[sweep]) : SUCCESSFUL);
	if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

	for(n = 0; n < 3; n++) {
		nowDir = fluxcall[n][sweep];
		/* FIXME: INSERT MAGNETIC FLUX ROUTINES HERE */

		if(fluid->dim[0] > 3) {
			stepParameters.stepDirection = nowDir;

			/* FIXME: This SHOULD NOT be necessary but I don't have time to track down
			 * FIXME: why the damn halos are O(t) dirty at this point */
			if(nowDir == fluid->partitionDir) {
				returnCode = MGA_exchangeLocalHalos(fluid, 5);
                		if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;
			}

			returnCode = performFluidUpdate_1D(fluid, stepParameters, parallelTopo);
			if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
			returnCode = setFluidBoundary(fluid, fluid->matlabClassHandle, nowDir);
			if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
			returnCode = exchange_MPI_Halos(fluid, 5, parallelTopo, nowDir);
			if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
		}

		returnCode = (permcall[n][sweep] != 0 ? flipArrayIndices(fluid, NULL, 5, permcall[n][sweep]) : SUCCESSFUL );
		if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
	}
}

return CHECK_IMOGEN_ERROR(returnCode);

/* Fluid half-step completed 
 * If order > 0, next call sourcing terms
 * If order < 0, next call fluid with order > 0 */
}

