
#ifndef NOMATLAB
#include "mex.h"
#endif

#include "nvToolsExt.h"
#include "cudaCommon.h"

#include "cudaFluidStep.h"
#include "cudaArrayRotateB.h"
#include "cudaHaloExchange.h"
#include "cudaStatics.h"

#include "flux.h"

#ifdef NOMATLAB
int getGPUTypeStreams(void *inputArray, cudaStream_t **streams, int *numel)
{
	ParallelTopology *p = (ParallelTopology *)inputArray;

	numel[0] = 2;
	streams[0] = &p->cudaStreamPtrs[0];
}
#endif


int performFluidUpdate_3D(MGArray *fluid, ParallelTopology* parallelTopo, FluidStepParams fsp, MGArray *tempStorage)
{
int sweep, flag_1D = 0;

// Choose our sweep number depending on whether we are 1- or 2-dimensional
//if(fluid[0].dim[2] > 1) { // if nz > 1, three-dimensional
	sweep = (fsp.stepNumber + 3*(fsp.stepDirection > 0)) % 6;

if((fluid->dim[1] == 1) && (fluid->dim[2] == 1)) {
	flag_1D = 1;
}
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

	returnCode = performFluidUpdate_1D(fluid, stepParameters, parallelTopo, (MGArray *)NULL);
	if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
	returnCode = setFluidBoundary(fluid, &fsp.geometry, nowDir);
	if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
	returnCode = exchange_MPI_Halos(fluid, 5, parallelTopo, nowDir);
	return CHECK_IMOGEN_ERROR(returnCode);
}

#ifdef USE_NVTX
	nvtxRangePush(__FUNCTION__);
#endif

// Put pointers for GPU storage here so we can acquire it once for this whole step, reusing for all 3 fluid calls
// and the array rotates.
MGArray localTempStorage;
localTempStorage.nGPUs = -1; // Use this as a "not allocated" marker.
int usingLocalTemp = 0;
if(tempStorage == NULL) {
	usingLocalTemp = 1;
	tempStorage = &localTempStorage;
}

cudaStream_t *cudaStreams;
int nstreams;

#ifdef NOMATLAB
int s = getGPUTypeStreams((void *)parallelTopo, &cudaStreams, &nstreams);
#else
int s = getGPUTypeStreams(fluid->boundaryConditions.externalData, &cudaStreams, &nstreams);
#endif

if(fsp.stepDirection > 0) { /* If we are doing forward sweep */
	// If we already have a buffer it's large, use it. Otherwise we actually /lose/ beause
	// performFluidUpdate below will have to cudaMalloc and free
	returnCode = (preperm[sweep] != 0 ? flipArrayIndices(fluid, (MGArray **)NULL, 5, preperm[sweep], cudaStreams,
		usingLocalTemp ? (MGArray *)NULL : tempStorage) : SUCCESSFUL);
	if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

	for(n = 0; n < 3; n++) {
		nowDir = fluxcall[n][sweep];
		if(fluid->dim[0] > 3) {
			stepParameters.stepDirection = nowDir;
			returnCode = performFluidUpdate_1D(fluid, stepParameters, parallelTopo, tempStorage);
			if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

			returnCode = setFluidBoundary(fluid, &fsp.geometry, nowDir);
			if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

			returnCode = exchange_MPI_Halos(fluid, 5, parallelTopo, nowDir);
			if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
		}
		/* FIXME: INSERT MAGNETIC FLUX ROUTINES HERE */

		returnCode = (permcall[n][sweep] != 0 ? flipArrayIndices(fluid, (MGArray **)NULL, 5, permcall[n][sweep], cudaStreams, tempStorage) : SUCCESSFUL );
		if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
	}
} else { /* If we are doing backwards sweep */
	// If we already have a buffer it's large, use it. Otherwise we actually /lose/ beause
	// performFluidUpdate below will have to cudaMalloc and free
	returnCode = (preperm[sweep] != 0 ? flipArrayIndices(fluid, (MGArray **)NULL, 5, preperm[sweep], cudaStreams,
		usingLocalTemp ? (MGArray *)NULL : tempStorage) : SUCCESSFUL);
	if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

	for(n = 0; n < 3; n++) {
		nowDir = fluxcall[n][sweep];
		/* FIXME: INSERT MAGNETIC FLUX ROUTINES HERE */

		if(fluid->dim[0] > 3) {
			stepParameters.stepDirection = nowDir;
			returnCode = performFluidUpdate_1D(fluid, stepParameters, parallelTopo, tempStorage);
			if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

			returnCode = setFluidBoundary(fluid, &fsp.geometry, nowDir);
			if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

			returnCode = exchange_MPI_Halos(fluid, 5, parallelTopo, nowDir);
			if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
		}

		returnCode = (permcall[n][sweep] != 0 ? flipArrayIndices(fluid, (MGArray **)NULL, 5, permcall[n][sweep], cudaStreams, tempStorage) : SUCCESSFUL );
		if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
	}
}
	
if(usingLocalTemp) {
	#ifdef USE_NVTX
	nvtxMark("flux.cu:131 large free");
	#endif
	returnCode = MGA_delete(&localTempStorage);
	localTempStorage.nGPUs = -1;
	if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);
}


#ifdef USE_NVTX
	nvtxRangePop();
#endif

return CHECK_IMOGEN_ERROR(returnCode);

/* Fluid half-step completed 
 * If order > 0, next call sourcing terms
 * If order < 0, next call fluid with order > 0 */
}

