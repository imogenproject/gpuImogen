#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif
#include "mex.h"
#include "mpi.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas.h"

// My stuff
#include "cudaCommon.h"
#include "mpi_common.h"
#include "cudaHaloExchange.h"

/* THIS ROUTINE
   This routine interfaces with the parallel gateway halo routines
   The N MGArrays *ed to by phi swap ghost cells as described by topo
   circularity[...
 */

#define HALO_DEPTH 3

#ifdef STANDALONE_MEX_FUNCTION
/* mexFunction call:
 * cudaHaloExchange(GPU array, direction, topology, exterior circularity) */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs!=4) mexErrMsgTxt("call form is cudaHaloExchange(arraytag, dimension_to_xchg, topology, circularity).\n");

	CHECK_CUDA_ERROR("entering cudaHaloExchange");
	int xchg = (int)*mxGetPr(prhs[1]);

	ParallelTopology parallelTopo;
	topoStructureToC(prhs[2], &parallelTopo);

	if(parallelTopo.nproc[xchg] == 1) return;
	// Do not waste time if we can't possibly have any work to do

	MGArray phi;
	int worked = MGA_accessMatlabArrays(prhs, 0, 0, &phi);
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) { DROP_MEX_ERROR("Failed to access GPU array."); }

	if(CHECK_IMOGEN_ERROR(exchange_MPI_Halos(&phi, 1, &parallelTopo, xchg)) != SUCCESSFUL) {
		DROP_MEX_ERROR("Failed to perform MPI halo exchange!");
	}
}
#endif

/* exchange_MPI_Halos(MGArray *phi, int narrays, ParallelTopo *t, int xchgdir):
 * phi     - pointer to 1 or more MGArrays to synchronize halos with
 * narrays - number of arrays 
 * t       - parallel topology
 * xchgDir - array direction to synchronize */
int exchange_MPI_Halos(MGArray *phi, int nArrays, ParallelTopology* topo, int xchgDir)
{
	int returnCode = CHECK_CUDA_ERROR("entering exchange_MPI_Halos");
	if(returnCode != SUCCESSFUL) { return returnCode; }

	xchgDir -= 1; // Convert 1-2-3 index into 0-1-2 memory index

	// Avoid wasting time...
	if(xchgDir+1 > topo->ndim) return SUCCESSFUL;
	if(topo->nproc[xchgDir] == 1) return SUCCESSFUL;

	int memDir;

	int i;
	for(i = 0; i < nArrays; i++) {
		/* Be told if the left and right sides of the dimension are circular or not */
		int leftCircular, rightCircular;
		switch(xchgDir) {
		case 0:
			leftCircular = (phi->circularBoundaryBits & MGA_BOUNDARY_XMINUS) ? 1 : 0;
			rightCircular = (phi->circularBoundaryBits & MGA_BOUNDARY_XPLUS) ? 1 : 0;
			break;
		case 1:
			leftCircular = (phi->circularBoundaryBits & MGA_BOUNDARY_YMINUS) ? 1 : 0;
			rightCircular = (phi->circularBoundaryBits & MGA_BOUNDARY_YPLUS) ? 1 : 0;
			break;

		case 2:
			leftCircular = (phi->circularBoundaryBits & MGA_BOUNDARY_YMINUS) ? 1 : 0;
			rightCircular = (phi->circularBoundaryBits & MGA_BOUNDARY_YPLUS) ? 1 : 0;
			break;
		default:
			PRINT_FAULT_HEADER;
			printf("Valid exchange directions are 1/2/3\nI was called with %i\n", xchgDir + 1);
			PRINT_FAULT_FOOTER;

			return ERROR_INVALID_ARGS;
		}

		memDir = phi->currentPermutation[xchgDir]; // The actual in-memory direction we're gonna be exchanging

		double *ptrHalo;

		// Find the size of the swap buffer
		int numPerHalo = MGA_wholeFaceHaloNumel(phi, memDir, HALO_DEPTH);

		cudaMallocHost((void **)&ptrHalo, 4*numPerHalo*sizeof(double));
		returnCode = CHECK_CUDA_ERROR("cudaHostAlloc");
		if(returnCode != SUCCESSFUL) return returnCode;

		MPI_Comm commune = MPI_Comm_f2c(topo->comm);

		double *ptmp = ptrHalo;
		// Fetch left face
		if(leftCircular)
			returnCode = MGA_wholeFaceToLinear(phi, memDir, 0, 0, HALO_DEPTH, &ptmp);
		if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);


		ptmp = ptrHalo + numPerHalo;
		// Fetch right face
		if(rightCircular)
			returnCode = MGA_wholeFaceToLinear(phi, memDir, 1, 0, HALO_DEPTH, &ptmp);
		if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

		// synchronize to make sure host sees what was uploaded
		int j;
		for(j = 0; j < phi->nGPUs; j++) {
			cudaSetDevice(phi->deviceID[j]);
			cudaDeviceSynchronize();
		}

		mpi_exchangeHalos(topo, xchgDir, (void*)ptrHalo,
				(void*)(ptrHalo + numPerHalo),\
				(void*)(ptrHalo+2*numPerHalo),\
				(void*)(ptrHalo+3*numPerHalo), numPerHalo, MPI_DOUBLE);
		MPI_Barrier(MPI_COMM_WORLD);

		// write left face
		ptmp = ptrHalo + 2*numPerHalo;
		if(leftCircular)
			returnCode = MGA_wholeFaceToLinear(phi, memDir, 0, 1, HALO_DEPTH, &ptmp);
		if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

		ptmp = ptrHalo + 3*numPerHalo;
		// Fetch right face
		if(rightCircular)
			returnCode = MGA_wholeFaceToLinear(phi, memDir, 1, 1, HALO_DEPTH, &ptmp);
		if(returnCode != SUCCESSFUL) return CHECK_IMOGEN_ERROR(returnCode);

		cudaFreeHost((void **)ptrHalo);

		// Move to the next array to exchange
		phi++;
	}

	return SUCCESSFUL;

}



