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

// PGW
#include "parallel_halo_arrays.h"

// My stuff
#include "cudaCommon.h"
#include "compiled_common.h"
#include "cudaHaloExchange.h"

/* THIS ROUTINE
   This routine interfaces with the parallel gateway halo routines
   The N MGArrays *ed to by phi swap ghost cells as described by topo
   circularity[
 */

#ifdef STANDALONE_MEX_FUNCTION
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Functional form:
    cudaHaloExchange(arraytag, dimension_to_exchange, parallel topology information, circularity)

    1. get neighbors from halo library in dimension_to_exchange direction
    2. determine which memory direction that currently is
    3. If it's x or y, rip it to a linear array
    4. Aquire some host-pinned memory and dump to that
    5. pass that host pointer to halo_exchange
    6. wait for MPI to return control
	 */
	if (nrhs!=4) mexErrMsgTxt("call form is cudaHaloExchange(arraytag, dimension_to_xchg, topology, circularity).\n");

	CHECK_CUDA_ERROR("entering cudaHaloExchange");
	int xchg = (int)*mxGetPr(prhs[1]) - 1;

	pParallelTopology parallelTopo = topoStructureToC(prhs[2]);

	if(parallelTopo->nproc[xchg] == 1) return;
	// Do not waste time if we can't possibly have any work to do

	MGArray phi;
	int worked = MGA_accessMatlabArrays(prhs, 0, 0, &phi);

	exchange_MPI_Halos(&phi, 1, parallelTopo, xchg);
}
#endif

int exchange_MPI_Halos(MGArray *phi, int nArrays, pParallelTopology topo, int xchgDir)
{
	CHECK_CUDA_ERROR("entering exchange_MPI_Halos");
	xchgDir -= 1; // Convert 1-2-3 index into 0-1-2 memory index

	// Avoid wasting time...
	if(xchgDir+1 > topo->ndim) return ERROR_NULL_OPERATION;
	if(topo->nproc[xchgDir] == 1) return ERROR_NULL_OPERATION;

	int memDir;

	int i;
	for(i = 0; i < nArrays; i++) {
		memDir = phi->currentPermutation[xchgDir]; // The actual in-memory direction we're gonna be exchanging

		double *ptrHalo;

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
		}

		// Find the size of the swap buffer
		int numPerHalo = MGA_wholeFaceHaloNumel(phi, memDir, 3);

		cudaError fail = cudaMallocHost((void **)&ptrHalo, 4*numPerHalo*sizeof(double));
		CHECK_CUDA_ERROR("cudaHostAlloc");

		MPI_Comm commune = MPI_Comm_f2c(topo->comm);

		double *ptmp = ptrHalo;
		// Fetch left face
		if(leftCircular)
			MGA_wholeFaceToLinear(phi, memDir, 0, 0, 3, &ptmp);

		ptmp = ptrHalo + numPerHalo;
		// Fetch right face
		if(rightCircular)
			MGA_wholeFaceToLinear(phi, memDir, 1, 0, 3, &ptmp);

		// synchronize to make sure host sees what was uploaded
		int j;
		for(j = 0; j < phi->nGPUs; j++) {
			cudaSetDevice(phi->deviceID[j]);
			cudaDeviceSynchronize();
		}

		parallel_exchange_dim_contig(topo, xchgDir, (void*)ptrHalo,
				(void*)(ptrHalo + numPerHalo),\
				(void*)(ptrHalo+2*numPerHalo),\
				(void*)(ptrHalo+3*numPerHalo), numPerHalo, MPI_DOUBLE);
		MPI_Barrier(MPI_COMM_WORLD);

		// write left face
		ptmp = ptrHalo + 2*numPerHalo;
		if(leftCircular)
			MGA_wholeFaceToLinear(phi, memDir, 0, 1, 3, &ptmp);

		ptmp = ptrHalo + 3*numPerHalo;
		// Fetch right face
		if(rightCircular)
			MGA_wholeFaceToLinear(phi, memDir, 1, 1, 3, &ptmp);

		cudaFreeHost((void **)ptrHalo);

		// Move to the next array to exchange
		phi++;
	}

	return SUCCESSFUL;

}



