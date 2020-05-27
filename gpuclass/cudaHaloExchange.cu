#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif
#ifdef STANDALONE_MEX_FUNCTION
#include "mex.h"
#endif

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas.h"
#include "nvToolsExt.h"

// My stuff
#include "cudaCommon.h"
#include "mpi_common.h"
#include "cudaHaloExchange.h"

/* THIS ROUTINE
   This routine interfaces with the parallel gateway halo routines
   The N MGArrays *ed to by phi swap ghost cells as described by topo
   circularity[...
 */

double *hostNormalBufferRoot;
double *hostPinnedBufferRoot;
int globalBufferLength;
int amRegistered = 0;

void exitFreeFunction(void)
{

if(hostNormalBufferRoot != NULL)
	free(hostNormalBufferRoot);

if(hostPinnedBufferRoot != NULL) {
	cudaHostUnregister((void *)hostPinnedBufferRoot);
	free(hostPinnedBufferRoot);
}

globalBufferLength = 0; 

}


void checkGlobalBufferLen(int newlen)
{

if(newlen > globalBufferLength) {
	int returnCode;
	// resize host memory
	hostNormalBufferRoot = (double *)realloc((void *)hostNormalBufferRoot, newlen * sizeof(double));

	// unregister
	if(hostPinnedBufferRoot != NULL) {
		cudaHostUnregister((void *)hostPinnedBufferRoot);
		returnCode = CHECK_CUDA_ERROR("cudaHostUnregister");
	        if(returnCode != SUCCESSFUL) return;
		}

	// resize
	hostPinnedBufferRoot = (double *)realloc((void *)hostPinnedBufferRoot, newlen * sizeof(double));
	// repin
	cudaHostRegister ((void *)hostPinnedBufferRoot, newlen * sizeof(double), cudaHostRegisterMapped);
	returnCode = CHECK_CUDA_ERROR("cudaHostUnregister");
	if(returnCode != SUCCESSFUL) return;
	
#ifdef USE_NVTX
	nvtxMark("cudaHaloExchange buffer expansion triggered");
#endif

	globalBufferLength = newlen;
}

}

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
int exchange_MPI_Halos(MGArray *theta, int nArrays, ParallelTopology* topo, int xchgDir)
{
	if(amRegistered == 0) {
#ifdef NOMATLAB
		atexit(&exitFreeFunction);
#else
		mexAtExit(&exitFreeFunction);
#endif
		amRegistered = 1; 
	}

	int returnCode = CHECK_CUDA_ERROR("entering exchange_MPI_Halos");
	if(returnCode != SUCCESSFUL) { return returnCode; }

	xchgDir -= 1; // Convert 1-2-3 index into 0-1-2 memory index

	// Avoid wasting time...
	if(xchgDir+1 > topo->ndim) return SUCCESSFUL;
	if(topo->nproc[xchgDir] == 1) return SUCCESSFUL;
#ifdef USE_NVTX
	nvtxRangePush(__FUNCTION__);
#endif

	int memDir;

	int i;

	int totalBlockAlloc = 0;
	int arraysOffset[nArrays], leftCirc[nArrays], rightCirc[nArrays], blockSize[nArrays];

	MGArray *phi = theta;
	// Run through the arrays and learn how much we need to allocate.
	for(i = 0; i < nArrays; i++) {
		/* Be told if the left and right sides of the dimension are circular or not */
		int leftCircular, rightCircular;
		switch(xchgDir) {
		case 0:
			leftCircular = (phi->mpiCircularBoundaryBits & MGA_BOUNDARY_XMINUS) ? 1 : 0;
			rightCircular = (phi->mpiCircularBoundaryBits & MGA_BOUNDARY_XPLUS) ? 1 : 0;
			break;
		case 1:
			leftCircular = (phi->mpiCircularBoundaryBits & MGA_BOUNDARY_YMINUS) ? 1 : 0;
			rightCircular = (phi->mpiCircularBoundaryBits & MGA_BOUNDARY_YPLUS) ? 1 : 0;
			break;

		case 2:
			leftCircular = (phi->mpiCircularBoundaryBits & MGA_BOUNDARY_YMINUS) ? 1 : 0;
			rightCircular = (phi->mpiCircularBoundaryBits & MGA_BOUNDARY_YPLUS) ? 1 : 0;
			break;
		default:
			PRINT_FAULT_HEADER;
			printf("Valid exchange directions are 1/2/3\nI was called with %i\n", xchgDir + 1);
			PRINT_FAULT_FOOTER;

			return ERROR_INVALID_ARGS;
		}

		memDir = phi->currentPermutation[xchgDir]; // The actual in-memory direction we're gonna be exchanging

		int haloDepth = phi->haloSize;

		// Find the size of the swap buffer needed for this array
		int numPerHalo = MGA_wholeFaceHaloNumel(phi, memDir, haloDepth);

		leftCirc[i] = leftCircular;
		rightCirc[i] = rightCircular;
		arraysOffset[i] = totalBlockAlloc; // the start index of the array's buffer is the current end
		totalBlockAlloc += 4*numPerHalo;   // grow buffer to be alloc'd
		blockSize[i] = numPerHalo;         // the size of each of the 4 blocks within an array buffer

		phi++;
	}


	// make sure we have enough space
	// NOTE: this can experience many allocations if totalBlockAlloc slowly increaes on subsequent calls
	// NOTE: but that will not happen irl (i.e. it does not double to get bounded # of reallocs)
	checkGlobalBufferLen(totalBlockAlloc);

	double *devbufptr;
	cudaHostGetDevicePointer ((void **)&devbufptr, (void *)hostPinnedBufferRoot, 0);
	returnCode = CHECK_CUDA_ERROR("cudaHostGetDevicePointer");
	if(returnCode != SUCCESSFUL) return returnCode;

	double *ptrHalo;

	phi = theta;
	// Read all halos into common block buffer
	for(i = 0; i < nArrays; i++) {
		int haloDepth = phi->haloSize;

		ptrHalo = devbufptr + arraysOffset[i];

		double *ptmp = ptrHalo;
		// Fetch left face
		if(leftCirc[i])
			returnCode = MGA_wholeFaceToLinear(phi, memDir, 0, 0, haloDepth, &ptmp);
		if(returnCode != SUCCESSFUL) break;

		ptmp = ptrHalo + blockSize[i];
		// Fetch right face
		if(rightCirc[i])
			returnCode = MGA_wholeFaceToLinear(phi, memDir, 1, 0, haloDepth, &ptmp);
		if(returnCode != SUCCESSFUL) break;
		phi++;
	}

	// make quite sure the face read/writes are complete so we can unregister
	int j;
	for(j = 0; j < theta->nGPUs; j++) {
		cudaSetDevice(theta->deviceID[j]);
		cudaDeviceSynchronize();
	}

	// This is necessary because otherwise a horrible silent failure occurs on RDMA-capable
	// MPI substrates - the mpi communication simply fails to transfer data without any
	// error of any kind
	//
	// God ****ing damn it took a long time to realize that that was what was going on

	// copy to the unpinned buffer
	memmove((void *)hostNormalBufferRoot, (void *)hostPinnedBufferRoot, totalBlockAlloc * sizeof(double));

	phi = theta;
	#ifdef USE_NVTX
	nvtxRangePush("mpi_exchangeHalos");
	#endif
	for(i = 0; i < nArrays; i++) {
		//int haloDepth = phi->haloSize;
		ptrHalo = hostNormalBufferRoot + arraysOffset[i];
		//double *ptmp = ptrHalo;

		mpi_exchangeHalos(topo, xchgDir, (void*)ptrHalo,
				(void*)(ptrHalo + blockSize[i]),\
				(void*)(ptrHalo+2*blockSize[i]),\
				(void*)(ptrHalo+3*blockSize[i]), blockSize[i], MPI_DOUBLE);
		MPI_Barrier(MPI_COMM_WORLD);

	}
	#ifdef USE_NVTX
	nvtxRangePop();
	#endif

	// fixme: these can move half as much in reality but it's shittily interleaved
	// fixme: either way it can't possible ever be as slow as the cuda malloc/free/register functions, OMG...
	memmove((void *)hostPinnedBufferRoot, (void *)hostNormalBufferRoot, totalBlockAlloc * sizeof(double));

	for(i = 0; i < nArrays; i++) {
		int haloDepth = phi->haloSize;
		ptrHalo = devbufptr + arraysOffset[i];
		double *ptmp = ptrHalo;
		// write left face
		ptmp = ptrHalo + 2*blockSize[i];
		if(leftCirc[i])
			returnCode = MGA_wholeFaceToLinear(phi, memDir, 0, 1, haloDepth, &ptmp);
		if(returnCode != SUCCESSFUL) break;

		ptmp = ptrHalo + 3*blockSize[i];
		// Fetch right face
		if(rightCirc[i])
			returnCode = MGA_wholeFaceToLinear(phi, memDir, 1, 1, haloDepth, &ptmp);
		if(returnCode != SUCCESSFUL) break;

		// Move to the next array to exchange
		phi++;
	}

	for(j = 0; j < theta->nGPUs; j++) {
                cudaSetDevice(theta->deviceID[j]);
                cudaDeviceSynchronize();
        }

	CHECK_IMOGEN_ERROR(returnCode);

#ifdef USE_NVTX
	nvtxRangePop();
#endif

	return returnCode;
}



