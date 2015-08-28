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
#include "cudaCommon.h"
#include "parallel_halo_arrays.h"

/* THIS ROUTINE
   This routine interfaces with the parallel gateway halo routines
   */
void haloTransfer(MGArray *phi, double *hostmem, int haloDir, int oneIffLeft, int oneIffToHost);
void memBlockCopy(double *S, dim3 sDim, dim3 sOffset, double *D, dim3 dDim, dim3 dOffset, dim3 num2cpy);
__global__ void cukern_MemBlockCopy(double *S, dim3 sDim, double *D, dim3 dDim, dim3 num2cpy);

/* X halo routines */
/* These are the suck; We have to grab 24-byte wide chunks en masse */
/* Fork ny by nz threads to do the job */
__global__ void cukern_HaloXToLinearL(double *mainarray, double *linarray, int nx);
__global__ void cukern_LinearToHaloXL(double *mainarray, double *linarray, int nx);
__global__ void cukern_HaloXToLinearR(double *mainarray, double *linarray, int nx);
__global__ void cukern_LinearToHaloXR(double *mainarray, double *linarray, int nx);

/* Y halo routines */
/* We grab an X-Z plane, making it easy to copy N linear strips of memory */
/* Fork off nz by 3 blocks to do the job */
__global__ void cukern_HaloYToLinearL(double *mainarray, double *linarray, int nx, int ny);
__global__ void cukern_LinearToHaloYL(double *mainarray, double *linarray, int nx, int ny);

__global__ void cukern_HaloYToLinearR(double *mainarray, double *linarray, int nx, int ny);
__global__ void cukern_LinearToHaloYR(double *mainarray, double *linarray, int nx, int ny);

/* Z halo routines */
/* The easiest; We make one copy of an Nx by Ny by 3 slab of memory */
/* No kernels necessary, we can simply memcpy our hearts out */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Functional form:
    cudaHaloExchange(arraytag, [orientation 3x1], dimension_to_exchange, parallel topology information)

    1. get neighbors from halo library in dimension_to_exchange direction
    2. determine which memory direction that currently is
    3. If it's x or y, rip it to a linear array
    4. Aquire some host-pinned memory and dump to that
    5. pass that host pointer to halo_exchange
    6. wait for MPI to return control
	 */
	if (nrhs!=5) mexErrMsgTxt("call form is cudaHaloExchange(arraytag, [3x1 orientation], dimension_to_xchg, topology, circularity\n");
	if(mxGetNumberOfElements(prhs[1]) != 3) mexErrMsgTxt("2nd argument must be a 3-element array\n");

	CHECK_CUDA_ERROR("entering cudaHaloExchange");
	int xchg = (int)*mxGetPr(prhs[2]) - 1;
	int mgadir = xchg+1;
	int orient[3];

	pParallelTopology parallelTopo = topoStructureToC(prhs[3]);

	if(parallelTopo->nproc[xchg] == 1) return;
	// Do not waste time if we can't possibly have any work to do

	MGArray phi;
	int worked = MGA_accessMatlabArrays(prhs, 0, 0, &phi);

	int ctr;	for(ctr = 0; ctr < 3; ctr++) { orient[ctr] = (int)*(mxGetPr(prhs[1]) + ctr); }

	int memDimension = orient[xchg]-1; // The actual in-memory direction we're gonna be exchanging

	CHECK_CUDA_ERROR("Entering cudaHaloExchange");
	cudaError_t fail;

	if(xchg+1 > parallelTopo->ndim) return; // The topology does not extend in this dimension
	if(parallelTopo->nproc[xchg] == 1) return; // Only 1 block in this direction.

	double *ptrHalo;

	/* Be told if the left and right sides of the dimension are circular or not */
	double *interior = mxGetPr(prhs[4]);
	int leftCircular  = (int)interior[2*memDimension];
	int rightCircular = (int)interior[2*memDimension+1];

	// Find the size of the swap buffer
	int numPerHalo = MGA_wholeFaceHaloNumel(&phi, mgadir, 3);

	fail = cudaMallocHost((void **)&ptrHalo, 4*numPerHalo*sizeof(double));
	CHECK_CUDA_ERROR("cudaHostAlloc");

	MPI_Comm commune = MPI_Comm_f2c(parallelTopo->comm);

	double *ptmp = ptrHalo;
	// Fetch left face
	if(leftCircular)
		MGA_wholeFaceToLinear(&phi, mgadir, 0, 0, 3, &ptmp);

	ptmp = ptrHalo + numPerHalo;
	// Fetch right face
	if(rightCircular)
		MGA_wholeFaceToLinear(&phi, mgadir, 1, 0, 3, &ptmp);

	// synchronize to make sure host sees what was uploaded
	int i;
	for(i = 0; i < phi.nGPUs; i++) {
		cudaSetDevice(phi.deviceID[i]);
		cudaDeviceSynchronize();
	}
	parallel_exchange_dim_contig(parallelTopo, 0, (void*)ptrHalo,
			(void*)(ptrHalo + numPerHalo),\
			(void*)(ptrHalo+2*numPerHalo),\
			(void*)(ptrHalo+3*numPerHalo), numPerHalo, MPI_DOUBLE);
	MPI_Barrier(MPI_COMM_WORLD);

	// write left face
	ptmp = ptrHalo + 2*numPerHalo;
	if(leftCircular)
		MGA_wholeFaceToLinear(&phi, mgadir, 0, 1, 3, &ptmp);

	ptmp = ptrHalo + 3*numPerHalo;
	// Fetch right face
	if(rightCircular)
		MGA_wholeFaceToLinear(&phi, mgadir, 1, 1, 3, &ptmp);

	free(parallelTopo);
	cudaFreeHost((void **)ptrHalo);
}



