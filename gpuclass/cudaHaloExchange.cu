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

	int swappingPartitionDirection = (mgadir == phi.partitionDir);

	// Find the size of the swap buffer
	int numPerHalo = 0;
	if(swappingPartitionDirection) { // It's only one face, any will do so use the 1st
		numPerHalo = MGA_partitionHaloNumel(&phi, 0, mgadir, 3);
	} else { // Operation is transverse to partitioning, we need to swap all partitions!
		for(ctr = 0; ctr < phi.nGPUs; ctr++) {
			numPerHalo += MGA_partitionHaloNumel(&phi, ctr, mgadir, 3);
		}
	}

	fail = cudaMallocHost((void **)&ptrHalo, 4*numPerHalo*sizeof(double));
	CHECK_CUDA_ERROR("cudaHostAlloc");

	MPI_Comm commune = MPI_Comm_f2c(parallelTopo->comm);

	double *ptmp;
	// Fetch left face
	if(leftCircular) {
			if(swappingPartitionDirection) {
				ptmp = ptrHalo;
				MGA_partitionHaloToLinear(&phi, 0, mgadir, 0, 0, 3, &ptmp);
			} else { // Fetch all halo partitions
				int q = 0;
				for(ctr = 0; ctr < phi.nGPUs; ctr++) {
					ptmp = ptrHalo + q;
					MGA_partitionHaloToLinear(&phi, ctr, mgadir, 0, 0, 3, &ptmp);
					q += MGA_partitionHaloNumel(&phi, ctr, mgadir, 3);
				}
			}
		}
	// Fetch right face
	if(rightCircular) {
			if(swappingPartitionDirection) {
				ptmp = ptrHalo + numPerHalo;
				MGA_partitionHaloToLinear(&phi, phi.nGPUs-1, mgadir, 1, 0, 3, &ptmp);
			} else { // Fetch all halo partitions
				int q = 0;
				for(ctr = 0; ctr < phi.nGPUs; ctr++) {
					ptmp = ptrHalo + numPerHalo + q;
					MGA_partitionHaloToLinear(&phi, phi.nGPUs-1, mgadir, 1, 0, 3, &ptmp);
					q += MGA_partitionHaloNumel(&phi, ctr, xchg, 3);
				}
			}
		}

	//if(leftCircular)  haloTransfer(&phi, ptrHalo, orient[xchg], 1, 1);
	//if(rightCircular) haloTransfer(&phi, ptrHalo + numPerHalo, orient[xchg], 0, 1);
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
	if(leftCircular) {
			if(swappingPartitionDirection) {
				ptmp = ptrHalo + 2*numPerHalo;
				MGA_partitionHaloToLinear(&phi, 0, mgadir, 0, 1, 3, &ptmp);
			} else { // Fetch all halo partitions
				int q = 0;
				for(ctr = 0; ctr < phi.nGPUs; ctr++) {
					ptmp = ptrHalo + 2*numPerHalo + q;
					MGA_partitionHaloToLinear(&phi, ctr, mgadir, 0, 1, 3, &ptmp);
					q += MGA_partitionHaloNumel(&phi, ctr, mgadir, 3);
				}
			}
		}
	// Fetch right face
	if(rightCircular) {
			if(swappingPartitionDirection) {
				ptmp = ptrHalo + 3*numPerHalo;
				MGA_partitionHaloToLinear(&phi, phi.nGPUs-1, mgadir, 1, 1, 3, &ptmp);
			} else { // Fetch all halo partitions
				int q = 0;
				for(ctr = 0; ctr < phi.nGPUs; ctr++) {
					ptmp = ptrHalo + 3*numPerHalo + q;
					MGA_partitionHaloToLinear(&phi, phi.nGPUs-1, mgadir, 1, 1, 3, &ptmp);
					q += MGA_partitionHaloNumel(&phi, ctr, xchg, 3);
				}
			}
		}

	//if(leftCircular)  haloTransfer(&phi, ptrHalo + 2*numPerHalo, orient[xchg], 1, 0);
	//if(rightCircular) haloTransfer(&phi, ptrHalo + 3*numPerHalo, orient[xchg], 0, 0);
    // don't cuda synchronize here since future kernel calls are synchronous w.r.t. other kernel calls & memcopies

	free(parallelTopo);
	cudaFreeHost((void **)ptrHalo);
}



void haloTransfer(MGArray *phi, double *hostmem, int haloDir, int left, int toCPU)
{
	int part[6];
	dim3 gpudim, gpuoffset, halodim, halooffset, copysize;

	double *gpuPointer;
	// haloDirection is given (1=x, 2=y, 3=z)
	// phi->partitionDir gives partition direction (1=x, 2=y, 3=z)
	// The directions which are not among the two above are spanning
	if(haloDir == phi->partitionDir) {
		if(left) {
			// Get the least-positive partition & access it
			cudaSetDevice(phi->deviceID[0]);
			gpuPointer = phi->devicePtr[0];
			calcPartitionExtent(phi, 0, part);
		} else {
			// Get the most-positive partition & access it
			cudaSetDevice(phi->deviceID[phi->nGPUs-1]);
			gpuPointer = phi->devicePtr[phi->nGPUs-1];
			calcPartitionExtent(phi, phi->nGPUs-1, part);
		}
		gpudim.x = part[3]; gpudim.y = part[4]; gpudim.z = part[5];
		copysize = gpudim;
		gpuoffset.x = 0; gpuoffset.y = 0; gpuoffset.z = 0;
		// At this point it would attempt to copy the entire array

		// By default believe that the halo array matches the full array
		halooffset = gpuoffset;
		halodim = gpudim;

		switch(haloDir + 3*left + 6*toCPU) {
		case 1: gpuoffset.x = gpudim.x - 3; halodim.x = copysize.x = 3; break; // x, right, upload to gpu
		case 2: gpuoffset.y = gpudim.y - 3; halodim.y = copysize.y = 3; break; // y, right, upload to gpu
		case 3: gpuoffset.z = gpudim.z - 3; halodim.z = copysize.z = 3; break; // z, right, upload to gpu
		case 4: gpuoffset.x = 0;            halodim.x = copysize.x = 3; break; // x, left, upload to gpu
		case 5: gpuoffset.y = 0;            halodim.y = copysize.y = 3; break; // y, left, upload to gpu
		case 6: gpuoffset.z = 0;            halodim.z = copysize.z = 3; break; // z, left, upload to gpu

		case 7: gpuoffset.x = gpudim.x - 6; halodim.x = copysize.x = 3; break; // x, right, read to host
		case 8: gpuoffset.y = gpudim.y - 6; halodim.y = copysize.y = 3; break; // y, right, read to host
		case 9: gpuoffset.z = gpudim.z - 6; halodim.z = copysize.z = 3; break; // z, right, read to host
		case 10: gpuoffset.x = 3;           halodim.x = copysize.x = 3; break; // x, left, read to host
		case 11: gpuoffset.y = 3;           halodim.y = copysize.y = 3; break; // y, left, read to host
		case 12: gpuoffset.z = 3;           halodim.z = copysize.z = 3; break; // z, left, read to host
		}

		if(toCPU) {
			memBlockCopy(gpuPointer, gpudim, gpuoffset, hostmem, halodim, halooffset, copysize);
		} else {
			memBlockCopy(hostmem, halodim, halooffset, gpuPointer, gpudim, gpuoffset, copysize);
		}

	} else {
		int i;
		for(i = 0; i < phi->nGPUs; i++) {
			cudaSetDevice(phi->deviceID[i]);
			gpuPointer = phi->devicePtr[i];
			calcPartitionExtent(phi, 0, part);

			// set the GPU dimension; This is always correct
			gpudim.x = part[3]; gpudim.y = part[4]; gpudim.z = part[5];

			// Default to a dimension-spanning copy in all dimensions
			copysize = gpudim;

			// With zero gpu or halo offset
			gpuoffset.x = 0; gpuoffset.y = 0; gpuoffset.z = 0;
			halooffset = gpuoffset; // zero!

			// Host array size is the unpartitioned size of the array
			halodim.x = phi->dim[0]; halodim.y = phi->dim[1]; halodim.z = phi->dim[2];

			// Calculate how much to trim due to MGArray's partitioning halo
			int trimleft = 0, trimright = 0;
			if(i > 0) trimleft = phi->haloSize;
			if(i < phi->nGPUs-1) trimright = phi->haloSize;

			switch(haloDir + 3*left + 6*toCPU) {
			case 1: gpuoffset.x = gpudim.x - 3; halodim.x = copysize.x = 3; break; // x, right, upload to gpu
			case 2: gpuoffset.y = gpudim.y - 3; halodim.y = copysize.y = 3; break; // y, right, upload to gpu
			case 3: gpuoffset.z = gpudim.z - 3; halodim.z = copysize.z = 3; break; // z, right, upload to gpu
			case 4: gpuoffset.x = 0;            halodim.x = copysize.x = 3; break; // x, left, upload to gpu
			case 5: gpuoffset.y = 0;            halodim.y = copysize.y = 3; break; // y, left, upload to gpu
			case 6: gpuoffset.z = 0;            halodim.z = copysize.z = 3; break; // z, left, upload to gpu

			case 7: gpuoffset.x = gpudim.x - 6; halodim.x = copysize.x = 3; break; // x, right, read to host
			case 8: gpuoffset.y = gpudim.y - 6; halodim.y = copysize.y = 3; break; // y, right, read to host
			case 9: gpuoffset.z = gpudim.z - 6; halodim.z = copysize.z = 3; break; // z, right, read to host
			case 10: gpuoffset.x = 3;           halodim.x = copysize.x = 3; break; // x, left, read to host
			case 11: gpuoffset.y = 3;           halodim.y = copysize.y = 3; break; // y, left, read to host
			case 12: gpuoffset.z = 3;           halodim.z = copysize.z = 3; break; // z, left, read to host
			}

			// reset the offsets and dimensions in the nonspanning direction
			switch(phi->partitionDir) {
			case PARTITION_X: gpuoffset.x = trimleft; halooffset.x = part[0]+trimleft; copysize.x -= (trimleft+trimright); break;
			case PARTITION_Y: gpuoffset.y = trimleft; halooffset.y = part[1]+trimleft; copysize.y -= (trimleft+trimright); break;
			case PARTITION_Z: gpuoffset.z = trimleft; halooffset.z = part[2]+trimleft; copysize.z -= (trimleft+trimright); break;
			}

			if(toCPU) {
				memBlockCopy(gpuPointer, gpudim, gpuoffset, hostmem, halodim, halooffset, copysize);
			} else {
				memBlockCopy(hostmem, halodim, halooffset, gpuPointer, gpudim, gpuoffset, copysize);
			}
		}
	}

}

// Copy from (S)ource, which is a 3d array sDim in size,
// from addresses of block (sOffset) to (sOffset+num2cpy - <1,1,1>) inclusive
// to (D)estination, of size dDim, at locations (dOffset) to (dOffset+num2cpy-<1,1,1>) inclusive
void memBlockCopy(double *S, dim3 sDim, dim3 sOffset, double *D, dim3 dDim, dim3 dOffset, dim3 num2cpy)
{
	// Move the pointer to the offset 3d address
	S = S + sOffset.x + sDim.x*(sOffset.y + sDim.y*sOffset.z);
	D = D + dOffset.x + dDim.x*(dOffset.y + dDim.y*dOffset.z);
	dim3 blocksize, gridsize;

	blocksize.x = (num2cpy.x > 8) ? 8 : num2cpy.x;
	blocksize.y = (blocksize.x < 8) ? 16 : 8;
	blocksize.z = (num2cpy.z < 8) ? 1 : 4;

	gridsize.x = num2cpy.x / blocksize.x; gridsize.x += (gridsize.x*blocksize.x < num2cpy.x);
	gridsize.y = num2cpy.y / blocksize.y; gridsize.y += (gridsize.y*blocksize.y < num2cpy.y);
	gridsize.z = 1;

	cukern_MemBlockCopy<<<gridsize, blocksize>>>(S, sDim, D, dDim, num2cpy);
	CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, NULL, -1, "in mem block copy");
}

// The invocation translates both S and D from their allocated values to the lowest-index point of the copy
// (geometrically, "the lower left corner") so we need only the dimensions & extent of copy
__global__ void cukern_MemBlockCopy(double *S, dim3 sDim, double *D, dim3 dDim, dim3 num2cpy)
{
	unsigned int x, y, z;

	x = threadIdx.x + blockIdx.x*blockDim.x;
	y = threadIdx.y + blockIdx.y*blockDim.y;
	z = threadIdx.z;

	if(x >= num2cpy.x) return;
	if(y >= num2cpy.y) return;

	// Translate to our first point
	S += x + sDim.x*(y + sDim.y*z);
	D += x + dDim.x*(y + dDim.y*z);

	for( ; z < num2cpy.z; z+= blockDim.z)
		D[dDim.x*dDim.y*z] = S[sDim.x*sDim.y*z];

}
