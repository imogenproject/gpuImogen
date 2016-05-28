#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif
// Matlab
#include "mex.h"

// CUDA
#include "cuda.h"

// Local
#include "cudaCommon.h"
#include "cudaArrayRotateB.h"

__global__ void cukern_ArrayTranspose2D(double *src, double *dst, int nx, int ny);
__global__ void cukern_ArrayExchangeXY(double *src,  double *dst, int nx, int ny, int nz);
__global__ void cukern_ArrayExchangeXZ(double *src,  double *dst, int nx, int ny, int nz);
__global__ void cukern_ArrayExchangeYZ(double *src,  double *dst, int nx, int ny, int nz);
__global__ void cukern_ArrayRotateRight(double *src, double *dst, int nx, int ny, int nz);
__global__ void cukern_ArrayRotateLeft(double *src,  double *dst, int nx, int ny, int nz);

#define BDIM 16

#ifdef STANDALONE_MEX_FUNCTION
/* Generates a directly Matlab-accessible entry point for the array index transposition routines.
 * [L0, L1, ..., Ln] = cudaArrayRotateB(R0, R1, ..., Rn, transpositionCode)
 * where {R0, R1, ..., Rn} are n GPU arrays,
 *       {L0, L1, ..., Ln} must be either not exist (nlhs=0) or be present in equal number
 * and the transpositionCode is one of 2 (switch X/Y), 3 (switch X/Z), 4 (switch Y/Z),
 * 5 (shift left XYZ->YZX) or 6 (shift right XYZ->ZXY).
 * If nz == 1: code 3 = 4 = identity (no change in memory layout)
 *             code 5 = 6 -> 2 (Permutation either direction reduces to XY transposition)
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int makeNew = 0;

	if(nrhs != 2) { mexErrMsgTxt("Input args must be cudaArrayRotateB(GPU array, dir to transpose with X)"); }
	switch(nlhs) {
	case 0: makeNew = 0; break;
	case 1: makeNew = 1; break;
	default: mexErrMsgTxt("cudaArrayRotate must return zero (in-place transpose) or one (new array) arguments."); break;
	}

	if(CHECK_CUDA_ERROR("entering cudaArrayRotateB") != SUCCESSFUL)
		{ DROP_MEX_ERROR("cudaArrayRotateB aborting due to errors at entry.\n"); }

	MGArray src;
	int returnCode = CHECK_IMOGEN_ERROR(MGA_accessMatlabArrays(prhs, 0, 0, &src));
	if(returnCode != SUCCESSFUL)
		{ DROP_MEX_ERROR("cudaArrayRotateB aborting: Unable to access array.\n"); }
	int indExchange = (int)*mxGetPr(prhs[1]);

	MGArray *novelty;
	returnCode = CHECK_IMOGEN_ERROR(flipArrayIndices(&src, (makeNew ? &novelty : NULL), 1, indExchange));
	if(returnCode != SUCCESSFUL)
		{ DROP_MEX_ERROR("cudaArrayRotateB aborting: Index transposition failed.\n"); }

	if(makeNew) { // return new tags
		MGA_returnOneArray(plhs, novelty);
	} else { // overwrite original tag
		serializeMGArrayToTag(&src, (int64_t *)mxGetData(prhs[0]));
	}

}
#endif

/* flipArrayIndices alters the memory layout of the input arrays according to the exchange code.
 * If new arrays are created, consumes additional memory equal to sum of input arrays. If changes
 * are in place, consumes temporary memory equal to largest single input array.
 *
 * phi: MGArray pointer to nArrays input MGArrays
 * psi: MGArray *.
 *   IF NOT NULL: allocates nArrays arrays & returns completely new MGArrays
 *   IF NULL:     overwrites phi[*], reusing original data *ers
 * nArrays: Positive integer
 * exchangeCode:
 * 	- 2: Exchange X and Y
 * 	- 3: Exchange X and Z
 * 	- 4: Exchange Y and Z
 * 	- 5: Permute indices left (XYZ -> YZX)
 * 	- 6: Permute indices right(XYZ -> ZXY)
 * 	- other than (2, 3, 4, 5, 6): Return ERROR_INVALID_ARGS
 * 	- If nz=1, codes 3/4 are identity and 5/6 reduce to transposition.
 */
int flipArrayIndices(MGArray *phi, MGArray **newArrays, int nArrays, int exchangeCode)
{
	int returnCode = SUCCESSFUL;

	MGArray trans;

	int i, sub[6];
	int is3d;

	MGArray *psi = NULL;
	if(newArrays != NULL) {
		newArrays[0] = (MGArray *)malloc(nArrays * sizeof(MGArray));
		psi = newArrays[0];
	}

	int j;
	for(j = 0; j < nArrays; j++) {
		trans = *phi;

		is3d = (phi->dim[2] > 3);
		// These choices assure that the 3D transformation semantics reduce properly for 2D
		if(is3d == 0) {
			if((exchangeCode == 3) || (exchangeCode == 4)) {
				// Identity operation:
				if(newArrays != NULL) {
					// If returning new, build deep copy
					MGArray *tp = NULL;
					returnCode = MGA_duplicateArray(&tp, phi);
					psi[0] = *tp;
					free(tp);

					if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) break;
				}   // Otherwise do nothing
				phi++;
				continue;
			}

			if(exchangeCode == 5) exchangeCode = 2; // Index rotate becomes XY transpose
			if(exchangeCode == 6) exchangeCode = 2;
		}

		switch(exchangeCode) {
		case 2: /* Transform XYZ -> YXZ */
			trans.dim[0] = phi->dim[1];
			trans.dim[1] = phi->dim[0];

			trans.currentPermutation[0] = phi->currentPermutation[1];
			trans.currentPermutation[1] = phi->currentPermutation[0];

			if(phi->partitionDir == PARTITION_X) { trans.partitionDir = PARTITION_Y; }
			if(phi->partitionDir == PARTITION_Y) { trans.partitionDir = PARTITION_X; }
  			break;
		case 3: /* Transform XYZ to ZYX */
			trans.dim[0] = phi->dim[2];
			trans.dim[2] = phi->dim[0];

			trans.currentPermutation[0] = phi->currentPermutation[2];
			trans.currentPermutation[2] = phi->currentPermutation[0];

			if(phi->partitionDir == PARTITION_X) { trans.partitionDir = PARTITION_Z; }
			if(phi->partitionDir == PARTITION_Z) { trans.partitionDir = PARTITION_X; }
			break;
		case 4: /* Transform XYZ to XZY */
			trans.dim[1] = phi->dim[2];
			trans.dim[2] = phi->dim[1];

			trans.currentPermutation[1] = phi->currentPermutation[2];
			trans.currentPermutation[2] = phi->currentPermutation[1];

			if(phi->partitionDir == PARTITION_Y) { trans.partitionDir = PARTITION_Z; }
			if(phi->partitionDir == PARTITION_Z) { trans.partitionDir = PARTITION_Y; }
			break;
		case 5: /* Rotate left, XYZ to YZX */
			for(i = 0; i < 3; i++) trans.dim[i] = phi->dim[(i+1)%3];

			for(i = 0; i < 3; i++) trans.currentPermutation[i] = phi->currentPermutation[(i+1)%3];

			if(phi->partitionDir == PARTITION_X) { trans.partitionDir = PARTITION_Z; }
			if(phi->partitionDir == PARTITION_Y) { trans.partitionDir = PARTITION_X; }
			if(phi->partitionDir == PARTITION_Z) { trans.partitionDir = PARTITION_Y; }
			break;
		case 6: /* Rotate right, XYZ to ZXY */
			for(i = 0; i < 3; i++) trans.dim[i] = phi->dim[(i+2)%3];

			for(i = 0; i < 3; i++) trans.currentPermutation[i] = phi->currentPermutation[(i+2)%3];

			if(phi->partitionDir == PARTITION_X) { trans.partitionDir = PARTITION_Y; }
			if(phi->partitionDir == PARTITION_Y) { trans.partitionDir = PARTITION_Z; }
			if(phi->partitionDir == PARTITION_Z) { trans.partitionDir = PARTITION_X; }
			break;
		default:
			PRINT_FAULT_HEADER;
			printf("Index to exchange is invalid: %i is not in 2-6 inclusive.\n", exchangeCode);
			PRINT_FAULT_FOOTER;
			fflush(stdout);
			return ERROR_INVALID_ARGS;
		}

		trans.permtag = MGA_numsToPermtag(&trans.currentPermutation[0]);

		// Recalculate the partition sizes
		// FIXME: This... remains the same, n'est ce pas?
		for(i = 0; i < trans.nGPUs; i++) {
			calcPartitionExtent(&trans, i, sub);
			trans.partNumel[i] = sub[3]*sub[4]*sub[5];
		}

		MGArray *nuClone;
		if((exchangeCode != -1) || (psi != NULL)) {
			nuClone = MGA_allocArrays(1, &trans);
		}

		dim3 blocksize, gridsize;

		for(i = 0; i < trans.nGPUs; i++) {
			cudaSetDevice(trans.deviceID[i]);
			returnCode = CHECK_IMOGEN_ERROR(CHECK_CUDA_ERROR("cudaSetDevice()"));
			if(returnCode != SUCCESSFUL) break;

			calcPartitionExtent(phi, i, sub);

			switch(exchangeCode) {
			case 2: // Flip XY
				blocksize.x = blocksize.y = BDIM; blocksize.z = 1;
				gridsize.x = ROUNDUPTO(phi->dim[0], BDIM) / BDIM;
				gridsize.y = ROUNDUPTO(phi->dim[1], BDIM) / BDIM;
				gridsize.z = 1;

				if(is3d) {
					cukern_ArrayExchangeXY<<<gridsize, blocksize>>>(phi->devicePtr[i], nuClone->devicePtr[i], sub[3],sub[4],sub[5]);
				} else {
					cukern_ArrayTranspose2D<<<gridsize, blocksize>>>(phi->devicePtr[i], nuClone->devicePtr[i], sub[3], sub[4]);
				}
				break;
			case 3: // Flip XZ
				blocksize.x = blocksize.y = BDIM; blocksize.z = 1;
				gridsize.x = ROUNDUPTO(phi->dim[0], BDIM) / BDIM;
				gridsize.y = ROUNDUPTO(phi->dim[2], BDIM) / BDIM;
				gridsize.z = 1;

				cukern_ArrayExchangeXZ<<<gridsize, blocksize>>>(phi->devicePtr[i], nuClone->devicePtr[i], sub[3], sub[4], sub[5]);
				break;

			case 4: // Flip YZ
				blocksize.x = 32;
				blocksize.y = blocksize.z = 4;

				gridsize.x = ROUNDUPTO(sub[3], blocksize.x) / blocksize.x;
				gridsize.y = ROUNDUPTO(sub[4], blocksize.y) / blocksize.y;
				gridsize.z = 1;

				cukern_ArrayExchangeYZ<<<gridsize, blocksize>>>(phi->devicePtr[i], nuClone->devicePtr[i], sub[3],sub[4],sub[5]);

				break;
			case 5: // Rotate left
				blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

				gridsize.x = ROUNDUPTO(phi->dim[0], BDIM)/BDIM;
				gridsize.y = ROUNDUPTO(phi->dim[1], BDIM)/BDIM;
				gridsize.z = 1;

				cukern_ArrayRotateLeft<<<gridsize, blocksize>>>(phi->devicePtr[i], nuClone->devicePtr[i], sub[3], sub[4], sub[5]);

				break;
			case 6: // Rotate right
				blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

				gridsize.x = ROUNDUPTO(phi->dim[0], BDIM)/BDIM;
				gridsize.y = ROUNDUPTO(phi->dim[2], BDIM)/BDIM;
				gridsize.z = 1;

				cukern_ArrayRotateRight<<<gridsize, blocksize>>>(phi->devicePtr[i], nuClone->devicePtr[i], sub[3], sub[4], sub[5]);

				break;
			}
			returnCode = CHECK_IMOGEN_ERROR(CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, phi, i, "array transposition call"));
			if(returnCode != SUCCESSFUL) break;
		}

		if(returnCode != SUCCESSFUL) break;

		if(psi == NULL) { // Overwrite original data:
			MGArray tmp = phi[0];
			phi[0] = *nuClone; // Nuke original MGArray

			phi->numSlabs = tmp.numSlabs; /* Retain original is/isnot allocated status */

			for(i = 0; i < trans.nGPUs; i++) {
				phi->devicePtr[i] = tmp.devicePtr[i]; // But keep same pointers
				cudaSetDevice(tmp.deviceID[i]); // And overwrite original data
				cudaMemcpyAsync(phi->devicePtr[i], nuClone->devicePtr[i], phi->partNumel[i]*sizeof(double), cudaMemcpyDeviceToDevice);
				returnCode = CHECK_IMOGEN_ERROR(CHECK_CUDA_ERROR("cudaMemcpyAsync()"));
				if(returnCode != SUCCESSFUL) break;
			}
			if(returnCode == SUCCESSFUL) returnCode = MGA_delete(nuClone); // Before deleting new pointer
			free(nuClone);
		} else { // Otherwise, simply write the new MGArray to the output pointer.
			psi[0] = *nuClone;
		}

		phi++;
		if(psi != NULL) psi++;
	}

	if(returnCode == SUCCESSFUL) returnCode = CHECK_IMOGEN_ERROR(CHECK_CUDA_ERROR("Departing cudaArrayRotateB"));
	return returnCode;
}

__global__ void cukern_ArrayTranspose2D(double *src, double *dst, int nx, int ny)
{
	__shared__ double tmp[BDIM][BDIM+1];

	int myx = threadIdx.x + BDIM*blockIdx.x;
	int myy = threadIdx.y + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
	int myAddr = myx + nx*myy;

	if((myx < nx) && (myy < ny)) tmp[threadIdx.y][threadIdx.x] = src[myAddr];

	__syncthreads();

	//myx = threadIdx.x + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
	myAddr = myy + threadIdx.x - threadIdx.y;
	//myy = threadIdx.y + BDIM*blockIdx.x;
	myy  = myx + threadIdx.y - threadIdx.x;
	myx = myAddr;

	myAddr = myx + ny*myy;

	if((myx < ny) && (myy < nx)) dst[myAddr] = tmp[threadIdx.x][threadIdx.y];

}

__global__ void cukern_ArrayExchangeXY(double *src, double *dst, int nx, int ny, int nz)
{

	__shared__ double tmp[BDIM][BDIM+1];

	int myx = threadIdx.x + BDIM*blockIdx.x;
	int myy = threadIdx.y + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
	int mySrcAddr = myx + nx*myy;
	bool doRead = 0;
	bool doWrite = 0;

	if((myx < nx) && (myy < ny)) doRead = 1;

	myx = threadIdx.x + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
	myy = threadIdx.y + BDIM*blockIdx.x;
	int myDstAddr = myx + ny*myy;

	if((myx < ny) && (myy < nx)) doWrite = 1;

	for(myx = 0; myx < nz; myx++) {
		if(doRead) tmp[threadIdx.y][threadIdx.x] = src[mySrcAddr];
		mySrcAddr += nx*ny;
		__syncthreads();

		if(doWrite) dst[myDstAddr] = tmp[threadIdx.x][threadIdx.y];
		myDstAddr += nx*ny;
		__syncthreads();
	}

}

__global__ void cukern_ArrayExchangeXZ(double*src, double *dst, int nx, int ny, int nz)
{
	__shared__ double tmp[BDIM][BDIM+1];

	int myx = threadIdx.x + BDIM*blockIdx.x;
	int myz = threadIdx.y + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
	int mySrcAddr = myx + nx*ny*myz;
	bool doRead = 0;
	bool doWrite = 0;

	if((myx < nx) && (myz < nz)) doRead = 1;

	myx = threadIdx.x + BDIM*((blockIdx.y + blockIdx.x) % gridDim.y);
	myz = threadIdx.y + BDIM*blockIdx.x;
	int myDstAddr = myx + nz*ny*myz;

	if((myx < nz) && (myz < nx)) doWrite = 1;

	for(myx = 0; myx < ny; myx++) {
		if(doRead) tmp[threadIdx.y][threadIdx.x] = src[mySrcAddr];
		mySrcAddr += nx;
		__syncthreads();

		if(doWrite) dst[myDstAddr] = tmp[threadIdx.x][threadIdx.y];
		myDstAddr += nz;
		__syncthreads();
	}


}

/* Assume we have threads that span X and input Y; step in input Z */
__global__ void cukern_ArrayExchangeYZ(double *src, double *dst,   int nx, int ny, int nz)
{
	int myx = threadIdx.x + blockDim.x*blockIdx.x;
	if(myx >= nx) return;

	int inputY = threadIdx.y + blockDim.y*blockIdx.y;
	if(inputY >= ny) return;

	int inputZ;
	int outputY, outputZ;

	/* Because the natively aligned index doesn't change, no need to stream through shmem to maintain global r/w coalescence */
	for(inputZ = threadIdx.z; inputZ < nz; inputZ += blockDim.z) {
		outputY = inputZ;
		outputZ = inputY;

		dst[myx + outputY*nx + outputZ*nx*nz] =
				src[myx + inputY*nx + inputZ *nx*ny];

	}

}


/* This kernel takes an input array src[i + nx(j+ny k)] and writes dst[k + nz(i + nx j)]
 * such that for input indices ordered [123] output indices are ordered [312]
 *
 * Strategy: blocks span I-K space and walk in J.
 */
__global__ void cukern_ArrayRotateRight(double *src, double *dst,  int nx, int ny, int nz)
{
	int i, j, k;
	i = threadIdx.x + blockDim.x*blockIdx.x;
	k = threadIdx.y + blockDim.y*blockIdx.y;
	bool doread = (i < nx) && (k < nz);
	src += (i + nx*ny*k);

	i = threadIdx.y + blockDim.x*blockIdx.x;
	k = threadIdx.x + blockDim.y*blockIdx.y;
	bool dowrite = (i < nx) && (k < nz);
	dst += (k + nz*i);

	__shared__ double tile[BDIM][BDIM+1];

	for(j = 0; j < ny; j++) {
		if(doread)
			tile[threadIdx.y][threadIdx.x] = src[nx*j];
		__syncthreads();
		if(dowrite)
			dst[nz*nx*j] = tile[threadIdx.x][threadIdx.y];
		__syncthreads();
	}
	
}

/* This kernel takes input array src[i + nx(j + ny k)] and writes dst[j + ny(k + nz i)]
 * such that input indices ordered [123] give output indices ordered [231]
 *
 * Strategy: Blocks span I-J space and walk K
 */
__global__ void cukern_ArrayRotateLeft(double *src, double *dst,  int nx, int ny, int nz)
{

	int i,j,k;
	i = threadIdx.x + blockDim.x*blockIdx.x;
	j = threadIdx.y + blockDim.y*blockIdx.y;
	bool doread = (i < nx) && (j < ny);
	src += (i + nx*j);

	i = threadIdx.y + blockDim.x*blockIdx.x;
	j = threadIdx.x + blockDim.y*blockIdx.y;
	bool dowrite = (i < nx) && (j < ny);
	dst += (j + ny*nz*i);

	__shared__ double tile[BDIM][BDIM+1];

	for(k = 0; k < nz; k++) {
		if(doread)
			tile[threadIdx.x][threadIdx.y] = src[nx*ny*k];
		__syncthreads();
		if(dowrite)
			dst[ny*k] = tile[threadIdx.y][threadIdx.x];
		__syncthreads();
	}
}
