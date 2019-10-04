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
#include "nvToolsExt.h"

// Local
#include "cudaCommon.h"
#include "cudaArrayRotateB.h"

__global__ void cukern_ArrayTranspose2D(double *src, double *dst, int nx, int ny);
__global__ void cukern_ArrayExchangeXY(double *src,  double *dst, int nx, int ny, int nz);
__global__ void cukern_ArrayExchangeXZ(double *src,  double *dst, int nx, int ny, int nz);
__global__ void cukern_ArrayExchangeYZ(double *src,  double *dst, int nx, int ny, int nz);
__global__ void cukern_ArrayRotateRight(double *src, double *dst, int nx, int ny, int nz);
__global__ void cukern_ArrayRotateLeft(double *src,  double *dst, int nx, int ny, int nz);

int flipArrayIndices_multisame(MGArray *phi, int nArrays, int exchangeCode, cudaStream_t *streamPtrs, MGArray *tempStorage);
int actuallyNeedToReorder(int *dims, int code);
int alterArrayMetadata(MGArray *src, MGArray *dst, int exchangeCode);

int checkSufficientFlipStorage(MGArray *phi, MGArray *tmp);

__global__ void cukern_memmove(double *s, double *d, long n);

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

	MGArray nouveau;
	MGArray *novelty = &nouveau;
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
 * newArrays: MGArray **.
 *   IF NOT NULL: returns completely new MGArrays
 *   	If newArrays[0] is not NULL, assumes it points to at least nArrays MGArrays.
 *   	if newarrays[0] is NULL, malloc()s nArrays MGArrays
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
int flipArrayIndices(MGArray *phi, MGArray **newArrays, int nArrays, int exchangeCode, cudaStream_t *streamPtrs, MGArray *tempStorage)
{

	int returnCode = SUCCESSFUL;

	MGArray trans;

	int i, sub[6];
	int is3d, isRZ;

	int j;
	// Determine if we can avoid making new arrays for every array we're doing this to
	// malloc/free are slow in cuda too
	int reallocatePerArray = 0;
	for(j = 1; j < nArrays; j++) {
		if(MGA_arraysAreIdenticallyShaped(phi, phi+j) == 0) {
			reallocatePerArray = 1;
		}
	}

	if((reallocatePerArray == 0) && (newArrays == NULL) && (streamPtrs != NULL)) {
		return flipArrayIndices_multisame(phi, nArrays, exchangeCode, streamPtrs, tempStorage);
	}

#ifdef USE_NVTX
	nvtxRangePush(__FUNCTION__);
#endif

	MGArray *psi = NULL;
	if(newArrays != NULL) {
		if(newArrays[0] == NULL) {
			newArrays[0] = (MGArray *)malloc(nArrays * sizeof(MGArray));
		}
		psi = newArrays[0];
		reallocatePerArray = 1; // definitely should do this if they're being returned!
	}

	MGArray *nuClone;
	for(j = 0; j < nArrays; j++) {

		if(actuallyNeedToReorder(&phi->dim[0], exchangeCode) == 0) {

			// Identity operation:
			if(newArrays != NULL) {
				// If returning new, build deep copy
				MGArray *tp = NULL;
				returnCode = MGA_duplicateArray(&tp, phi);
				psi[0] = *tp;
				free(tp);
				// Then rewrite its metadata
				alterArrayMetadata(phi, psi, exchangeCode);
				if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) break;
			} else {
				// If not, just rewrite input metadata
				alterArrayMetadata(phi, phi, exchangeCode);
			}

			phi++;
			if(psi != NULL) psi++;
			continue;
		}

		// Oh, it looks like we got actual work to do. Womp-womp-woooomp.
		trans = *phi;
		alterArrayMetadata(phi, &trans, exchangeCode);

		is3d = (phi->dim[2] > 3);
		isRZ = (phi->dim[1] == 1) && (phi->dim[2] > 3);

		if((j == 0) || reallocatePerArray) {
			if((exchangeCode != -1) || (psi != NULL)) {
				returnCode = MGA_allocArrays(&nuClone, 1, &trans);
			}
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

				gridsize.z = 1;

				if(isRZ) {
					gridsize.x = ROUNDUPTO(phi->dim[0], BDIM) / BDIM;
					gridsize.y = ROUNDUPTO(phi->dim[2], BDIM) / BDIM;
					cukern_ArrayTranspose2D<<<gridsize, blocksize>>>(phi->devicePtr[i], nuClone->devicePtr[i], sub[3], sub[5]);
				} else {
				if(is3d) {
					gridsize.x = ROUNDUPTO(phi->dim[0], BDIM) / BDIM;
					gridsize.y = ROUNDUPTO(phi->dim[2], BDIM) / BDIM;
					cukern_ArrayExchangeXZ<<<gridsize, blocksize>>>(phi->devicePtr[i], nuClone->devicePtr[i], sub[3], sub[4], sub[5]);
				} else {
					gridsize.x = ROUNDUPTO(phi->dim[0], BDIM) / BDIM;
					gridsize.y = ROUNDUPTO(phi->dim[1], BDIM) / BDIM;
					cukern_ArrayTranspose2D<<<gridsize, blocksize>>>(phi->devicePtr[i], nuClone->devicePtr[i], sub[3], sub[4]);
				}
				}
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
			phi[0] = trans; // Nuke original MGArray

			//phi->numSlabs = tmp.numSlabs; /* Retain original is/isnot allocated status */
			//phi->vectorComponent = tmp.vectorComponent; /* retain original vector component. */

			for(i = 0; i < trans.nGPUs; i++) {
				phi->devicePtr[i] = tmp.devicePtr[i]; // But keep same pointers
				cudaSetDevice(tmp.deviceID[i]); // And overwrite original data
//				cudaMemcpyAsync(phi->devicePtr[i], nuClone->devicePtr[i], phi->partNumel[i]*sizeof(double), cudaMemcpyDeviceToDevice);
				cukern_memmove<<<128, 256>>>(nuClone->devicePtr[i], phi->devicePtr[i], phi->partNumel[i]);
				returnCode = CHECK_IMOGEN_ERROR(CHECK_CUDA_ERROR("cudaMemcpyAsync()"));
				if(returnCode != SUCCESSFUL) break;
			}
			if(returnCode == SUCCESSFUL) {
				if((j == (nArrays-1)) || reallocatePerArray) {
					returnCode = MGA_delete(nuClone); // Before deleting new pointer
				}
			}
			if((j == (nArrays-1)) || reallocatePerArray) { free(nuClone); }

		} else { // Otherwise, simply write the new MGArray to the output pointer.
			psi[0] = *nuClone;
		}

		phi++;
		if(psi != NULL) psi++;
	}

	if(returnCode == SUCCESSFUL) returnCode = CHECK_IMOGEN_ERROR(CHECK_CUDA_ERROR("Departing cudaArrayRotateB"));
#ifdef USE_NVTX
	nvtxRangePop();
#endif
	return returnCode;
}

/* Given a reference array "of", reads the orientation of "in" and applies the appropriate permutation
 * to make it match the orientation of "of" and the result is found in "out".
 *
 * If in==out, the result comes back in place; This requires a buffer.
 * If this is the case and tempStorage is not NULL, tempStorage will be used as the staging area
 * Otherwise, a buffer will be allocated/freed. tempStorage is permitted to be NULL.
 */
int matchArrayOrientation(MGArray *of, MGArray *in, MGArray *out, MGArray *tempStorage)
{
	/*
     * exchange codes for flipArrayIndices: F12 = 2, F13 = 3, F23 = 4, RL = 5, RR = 6;
     *
			1		2		3		4		5		6 (permtag values)
   from\to  123     132     213     231     312     321
	123     ID      F23     F12     RL      RR      F13
	132     F23     ID      RR      F13     F12     RL
	213     F12     RL      ID      F23     F13     RR
	231     RR      F13     F23     ID      RL      F12
	312     RL      F12     F13     RR      ID      F23
	321     F13     RR      RL      F12     F23     ID
	*/
	int flagtable[36] = {
0,     4,      2,     5,      6,     3,
4,     0,      6,     3,      2,     5,
2,     5,      0,     4,      3,     6,
6,     3,      4,     0,      5,     2,
5,     2,      3,     6,      0,     4,
3,     6,      5,     2,      4,     0};

	int toperm = of->permtag;
	int fromperm = in->permtag;

	int flag = flagtable[(toperm-1) + 6*(fromperm-1)];

	int worked = SUCCESSFUL;
	if(out == in) {
		// in place flip
		if(flag != 0) {
			worked = flipArrayIndices(in, NULL, 1, flag, NULL, tempStorage);
		} // identity in-place requires nothing
	} else {
		// out of place flip
		if(flag != 0) {
			worked = flipArrayIndices(in, &out, 1, flag, NULL, tempStorage);
		} else {
			// fixme: copy it!
			worked = ERROR_NOIMPLEMENT;
		}
	}

	return CHECK_IMOGEN_ERROR(worked);
}

/* This routine performs optimized out-of-place exchange on multiple identical arrays
 * at once by taking advantage of concurrent copy + execute; Don't call it directly, flipArrayIndices
 * will determine correctly if it can be called! */
int flipArrayIndices_multisame(MGArray *phi, int nArrays, int exchangeCode, cudaStream_t *streamPtrs, MGArray *tempStorage)
{
#ifdef USE_NVTX
	nvtxRangePush(__FUNCTION__);
#endif
	int returnCode = SUCCESSFUL;

	MGArray trans;

	int i, sub[6];
	int is3d, isRZ;

	int j;

	int usingLocalTemp = 0;
	MGArray localTempStorage;

	usingLocalTemp = checkSufficientFlipStorage(phi, tempStorage) ? 0 : 1;

	// FIXME need to check that tempStorage has sufficient array size available as well!!!!
	// FIXME this is definitely the case when it's being called from flux.cu but not so otherwise.
	if(usingLocalTemp) {
		// allocate it
		#ifdef USE_NVTX
		nvtxMark("cudaArrayRotateB.cu:375 large alloc 2 arrays");
		#endif
		returnCode = MGA_allocSlab(phi, &localTempStorage, 2);
		tempStorage = &localTempStorage;
		usingLocalTemp = 1;
	}

	MGArray tempB;
	MGArray *nuCloneA;
	MGArray *nuCloneB;

	if(CHECK_IMOGEN_ERROR(returnCode) != SUCCESSFUL) return returnCode;

	// advance to 2nd slab of local temp 
	tempB = *tempStorage;
	for(i = 0; i < tempStorage->nGPUs; i++) {
		tempB.devicePtr[i] = tempStorage->devicePtr[i] + (tempStorage->slabPitch[i] / 8);
	}

	nuCloneA = tempStorage;
	nuCloneB = &tempB;

	int tempBlock = 1;
	MGArray *tmparr;
	cudaStream_t tmpstr;

	for(j = 0; j < nArrays; j++) {

		if(actuallyNeedToReorder(&phi->dim[0], exchangeCode) == 0) {
			// Just rewrite input metadata
			alterArrayMetadata(phi, phi, exchangeCode);
			phi++;
			continue;
		}

		if(tempBlock==0) {
			tmparr = nuCloneA;
		} else {
			tmparr = nuCloneB;
		}

		// Oh, it looks like we got actual work to do. Womp-womp-woooomp.
		trans = *phi;
		alterArrayMetadata(phi, &trans, exchangeCode);

		is3d = (phi->dim[2] > 3);
		isRZ = (phi->dim[1] == 1) && (phi->dim[2] > 3);

		dim3 blocksize, gridsize;

		for(i = 0; i < trans.nGPUs; i++) {
			cudaSetDevice(trans.deviceID[i]);
			returnCode = CHECK_IMOGEN_ERROR(CHECK_CUDA_ERROR("cudaSetDevice()"));
			if(returnCode != SUCCESSFUL) break;

			calcPartitionExtent(phi, i, sub);

			tmpstr = (tempBlock) ? streamPtrs[trans.nGPUs+i] : streamPtrs[i];

			switch(exchangeCode) {
			case 2: // Flip XY
				blocksize.x = blocksize.y = BDIM; blocksize.z = 1;
				gridsize.x = ROUNDUPTO(phi->dim[0], BDIM) / BDIM;
				gridsize.y = ROUNDUPTO(phi->dim[1], BDIM) / BDIM;
				gridsize.z = 1;

				if(is3d) {
					cukern_ArrayExchangeXY<<<gridsize, blocksize, 0, tmpstr>>>(phi->devicePtr[i], tmparr->devicePtr[i], sub[3],sub[4],sub[5]);
				} else {
					cukern_ArrayTranspose2D<<<gridsize, blocksize, 0, tmpstr>>>(phi->devicePtr[i], tmparr->devicePtr[i], sub[3], sub[4]);
				}
				break;
			case 3: // Flip XZ
				blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

				gridsize.z = 1;

				if(isRZ) {
					gridsize.x = ROUNDUPTO(phi->dim[0], BDIM) / BDIM;
					gridsize.y = ROUNDUPTO(phi->dim[2], BDIM) / BDIM;
					cukern_ArrayTranspose2D<<<gridsize, blocksize, 0, tmpstr>>>(phi->devicePtr[i], tmparr->devicePtr[i], sub[3], sub[5]);
				} else {
				if(is3d) {
					gridsize.x = ROUNDUPTO(phi->dim[0], BDIM) / BDIM;
					gridsize.y = ROUNDUPTO(phi->dim[2], BDIM) / BDIM;
					cukern_ArrayExchangeXZ<<<gridsize, blocksize, 0, tmpstr>>>(phi->devicePtr[i], tmparr->devicePtr[i], sub[3], sub[4], sub[5]);
				} else {
					gridsize.x = ROUNDUPTO(phi->dim[0], BDIM) / BDIM;
					gridsize.y = ROUNDUPTO(phi->dim[1], BDIM) / BDIM;
					cukern_ArrayTranspose2D<<<gridsize, blocksize, 0, tmpstr>>>(phi->devicePtr[i], tmparr->devicePtr[i], sub[3], sub[4]);
				}
				}
				break;

			case 4: // Flip YZ
				blocksize.x = 32;
				blocksize.y = blocksize.z = 4;

				gridsize.x = ROUNDUPTO(sub[3], blocksize.x) / blocksize.x;
				gridsize.y = ROUNDUPTO(sub[4], blocksize.y) / blocksize.y;
				gridsize.z = 1;

				cukern_ArrayExchangeYZ<<<gridsize, blocksize, 0, tmpstr>>>(phi->devicePtr[i], tmparr->devicePtr[i], sub[3],sub[4],sub[5]);

				break;
			case 5: // Rotate left
				blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

				gridsize.x = ROUNDUPTO(phi->dim[0], BDIM)/BDIM;
				gridsize.y = ROUNDUPTO(phi->dim[1], BDIM)/BDIM;
				gridsize.z = 1;

				cukern_ArrayRotateLeft<<<gridsize, blocksize, 0, tmpstr>>>(phi->devicePtr[i], tmparr->devicePtr[i], sub[3], sub[4], sub[5]);

				break;
			case 6: // Rotate right
				blocksize.x = blocksize.y = BDIM; blocksize.z = 1;

				gridsize.x = ROUNDUPTO(phi->dim[0], BDIM)/BDIM;
				gridsize.y = ROUNDUPTO(phi->dim[2], BDIM)/BDIM;
				gridsize.z = 1;

				cukern_ArrayRotateRight<<<gridsize, blocksize, 0, tmpstr>>>(phi->devicePtr[i], tmparr->devicePtr[i], sub[3], sub[4], sub[5]);

				break;
			}
			returnCode = CHECK_IMOGEN_ERROR(CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, phi, i, "array transposition call"));
			if(returnCode != SUCCESSFUL) break;

		}

		if(returnCode != SUCCESSFUL) break;

		MGArray tmp = phi[0];
		phi[0] = trans; // Nuke original MGArray

		//phi->numSlabs = tmp.numSlabs; /* Retain original is/isnot allocated status */
		//phi->vectorComponent = tmp.vectorComponent; /* retain original vector component. */

		for(i = 0; i < trans.nGPUs; i++) {
			phi->devicePtr[i] = tmp.devicePtr[i]; // But keep same pointers
			cudaSetDevice(tmp.deviceID[i]); // And overwrite original data

			tmpstr = (tempBlock) ? streamPtrs[trans.nGPUs+i] : streamPtrs[i];
			//cudaMemcpyAsync(phi->devicePtr[i], tmparr->devicePtr[i], phi->partNumel[i]*sizeof(double), cudaMemcpyDeviceToDevice, tmpstr);
			cukern_memmove<<<128, 256, 0, tmpstr>>>(tmparr->devicePtr[i], phi->devicePtr[i], phi->partNumel[i]);
			returnCode = CHECK_IMOGEN_ERROR(CHECK_CUDA_ERROR("cudaMemcpyAsync()"));
			if(returnCode != SUCCESSFUL) break;
		}

		tempBlock = 1 - tempBlock; // alternate between 0 and 1
		phi++;
	}

	phi -= nArrays; // return to beginning

	// Clean up allocated arrays
	if(returnCode == SUCCESSFUL) {
		if(usingLocalTemp) {
			#ifdef USE_NVTX
			nvtxMark("cudaArrayRotateB.cu:476 large free");
			#endif
			MGA_delete(&localTempStorage);
		}
	}

	if(returnCode == SUCCESSFUL) returnCode = CHECK_IMOGEN_ERROR(CHECK_CUDA_ERROR("Departing cudaArrayRotateB"));
#ifdef USE_NVTX
	nvtxRangePop();
#endif
	return returnCode;
}

// Identifies cases of index exchanging which do not result in any change of
// actual in-memory ordering: These cases smile and wave as they return 0.
int actuallyNeedToReorder(int *dims, int code)
{
	switch(code) {
	case 2: // flip XY
		if((dims[0] == 1) || (dims[1] == 1)) return 0;
		break;
	case 3: // flip XZ
		if(dims[1] == 1) {
			if((dims[0] == 1) || (dims[2] == 1)) return 0;
		}
		break;
	case 4: // flip YZ
		if((dims[1] == 1) || (dims[2] == 1)) return 0;
		break;
	case 5: // rotate left
		if(dims[0] == 1) return 0; // ???
		if((dims[0] == 1) && (dims[1] == 1)) return 0;
		if((dims[1] == 1) && (dims[2] == 1)) return 0;
		if((dims[0] == 1) && (dims[2] == 1)) return 0;
		break;
	case 6: // rotate right
		if(dims[2] == 1) return 0; // ???
		if((dims[0] == 1) && (dims[1] == 1)) return 0;
		if((dims[1] == 1) && (dims[2] == 1)) return 0;
		if((dims[0] == 1) && (dims[2] == 1)) return 0;
		break;
	}
	return 1; // All other cases return true, we do reorder.
}

int alterArrayMetadata(MGArray *src, MGArray *dst, int exchangeCode)
{
	MGArray origsrc = src[0]; // in event (dst == src)
	int sub[6]; int i;

	switch(exchangeCode) {
	case 2: /* Transform XYZ -> YXZ */
		dst->dim[0] = origsrc.dim[1];
		dst->dim[1] = origsrc.dim[0];

		dst->currentPermutation[0] = origsrc.currentPermutation[1];
		dst->currentPermutation[1] = origsrc.currentPermutation[0];

		if(origsrc.partitionDir == PARTITION_X) { dst->partitionDir = PARTITION_Y; }
		if(origsrc.partitionDir == PARTITION_Y) { dst->partitionDir = PARTITION_X; }
			break;
	case 3: /* Transform XYZ to ZYX */
		dst->dim[0] = origsrc.dim[2];
		dst->dim[2] = origsrc.dim[0];

		dst->currentPermutation[0] = origsrc.currentPermutation[2];
		dst->currentPermutation[2] = origsrc.currentPermutation[0];

		if(origsrc.partitionDir == PARTITION_X) { dst->partitionDir = PARTITION_Z; }
		if(origsrc.partitionDir == PARTITION_Z) { dst->partitionDir = PARTITION_X; }
		break;
	case 4: /* Transform XYZ to XZY */
		dst->dim[1] = origsrc.dim[2];
		dst->dim[2] = origsrc.dim[1];

		dst->currentPermutation[1] = origsrc.currentPermutation[2];
		dst->currentPermutation[2] = origsrc.currentPermutation[1];

		if(origsrc.partitionDir == PARTITION_Y) { dst->partitionDir = PARTITION_Z; }
		if(origsrc.partitionDir == PARTITION_Z) { dst->partitionDir = PARTITION_Y; }
		break;
	case 5: /* Rotate left, XYZ to YZX */
		for(i = 0; i < 3; i++) dst->dim[i] = origsrc.dim[(i+1)%3];

		for(i = 0; i < 3; i++) dst->currentPermutation[i] = origsrc.currentPermutation[(i+1)%3];

		if(origsrc.partitionDir == PARTITION_X) { dst->partitionDir = PARTITION_Z; }
		if(origsrc.partitionDir == PARTITION_Y) { dst->partitionDir = PARTITION_X; }
		if(origsrc.partitionDir == PARTITION_Z) { dst->partitionDir = PARTITION_Y; }
		break;
	case 6: /* Rotate right, XYZ to ZXY */
		for(i = 0; i < 3; i++) dst->dim[i] = origsrc.dim[(i+2)%3];

		for(i = 0; i < 3; i++) dst->currentPermutation[i] = origsrc.currentPermutation[(i+2)%3];

		if(origsrc.partitionDir == PARTITION_X) { dst->partitionDir = PARTITION_Y; }
		if(origsrc.partitionDir == PARTITION_Y) { dst->partitionDir = PARTITION_Z; }
		if(origsrc.partitionDir == PARTITION_Z) { dst->partitionDir = PARTITION_X; }
		break;
	default:
		PRINT_FAULT_HEADER;
		printf("Index to exchange is invalid: %i is not in 2-6 inclusive.\n", exchangeCode);
		PRINT_FAULT_FOOTER;
		fflush(stdout);
		return ERROR_INVALID_ARGS;
	}

	dst->permtag = MGA_numsToPermtag(&dst->currentPermutation[0]);

	// Recalculate the partition sizes


	for(i = 0; i < dst->nGPUs; i++) {
		calcPartitionExtent(dst, i, sub);
		dst->partNumel[i] = sub[3]*sub[4]*sub[5];
	}

	return SUCCESSFUL;
}

/* Examines *tmp and determines if it can hold two copies of *phi (sufficient for index flip multisame)
 * or not */
int checkSufficientFlipStorage(MGArray *phi, MGArray *tmp)
{

if(tmp == NULL) return 0; 
//printf("in checkSufficientFlipStorage: tmp != NULL\n");
if(phi->nGPUs != tmp->nGPUs) return 0; // that much is obvious
//printf("in checkSufficientFlipStorage: nGPUs matches\n");
int i;
long needNumel;
long haveNumel;
for(i = 0; i < phi->nGPUs; i++) {
	needNumel = 2*phi->partNumel[i];
	haveNumel = (tmp->slabPitch[i] * tmp->numSlabs) / 8;
//	printf("in checkSufficientFlipStorage: partition %i: need %li elements have %li\n", i, needNumel, haveNumel);
	if(haveNumel < needNumel) return 0;
}

//printf("in checkSufficientFlipStorage: we do! Yay, no big mallocs!\n");
return 1;

}

__global__ void cukern_memmove(double *s, double *d, long n)
{
long h = blockDim.x*gridDim.x;
long x = threadIdx.x + blockIdx.x*blockDim.x;

while(x < n) {
	d[x] = s[x];
	x += h;  
}
}


/* Very efficiently swaps XY for YX ordering in a 2D array */
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


/* This kernel takes an input array src[i + nx(j + ny k)] and writes dst[k + nz(i + nx j)]
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
