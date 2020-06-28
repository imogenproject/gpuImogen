#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef UNIX
#include <stdint.h>
#include <unistd.h>
#endif
#include "mex.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas.h"
#include "nvToolsExt.h"

#include "cudaCommon.h"
#include "cudaStatics.h"
#include "cudaArrayRotateB.h"
#include "fluidMethod.h" // to know if we need 3 or 4 bdy cells

#define ENFORCE_FLUIDMIN

/* THIS FUNCTION:
   cudaStatics is used in the imposition of several kinds of boundary conditions
   upon arrays. Given a list of indices I, coefficients C and values V, it
   writes out

   phi[I] = (1-C)*phi[I] + C[i]*V[i],
   causing phi[I] to fade to V[i] at an exponential rate.

   It is also able to set mirror boundary conditions
 */

/* FIXME: rewrite this crap with template<>s */
/* X DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS FOR MIRROR BCS */
/* Assume a block size of [3 A B] */
__global__ void cukern_xminusSymmetrize(double *phi, int nx, int ny, int nz);
__global__ void cukern_xminusAntisymmetrize(double *phi, int nx, int ny, int nz);
__global__ void cukern_xplusSymmetrize(double *phi, int nx, int ny, int nz);
__global__ void cukern_xplusAntisymmetrize(double *phi, int nx, int ny, int nz);
/* Y DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS */
/* assume a block size of [N 1 M] */
__global__ void cukern_yminusSymmetrize(double *phi, int nx, int ny, int nz, int depth);
__global__ void cukern_yminusAntisymmetrize(double *phi, int nx, int ny, int nz, int depth);
__global__ void cukern_yplusSymmetrize(double *phi, int nx, int ny, int nz, int depth);
__global__ void cukern_yplusAntisymmetrize(double *phi, int nx, int ny, int nz, int depth);
/* Z DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS */
/* Assume launch with size [U V 1] */
__global__ void cukern_zminusSymmetrize(double *Phi, int nx, int ny, int nz, int depth);
__global__ void cukern_zminusAntisymmetrize(double *Phi, int nx, int ny, int nz, int depth);
__global__ void cukern_zplusSymmetrize(double *Phi, int nx, int ny, int nz, int depth);
__global__ void cukern_zplusAntisymmetrize(double *Phi, int nx, int ny, int nz, int depth);

/* X direction extrapolated boundary conditions */
/* Launch size [3 A B] */
__global__ void cukern_extrapolateLinearBdyXMinus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateLinearBdyXPlus(double *phi, int nx, int ny, int nz);

__global__ void cukern_extrapolateFlatConstBdyXMinus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateFlatConstBdyXPlus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateFlatConstBdyYMinus(double *phi, int nx, int ny, int nz, int depth);
__global__ void cukern_extrapolateFlatConstBdyYPlus(double *phi, int nx, int ny, int nz, int depth);
__global__ void cukern_extrapolateFlatConstBdyZMinus(double *phi, int nx, int ny, int nz, int depth);
__global__ void cukern_extrapolateFlatConstBdyZPlus(double *phi, int nx, int ny, int nz, int depth);

__global__ void cukern_applySpecial_fade(double *phi, double *statics, int nSpecials, int blkOffset);

int setFreeBalanceCondition(MGArray *fluid, GeometryParams *geom, int rightside, int direction, MGArray *gravpot, double fluidGamma);

int setOutflowCondition(MGArray *fluid, GeometryParams *geom, int rightside, int memdir);
template <int direct>
__global__ void cukern_SetOutflowX(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection, int depth);
template <int direct>
__global__ void cukern_SetOutflowY(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection, int depth);
template <int direct>
__global__ void cukern_SetOutflowZ(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection, int depth);

int setBoundarySAS(MGArray *phi, int side, int direction, int sas);

__constant__ __device__ double restFrmSpeed[8];
__constant__ __device__ double thermoVariables[4];

#ifdef STANDALONE_MEX_FUNCTION

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if( (nlhs != 0) || (nrhs != 4)) { mexErrMsgTxt("cudaStatics operator is cudaStatics(ImogenArray, blockdim, GeometryManager, direction)"); }

	CHECK_CUDA_ERROR("entering cudaStatics");

	GeometryParams geom = accessMatlabGeometryClass(prhs[2]);

	int worked;
	MGArray phi;
	worked = MGA_accessMatlabArrays(prhs, 0, 0, &phi);
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) {
		DROP_MEX_ERROR("Access array dumped.");
	}

	worked = setArrayBoundaryConditions(&phi, &geom, (int)*mxGetPr(prhs[3]), 0);
	worked = setArrayBoundaryConditions(&phi, &geom, (int)*mxGetPr(prhs[3]), 1);
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) {
		DROP_MEX_ERROR("setBoundaryCondition called from standalone has crashed: Crashing interpreter. If this happens on entry and before any iterations, check for valid BC mode settings\n");
	}
}
#endif

/* Takes a pointer to MGArray *fluid, which must refer to five MGArrays (rho, px, py, pz, etotal), the Matlab class handle for the density
 * array in case props need to be accessed, the geometry parameters ifnormation *geo, and the direction to set in.
 *
 * direction is 123 for XYZ; This is the case regardless of memory orientation.
 */
int setFluidBoundary(MGArray *fluid, GeometryParams *geo, int direction)
{
#ifdef USE_NVTX
	nvtxRangePush(__FUNCTION__);
#endif

	CHECK_CUDA_ERROR("entering setFluidBoundary");

	int worked;

	// On rereading this seems dumb but there was DEFINITELY a good reason involving
	// c++-level updates magically failing to manifest in ML
	if(fluid == NULL) {
		return ERROR_NULL_POINTER;
	}

	worked = setArrayStaticCells(fluid);
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) { return worked; }

	/* Grabs the whole boundaryData struct from the ImogenArray class */
	mxArray *boundaryData = (mxArray *)fluid->boundaryConditions.externalData;

	if((direction < 1) || (direction > 3)) {
		PRINT_FAULT_HEADER;
		printf("Direction passed to cudaStatics was %i which is not one of 1 (X/R) 2 (Y/theta) 3 (Z)", direction);
		PRINT_FAULT_FOOTER;
		return ERROR_INVALID_ARGS;
	}

	// FIXME: okay here's the shitty way we're doing this,
	// FIXME: If the density array says 'outflow', we call setOutflowCondition once.
	// FIXME: If not, we loop

	int sideNum; // coordinate face: 0 = negative, 1 = positive
	for(sideNum = 0; sideNum < 2; sideNum++) {
		BCModeTypes bt = fluid->boundaryConditions.mode[2*(direction-1) + sideNum];

		if(bt == outflow) {
			int i;
			// make sure we go through and set statics
			for(i = 0; i < 5; i++) {
				worked = setArrayStaticCells(fluid+i);
				if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;
			}
			// Then set the outflow
			worked = setOutflowCondition(fluid, geo, sideNum, direction);
		} else if(bt == freebalance) {
			int i;
			// make sure statics are set
			for(i = 0; i < 5; i++) {
				worked = setArrayStaticCells(fluid+i);
				if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;
			}

			// Now we need to grab the gravitational potential array
			MGArray gravpot;
			mxArray *q = derefXdotAdotB(boundaryData, "gravpot", "field");
			worked = MGA_accessMatlabArrays((const mxArray **)&q, 0, 0, &gravpot);
			if(worked != SUCCESSFUL) {
				PRINT_SIMPLE_FAULT("Failure: Attempted to access fluid[0] (which should be mass) .boundaryData.gravpot and was unsuccessful.\nThe freebalance boundary condition requires that a gravity potential array be set.\n")
				return worked;
			}

			double fluidGamma = derefXdotAdotB_scalar((mxArray *)boundaryData, "gamma", NULL);

			// Then set the free balance condition
			worked = setFreeBalanceCondition(fluid, geo, sideNum, direction, &gravpot, fluidGamma);
		} else {
			int i;
			// iterate through the 5 fluid arrays. setA.B.C. sets statics for the array it's passed automatically.
			for(i = 0; i < 5; i++) {
				worked = setArrayBoundaryConditions(fluid+i, geo, direction, sideNum);
				if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;
			}
		}
		if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;
	}

#ifdef USE_NVTX
	nvtxRangePop();
#endif

	return CHECK_IMOGEN_ERROR(worked);
}

/* Accept a pointer to an MGArray, its matlab handle, geometry information, the physical direction to set the
 * boundary conditions on, and the side (0 = negative coord side, 1 = positive coord side) */
int setArrayBoundaryConditions(MGArray *phi, GeometryParams *geo, int direction, int side)
{
	CHECK_CUDA_ERROR("entering setBoundaryConditions");

	int worked;
	if(phi == NULL) {
		return ERROR_NULL_POINTER;
	}

	worked = setArrayStaticCells(phi);
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) return worked;

	if(direction == 0) return SUCCESSFUL; /* Skips edge BCs if desired. */

	/* Grabs the whole boundaryData struct from the ImogenArray class */
	mxArray *boundaryData = (mxArray *)phi->boundaryConditions.externalData;
	if(boundaryData == NULL) {
		PRINT_SIMPLE_FAULT("FATAL: field 'boundaryData' D.N.E. in class. Not a class? Not an ImogenArray/FluidArray?\n");
		return ERROR_INVALID_ARGS;
	}

	int vectorComponent = phi->vectorComponent;

	mxArray *bcModes = mxGetField(boundaryData, 0, "bcModes");
	if(bcModes == NULL) {
		PRINT_SIMPLE_FAULT("FATAL: bcModes structure not present. Not an ImogenArray? Not initialized?\n");
		return ERROR_INVALID_ARGS;
	}

	int *perm = &phi->currentPermutation[0];

	int memoryDirection = MGA_dir2memdir(perm, direction);

	BCModeTypes bm = phi->boundaryConditions.mode[2*(direction-1)+side];
	// Sets a mirror BC: scalar, vector_perp f(b+x) = f(b-x), vector normal f(b+x) = -f(b-x)
	if(bm == mirror)
		worked = setBoundarySAS(phi, side, memoryDirection, vectorComponent == direction);

	// Extrapolates f(b+x) = f(b)
	if(bm == extrapConstant) {
		worked = setBoundarySAS(phi, side, memoryDirection, 2);
	}

	// Extrapolates f(b+x) = f(b) + x f'(b)
	// WARNING: This is unconditionally unstable unless normal flow rate is supersonic
	if(bm == extrapLinear) {
		worked = setBoundarySAS(phi, side, memoryDirection, 3);
	}

	if(bm == wall) {
		PRINT_FAULT_HEADER;
		printf("Wall BC is not implemented!\n");
		PRINT_FAULT_FOOTER;
		worked = ERROR_INVALID_ARGS;
	}

	if(bm == outflow) {
		if(phi->numSlabs == 0) { // primitive check that this in fact the rho array setOutflowCondition needs...
			worked = setOutflowCondition(phi, geo, side, direction);
		}
	}

	if(bm == freebalance) {
		if(phi->numSlabs == 0) { // primitive check that this in fact the rho array setOutflowCondition needs...
			// Now we need to grab the gravitational potential array
			MGArray gravpot;
			mxArray *q = derefXdotAdotB(boundaryData, "gravpot", "field");
			if(q != NULL) {
				worked = MGA_accessMatlabArrays((const mxArray **)&q, 0, 0, &gravpot);
				if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) {
					printf("Failure: Attempted to access fluid[0] (which should be mass) .boundaryData.gravpot and was unsuccessful.\nThe freebalance boundary condition requires that a gravity potential array be set.\n");
					return worked;
				}

				double fluidGamma = derefXdotAdotB_scalar((mxArray *)phi->boundaryConditions.externalData, "gamma", NULL);

				// Then set the free balance condition
				worked = setFreeBalanceCondition(phi, geo, side, direction, &gravpot, fluidGamma);
			}
		}
	}


		// whatever we just did, check...
		if(worked != SUCCESSFUL) break;

	return SUCCESSFUL;
}

int setArrayStaticCells(MGArray *phi)
{
	int worked;
	MGArray statics;

	/* Grabs the whole boundaryData struct from the ImogenArray class */
	mxArray *boundaryData = (mxArray *)phi->boundaryConditions.externalData;
	if(boundaryData == NULL) {
		printf("FATAL: field 'boundaryData' D.N.E. in class. Not a class? Not an ImogenArray/FluidArray?\n");
		return ERROR_INVALID_ARGS;
	}

	// fixme: we can't do this in this function yet. Oh boy
	/* The statics describe "solid" structures which we force the grid to have */
	mxArray *gpuStatics = mxGetField(boundaryData, 0, "staticsData");
	if(gpuStatics == NULL) {
		PRINT_SIMPLE_FAULT("FATAL: field 'staticsData' D.N.E. in boundaryData struct. Statics not compiled?\n");
		return ERROR_INVALID_ARGS;
	}
	worked = MGA_accessMatlabArrays((const mxArray **)(&gpuStatics), 0, 0, &statics);
	BAIL_ON_FAIL(worked)

	int *perm = &phi->currentPermutation[0];
	int offsetIdx = 2*(perm[0]-1) + 1*(perm[1] > perm[2]);

	/* The offset array describes the index offsets for the data in the gpuStatics array */
	mxArray *offsets    = mxGetField(boundaryData, 0, "compOffset");
	if(offsets == NULL) {
		PRINT_SIMPLE_FAULT("FATAL: field 'compOffset' D.N.E. in boundaryData. Not an ImogenArray? Statics not compiled?\n");
		return ERROR_INVALID_ARGS;
	}
	double *offsetCount = mxGetPr(offsets);
	long int staticsOffset = (long int)offsetCount[2*offsetIdx];
	int staticsNumel  = (int)offsetCount[2*offsetIdx+1];

	/* Parameter describes what block size to launch with... */
	int blockdim = 32;

	dim3 griddim; griddim.x = staticsNumel / blockdim + 1;
	if(griddim.x > 1024) {
		griddim.x = 1024;
		griddim.y = staticsNumel/(blockdim*griddim.x) + 1;
	}

	/* Every call results in applying specials */
	int i;
	for(i = 0; i < phi->nGPUs; i++) {
		staticsOffset = (long int)offsetCount[12*i + 2*offsetIdx];
		staticsNumel = (int)offsetCount[12*i + 2*offsetIdx + 1];

		if(staticsNumel > 0) {
			cudaSetDevice(phi->deviceID[i]);
			CHECK_CUDA_ERROR("setDevice");

			cukern_applySpecial_fade<<<griddim, blockdim>>>(phi->devicePtr[i], statics.devicePtr[i] + staticsOffset, staticsNumel, statics.dim[0]);

//			cudaDeviceSynchronize(); // NOTE for debugging only!
			worked = CHECK_CUDA_LAUNCH_ERROR(blockdim, griddim, phi, 0, "cuda statics application in setArrayStaticCells");
			if(worked != SUCCESSFUL) return worked;
		}
	}
return SUCCESSFUL;
}

int doBCForPart(MGArray *fluid, int part, int direct, int rightside)
{
	if(fluid->partitionDir == direct) {
		if(rightside && (part != fluid->nGPUs-1)) return 0;
		if(!rightside && (part != 0)) return 0;
	}
	return 1;
}

/* Sets boundary condition as follows:
 * fluid boundary = (v . outwardNormal > 0) ? constant : mirror
 * This must be passed ONLY the density array (FIXME: awful testing hack)
 */
int setOutflowCondition(MGArray *fluid, GeometryParams *geom, int rightside, int direction)
{
	dim3 blockdim, griddim;
	int i;
	int sub[6];

	int status;

	int memdir = MGA_dir2memdir(&fluid->currentPermutation[0], direction);

	int h = fluid->haloSize;
	double rfVelocity[2*h];
	// When setting the radial face for outflow and the frame is rotating,
	// the inertial rest frame zero (imposed if inward flow is attempted)
	// is not zero in the rotating frame:

	//for(i = 0; i < h; i++) { rfVelocity[i] = -(geom->Rinner + i*geom->h[0])*geom->frameOmega; }
	for(i = 0; i < h; i++) { rfVelocity[i] = 1.05; }
	//for(i = 0; i < h; i++) { rfVelocity[h+i] = -(geom->Rinner + (fluid->dim[0]-h+i)*geom->h[0])*geom->frameOmega;  }
	for(i = 0; i < h; i++) { rfVelocity[h+i] = -1.198 - .0047*i;  }

	for(i = 0; i < fluid->nGPUs; i++) {
		// Prevent BC from being done to internal partition boundaries if we are partitioned in this direction
		if(doBCForPart(fluid, i, PARTITION_X, rightside)) {
			cudaSetDevice(fluid->deviceID[i]);
			cudaMemcpyToSymbol((const void *)restFrmSpeed, &rfVelocity[0], 2*h*sizeof(double), 0, cudaMemcpyHostToDevice);
		}
	}

	int depth = fluid->haloSize;

	status = CHECK_CUDA_ERROR("__constant__ uploads for outflow condition");
	if(status != SUCCESSFUL) return status;

	switch(memdir) {
	case 1: // x
		blockdim.x = depth; blockdim.y = 8; blockdim.z = 16;

		for(i = 0; i < fluid->nGPUs; i++) {
			if(doBCForPart(fluid, i, PARTITION_X, rightside) == 0) continue;

			cudaSetDevice(fluid->deviceID[i]);
			calcPartitionExtent(fluid, i, &sub[0]);

			griddim.x = ROUNDUPTO(sub[4], blockdim.y)/blockdim.y;
			griddim.y = ROUNDUPTO(sub[5], blockdim.z)/blockdim.z;

			if(rightside) {
				double *base = fluid->devicePtr[i] + sub[3] - 1 - depth;
				cukern_SetOutflowX<1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction, depth);
			} else {
				double *base = fluid->devicePtr[i];
				cukern_SetOutflowX<-1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction, depth);
			}
		}
		break;
	case 2: // y
		blockdim.x = 32; blockdim.y = depth; blockdim.z = 1;
		for(i = 0; i < fluid->nGPUs; i++) {
			if(doBCForPart(fluid, i, PARTITION_Y, rightside) == 0) continue;
			cudaSetDevice(fluid->deviceID[i]);
			calcPartitionExtent(fluid, i, &sub[0]);

			griddim.x = ROUNDUPTO(sub[4], blockdim.x)/blockdim.x;
			griddim.y = sub[5];
			griddim.z = 1;

			if(rightside) {
				double *base = fluid->devicePtr[i] + (sub[4]-1-2*depth)*sub[3];
				cukern_SetOutflowY<1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction, depth);
			} else {
				cukern_SetOutflowY<-1><<<griddim, blockdim>>>(fluid->devicePtr[i], sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction, depth);
			}
		}
		break;
	case 3: // z
		blockdim.x = 32; blockdim.y = 4; blockdim.z = depth;
		for(i = 0; i < fluid->nGPUs; i++) {
			if(doBCForPart(fluid, i, PARTITION_Z, rightside) == 0) continue;

			cudaSetDevice(fluid->deviceID[i]);
			calcPartitionExtent(fluid, i, &sub[0]);
			griddim.x = ROUNDUPTO(sub[3], blockdim.x)/blockdim.x;
			griddim.y = ROUNDUPTO(sub[4], blockdim.y)/blockdim.y;
			griddim.z = 1;

			if(rightside) {
				double *base = fluid->devicePtr[i] + sub[3]*sub[4]*(sub[5]-1 - 2*depth);
				cukern_SetOutflowZ<1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction, depth);
			} else {
				cukern_SetOutflowZ<-1><<<griddim, blockdim>>>(fluid->devicePtr[i], sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction, depth);
			}
		}
		break;
	}

	status = CHECK_CUDA_LAUNCH_ERROR(blockdim, griddim, fluid, direction, "Setting outflow BC on array");
	return status;
}

__device__ double outflowBC_radial_vel(double p0, int normdir, int i)
{
	return 0.0;
}

// may return nonzero because inertial rest frame zero velocity is nonzero in rotating frame
__device__ double outflowBC_phi_vel(double v0, int normdir, int i)
{
	if(normdir == 1) { return restFrmSpeed[i]; } else { return v0; }
}

__device__ double outflowBC_zee_vel(double v0, int normdir, int i)
{
	return 0.0;
}

// Assume we are passed &rho(depth,0,0) if direct == 0
// assume we are passed &rho(nx-1-depth,0,0) if direct == 1
// Launch an depthxAxB block with an NxM grid such that AN >= ny and BM >= nz
template <int direct>
__global__ void cukern_SetOutflowX(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection, int depth)
{
	int tix = threadIdx.x;

	int y = threadIdx.y + blockDim.y*blockIdx.x;
	int z = threadIdx.z + blockDim.z*blockIdx.y;
	if((y >= ny) || (z >= nz)) return;

	// Y-Z translate
	base += nx*(y+ny*z);

	double a;

	if(direct > 0) {
		// + edge: [0] is base
		a = base[(1+normalDirection)*slabNumel];
	} else {
		// - edge: [0] is edge
		a = base[(1+normalDirection)*slabNumel + depth];
	}

	double rho, E, p1, p2;

	if(a*direct > 0) {
		// boundary normal momentum positive: extrapolate as constant
		if(direct == 1) { // +x: base = &rho[nx-4, 0 0]
			tix++; // 0123 to 1234
			a = base[0]; base[tix] = a; base += slabNumel;
			a = base[0]; base[tix] = a; base += slabNumel;
			a = base[0]; base[tix] = a; base += slabNumel;
			a = base[0]; base[tix] = a; base += slabNumel;
			a = base[0]; base[tix] = a; base += slabNumel;
		} else { // -x: base = &rho[3,0,0]
			base = base - blockDim.x; // move back to zero
			a = base[depth]; base[tix] = a; base += slabNumel;
			a = base[depth]; base[tix] = a; base += slabNumel;
			a = base[depth]; base[tix] = a; base += slabNumel;
			a = base[depth]; base[tix] = a; base += slabNumel;
			a = base[depth]; base[tix] = a; base += slabNumel;
		}
	} else {
		// boundary normal momentum negative: null normal velocity, otherwise mirror
		// phi velocity
		if(direct == 1) { // +x
			rho = a = base[0]; base[1+tix] = a; base += slabNumel; // rho
			E =       base[0]; base += slabNumel; // E

			a = base[0]; p1 = a; p2 = rho*outflowBC_radial_vel(a/rho, normalDirection, depth+tix); base[1+tix] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[0]; p1 = a; p2 = rho*outflowBC_phi_vel(a/rho, normalDirection, depth+tix); base[1+tix] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[0]; p1 = a; p2 = rho*outflowBC_zee_vel(a/rho, normalDirection, depth+tix); base[1+tix] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho;
			base[1+tix-3*slabNumel] = E;
		} else { // -x
			base=base-depth;

			rho = a = base[depth]; base[tix] = a; base += slabNumel; // rho
			E =       base[depth]; base += slabNumel; // E

			a = base[depth]; p1 = a; p2 = rho*outflowBC_radial_vel(a/rho, normalDirection, tix); base[tix] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[depth]; p1 = a; p2 = rho*outflowBC_phi_vel(a/rho, normalDirection, tix); base[tix] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[depth]; p1 = a; p2 = rho*outflowBC_zee_vel(a/rho, normalDirection, tix); base[tix] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho;
			base[tix-3*slabNumel] = E;
		}

		// The potential for evacuating flows created by this mode of operation can result in crappy behavior
		// We need to re-enforce. maybe?
#ifdef ENFORCE_FLUIDMIN

#endif

	}

}

// Use 32x3 threads and dim3(roundup(nx/32), nz, 1) blocks
// if(direct == -1) {minus coordinate edge} assume base is &rho[0]
// if(direct == 1)  {plus coordinate edge}  assume base is &rho[0,ny-1-2*depth (base of altered region),0]
template <int direct>
__global__ void cukern_SetOutflowY(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection, int depth)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int z = blockIdx.y;
	int tiy = threadIdx.y;

	if((x >= nx) || (z >= nz)) return;

	// Y-Z translate
	base += x + nx*(ny*z);

	double a, rho, E, p1, p2;

	a = base[(1+normalDirection)*slabNumel+4*depth*nx];

	if(a*direct > 0) {
		// normal mom is positive: extrapolate constant
		if(direct == 1) { // +y edge
			a = base[depth*nx]; base[(depth+1+tiy)*nx] = a; base += slabNumel;
			a = base[depth*nx]; base[(depth+1+tiy)*nx] = a; base += slabNumel;
			a = base[depth*nx]; base[(depth+1+tiy)*nx] = a; base += slabNumel;
			a = base[depth*nx]; base[(depth+1+tiy)*nx] = a; base += slabNumel;
			a = base[depth*nx]; base[(depth+1+tiy)*nx] = a; base += slabNumel;
		} else { // -y edge
			a = base[depth*nx]; base[(tiy)*nx] = a; base += slabNumel;
			a = base[depth*nx]; base[(tiy)*nx] = a; base += slabNumel;
			a = base[depth*nx]; base[(tiy)*nx] = a; base += slabNumel;
			a = base[depth*nx]; base[(tiy)*nx] = a; base += slabNumel;
			a = base[depth*nx]; base[(tiy)*nx] = a; base += slabNumel;
		}
	} else {
		// normal mom negative: set norm momentum to rest frame zero, otherwise constant
		if(direct == 1) { // +y edge
			rho = a = base[depth*nx]; base[(depth+1+tiy)*nx] = a; base += slabNumel; // rho
			E =       base[depth*nx]; base += slabNumel; // E

			a = base[depth*nx]; p1 = a; p2 = rho*outflowBC_radial_vel(a/rho, normalDirection, depth+tiy); base[(depth+1+tiy)*nx] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[depth*nx]; p1 = a; p2 = rho*outflowBC_phi_vel(a/rho, normalDirection, depth+tiy); base[(depth+1+tiy)*nx] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[depth*nx]; p1 = a; p2 = rho*outflowBC_zee_vel(a/rho, normalDirection, depth+tiy); base[(depth+1+tiy)*nx] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho;
			base[(depth+1+tiy)*nx - 3*slabNumel] = E;
		} else { // -y edge
			rho = a = base[depth*nx]; base[(4+tiy)*nx] = a; base += slabNumel; // rho
			E =       base[depth*nx]; base += slabNumel; // E

			a = base[depth*nx]; p1 = a; p2 = rho*outflowBC_radial_vel(a/rho, normalDirection, tiy); base[(tiy)*nx] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[depth*nx]; p1 = a; p2 = rho*outflowBC_phi_vel(a/rho, normalDirection, tiy); base[(tiy)*nx] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[depth*nx]; p1 = a; p2 = rho*outflowBC_zee_vel(a/rho, normalDirection, tiy); base[(tiy)*nx] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho;
			base[(tiy)*nx-3*slabNumel] = E;
		}
	}

}

// XY threads & block span the XY space, 'depth' Z threads
// one plane of threads
// if(direct == -1) base should be &rho[0,0,0]
// if(direct == 1) base should be &rho[0,0,nz-1 - 2*depth]
template<int direct>
__global__ void cukern_SetOutflowZ(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection, int depth)
{
int x = threadIdx.x + blockDim.x*blockIdx.x;
int y = threadIdx.y + blockDim.y*blockIdx.y;
int tiz= threadIdx.z;

if((x >= nx) || (y >= ny)) return;

base += x + nx*y;

int delta = nx*ny;
double a = base[(1+normalDirection)*slabNumel];
int i;
double rho, E, p1, p2;

if(a*direct > 0) {
	// positive normal momentum: constant extrap
	if(direct == 1) { // +z edge
		for(i = 0; i < 5; i++) { a = base[depth*delta]; base[(depth+1+tiz)*delta] = a; base += slabNumel; }
	} else { // -z edge
		for(i = 0; i < 5; i++) { a = base[depth*delta]; base[tiz*delta] = a; base += slabNumel; }
	}
} else {
	// negative normal momentum: set normal mom to null
	if(direct == 1) { // +z edge
		rho = a = base[depth*delta]; base[(depth+1+tiz)*delta] = a; base += slabNumel;
		E = base[depth*delta]; base += slabNumel;

		a = base[depth*delta]; p1 = a; p2 = rho*outflowBC_radial_vel(a/rho, normalDirection, 3+tiz); base[(depth+1+tiz)*delta] = p2;
		E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
		a = base[depth*delta]; p1 = a; p2 = rho*outflowBC_phi_vel(a/rho, normalDirection, 3+tiz); base[(depth+1+tiz)*delta] = p2;
		E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
		a = base[depth*delta]; p1 = a; p2 = rho*outflowBC_zee_vel(a/rho, normalDirection, 3+tiz); base[(depth+1+tiz)*delta] = p2;
		E += .5*(p2-p1)*(p2+p1)/rho;
		base[(depth+1+tiz)*delta - 3*slabNumel] = E;
	} else { // -z edge : defective at present
		rho = a = base[depth*delta]; base[(tiz)*delta] = a; base += slabNumel;
		E = base[depth*delta]; base += slabNumel;

		a = base[depth*delta]; p1 = a; p2 = rho*outflowBC_radial_vel(a/rho, normalDirection, tiz); base[(tiz)*delta] = p2;
		E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
		a = base[depth*delta]; p1 = a; p2 = rho*outflowBC_phi_vel(a/rho, normalDirection, tiz); base[(tiz)*delta] = p2;
		E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
		a = base[depth*delta]; p1 = a; p2 = rho*outflowBC_zee_vel(a/rho, normalDirection, tiz); base[(tiz)*delta] = p2;
		E += .5*(p2-p1)*(p2+p1)/rho;
		base[tiz*delta - 3*slabNumel] = E;
	}
}

}

template <int direct>
__global__ void cukern_SetFreeBalanceX(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection, int depth, double *gravpot, double vFrame);
template <int direct>
__global__ void cukern_SetFreeBalanceY(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection, int depth, double *gravpot, double vFrame);
template<int direct>
__global__ void cukern_SetFreeBalanceZ(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection, int depth, double *gravpot, double vFrame);

__constant__ double radialQ[64];

/* Sets boundary condition as follows:
 * Under the presumption of gravitational balance at the boundary in question, solves the five fluid variables via
 * v_transverse = free (extrapolate constant)
 * v_normal = 0 (no flow)
 * d(v_normal)/dt = 0 (no comoving acceleration) via dP/dnormal = rho d(phi_grav)/dnormal)
 * Balance is computed by extrapolating an isothermal condition (a BC of T=constant and rho solved to give the pressure gradient)
 * In the absence of gravity, this reduces to a slip wall
 */
int setFreeBalanceCondition(MGArray *fluid, GeometryParams *geom, int rightside, int direction, MGArray *gravpot, double fluidGamma)
{
	dim3 blockdim, griddim;
	int i;
	int sub[6];

	int status;

	int memdir = MGA_dir2memdir(&fluid->currentPermutation[0], direction);

	int h = fluid->haloSize;

	double thermo[4];
	thermo[0] = fluidGamma;
	thermo[1] = 1/(fluidGamma - 1); // this is the only one that's used...
	thermo[2] = fluidGamma - 1;

	double q, vFrameInner, vFrameOuter, x0;
	double hostRadQ[6*h];

	vFrameInner = geom->frameOmega * (geom->Rinner + (h)*geom->h[0]); // at reference cell
	// compute the frame term for the inner cells
	for(i = 0; i < h; i++) {
		// q is a dimensionless radius ratio
		q = (geom->Rinner + (i)*geom->h[0]) / (geom->Rinner + (h)*geom->h[0]);

		// centripetal potential integral from cell h to cell i = v0^2 x0 + v0 x1 + x2
		x0 = (q-1)/q; // on v0^2

		hostRadQ[    i] = x0;
		hostRadQ[h + i] = pow(q, -0.5);
		hostRadQ[2*h+i] = q;
	}

	// compute the frame term for the outer cells
	vFrameOuter = geom->frameOmega * (geom->Rinner + (fluid->dim[0] -1 - h)*geom->h[0]); // at reference cell
	for(i = 0; i < h; i++) {
		// a dimensionless radius ratio
		q = (geom->Rinner + (fluid->dim[0] - h + i)*geom->h[0]) / (geom->Rinner + (fluid->dim[0] - 1 - h)*geom->h[0]);

		// centripetal potential integral from cell h to cell i = v0^2 x0 + v0 x1 + x2
		x0 = (q-1)/q; // on v0^2

		hostRadQ[3*h + i] = x0;
		hostRadQ[4*h + i] = pow(q, -0.5);
		hostRadQ[5*h + i] = q;
	}

	for(i = 0; i < fluid->nGPUs; i++) {
		// Prevent BC from being done to internal partition boundaries if we are partitioned in this direction
		if(doBCForPart(fluid, i, PARTITION_X, rightside)) {
			cudaSetDevice(fluid->deviceID[i]);
			if(direction == 1) { cudaMemcpyToSymbol((const void *)radialQ, &hostRadQ[0], 6*h*sizeof(double), 0, cudaMemcpyHostToDevice); }
			cudaMemcpyToSymbol((const void *)thermoVariables, &thermo[0], 4*sizeof(double), 0, cudaMemcpyHostToDevice);
		}
	}

	int depth = fluid->haloSize;

	status = CHECK_CUDA_ERROR("__constant__ uploads for freebalance condition");
	if(status != SUCCESSFUL) return status;

	MGArray gravpotOriented;
	int createdNewGravpot = 0;

	if(fluid->permtag != gravpot->permtag) {
		MGArray *q = &gravpotOriented;

		status = matchArrayOrientation(fluid, gravpot, q, (MGArray *)NULL);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

		createdNewGravpot = 1;
	} else {
		gravpotOriented = *gravpot;
	}

	switch(memdir) {
	case 1: // X
		blockdim.x = depth; blockdim.y = blockdim.z = 16;

		for(i = 0; i < fluid->nGPUs; i++) {
			if(doBCForPart(fluid, i, PARTITION_X, rightside) == 0) continue;

			cudaSetDevice(fluid->deviceID[i]);
			calcPartitionExtent(fluid, i, &sub[0]);

			griddim.x = ROUNDUPTO(sub[4], 16)/16;
			griddim.y = ROUNDUPTO(sub[5], 16)/16;

			if(rightside) {
				double *base = fluid->devicePtr[i] + sub[3] - 1 - depth;
				double *beta = gravpotOriented.devicePtr[i] + sub[3] - 1 - depth;
				cukern_SetFreeBalanceX<1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction, depth, beta, vFrameOuter);
			} else {
				double *base = fluid->devicePtr[i];
				double *beta = gravpotOriented.devicePtr[i];
				cukern_SetFreeBalanceX<-1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction, depth, beta, vFrameInner);
			}
		}
		break;
	case 2: // Y
		blockdim.x = 32; blockdim.y = depth; blockdim.z = 1;
		for(i = 0; i < fluid->nGPUs; i++) {
			if(doBCForPart(fluid, i, PARTITION_Y, rightside) == 0) continue;
			cudaSetDevice(fluid->deviceID[i]);
			calcPartitionExtent(fluid, i, &sub[0]);

			griddim.x = ROUNDUPTO(sub[4], 32)/32;
			griddim.y = sub[5];
			griddim.z = 1;

			if(rightside) {
				double *base = fluid->devicePtr[i]   + sub[3] * (sub[4]-1-depth);
				double *beta = gravpotOriented.devicePtr[i] + sub[3] * (sub[4]-1-depth);
				cukern_SetFreeBalanceY<1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction, depth, beta, vFrameOuter);
			} else {
				double *base = fluid->devicePtr[i];
				double *beta = gravpotOriented.devicePtr[i];
				cukern_SetFreeBalanceY<-1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction, depth, beta, vFrameInner);
			}
		}
		break;
	case 3: // Z
		blockdim.x = 32; blockdim.y = 8; blockdim.z = 1;
		for(i = 0; i < fluid->nGPUs; i++) {
			if(doBCForPart(fluid, i, PARTITION_Z, rightside) == 0) continue;

			cudaSetDevice(fluid->deviceID[i]);
			calcPartitionExtent(fluid, i, &sub[0]);
			griddim.x = ROUNDUPTO(sub[3], blockdim.x)/blockdim.x;
			griddim.y = ROUNDUPTO(sub[4], blockdim.y)/blockdim.y;
			griddim.z = 1;

			if(rightside) {
				double *base = fluid->devicePtr[i] + sub[3]*sub[4]*(sub[5]-1 - depth);
				double *beta = gravpotOriented.devicePtr[i] + sub[3]*sub[4]*(sub[5] - 1 - depth);
				cukern_SetFreeBalanceZ<1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction, depth, beta, vFrameOuter);
			} else {
				double *base = fluid->devicePtr[i];
				double *beta = gravpotOriented.devicePtr[i];
				cukern_SetFreeBalanceZ<-1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction, depth, beta, vFrameInner);
			}
		}
		break;
	}

	status = CHECK_CUDA_LAUNCH_ERROR(blockdim, griddim, fluid, direction, "Setting freebalance BC on array");

	if(createdNewGravpot) {
		status = CHECK_IMOGEN_ERROR(MGA_delete(&gravpotOriented));
	}

	return status;
}

// We are passed &rho(0,0,0) if direct == -1
// We are passed &rho(nx-1-depth,0,0) if direct == 1
// Launch an depthxAxB block with an NxM grid such that AN >= ny and BM >= nz
template <int direct>
__global__ void cukern_SetFreeBalanceX(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection, int depth, double *gravpot, double vFrame)
{
	int tix = threadIdx.x;

	int y = threadIdx.y + blockDim.y*blockIdx.x;
	int z = threadIdx.z + blockDim.z*blockIdx.y;
	if((y >= ny) || (z >= nz)) return;

	// Y-Z translate
	base    += nx*(y+ny*z);
	gravpot += nx*(y+ny*z);

	double rho, E, vx, vy, vz, T, phi0, phi1, rho2;

	if(direct > 0) { } else { base += depth; }
	// + boundary
	// mind all [depth] threads are hitting the same read points here
	rho = base[0];
	vx = base[2*slabNumel]/rho;
	vy = base[3*slabNumel]/rho;
	vz = base[4*slabNumel]/rho;
	E = base[slabNumel] - .5*rho*(vx*vx+vy*vy+vz*vz);

	if(direct > 0) { } else { base -= depth; }

	T = E / rho; // actually eint / rho = (P / rho)  / (gamma-1)

	if(direct > 0) { phi0 = gravpot[0]; } else { phi0 = gravpot[depth]; }
	if(direct > 0) { phi1 = gravpot[tix+1]; } else { phi1 = gravpot[tix]; } // load phi at current altitude

	// If difference in potential is zero, this reduces to extrapolate-as-constant.
	// FIXME HACK TESTING: this is used here to, based on density, discriminate between 
	// disk ("dense") and solar wind ("tenuous") conditions

	if(rho < 1e-7) { phi1 = phi0; vz = (vz*direct < 0) ? 0 : vz ;  }

	if(normalDirection != 1) {
		// Non-radial cases: just balance gravity potential against pressure at fixed temperature
		// Isothermal force balance condition:
		// rho dphi/dz = -dP/dz = -T drho/dz
		// -dphi/T = drho / rho
		// -(phi2 - phi1)/T = ln(rho2) - ln(rho1)
		// rho2 = rho1 exp(-(phi2 - phi1)/T)
		rho2 = rho * exp(-thermoVariables[1]*(phi1-phi0)/T);
	} else {
		// Accounts for the centripetal potential, asserting kepler curve (omega ~ r^-1.5) extrapolated from first valid cell
		double centripetalPotential;

		if(direct > 0) {
			vy += vFrame;
			centripetalPotential = vy*vy*radialQ[tix+3*depth];
			vy = vy * radialQ[tix + 4*depth] - vFrame * radialQ[tix + 5*depth];
		} else {
			vy += vFrame;
			centripetalPotential = vy*vy*radialQ[tix+0*depth];
			vy = vy * radialQ[tix + 1*depth] - vFrame * radialQ[tix + 2*depth];
		}
		rho2 = rho * exp(-thermoVariables[1]*(phi1-phi0 - centripetalPotential)/T);
	}

	if(direct > 0) { base = base + tix + 1; } else { base = base + tix; }

	base[0] = rho2; // density falls off with altitude
	base[2*slabNumel] = rho2*vx; // "slip wall" condition in the transverse directions
	base[3*slabNumel] = rho2*vy;
	base[4*slabNumel] = rho2*vz;      // Assert no motion vertically
	E = rho2*T + .5*rho2*(vx*vx+vy*vy+vz*vz);
	base[slabNumel] = E;        // Assert a pressure condition vertically to maintain zero comoving derivative

}

// Use 32x3 threads and dim3(roundup(nx/32), nz, 1) blocks
// if(direct == -1) {minus coordinate edge} base is &rho[0, 0, 0]
// if(direct == 1)  {plus coordinate edge}  base is &rho[0,ny-1-depth ,0]
template <int direct>
__global__ void cukern_SetFreeBalanceY(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection, int depth, double *gravpot, double vFrame)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int z = blockIdx.y;
	int tiy = threadIdx.y;

	if((x >= nx) || (z >= nz)) return;

	// X-Z translate
	base += x + nx*(ny*z);
	gravpot += x + nx*ny*z;

	double rho, E, vx, vy, vz, T, phi0, phi1, rho2;

	if(direct > 0) { } else { base = base + depth*nx; }
	// + boundary
	// mind all [depth] threads are hitting the same read points here
	rho = base[0];
	vx = base[2*slabNumel]/rho;
	vy = base[3*slabNumel]/rho;
	vz = base[4*slabNumel]/rho;
	E = base[slabNumel] - .5*rho*(vx*vx+vy*vy+vz*vz);
	if(direct > 0) { } else { base = base - depth*nx; }

	T = E / rho;
	if(direct > 0) { phi0 = gravpot[0]; } else { phi0 = gravpot[nx*depth]; }
	if(direct > 0) { phi1 = gravpot[nx*(tiy+1)]; } else { phi1 = gravpot[nx*tiy]; }

	if(rho < 1e-7) { phi1 = phi0; vz = (vz*direct < 0) ? 0 : vz; }

	if(normalDirection != 1) {
		// Non-radial cases: just balance gravity potential against pressure at fixed temperature
		// Isothermal force balance condition:
		// rho dphi/dz = -dP/dz = -T drho/dz
		// -dphi/T = drho / rho
		// -(phi2 - phi1)/T = ln(rho2) - ln(rho1)
		// rho2 = rho1 exp(-(phi2 - phi1)/T)
		rho2 = rho * exp(-thermoVariables[1]*(phi1-phi0)/T);
	} else {
		// Accounts for the centripetal potential, asserting kepler curve (omega ~ r^-1.5) extrapolated from first valid cell
		double centripetalPotential;

		if(direct > 0) {
			vy += vFrame;
			centripetalPotential = vy*vy*radialQ[tiy+3*depth];
			vy = vy * radialQ[tiy + 4*depth] - vFrame * radialQ[tiy + 5*depth];
		} else {
			vy += vFrame;
			centripetalPotential = vy*vy*radialQ[tiy+0*depth];
			vy = vy * radialQ[tiy + 1*depth] - vFrame * radialQ[tiy + 2*depth];
		}
		rho2 = rho * exp(-thermoVariables[1]*(phi1-phi0 - centripetalPotential)/T);
	}

	// Isothermal vertical balance condition:
	// rho dphi/dz = -dP/dz = -T drho/dz
	// -dphi/T = drho / rho
	// -(phi2 - phi1)/T = ln(rho2) - ln(rho1)
	// rho2 = rho1 exp(-(phi2 - phi1)/T)

	if(direct > 0) { base = base + nx*(tiy+1); } else { base = base + tiy*nx; }

	base[0] = rho2; // density falls off with altitude
	base[2*slabNumel] = rho2*vx; // "slip wall" condition in the transverse directions
	base[3*slabNumel] = rho2*vy;
	base[4*slabNumel] = rho2*vz;      // Assert no motion vertically
	E = rho2*T + .5*rho2*(vx*vx+vy*vy+vz*vz);
	base[slabNumel] = E;        // Assert a pressure condition vertically to maintain zero comoving derivative

}

// XY threads & block span the XY space, 1 Z thread
// one plane of threads
// if(direct == -1) base should be &rho[0,0,0]
// if(direct == 1) base should be &rho[0,0,nz-1 - depth]
template<int direct>
__global__ void cukern_SetFreeBalanceZ(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection, int depth, double *gravpot, double vFrame)
{
int x = threadIdx.x + blockDim.x*blockIdx.x;
int y = threadIdx.y + blockDim.y*blockIdx.y;

if((x >= nx) || (y >= ny)) return;

base += x + nx*y;
gravpot += x + nx*y;

int delta = nx*ny;
int i;
double rho, E, vx, vy, vz, T, phi0, phi1, rho2, theta;

if(direct > 0) { } else { base = base + depth*delta; }
rho = base[0];
vx = base[2*slabNumel] / rho;
vy = base[3*slabNumel] / rho;
vz = base[4*slabNumel] / rho;
E = base[slabNumel] - .5*rho*(vx*vx+vy*vy+vz*vz);
if(direct > 0) { } else { base = base - depth*delta; }

T = E / rho;
if(direct > 0) { phi0 = gravpot[0]; } else { phi0 = gravpot[depth*delta]; }

if(direct > 0) { base += delta; } else {  }

for(i = 0; i < depth; i++) {
	// load phi at current cell
	if(direct > 0) { phi1 = gravpot[(i+1)*delta]; } else { phi1 = gravpot[i*delta]; }

	if(rho < 1e-7) { phi1 = phi0; vz = (vz*direct < 0) ? 0 : vz ; }

	if(normalDirection != 1) {
			// Non-radial cases: just balance gravity potential against pressure at fixed temperature
			// Isothermal force balance condition:
			// rho dphi/dz = -dP/dz = -T drho/dz
			// -dphi/T = drho / rho
			// -(phi2 - phi1)/T = ln(rho2) - ln(rho1)
			// rho2 = rho1 exp(-(phi2 - phi1)/T)
			rho2 = rho * exp(-thermoVariables[1]*(phi1-phi0)/T);
			theta = vy;
		} else {
			// Accounts for the centripetal potential, asserting Kepler curve (omega ~ r^-1.5) extrapolated from first valid cell
			// Note, importantly, that this does not choose the correct Kepler prefactor (i.e. circular orbit),
			// merely the exponent. This is simply because the balance equation gives only one constraint, and therefore
			// two of rho, v and P must be defined & we choose v and T.
			double centripetalPotential;

			if(direct > 0) {
				theta = vy + vFrame; // return to inertial frame
				centripetalPotential = theta*theta*radialQ[i+3*depth];
				theta = theta * radialQ[i+4*depth] - vFrame * radialQ[i + 5*depth];
			} else {
				theta = vy + vFrame;
				centripetalPotential = theta*theta*radialQ[i+0*depth];
				theta = theta * radialQ[i+1*depth] - vFrame * radialQ[i + 2*depth];
			}
			rho2 = rho * exp(-thermoVariables[1]*(phi1-phi0 - centripetalPotential)/T);
		}

	base[0] = rho2; // density falls off with altitude
	base[2*slabNumel] = rho2*vx; // "slip wall" condition in the transverse directions
	base[3*slabNumel] = rho2*theta;
	base[4*slabNumel] = rho2*vz;      // Assert no motion vertically
	E = rho2*T + .5*rho2*(vx*vx+theta*theta+vz*vz);
	base[slabNumel] = E;        // Assert a pressure condition vertically to maintain zero comoving derivative

	base += delta; // move up a plane
}

}


int callBCKernel(dim3 griddim, dim3 blockdim, double *x, dim3 domainRez, int ktable, int bcDepth);

int setBoundarySAS(MGArray *phi, int side, int direction, int sas)
{
	dim3 blockdim, griddim;
	int i, sub[6];

	int returnCode;

	switch(direction) {
	case 1: { blockdim.x = phi->haloSize; blockdim.y = 16; blockdim.z = 8; } break;
	case 2: { blockdim.x = 16; blockdim.y = 1; blockdim.z = 16; } break;
	case 3: { blockdim.x = 16; blockdim.y = 16; blockdim.z = 1; } break;
	}

	// This is the easy case; We just have to apply a left-side condition to the leftmost partition and a
	// right-side condition to the rightmost partition and we're done
	if(direction == phi->partitionDir) {
		switch(direction) {
		case 1: {
			griddim.x = phi->dim[1] / blockdim.y; griddim.x += (griddim.x*blockdim.y < phi->dim[1]);
			griddim.y = phi->dim[2] / blockdim.z; griddim.y += (griddim.y*blockdim.z < phi->dim[2]);
		} break;
		case 2: {
			griddim.x = phi->dim[0] / blockdim.x; griddim.x += (griddim.x*blockdim.x < phi->dim[0]);
			griddim.y = phi->dim[2] / blockdim.z; griddim.y += (griddim.y*blockdim.z < phi->dim[2]);
		} break;
		case 3: {
			griddim.x = phi->dim[0] / blockdim.x; griddim.x += (griddim.x*blockdim.x < phi->dim[0]);
			griddim.y = phi->dim[1] / blockdim.y; griddim.y += (griddim.y*blockdim.y < phi->dim[1]);
		} break;
		}
		i = (side == 0) ? 0 : (phi->nGPUs - 1);
		cudaSetDevice(phi->deviceID[i]);
		returnCode = CHECK_CUDA_ERROR("cudaSetDevice()");
		if(returnCode != SUCCESSFUL) return returnCode;

		calcPartitionExtent(phi, i, sub);

		dim3 rez; rez.x = sub[3]; rez.y = sub[4]; rez.z = sub[5];
		returnCode = callBCKernel(griddim, blockdim, phi->devicePtr[i], rez, sas + 8*side + 16*(direction-1), phi->haloSize);
		if(returnCode != SUCCESSFUL) return returnCode;
	} else {
		// If the BC isn't on a face that's aimed in the partitioned direction,
		// we have to loop and apply it to all partitions.
		for(i = 0; i < phi->nGPUs; i++) {
			calcPartitionExtent(phi, i, sub);
			// Set the launch size based on partition extent
			switch(direction) {
			case 1: {
				griddim.x = sub[4] / blockdim.y; griddim.x += (griddim.x*blockdim.y < sub[4]);
				griddim.y = sub[5] / blockdim.z; griddim.y += (griddim.y*blockdim.z < sub[5]);
			} break;
			case 2: {
				griddim.x = sub[3] / blockdim.x; griddim.x += (griddim.x*blockdim.x < sub[3]);
				griddim.y = sub[5] / blockdim.z; griddim.y += (griddim.y*blockdim.z < sub[5]);
			} break;
			case 3: {
				griddim.x = sub[3] / blockdim.x; griddim.x += (griddim.x*blockdim.x < sub[3]);
				griddim.y = sub[4] / blockdim.y; griddim.y += (griddim.y*blockdim.y < sub[4]);
			} break;
			}
			cudaSetDevice(phi->deviceID[i]);
			returnCode = CHECK_CUDA_ERROR("cudaSetDevice()");
			if(returnCode != SUCCESSFUL) return returnCode;

			dim3 rez; rez.x = sub[3]; rez.y = sub[4]; rez.z = sub[5];
			returnCode = callBCKernel(griddim, blockdim, phi->devicePtr[i], rez, sas + 8*side + 16*(direction-1), phi->haloSize);
			if(returnCode != SUCCESSFUL) return returnCode;
		}

	}

	return SUCCESSFUL;
}

/* Sets the given array's boundary in the following manner:
   side      -> 0 = negative edge  1 = positive edge
   direction -> 1 = X	      2 = Y               3 = Z*
   sas       -> 0 = symmetrize      1 => antisymmetrize
	     -> 2 = extrap constant 3-> extrap linear
 * ktable index is computed as (0=symm, 1=antisymm) + 8*(0 = minus, 1 = plus side) + 16*(0=X, 1=Y, 2=Z)
 * As passed, assuming ImogenArray's indexPermute has been handled for us.
 * This function is just a thin veil around a switch(){}ed cukern_ invocation for setBoundarySAS.
 */
int callBCKernel(dim3 griddim, dim3 blockdim, double *x, dim3 domainRez, int ktable, int bcDepth)
{
	unsigned int nx = domainRez.x;
	unsigned int ny = domainRez.y;
	unsigned int nz = domainRez.z;
	int problem = 0;

	switch(ktable) {
	// 0-15: X direction
	case 0: cukern_xminusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 1: cukern_xminusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 2: cukern_extrapolateFlatConstBdyXMinus<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 3: cukern_extrapolateLinearBdyXMinus<<<griddim, blockdim>>>(x, nx, ny, nz); break;

	case 8: cukern_xplusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 9: cukern_xplusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 10: cukern_extrapolateFlatConstBdyXPlus<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 11: cukern_extrapolateLinearBdyXPlus<<<griddim, blockdim>>>(x, nx, ny, nz); break; // not done

	// 16-31: Y direction
	case 16: cukern_yminusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz, bcDepth); break;
	case 17: cukern_yminusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz, bcDepth); break;
	case 18: cukern_extrapolateFlatConstBdyYMinus<<<griddim, blockdim>>>(x, nx, ny, nz, bcDepth); break;
	case 19:
		problem = ERROR_NOIMPLEMENT;
		PRINT_FAULT_HEADER;
		printf("Fatal: This boundary condition (y-minus, linear) has not been implemented yet.\n");
		break;
	case 24: cukern_yplusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz, bcDepth); break;
	case 25: cukern_yplusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz, bcDepth); break;
	case 26: cukern_extrapolateFlatConstBdyYPlus<<<griddim, blockdim>>>(x, nx, ny, nz, bcDepth); break;
	case 27:
		problem = ERROR_NOIMPLEMENT;
		PRINT_FAULT_HEADER;
		printf("Fatal: This boundary condition (y-plus, linear) has not been implemented yet.\n");
		PRINT_FAULT_FOOTER; break;

	// 32-40: Z direction
	case 32: cukern_zminusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz, bcDepth); break;
	case 33: cukern_zminusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz, bcDepth); break;
	case 34: cukern_extrapolateFlatConstBdyZMinus<<<griddim, blockdim>>>(x, nx, ny, nz, bcDepth); break;
	case 35:
		problem = ERROR_NOIMPLEMENT;
		PRINT_FAULT_HEADER;
		printf("Fatal: This boundary condition (z-minus linear) has not been implemented yet.\n");
		PRINT_FAULT_FOOTER; break;

	case 40: cukern_zplusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz, bcDepth); break;
	case 41: cukern_zplusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz, bcDepth); break;
	case 42: cukern_extrapolateFlatConstBdyZPlus<<<griddim, blockdim>>>(x, nx, ny, nz, bcDepth); break;
	case 43:
		problem = ERROR_NOIMPLEMENT;
		PRINT_FAULT_HEADER;
		printf("Fatal: This boundary condition (z-plus linear) has not been implemented yet.\n");
		PRINT_FAULT_FOOTER; break;

	default:
		problem = ERROR_INVALID_ARGS;
		PRINT_FAULT_HEADER;
		printf("ERROR: callBCKernel invoked with INVALID ktable argument of %i\n", ktable);
		PRINT_FAULT_FOOTER;
		break;

	}

	if(problem == SUCCESSFUL) {
	    return CHECK_CUDA_ERROR("callBCKernel call");
	} else {
		return problem;
	}

}

/* CUDA kernel which reads through the preprocessed list of specials in statics[] and writes to
 * phi.
 * 'statics' is an array of size [Nx3] in which
 *   [N,0] contains the phi address to write,
 *   [N,1] contains the value to fade to and
 *   [N,2] contains the rate at which to fade.
 * The x length of the array is given by blkOffset; We are to use [0, nSpecials-1], and the initial
 * x offset has been taken care of by the invoking host function.
 */
__global__ void cukern_applySpecial_fade(double *phi, double *statics, int nSpecials, int blkOffset)
{
	int myAddr = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x*blockIdx.y);
	if(myAddr >= nSpecials) return;
	statics += myAddr;

	long int xaddr = (long int)statics[0];
	double f0      =	   statics[blkOffset];
	double c       =	   statics[blkOffset*2];
	double wval;

	//	if(c >= 0) {
	// Fade condition: Exponentially pulls cell towards c with rate constant f0;
	if(c != 1.0) {
		wval = f0*c + (1.0-c)*phi[xaddr];
		if(isnan(wval)) { // Should almost never be triggered!!!!
			wval = f0*c;
		}
		phi[xaddr] = wval;
	} else {
		phi[xaddr] = f0;
	}
	//	} else {
	// Wall condition: Any transfer between the marked cells is reversed
	// Assumptions: 2nd cell (xprimeaddr) must be in a stationary, no-flux region
	//		long int xprimeaddr = (long int) statics[myAddr + blkOffset*3];
	//		phi[xaddr] += (phi[xprimeaddr]-f0);
	//		phi[xprimaddr] = f0;
	//	}

}

/* This is where we finally meet the metal - these are the actual kernels that R/W the boundaries
 * to assert various boundary extrapolation functions.
 */


/* X DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS FOR MIRROR BCS */
/* Assume a block size of [3 A B] with grid dimensions [M N 1] s.t. AM >= ny, BN >= nz*/
/* Define the preamble common to all of these kernels: */
#define XSASKERN_PREAMBLE \
		int stridey = nx; int stridez = nx*ny; \
		int yidx = threadIdx.y + blockIdx.x*blockDim.y; \
		int zidx = threadIdx.z + blockIdx.y*blockDim.z; \
		if(yidx >= ny) return; if(zidx >= nz) return; \
		phi += stridey*yidx + stridez*zidx;

/* We establish symmetry or antisymmetry such that we have 
 * [... A B C D  C  B  A|-> BOUNDARY
 * [... A B C D -C -B -A|-> BOUNDARY 
 * i.e. symmetry is about the 4th cell from the boundary */
// X direction kernels just use 3 threads in order to acheive slightly less terrible
// memory access patterns

#define MIRROR_OFFSET 1

// The X directions we can use blockSize.x to tell us the halo depth
__global__ void cukern_xminusSymmetrize(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	int depth = blockDim.x;
	phi[depth-1-threadIdx.x] = phi[depth+MIRROR_OFFSET+threadIdx.x];
}

__global__ void cukern_xminusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	int depth = blockDim.x;
	phi[depth-1-threadIdx.x] = -phi[depth+MIRROR_OFFSET+threadIdx.x];
}

__global__ void cukern_xplusSymmetrize(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	int depth = blockDim.x;
	phi[nx-depth+threadIdx.x] = phi[nx-depth-1-MIRROR_OFFSET-threadIdx.x];
}

__global__ void cukern_xplusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	int depth = blockDim.x;
	phi[nx-depth+threadIdx.x] = -phi[nx-depth-1-MIRROR_OFFSET-threadIdx.x];
}

/* These are called when a BC is set to 'const' or 'linear' */

// Extrapolate a constant boundary on the -x side of a cartesian block
__global__ void cukern_extrapolateFlatConstBdyXMinus(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	int depth = blockDim.x;
	phi[threadIdx.x] = phi[depth];
}

// oh dear... not implemented...
__constant__ int radExtrapCoeffs[5];
#define RAD_EX_X0 
#define RAD_EX_Y0
#define RAD_EX_Z0

// Extrapolate a constant boundary on the -r side of a cylindrical block
__global__ void cukern_extrapolateRadialConstBdyXMinus(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE

	// 	
}

// Extrapolate a constant boundary on the +r side of a cartesian block
__global__ void cukern_extrapolateFlatConstBdyXPlus(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	int depth = blockDim.x;
	phi[nx-depth+threadIdx.x] = phi[nx-depth-1];
}

// NOTE:
// These conditions are GENERALLY UNSTABLE and result in catastrophic backflow
// 
// fixme: Oh dear this is catastrophically broken in its use of shmem. I am stupid
__global__ void cukern_extrapolateLinearBdyXMinus(double *phi, int nx, int ny, int nz)
{
	__shared__ double f[4];
	XSASKERN_PREAMBLE
	int depth = blockDim.x;
	f[threadIdx.x] = phi[threadIdx.x+depth];
	__syncthreads();
	phi[threadIdx.x] = phi[depth] + (depth-threadIdx.x)*(f[0]-f[1]);

	// Correct procedure:
	// Compute the local gradient at the boundary,
	// Compute the characteristic transform,
	// Eliminate the unacceptable (flowing-onto-the-grid) characteristic,
	// Convert back,
	// And use those gradients to backfill the boundary instead
	// However this requires a much "smarter" notion of boundaries than Imogen presently has.
	//
	// Assuming an acceptable method has yielded [rho'; v'; P'], the transform
	// |rho'_out|   / 1, (L-R) rho/2c , (L/2+R/2-1)/c^2\|rho'|
	// |v'_out  | = | 0, (L+R)/2      , (L-R)/2c rho   ||v'  |
	// |P'_out  |   \ 0, (L-R) c rho/2, (L+R)/2        /|P'  |
	// Wherein at the right boundary R=0 and L=1, and at the left boundary R=1 and L=0,
	// will eliminate (to order h^2) the characteristic entering the grid (which of course
	// we can't know without extending the grid)

	// Note: This calculation needs to be redone in the frame which is NOT comoving with
	// the fluid to be used on an eulerian grid which is also not in general comoving.
	// The general procedure:
	// We multiply the column vector of primitive gradients, by the inverse of the matrix
	// of eigenvectors, by diag([R,1,1,1,L]), by the matrix of eigenvectors
	// That product of matrices, in the 1D case, for the comoving frame, is the one above
}


__global__ void cukern_extrapolateLinearBdyXPlus(double *phi, int nx, int ny, int nz)
{
	__shared__ double f[4];
	XSASKERN_PREAMBLE
	int depth = blockDim.x;
	phi += nx - depth - 1;
	f[threadIdx.x] = phi[threadIdx.x];
	__syncthreads();
	phi[threadIdx.x+2] = f[1] + (threadIdx.x+1)*(f[1]-f[0]);
}


/* Y DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS */
/* assume a block size of [A 1 B] with grid dimensions [M N 1] s.t. AM >= nx, BN >=nz */
#define YSASKERN_PREAMBLE \
		int stridez = nx*ny; \
		int xidx = threadIdx.x + blockIdx.x*blockDim.x; \
		int zidx = threadIdx.z + blockIdx.y*blockDim.y; \
		if(xidx >= nx) return; if(zidx >= nz) return;   \
		phi += xidx + stridez*zidx; \
		int q;


__global__ void cukern_yminusSymmetrize(double *phi, int nx, int ny, int nz, int depth)
{
	YSASKERN_PREAMBLE

	for(q = 0; q < depth; q++) { phi[nx*q] = phi[nx*(2*depth-1+MIRROR_OFFSET-q)]; }
}

__global__ void cukern_yminusAntisymmetrize(double *phi, int nx, int ny, int nz, int depth)
{
	YSASKERN_PREAMBLE
	for(q = 0; q < depth; q++) { phi[nx*q] = -phi[nx*(2*depth-1+MIRROR_OFFSET-q)]; }
}

__global__ void cukern_yplusSymmetrize(double *phi, int nx, int ny, int nz, int depth)
{
	YSASKERN_PREAMBLE
	for(q = 0; q < depth; q++) { phi[nx*(ny-1-q)] = phi[nx*(ny-MIRROR_OFFSET-2*depth+q)]; }
}

__global__ void cukern_yplusAntisymmetrize(double *phi, int nx, int ny, int nz, int depth)
{
	YSASKERN_PREAMBLE
	for(q = 0; q < depth; q++) { phi[nx*(ny-1-q)] = -phi[nx*(ny-MIRROR_OFFSET-2*depth+q)]; }
}

__global__ void cukern_extrapolateFlatConstBdyYMinus(double *phi, int nx, int ny, int nz, int depth)
{
	YSASKERN_PREAMBLE
	double f = phi[depth*nx];
	for(q = 0; q < depth; q++) { phi[q*nx] = f; }
}

__global__ void cukern_extrapolateFlatConstBdyYPlus(double *phi, int nx, int ny, int nz, int depth)
{
	YSASKERN_PREAMBLE
	double f = phi[(ny-depth)*nx];
	for(q = 0; q < depth; q++) { phi[(ny-depth+q)*nx] = f; }
}

__global__ void cukern_yminusDoubleMach(double *phi, int nx, int ny, int nz, int depth)
{

}

/* Z DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS */
/* Assume launch with size [A B 1] and grid of size [M N 1] s.t. AM >= nx, BN >= ny*/
#define ZSASKERN_PREAMBLE \
		int xidx = threadIdx.x + blockIdx.x * blockDim.x; \
		int yidx = threadIdx.y + blockIdx.y * blockDim.y; \
		if(xidx >= nx) return; if(yidx >= ny) return; \
		phi += xidx + nx*yidx; \
		int stride = nx*ny;

__global__ void cukern_zminusSymmetrize(double *phi, int nx, int ny, int nz, int depth)
{
	ZSASKERN_PREAMBLE

	int q;
	for(q = 0; q < depth; q++) // nvcc will unroll it
		phi[q*stride] = phi[(2*depth+MIRROR_OFFSET-q)*stride];
}

__global__ void cukern_zminusAntisymmetrize(double *phi, int nx, int ny, int nz, int depth)
{
	ZSASKERN_PREAMBLE
	int q;
	for(q = 0; q < depth; q++)
		phi[q*stride] = -phi[(2*depth+MIRROR_OFFSET-q)*stride];
}

__global__ void cukern_zplusSymmetrize(double *phi, int nx, int ny, int nz, int depth)
{
	ZSASKERN_PREAMBLE
	int q;
	for(q = 0; q < depth; q++)
		phi[stride*(nz-1-q)] = phi[stride*(nz-MIRROR_OFFSET-2*depth+q)];
}

__global__ void cukern_zplusAntisymmetrize(double *phi, int nx, int ny, int nz, int depth)
{
	ZSASKERN_PREAMBLE
	int q;
	for(q = 0; q < depth; q++)
		phi[stride*(nz-1-q)] = -phi[stride*(nz-MIRROR_OFFSET-2*depth+q)];

}

__global__ void cukern_extrapolateFlatConstBdyZMinus(double *phi, int nx, int ny, int nz, int depth)
{
	ZSASKERN_PREAMBLE
	int q;
	for(q = 0; q < depth; q++)
		phi[stride*q] = phi[stride*depth];
}

__global__ void cukern_extrapolateFlatConstBdyZPlus(double *phi, int nx, int ny, int nz, int depth)
{

	ZSASKERN_PREAMBLE
	int q;
	for(q = 0; q < depth; q++)
		phi[stride*(nz-1-q)] = phi[stride*(nz-depth)];

}
