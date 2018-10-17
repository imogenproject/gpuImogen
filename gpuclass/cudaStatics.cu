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

#include "cudaCommon.h"
#include "cudaStatics.h"

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
__global__ void cukern_yminusSymmetrize(double *phi, int nx, int ny, int nz);
__global__ void cukern_yminusAntisymmetrize(double *phi, int nx, int ny, int nz);
__global__ void cukern_yplusSymmetrize(double *phi, int nx, int ny, int nz);
__global__ void cukern_yplusAntisymmetrize(double *phi, int nx, int ny, int nz);
/* Z DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS */
/* Assume launch with size [U V 1] */
__global__ void cukern_zminusSymmetrize(double *Phi, int nx, int ny, int nz);
__global__ void cukern_zminusAntisymmetrize(double *Phi, int nx, int ny, int nz);
__global__ void cukern_zplusSymmetrize(double *Phi, int nx, int ny, int nz);
__global__ void cukern_zplusAntisymmetrize(double *Phi, int nx, int ny, int nz);

/* X direction extrapolated boundary conditions */
/* Launch size [3 A B] */
__global__ void cukern_extrapolateLinearBdyXMinus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateLinearBdyXPlus(double *phi, int nx, int ny, int nz);

__global__ void cukern_extrapolateFlatConstBdyXMinus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateFlatConstBdyXPlus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateFlatConstBdyYMinus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateFlatConstBdyYPlus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateFlatConstBdyZMinus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateFlatConstBdyZPlus(double *phi, int nx, int ny, int nz);

__global__ void cukern_applySpecial_fade(double *phi, double *statics, int nSpecials, int blkOffset);

int setOutflowCondition(MGArray *fluid, GeometryParams *geom, int rightside, int memdir);
template <int direct>
__global__ void cukern_SetOutflowX(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection);
template <int direct>
__global__ void cukern_SetOutflowY(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection);
template <int direct>
__global__ void cukern_SetOutflowZ(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection);

int setBoundarySAS(MGArray *phi, int side, int direction, int sas);

__constant__ __device__ double restFrmSpeed[6];

#ifdef STANDALONE_MEX_FUNCTION

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if( (nlhs != 0) || (nrhs != 4)) { mexErrMsgTxt("cudaStatics operator is cudaStatics(ImogenArray, blockdim, GeometryManager, direction)"); }

	CHECK_CUDA_ERROR("entering cudaStatics");

	GeometryParams geom = accessMatlabGeometryClass(prhs[2]);

	int worked = setBoundaryConditions(NULL, prhs[0], &geom, (int)*mxGetPr(prhs[3]));
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) {
		DROP_MEX_ERROR("setBoundaryCondition called from standalone has crashed: Crashing interpreter.");
	}
}
#endif

int setFluidBoundary(MGArray *fluid, const mxArray *matlabhandle, GeometryParams *geo, int direction)
{
	CHECK_CUDA_ERROR("entering setBoundaryConditions");

	MGArray phi, statics;
	int worked;
	if(fluid == NULL) {
		worked = MGA_accessMatlabArrays((const mxArray **)&matlabhandle, 0, 0, &phi);
		BAIL_ON_FAIL(worked)
	} else {
		worked = (fluid->matlabClassHandle == matlabhandle) ? SUCCESSFUL : ERROR_CRASH;
		if(worked != SUCCESSFUL) {
			PRINT_FAULT_HEADER;
			printf("setBoundaryConditions permits both the MGArray and its Matlab handle to be passed because the MGA may have been internally modified without the handle having been, but the MGArray must name that Matlab handle as its originator.\n");
			PRINT_FAULT_FOOTER;
			BAIL_ON_FAIL(worked);
		}
		phi = fluid[0];
	}

	/* Grabs the whole boundaryData struct from the ImogenArray class */
	mxArray *boundaryData = mxGetProperty(matlabhandle, phi.mlClassHandleIndex, "boundaryData");
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

	int *perm = &phi.currentPermutation[0];
	int offsetidx = 2*(perm[0]-1) + 1*(perm[1] > perm[2]);

	/* The offset array describes the index offsets for the data in the gpuStatics array */
	mxArray *offsets    = mxGetField(boundaryData, 0, "compOffset");
	if(offsets == NULL) {
		PRINT_SIMPLE_FAULT("FATAL: field 'compOffset' D.N.E. in boundaryData. Not an ImogenArray? Statics not compiled?\n");
		return ERROR_INVALID_ARGS;
	}
	double *offsetcount = mxGetPr(offsets);
	long int staticsOffset = (long int)offsetcount[2*offsetidx];
	int staticsNumel  = (int)offsetcount[2*offsetidx+1];

	/* Parameter describes what block size to launch with... */
	int blockdim = 32;

	dim3 griddim; griddim.x = staticsNumel / blockdim + 1;
	if(griddim.x > 1024) {
		griddim.x = 1024;
		griddim.y = staticsNumel/(blockdim*griddim.x) + 1;
	}

	/* Every call results in applying specials */
	if(staticsNumel > 0) {
		//PAR_WARN(phi);
		cukern_applySpecial_fade<<<griddim, blockdim>>>(phi.devicePtr[0], statics.devicePtr[0] + staticsOffset, staticsNumel, statics.dim[0]);
		worked = CHECK_CUDA_LAUNCH_ERROR(blockdim, griddim, &phi, 0, "cuda statics application");
		if(worked != SUCCESSFUL) return worked;
	}

	mxArray *bcModes = mxGetField(boundaryData, 0, "bcModes");
	if(bcModes == NULL) {
		PRINT_SIMPLE_FAULT("FATAL: bcModes structure not present. Not an ImogenArray? Not initialized?\n");
		return ERROR_INVALID_ARGS;
	}


	if((direction < 1) || (direction > 3)) {
		PRINT_FAULT_HEADER;
		printf("Direction passed to cudaStatics was %i which is not one of 1 (X/R) 2 (Y/theta) 3 (Z)", direction);
		PRINT_FAULT_FOOTER;
		return ERROR_INVALID_ARGS;
	}

	/* So this is utterly stupid, but the boundary condition modes are stored in the form
       { 'type minus x', 'type minus y', 'type minus z';
	 'type plus  x', 'type plus y',  'type plus z'};
       Yes, strings in a cell array. No I'm not fixing it, because I'm not diving into that pile of burning garbage. */

	mxArray *bcstr; char *bs;

	// FIXME: okay here's the shitty way we're doing this,
	// FIXME: If the density array says 'outflow', we call setOutflowCondition once.
	// FIXME: If not, we loop

	int d; // coordinate face: 0 = negative, 1 = positive
	for(d = 0; d < 2; d++) {
		bcstr = mxGetCell(bcModes, 2*(direction-1) + d);
		bs = (char *)malloc(sizeof(char) * (mxGetNumberOfElements(bcstr)+1));
		mxGetString(bcstr, bs, mxGetNumberOfElements(bcstr)+1);

		if(strcmp(bs,"outflow") == 0) {
			worked = setOutflowCondition(&phi, geo, d, direction);
		} else {
			int i;
			for(i = 0; i < 5; i++) {
				worked = setBoundaryConditions(fluid+i, fluid[i].matlabClassHandle, geo, direction);
				if(worked != SUCCESSFUL) break;
			}
		}
		free(bs);
		if(worked != SUCCESSFUL) break;
	}

	return CHECK_IMOGEN_ERROR(worked);
}

int setBoundaryConditions(MGArray *array, const mxArray *matlabhandle, GeometryParams *geo, int direction)
{
	CHECK_CUDA_ERROR("entering setBoundaryConditions");

	MGArray phi, statics;
	int worked;
	if(array == NULL) {
		worked = MGA_accessMatlabArrays((const mxArray **)&matlabhandle, 0, 0, &phi);
		BAIL_ON_FAIL(worked)
	} else {
		worked = (array->matlabClassHandle == matlabhandle) ? SUCCESSFUL : ERROR_CRASH;
		if(worked != SUCCESSFUL) {
			PRINT_FAULT_HEADER;
			printf("setBoundaryConditions permits both the MGArray and its Matlab handle to be passed because the MGA may have been internally modified without the handle having been, but the MGArray must name that Matlab handle as its originator.\n");
			PRINT_FAULT_FOOTER;
			BAIL_ON_FAIL(worked);
		}
		phi = array[0];
	}

	/* Grabs the whole boundaryData struct from the ImogenArray class */
	mxArray *boundaryData = mxGetProperty(matlabhandle, phi.mlClassHandleIndex, "boundaryData");
	if(boundaryData == NULL) {
		PRINT_SIMPLE_FAULT("FATAL: field 'boundaryData' D.N.E. in class. Not a class? Not an ImogenArray/FluidArray?\n");
		return ERROR_INVALID_ARGS;
	}

	/* The statics describe "solid" structures which we force the grid to have */
	mxArray *gpuStatics = mxGetField(boundaryData, 0, "staticsData");
	if(gpuStatics == NULL) {
		PRINT_SIMPLE_FAULT("FATAL: field 'staticsData' D.N.E. in boundaryData struct. Statics not compiled?\n");
		return ERROR_INVALID_ARGS;
	}
	worked = MGA_accessMatlabArrays((const mxArray **)(&gpuStatics), 0, 0, &statics);
	BAIL_ON_FAIL(worked)

	int *perm = &phi.currentPermutation[0];
	int offsetidx = 2*(perm[0]-1) + 1*(perm[1] > perm[2]);

	/* The offset array describes the index offsets for the data in the gpuStatics array */
	mxArray *offsets    = mxGetField(boundaryData, 0, "compOffset");
	if(offsets == NULL) {
		PRINT_SIMPLE_FAULT("FATAL: field 'compOffset' D.N.E. in boundaryData. Not an ImogenArray? Statics not compiled?\n");
		return ERROR_INVALID_ARGS;
	}
	double *offsetcount = mxGetPr(offsets);
	long int staticsOffset = (long int)offsetcount[2*offsetidx];
	int staticsNumel  = (int)offsetcount[2*offsetidx+1];

	/* Parameter describes what block size to launch with... */
	int blockdim = 32;

	dim3 griddim; griddim.x = staticsNumel / blockdim + 1;
	if(griddim.x > 1024) {
		griddim.x = 1024;
		griddim.y = staticsNumel/(blockdim*griddim.x) + 1;
	}

	/* Every call results in applying specials */
	if(staticsNumel > 0) {
//		PAR_WARN(phi);
		cukern_applySpecial_fade<<<griddim, blockdim>>>(phi.devicePtr[0], statics.devicePtr[0] + staticsOffset, staticsNumel, statics.dim[0]);
		worked = CHECK_CUDA_LAUNCH_ERROR(blockdim, griddim, &phi, 0, "cuda statics application");
		if(worked != SUCCESSFUL) return worked;
	}


	int vectorComponent = phi.vectorComponent;

	/* BEGIN DETERMINATION OF ANALYTIC BOUNDARY CONDITIONS */
	int numDirections = 1;

	mxArray *bcModes = mxGetField(boundaryData, 0, "bcModes");
	if(bcModes == NULL) {
		PRINT_SIMPLE_FAULT("FATAL: bcModes structure not present. Not an ImogenArray? Not initialized?\n");
		return ERROR_INVALID_ARGS;
	}

	int j;
	for(j = 0; j < numDirections; j++) {
		if(direction == 0) continue; /* Skips edge BCs if desired. */
		int memoryDirection = MGA_dir2memdir(perm, direction);

		/* So this is kinda brain-damaged, but the boundary condition modes are stored in the form
       { 'type minus x', 'type minus y', 'type minus z';
	 'type plus  x', 'type plus y',  'type plus z'};
       Yes, strings in a cell array. */
		/* Okay, that's not kinda, it's straight-up stupid. */

		mxArray *bcstr; char *bs;

		int d; for(d = 0; d < 2; d++) {
			bcstr = mxGetCell(bcModes, 2*(direction-1) + d);
			bs = (char *)malloc(sizeof(char) * (mxGetNumberOfElements(bcstr)+1));
			mxGetString(bcstr, bs, mxGetNumberOfElements(bcstr)+1);

			// Sets a mirror BC: scalar, vector_perp f(b+x) = f(b-x), vector normal f(b+x) = -f(b-x)
			if(strcmp(bs, "mirror") == 0)
				worked = setBoundarySAS(&phi, d, memoryDirection, vectorComponent == direction);

			// Extrapolates f(b+x) = f(b)
			if(strcmp(bs, "const") == 0) {
				worked = setBoundarySAS(&phi, d, memoryDirection, 2);
			}

			// Extrapolates f(b+x) = f(b) + x f'(b)
			// WARNING: This is unconditionally unstable unless normal flow rate is supersonic
			if(strcmp(bs, "linear") == 0) {
				worked = setBoundarySAS(&phi, d, memoryDirection, 3);
			}

			if(strcmp(bs, "wall") == 0) {
				PRINT_FAULT_HEADER;
				printf("Wall BC is not implemented!\n");
				PRINT_FAULT_FOOTER;
				worked = ERROR_INVALID_ARGS;
			}

			if(strcmp(bs, "outflow") == 0) {
				if(phi.numSlabs == 0) { // primitive check that this in fact the rho array setOutflowCondition needs...
					worked = setOutflowCondition(&phi, geo, d, direction);
				}
			}
			// whatever we just did, check...
			if(worked != SUCCESSFUL) break;

		}
		if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) return worked;
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
 * boundary = (v_normal > 0) ? constant : mirror
 * This must be passed ONLY the density array (FIXME: awful testing hack)
 */
int setOutflowCondition(MGArray *fluid, GeometryParams *geom, int rightside, int direction)
{
	dim3 blockdim, griddim;
	int i;
	int sub[6];

	int status;

	int memdir = MGA_dir2memdir(&fluid->currentPermutation[0], direction);

	double rfVelocity[6];
	// When setting the radial face for outflow and the frame is rotating,
	// the inertial rest frame zero (imposed if inward flow is attempted)
	// is not zero in the rotating frame:

	for(i = 0; i < 3; i++) { rfVelocity[i] = -(geom->Rinner + i*geom->h[0])*geom->frameOmega; }
	for(i = 3; i < 6; i++) { rfVelocity[i] = -(geom->Rinner + (fluid->dim[0]-6+i)*geom->h[0])*geom->frameOmega;  }

	for(i = 0; i < fluid->nGPUs; i++) {
		// Prevent BC from being done to internal partition boundaries if we are partitioned in this direction
		if(doBCForPart(fluid, i, PARTITION_X, rightside)) {
			cudaSetDevice(fluid->deviceID[i]);
			cudaMemcpyToSymbol((const void *)restFrmSpeed, &rfVelocity[0], 6*sizeof(double), 0, cudaMemcpyHostToDevice);
		}
	}

	status = CHECK_CUDA_ERROR("__constant__ uploads for outflow condition");
	if(status != SUCCESSFUL) return status;

	switch(memdir) {
	case 1: // x
		blockdim.x = 3; blockdim.y = blockdim.z = 16;

		for(i = 0; i < fluid->nGPUs; i++) {
			if(doBCForPart(fluid, i, PARTITION_X, rightside) == 0) continue;

			cudaSetDevice(fluid->deviceID[i]);
			calcPartitionExtent(fluid, i, &sub[0]);

			griddim.x = ROUNDUPTO(sub[4], 16)/16;
			griddim.y = ROUNDUPTO(sub[5], 16)/16;

			if(rightside) {
				double *base = fluid->devicePtr[i] + sub[3] - 4;
				cukern_SetOutflowX<1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction);
			} else {
				double *base = fluid->devicePtr[i] + 3;
				cukern_SetOutflowX<-1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction);
			}
		}
		break;
	case 2: // y
		blockdim.x = 32; blockdim.y = 3; blockdim.z = 1;
		for(i = 0; i < fluid->nGPUs; i++) {
			if(doBCForPart(fluid, i, PARTITION_Y, rightside) == 0) continue;
			cudaSetDevice(fluid->deviceID[i]);
			calcPartitionExtent(fluid, i, &sub[0]);

			griddim.x = ROUNDUPTO(sub[4], 32)/32;
			griddim.y = sub[5];
			griddim.z = 1;

			if(rightside) {
				double *base = fluid->devicePtr[i] + (sub[4]-7)*sub[3];
				cukern_SetOutflowY<1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction);
			} else {
				cukern_SetOutflowY<-1><<<griddim, blockdim>>>(fluid->devicePtr[i], sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction);
			}
		}
		break;
	case 3: // z
		blockdim.x = 32; blockdim.y = 4; blockdim.z = 3;
		for(i = 0; i < fluid->nGPUs; i++) {
			if(doBCForPart(fluid, i, PARTITION_Z, rightside) == 0) continue;

			cudaSetDevice(fluid->deviceID[i]);
			calcPartitionExtent(fluid, i, &sub[0]);
			griddim.x = ROUNDUPTO(sub[3], blockdim.x)/blockdim.x;
			griddim.y = ROUNDUPTO(sub[4], blockdim.y)/blockdim.y;
			griddim.z = 1;

			if(rightside) {
				double *base = fluid->devicePtr[i] + sub[3]*sub[4]*(sub[5]-7);
				cukern_SetOutflowZ<1><<<griddim, blockdim>>>(base, sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction);
			} else {
				cukern_SetOutflowZ<-1><<<griddim, blockdim>>>(fluid->devicePtr[i], sub[3], sub[4], sub[5], (int32_t)fluid->slabPitch[i]/8, direction);
			}
		}
		break;
	}

	status = CHECK_CUDA_LAUNCH_ERROR(blockdim, griddim, fluid, direction, "Setting outflow BC on array");
	return status;
}

// given original value
__device__ double outflowBC_radial_vel(double p0, int normdir, int i)
{
	return 0.0;
}

__device__ double outflowBC_phi_vel(double v0, int normdir, int i)
{
	if(normdir == 1) { return restFrmSpeed[i]; } else { return v0; }
}

__device__ double outflowBC_zee_vel(double v0, int normdir, int i)
{
	return 0.0;
}

// Assume we are passed &rho(3,0,0) if direct == 0
// assume we are passed &rho(nx-4,0,0) if direct == 1
// Launch an 3xAxB block with an NxM grid such that AN >= ny and BM >= nz
template <int direct>
__global__ void cukern_SetOutflowX(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection) // FIXME slabNumel should be int64 type
{
	int tix = threadIdx.x;

	int y = threadIdx.y + blockDim.y*blockIdx.x;
	int z = threadIdx.z + blockDim.z*blockIdx.y;
	if((y >= ny) || (z >= nz)) return;

	// Y-Z translate
	base += nx*(y+ny*z);

	double a;

	a = base[(1+normalDirection)*slabNumel];

	double rho, E, p1, p2;

	if(a*direct > 0) {
		// boundary normal momentum positive: extrapolate as constant
		if(direct == 1) { // +x: base = &rho[nx-4, 0 0]
			tix++;
			a = base[0]; base[tix] = a; base += slabNumel;
			a = base[0]; base[tix] = a; base += slabNumel;
			a = base[0]; base[tix] = a; base += slabNumel;
			a = base[0]; base[tix] = a; base += slabNumel;
			a = base[0]; base[tix] = a; base += slabNumel;
		} else { // -x: base = &rho[3,0,0]
			base = base - 3; // move back to zer0
			a = base[3]; base[tix] = a; base += slabNumel;
			a = base[3]; base[tix] = a; base += slabNumel;
			a = base[3]; base[tix] = a; base += slabNumel;
			a = base[3]; base[tix] = a; base += slabNumel;
			a = base[3]; base[tix] = a; base += slabNumel;
		}
	} else {
		// boundary normal momentum negative: null normal velocity, otherwise mirror
		// phi velocity
		if(direct == 1) { // +x
			//base = base - 3;
			rho = a = base[0]; base[1+tix] = a; base += slabNumel; // rho
			E =       base[0]; base += slabNumel; // E

			a = base[0]; p1 = a; p2 = rho*outflowBC_radial_vel(a/rho, normalDirection, 3+tix); base[1+tix] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[0]; p1 = a; p2 = rho*outflowBC_phi_vel(a/rho, normalDirection, 3+tix); base[1+tix] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[0]; p1 = a; p2 = rho*outflowBC_zee_vel(a/rho, normalDirection, 3+tix); base[1+tix] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho;
			base[1+tix-3*slabNumel] = E;
		} else { // -x
			base=base-3;

			rho = a = base[3]; base[tix] = a; base += slabNumel; // rho
			E =       base[3]; base += slabNumel; // E

			a = base[3]; p1 = a; p2 = rho*outflowBC_radial_vel(a/rho, normalDirection, tix); base[tix] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[0]; p1 = a; p2 = rho*outflowBC_phi_vel(a/rho, normalDirection, tix); base[tix] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[0]; p1 = a; p2 = rho*outflowBC_zee_vel(a/rho, normalDirection, tix); base[tix] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho;
			base[tix-3*slabNumel] = E;
		}

		// The potential for evacuating flows created by this mode of operation can result in crappy behavior
		// We need to re-enforce
#ifdef ENFORCE_FLUIDMIN

#endif

	}

}

// Use 32x3 threads and dim3(roundup(nx/32), nz, 1) blocks
// if(direct == -1) {minus coordinate edge} assume base is &rho[0]
// if(direct == 1)  {plus coordinate edge}  assume base is &rho[0,ny-7,0]
template <int direct>
__global__ void cukern_SetOutflowY(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int z = blockIdx.y;
	int tiy = threadIdx.y;

	if((x >= nx) || (z >= nz)) return;

	// Y-Z translate
	base += x + nx*(ny*z);

	double a, rho, E, p1, p2;

	a = base[(1+normalDirection)*slabNumel];

	if(a*direct > 0) {
		// normal mom is positive: extrapolate constant
		if(direct == 1) { // +y edge
			a = base[3*nx]; base[(4+tiy)*nx] = a; base += slabNumel;
			a = base[3*nx]; base[(4+tiy)*nx] = a; base += slabNumel;
			a = base[3*nx]; base[(4+tiy)*nx] = a; base += slabNumel;
			a = base[3*nx]; base[(4+tiy)*nx] = a; base += slabNumel;
			a = base[3*nx]; base[(4+tiy)*nx] = a; base += slabNumel;
		} else { // -y edge
			a = base[3*nx]; base[(tiy)*nx] = a; base += slabNumel;
			a = base[3*nx]; base[(tiy)*nx] = a; base += slabNumel;
			a = base[3*nx]; base[(tiy)*nx] = a; base += slabNumel;
			a = base[3*nx]; base[(tiy)*nx] = a; base += slabNumel;
			a = base[3*nx]; base[(tiy)*nx] = a; base += slabNumel;
		}
	} else {
		// normal mom negative: set norm momentum to null
		if(direct == 1) { // +y edge
			rho = a = base[3*nx]; base[(4+tiy)*nx] = a; base += slabNumel; // rho
			E =       base[3*nx]; base += slabNumel; // E

			a = base[3*nx]; p1 = a; p2 = rho*outflowBC_radial_vel(a/rho, normalDirection, 3+tiy); base[(4+tiy)*nx] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[3*nx]; p1 = a; p2 = rho*outflowBC_phi_vel(a/rho, normalDirection, 3+tiy); base[(4+tiy)*nx] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[3*nx]; p1 = a; p2 = rho*outflowBC_zee_vel(a/rho, normalDirection, 3+tiy); base[(4+tiy)*nx] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho;
			base[(4+tiy)*nx-3*slabNumel] = E;
		} else { // -y edge
			rho = a = base[3*nx]; base[(4+tiy)*nx] = a; base += slabNumel; // rho
			E =       base[3*nx]; base += slabNumel; // E

			a = base[3*nx]; p1 = a; p2 = rho*outflowBC_radial_vel(a/rho, normalDirection, tiy); base[(tiy)*nx] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[3*nx]; p1 = a; p2 = rho*outflowBC_phi_vel(a/rho, normalDirection, tiy); base[(tiy)*nx] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
			a = base[3*nx]; p1 = a; p2 = rho*outflowBC_zee_vel(a/rho, normalDirection, tiy); base[(tiy)*nx] = p2;
			E += .5*(p2-p1)*(p2+p1)/rho;
			base[(tiy)*nx-3*slabNumel] = E;
		}
	}

}

// XY threads & block span the XY space, 3 Z threads 
// one plane of threads
// if(direct == -1) base should be &rho[0,0,0]
// if(direct == 1) base should be &rho[0,0,nz-7]
template<int direct>
__global__ void cukern_SetOutflowZ(double *base, int nx, int ny, int nz, int slabNumel, int normalDirection)
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
	// positive normal momentum: constant
	if(direct == 1) { // +z edge
		for(i = 0; i < 5; i++) { a = base[3*delta]; base[(4+tiz)*delta] = a; base += slabNumel; }
	} else { // -z edge
		for(i = 0; i < 5; i++) { a = base[3*delta]; base[tiz*delta] = a; base += slabNumel; }
	}
} else {
	// negative normal momentum: set normal mom to null
	if(direct == 1) { // +z edge
		rho = a = base[3*delta]; base[(3+tiz)*delta] = a; base += slabNumel;
		E = base[3*delta]; base += slabNumel;

		a = base[3*delta]; p1 = a; p2 = rho*outflowBC_radial_vel(a/rho, normalDirection, 3+tiz); base[(4+tiz)*delta] = p2;
		E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
		a = base[3*delta]; p1 = a; p2 = rho*outflowBC_phi_vel(a/rho, normalDirection, 3+tiz); base[(4+tiz)*delta] = p2;
		E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
		a = base[3*delta]; p1 = a; p2 = rho*outflowBC_zee_vel(a/rho, normalDirection, 3+tiz); base[(4+tiz)*delta] = p2;
		E += .5*(p2-p1)*(p2+p1)/rho;
		base[(4+tiz)*delta-3*slabNumel] = E;
	} else { // -z edge : defective at present
		rho = a = base[3*delta]; base[(tiz)*delta] = a; base += slabNumel;
		E = base[3*delta]; base += slabNumel;

		a = base[3*delta]; p1 = a; p2 = rho*outflowBC_radial_vel(a/rho, normalDirection, tiz); base[(tiz)*delta] = p2;
		E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
		a = base[3*delta]; p1 = a; p2 = rho*outflowBC_phi_vel(a/rho, normalDirection, tiz); base[(tiz)*delta] = p2;
		E += .5*(p2-p1)*(p2+p1)/rho; base += slabNumel;
		a = base[3*delta]; p1 = a; p2 = rho*outflowBC_zee_vel(a/rho, normalDirection, tiz); base[(tiz)*delta] = p2;
		E += .5*(p2-p1)*(p2+p1)/rho;
		base[tiz*delta - 3*slabNumel] = E;
	}
}

}


/* Sets the given array's boundary in the following manner:
   side      -> 0 = negative edge  1 = positive edge
   direction -> 1 = X	      2 = Y               3 = Z*
   sas       -> 0 = symmetrize      1 => antisymmetrize
	     -> 2 = extrap constant 3-> extrap linear

 *: As passed, assuming ImogenArray's indexPermute has been handled for us.
 */
int callBCKernel(dim3 griddim, dim3 blockdim, double *x, dim3 domainRez, int ktable)
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
	case 11: cukern_extrapolateLinearBdyXPlus<<<griddim, blockdim>>>(x, nx, ny, nz); break;

	// 16-31: Y direction
	case 16: cukern_yminusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 17: cukern_yminusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 18: cukern_extrapolateFlatConstBdyYMinus<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 19:
		problem = ERROR_NOIMPLEMENT;
		PRINT_FAULT_HEADER;
		printf("Fatal: This boundary condition (y-minus, linear) has not been implemented yet.\n");
		break;
	case 24: cukern_yplusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 25: cukern_yplusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 26: cukern_extrapolateFlatConstBdyYPlus<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 27:
		problem = ERROR_NOIMPLEMENT;
		PRINT_FAULT_HEADER;
		printf("Fatal: This boundary condition (y-plus, linear) has not been implemented yet.\n");
		PRINT_FAULT_FOOTER; break;

	// 32-40: Z direction
	case 32: cukern_zminusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 33: cukern_zminusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 34: cukern_extrapolateFlatConstBdyZMinus<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 35:
		problem = ERROR_NOIMPLEMENT;
		PRINT_FAULT_HEADER;
		printf("Fatal: This boundary condition (z-minus linear) has not been implemented yet.\n");
		PRINT_FAULT_FOOTER; break;

	case 40: cukern_zplusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 41: cukern_zplusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 42: cukern_extrapolateFlatConstBdyZPlus<<<griddim, blockdim>>>(x, nx, ny, nz); break;
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

int setBoundarySAS(MGArray *phi, int side, int direction, int sas)
{
	dim3 blockdim, griddim;
	int i, sub[6];

	int returnCode;

	switch(direction) {
	case 1: { blockdim.x = 3; blockdim.y = 16; blockdim.z = 8; } break;
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
		returnCode = callBCKernel(griddim, blockdim, phi->devicePtr[i], rez, sas + 8*side + 16*(direction-1));
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
			returnCode = callBCKernel(griddim, blockdim, phi->devicePtr[i], rez, sas + 8*side + 16*(direction-1));
			if(returnCode != SUCCESSFUL) return returnCode;
		}

	}

	return SUCCESSFUL;
}


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
__global__ void cukern_xminusSymmetrize(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	phi[2-threadIdx.x] = phi[4+threadIdx.x];
}

__global__ void cukern_xminusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	phi[2-threadIdx.x] = -phi[4+threadIdx.x];
}

__global__ void cukern_xplusSymmetrize(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	phi[nx-3+threadIdx.x] = phi[nx-5-threadIdx.x];
}

__global__ void cukern_xplusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	phi[nx-3+threadIdx.x] = -phi[nx-5-threadIdx.x];
}

/* These are called when a BC is set to 'const' or 'linear' */
__global__ void cukern_extrapolateFlatConstBdyXMinus(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	phi[threadIdx.x] = phi[3];
}

__constant__ int radExtrapCoeffs[5];
#define RAD_EX_X0 
#define RAD_EX_Y0
#define RAD_EX_Z0


__global__ void cukern_extrapolateRadialConstBdyXMinus(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE

	// 	
}

__global__ void cukern_extrapolateFlatConstBdyXPlus(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	phi[nx-3+threadIdx.x] = phi[nx-4];
}

// NOTE:
// These conditions are GENERALLY UNSTABLE and result in catastrophic backflow
// 
__global__ void cukern_extrapolateLinearBdyXMinus(double *phi, int nx, int ny, int nz)
{
	__shared__ double f[3];
	XSASKERN_PREAMBLE
	f[threadIdx.x] = phi[threadIdx.x+3];
	__syncthreads();
	phi[threadIdx.x] = phi[3] + (3-threadIdx.x)*(f[0]-f[1]);

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
	__shared__ double f[3];
	XSASKERN_PREAMBLE
	phi += nx - 5;
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

__global__ void cukern_yminusSymmetrize(double *phi, int nx, int ny, int nz)
{
	YSASKERN_PREAMBLE
	for(q = 0; q < 3; q++) { phi[nx*q] = phi[nx*(6-q)]; }
}

__global__ void cukern_yminusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
	YSASKERN_PREAMBLE
	for(q = 0; q < 3; q++) { phi[nx*q] = -phi[nx*(6-q)]; }
}

__global__ void cukern_yplusSymmetrize(double *phi, int nx, int ny, int nz)
{
	YSASKERN_PREAMBLE
	for(q = 0; q < 3; q++) { phi[nx*(ny-1-q)] = phi[nx*(ny-7+q)]; }
}

__global__ void cukern_yplusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
	YSASKERN_PREAMBLE
	for(q = 0; q < 3; q++) { phi[nx*(ny-1-q)] = -phi[nx*(ny-7+q)]; }
}

__global__ void cukern_extrapolateFlatConstBdyYMinus(double *phi, int nx, int ny, int nz)
{
	YSASKERN_PREAMBLE
	double f = phi[3*nx];
	for(q = 0; q < 3; q++) { phi[q*nx] = f; }
}

__global__ void cukern_extrapolateFlatConstBdyYPlus(double *phi, int nx, int ny, int nz)
{
	YSASKERN_PREAMBLE
	double f = phi[(ny-4)*nx];
	for(q = 0; q < 3; q++) { phi[(ny-3+q)*nx] = f; }
}

/* Z DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS */
/* Assume launch with size [A B 1] and grid of size [M N 1] s.t. AM >= nx, BN >= ny*/
#define ZSASKERN_PREAMBLE \
		int xidx = threadIdx.x + blockIdx.x * blockDim.x; \
		int yidx = threadIdx.y + blockIdx.y * blockDim.y; \
		if(xidx >= nx) return; if(yidx >= ny) return; \
		phi += xidx + nx*yidx; \
		int stride = nx*ny;

__global__ void cukern_zminusSymmetrize(double *phi, int nx, int ny, int nz)
{
	ZSASKERN_PREAMBLE

	int q;
	for(q = 0; q < 3; q++) // nvcc will unroll it
		phi[q*stride] = phi[(6-q)*stride];
}

__global__ void cukern_zminusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
	ZSASKERN_PREAMBLE
	int q;
	for(q = 0; q < 3; q++) 
		phi[q*stride] = -phi[(6-q)*stride];
}

__global__ void cukern_zplusSymmetrize(double *phi, int nx, int ny, int nz)
{
	ZSASKERN_PREAMBLE
	int q;
	for(q = 0; q < 3; q++)
		phi[stride*(nz-1-q)] = phi[stride*(nz-7+q)];
}

__global__ void cukern_zplusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
	ZSASKERN_PREAMBLE
	int q;
	for(q = 0; q < 3; q++)
		phi[stride*(nz-1-q)] = -phi[stride*(nz-7+q)];

}

__global__ void cukern_extrapolateFlatConstBdyZMinus(double *phi, int nx, int ny, int nz)
{
	ZSASKERN_PREAMBLE
	int q;
	for(q = 0; q < 3; q++)
		phi[stride*q] = phi[stride*4];
}

__global__ void cukern_extrapolateFlatConstBdyZPlus(double *phi, int nx, int ny, int nz)
{

	ZSASKERN_PREAMBLE
	int q;
	for(q = 0; q < 3; q++)
		phi[stride*(nz-1-q)] = phi[stride*(nz-4)];

}
