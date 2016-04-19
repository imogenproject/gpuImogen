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

/* THIS FUNCTION:
   cudaStatics is used in the imposition of several kinds of boundary conditions
   upon arrays. Given a list of indices I, coefficients C and values V, it
   writes out

   phi[I] = (1-C)*phi[I] + C[i]*V[i],
   causing phi[I] to fade to V[i] at an exponential rate.

   It is also able to set mirror boundary conditions (FIXME: Not fully tested!)
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

__global__ void cukern_extrapolateConstBdyXMinus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateConstBdyXPlus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateConstBdyYMinus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateConstBdyYPlus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateConstBdyZMinus(double *phi, int nx, int ny, int nz);
__global__ void cukern_extrapolateConstBdyZPlus(double *phi, int nx, int ny, int nz);

__global__ void cukern_applySpecial_fade(double *phi, double *statics, int nSpecials, int blkOffset);

int setBoundarySAS(MGArray *phi, int side, int direction, int sas);

#ifdef STANDALONE_MEX_FUNCTION

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if( (nlhs != 0) || (nrhs != 3)) { mexErrMsgTxt("cudaStatics operator is cudaStatics(ImogenArray, blockdim, direction)"); }

	CHECK_CUDA_ERROR("entering cudaStatics");

	setBoundaryConditions(NULL, prhs[0], (int)*mxGetPr(prhs[2]));
}
#endif

/* FIXME: This is terrible.
 * FIXME: MGArray needs to provision carrying its own boundary condition metadata around somehow.
 */
int setBoundaryConditions(MGArray *array, const mxArray *matlabhandle, int direction)
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
	mxArray *boundaryData = mxGetProperty(matlabhandle, 0, "boundaryData");
	if(boundaryData == NULL) {
		printf("FATAL: field 'boundaryData' D.N.E. in class. Not a class? Not an ImogenArray/FluidArray?\n");
		return ERROR_INVALID_ARGS;
	}

	/* The statics describe "solid" structures which we force the grid to have */
	mxArray *gpuStatics = mxGetField(boundaryData, 0, "staticsData");
	if(gpuStatics == NULL) {
		printf("FATAL: field 'staticsData' D.N.E. in boundaryData struct. Statics not compiled?\n");
		return ERROR_INVALID_ARGS;
	}
	worked = MGA_accessMatlabArrays((const mxArray **)(&gpuStatics), 0, 0, &statics);
	BAIL_ON_FAIL(worked)

	int *perm = &phi.currentPermutation[0];
	int offsetidx = 2*(perm[0]-1) + 1*(perm[1] > perm[2]);

	/* The offset array describes the index offsets for the data in the gpuStatics array */
	mxArray *offsets    = mxGetField(boundaryData, 0, "compOffset");
	if(offsets == NULL) {
		printf("FATAL: field 'compOffset' D.N.E. in boundaryData. Not an ImogenArray? Statics not compiled?\n");
		return ERROR_INVALID_ARGS;
	}
	double *offsetcount = mxGetPr(offsets);
	long int staticsOffset = (long int)offsetcount[2*offsetidx];
	int staticsNumel  = (int)offsetcount[2*offsetidx+1];

	/* Parameter describes what block size to launch with... */
	int blockdim = 8;

	dim3 griddim; griddim.x = staticsNumel / blockdim + 1;
	if(griddim.x > 32768) {
		griddim.x = 32768;
		griddim.y = staticsNumel/(blockdim*griddim.x) + 1;
	}

	/* Every call results in applying specials */
	if(staticsNumel > 0) {
		PAR_WARN(phi);
		cukern_applySpecial_fade<<<griddim, blockdim>>>(phi.devicePtr[0], statics.devicePtr[0] + staticsOffset, staticsNumel, statics.dim[0]);
		worked = CHECK_CUDA_LAUNCH_ERROR(blockdim, griddim, &phi, 0, "cuda statics application");
		if(worked != SUCCESSFUL) return worked;
	}

	/* Indicates which part of a 3-vector this array is (0 = scalar, 123=XYZ) */
	mxArray *comp = mxGetProperty(matlabhandle, 0, "component");
	int vectorComponent;
	if(comp != NULL) {
		vectorComponent = (int)(*mxGetPr(comp));
	} else {
		printf("Failed to fetch 'component' field of class: Not an ImogenArray? Bailing.\n");
		return ERROR_INVALID_ARGS;
	}

	/* BEGIN DETERMINATION OF ANALYTIC BOUNDARY CONDITIONS */
	int numDirections = 1;

	mxArray *bcModes = mxGetField(boundaryData, 0, "bcModes");
	if(bcModes == NULL) {
		printf("FATAL: bcModes structure not present. Not an ImogenArray? Not initialized?\n");
		return ERROR_INVALID_ARGS;
	}

	int j;
	for(j = 0; j < numDirections; j++) {
		if(direction == 0) continue; /* Skips edge BCs if desired. */
		int memoryDirection = perm[direction-1];

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
				printf("Wall BC is not implemented!\n");
				return ERROR_INVALID_ARGS;
			} 

		}
		if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) return worked;
	}

	return SUCCESSFUL;
}

/* Sets the given array+AMD's boundary in the following manner:
   side      -> 0 = negative edge  1 = positive edge
   direction -> 1 = X	      2 = Y               3 = Z*
   sas       -> 0 = symmetrize      1 => antisymmetrize
	     -> 2 = extrap constant 3-> extrap linear

 *: As passed, assuming ImogenArray's indexPermute has been handled for us.
 */

void callBCKernel(dim3 griddim, dim3 blockdim, double *x, int nx, int ny, int nz, int ktable)
{
	switch(ktable) {
	case 0: cukern_xminusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 1: cukern_xminusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 2: cukern_extrapolateConstBdyXMinus<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 3: cukern_extrapolateLinearBdyXMinus<<<griddim, blockdim>>>(x, nx, ny, nz); break;

	case 4: cukern_xplusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 5: cukern_xplusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 6: cukern_extrapolateConstBdyXPlus<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 7: cukern_extrapolateLinearBdyXPlus<<<griddim, blockdim>>>(x, nx, ny, nz); break;

	case 8: cukern_yminusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 9: cukern_yminusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 10: cukern_extrapolateConstBdyYMinus<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 11: mexErrMsgTxt("Fatal: This boundary condition (y-minus, linear) has not been implemented yet."); break;

	case 12: cukern_yplusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 13: cukern_yplusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 14: cukern_extrapolateConstBdyYPlus<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 15: mexErrMsgTxt("Fatal: This boundary condition (y-plus, linear) has not been implemented yet."); break;

	case 16: cukern_zminusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 17: cukern_zminusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 18: cukern_extrapolateConstBdyZMinus<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 19: mexErrMsgTxt("Fatal: This boundary condition (z-minus linear) has not been implemented yet."); break;

	case 20: cukern_zplusSymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 21: cukern_zplusAntisymmetrize<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 22: cukern_extrapolateConstBdyZPlus<<<griddim, blockdim>>>(x, nx, ny, nz); break;
	case 23: mexErrMsgTxt("Fatal: This boundary condition (z-plus linear) has not been implemented yet."); break;
	}

}
void *getBCKernel(int X)
{
	void *PLACEHOLDER = NULL;

	void *kerntable[24] = {(void *)&cukern_xminusSymmetrize, \
			(void *)&cukern_xminusAntisymmetrize, \
			(void *)&cukern_extrapolateConstBdyXMinus, \
			(void *)&cukern_extrapolateLinearBdyXMinus, \

			(void *)&cukern_xplusSymmetrize, \
			(void *)&cukern_xplusAntisymmetrize,
			(void *)&cukern_extrapolateConstBdyXPlus, \
			(void *)&cukern_extrapolateLinearBdyXPlus, \

			(void *)&cukern_yminusSymmetrize, \
			(void *)&cukern_yminusAntisymmetrize, \
			PLACEHOLDER, \
			PLACEHOLDER, \

			(void *)&cukern_yplusSymmetrize, \
			(void *)&cukern_yplusAntisymmetrize,
			PLACEHOLDER, \
			PLACEHOLDER, \

			(void *)&cukern_zminusSymmetrize, \
			(void *)&cukern_zminusAntisymmetrize, \
			PLACEHOLDER, \
			PLACEHOLDER, \

			(void *)&cukern_zplusSymmetrize, \
			(void *)&cukern_zplusAntisymmetrize, \
			PLACEHOLDER, \
			PLACEHOLDER };

	return kerntable[X];
}

int setBoundarySAS(MGArray *phi, int side, int direction, int sas)
{
	dim3 blockdim, griddim;
	void (* bckernel)(double *, int, int, int);
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

		//bckernel = (void (*)(double *, int, int, int))getBCKernel(sas + 4*side + 8*(direction-1));
		//if((void *)bckernel == NULL) mexErrMsgTxt("Fatal: This boundary condition has not been implemented yet.");

		//bckernel<<<griddim, blockdim>>>(phi->devicePtr[i], phi->dim[0], phi->dim[1], phi->dim[2]);
		calcPartitionExtent(phi, i, sub);

		callBCKernel(griddim, blockdim, phi->devicePtr[i], sub[3], sub[4], sub[5], sas + 4*side + 8*(direction-1));
		returnCode = CHECK_CUDA_LAUNCH_ERROR(blockdim, griddim, phi, sas + 2*side + 4*direction, "In setBoundarySAS; integer -> cukern table index");
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

			//bckernel = (void (*)(double *, int, int, int))getBCKernel(sas + 4*side + 8*(direction-1));
			//if((void *)bckernel == NULL)

			callBCKernel(griddim, blockdim, phi->devicePtr[i], sub[3], sub[4], sub[5], sas + 4*side + 8*(direction-1));

			//bckernel<<<griddim, blockdim>>>(phi->devicePtr[i], sub[3], sub[4], sub[5]);
			returnCode = CHECK_CUDA_LAUNCH_ERROR(blockdim, griddim, phi, sas + 4*side + 8*(direction-1), "In setBoundarySAS; integer -> cukern table index");
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

	//	if(c >= 0) {
	// Fade condition: Exponentially pulls cell towards c with rate constant f0;
	phi[xaddr] = f0*c + (1.0-c)*phi[xaddr];
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
__global__ void cukern_extrapolateConstBdyXMinus(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	phi[threadIdx.x] = phi[3];
}

__global__ void cukern_extrapolateConstBdyXPlus(double *phi, int nx, int ny, int nz)
{
	XSASKERN_PREAMBLE
	phi[nx-3+threadIdx.x] = phi[nx-4];
}

__global__ void cukern_extrapolateLinearBdyXMinus(double *phi, int nx, int ny, int nz)
{
	__shared__ double f[3];
	XSASKERN_PREAMBLE
	f[threadIdx.x] = phi[threadIdx.x+3];
	__syncthreads();
	phi[threadIdx.x] = phi[3] + (3-threadIdx.x)*(f[0]-f[1]);
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

__global__ void cukern_extrapolateConstBdyYMinus(double *phi, int nx, int ny, int nz)
{
	YSASKERN_PREAMBLE
	double f = phi[3*nx];
	for(q = 0; q < 3; q++) { phi[q*nx] = f; }
}

__global__ void cukern_extrapolateConstBdyYPlus(double *phi, int nx, int ny, int nz)
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

__global__ void cukern_extrapolateConstBdyZMinus(double *phi, int nx, int ny, int nz)
{
	ZSASKERN_PREAMBLE
	int q;
	for(q = 0; q < 3; q++)
		phi[stride*q] = phi[stride*4];
}

__global__ void cukern_extrapolateConstBdyZPlus(double *phi, int nx, int ny, int nz)
{

	ZSASKERN_PREAMBLE
	int q;
	for(q = 0; q < 3; q++)
		phi[stride*(nz-1-q)] = phi[stride*(nz-4)];

}
