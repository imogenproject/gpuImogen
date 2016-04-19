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
#include "freezeAndPtot.h"

/* THIS FUNCTION:
   Calculates the maximum in the x direction of the freezing speed c_f, defined
   as the fastest characteristic velocity in the x direction.

   In the hydrodynamic case this is the adiabatic sounds speed, in the MHD case
   this is the fast magnetosonic speed.

 */

#define BLOCKDIM 64
#define MAXPOW   5

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	// At least 2 arguments expected
	// Input and result
	if(nrhs!=12)
		mexErrMsgTxt("Wrong number of arguments. Call using [ptot freeze] = FreezeAndPtot(mass, ener, momx, momy, momz, bz, by, bz, gamma, direct=1, csmin, topology)");

	if(nlhs == 0) mexErrMsgTxt("0 LHS argument: Must return at least Ptotal");
	if(nlhs > 2)  mexErrMsgTxt(">2 LHS arguments: Can only return [Ptot c_freeze]");

	CHECK_CUDA_ERROR("entering freezeAndPtot");

	pParallelTopology topology = topoStructureToC(prhs[11]);

	int ispurehydro = (int)*mxGetPr(prhs[9]);

	int nArrays;
	if(ispurehydro) { nArrays = 5; } else { nArrays = 8; }

	MGArray fluid[8];
	MGA_accessMatlabArrays(prhs, 0, nArrays-1, fluid);

	dim3 arraySize;
	arraySize.x = fluid->dim[0];
	arraySize.y = fluid->dim[1];
	arraySize.z = fluid->dim[2];
	dim3 blocksize, gridsize;

	blocksize.x = BLOCKDIM; blocksize.y = blocksize.z = 1;

	MGArray clone;
	MGArray *POut;
	MGArray *cfOut;

	clone = fluid[0];
	if(fluid->partitionDir == PARTITION_X) {
		clone.dim[0] = fluid->nGPUs;
	} else {
		clone.dim[0] = 1;
	}
	clone.dim[1] = arraySize.y;
	clone.dim[2] = arraySize.z;

	clone.haloSize = 0;

	POut = MGA_createReturnedArrays(plhs, 1, fluid);
	MGArray *cfLocal;
	cfLocal= MGA_allocArrays(1, &clone);

	double hostgf[6];
	double gam = *mxGetPr(prhs[8]);
	hostgf[0] = gam;
	hostgf[1] = gam - 1.0;
	hostgf[2] = gam*(gam-1.0);
	hostgf[3] = (1.0 - .5*gam);
	hostgf[4] = (*mxGetPr(prhs[10]))*(*mxGetPr(prhs[10])); // min c_s squared ;
	hostgf[5] = (ALFVEN_CSQ_FACTOR - .5*gam*(gam-1.0));

	int i;
	int sub[6];

	for(i = 0; i < fluid->nGPUs; i++) {
		cudaSetDevice(fluid->deviceID[i]);
		cudaMemcpyToSymbol(gammafunc, &hostgf[0],     6*sizeof(double), 0, cudaMemcpyHostToDevice);
		CHECK_CUDA_ERROR("cfreeze symbol upload");
	}

	if(ispurehydro) {
		for(i = 0; i < fluid->nGPUs; i++) {
			cudaSetDevice(fluid->deviceID[i]);
			CHECK_CUDA_ERROR("cudaSetDevice()");
			calcPartitionExtent(fluid, i, sub);
			gridsize.x = sub[4];
			gridsize.y = sub[5];
			cukern_FreezeSpeed_hydro<<<gridsize, blocksize>>>(
					fluid[0].devicePtr[i],
					fluid[1].devicePtr[i],
					fluid[2].devicePtr[i],
					fluid[3].devicePtr[i],
					fluid[4].devicePtr[i],
					cfLocal->devicePtr[i], POut->devicePtr[i], sub[3]);
			CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "Freeze speed hydro");
		}
	} else {
		for(i = 0; i < fluid->nGPUs; i++) {
			cudaSetDevice(fluid->deviceID[i]);
			calcPartitionExtent(fluid, i, sub);
			cukern_FreezeSpeed_mhd<<<gridsize, blocksize>>>(
					fluid[0].devicePtr[i],
					fluid[1].devicePtr[i],
					fluid[2].devicePtr[i],
					fluid[3].devicePtr[i],
					fluid[4].devicePtr[i],
					fluid[5].devicePtr[i],
					fluid[6].devicePtr[i],
					fluid[7].devicePtr[i],
					cfLocal->devicePtr[i], POut->devicePtr[i], sub[3]);
			CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "freeze speed MHD");
		}

	}


	cfOut = NULL;

	MGA_globalReduceDimension(cfLocal, &cfOut, MGA_OP_MAX, 1, 0, 1, topology);

	MGA_delete(cfLocal);

	MGA_returnOneArray(plhs+1, cfOut);

	free(POut);
	free(cfLocal);
	free(cfOut);

}

#define gam gammafunc[0]
#define gm1 gammafunc[1]
#define gg1 gammafunc[2]
#define MHD_PRESS_B gammafunc[3]
#define cs0sq gammafunc[4]
#define MHD_CS_B gammafunc[5]

__global__ void cukern_FreezeSpeed_mhd(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *freeze, double *ptot, int nx)
{
	int tix = threadIdx.x;
	/* gridDim = [ny nz], nx = nx */
	int x = tix + nx*(blockIdx.x + gridDim.x*blockIdx.y);
	nx += nx*(blockIdx.x + gridDim.x*blockIdx.y);
	//int addrMax = nx + nx*(blockIdx.x + gridDim.x*blockIdx.y);

	double pressvar;
	double T, bsquared;
	double rhoinv;
	__shared__ double locBloc[BLOCKDIM];

	//CsMax = 0.0;
	locBloc[tix] = 0.0;

	if(x >= nx) return; // If we get a very low resolution

	while(x < nx) {
		rhoinv = 1.0/rho[x];
		T = .5*rhoinv*(px[x]*px[x] + py[x]*py[x] + pz[x]*pz[x]);
		bsquared = bx[x]*bx[x] + by[x]*by[x] + bz[x]*bz[x];

		// Calculate internal + magnetic energy
		pressvar = E[x] - T;

		// Assert minimum thermal soundspeed / temperature
		/*  if(gam*pressvar*rhoinv < cs0sq) {
    E[x] = T + bsqhf + cs0sq/(gam*rhoinv);
    pressvar = cs0sq/(gam*rhoinv);
    } */

		// Calculate gas + magnetic pressure
		ptot[x] = gm1*pressvar + MHD_PRESS_B*bsquared;

		// We calculate the freezing speed in the X direction: max of |v|+c_fast
		// MHD_CS_B includes an "alfven factor" to stabilize the code in low-beta situations
		pressvar = (gg1*pressvar + MHD_CS_B*bsquared)*rhoinv;
		pressvar = sqrt(abs(pressvar)) + abs(px[x]*rhoinv);

		if(pressvar > locBloc[tix]) locBloc[tix] = pressvar;

		x += BLOCKDIM;
	}

	__syncthreads();

	if(tix >= 32) return;
	if(locBloc[tix+32] > locBloc[tix]) { locBloc[tix] = locBloc[tix+32]; }
	__syncthreads(); // compute 2 and later schedule by half-warps so we need to be down to 16 before no syncthreads

	if(tix >= 16) return;
	if(locBloc[tix+16] > locBloc[tix]) { locBloc[tix] = locBloc[tix+16]; }

	if(tix >= 8) return;
	if(locBloc[tix+8] > locBloc[tix]) {  locBloc[tix] = locBloc[tix+8];  }

	if(tix >= 4) return;
	if(locBloc[tix+4] > locBloc[tix]) {  locBloc[tix] = locBloc[tix+4];  }

	if(tix >= 2) return;
	if(locBloc[tix+2] > locBloc[tix]) {  locBloc[tix] = locBloc[tix+2];  }

	if(tix == 0) {
		if(locBloc[1] > locBloc[0]) {  locBloc[0] = locBloc[1];  }

		freeze[blockIdx.x + gridDim.x*blockIdx.y] = locBloc[0];
	}

}

#define PRESSURE Cs
// cs0sq = gamma rho^(gamma-1))
__global__ void cukern_FreezeSpeed_hydro(double *rho, double *E, double *px, double *py, double *pz, double *freeze, double *ptot, int nx)
{
	int tix = threadIdx.x;
	int x = tix + nx*(blockIdx.x + gridDim.x*blockIdx.y);
	int addrMax = nx + nx*(blockIdx.x + gridDim.x*blockIdx.y);

	double Cs, CsMax;
	double psqhf, rhoinv;
	//double gg1 = gam*(gam-1.0);
	//double gm1 = gam - 1.0;

	__shared__ double locBloc[BLOCKDIM];

	CsMax = 0.0;
	locBloc[tix] = 0.0;

	if(x >= addrMax) return; // If we get a very low resolution

	while(x < addrMax) {
		rhoinv   = 1.0/rho[x];
		psqhf    = .5*(px[x]*px[x]+py[x]*py[x]+pz[x]*pz[x]);

		PRESSURE = gm1*(E[x] - psqhf*rhoinv);
		if(gam*PRESSURE*rhoinv < cs0sq) {
			PRESSURE = cs0sq/(gam*rhoinv);
			E[x] = psqhf*rhoinv + PRESSURE/gm1;
		} /* Constrain temperature to a minimum value */
		ptot[x] = PRESSURE;

		Cs      = sqrt(gam * PRESSURE *rhoinv) + abs(px[x]*rhoinv);
		if(Cs > CsMax) CsMax = Cs;

		x += BLOCKDIM;
	}

	locBloc[tix] = CsMax;

	__syncthreads();

	if(tix >= 32) return;
	if(locBloc[tix+32] > locBloc[tix]) { locBloc[tix] = locBloc[tix+32]; }
	__syncthreads(); // compute 2 and later schedule by half-warps so we need to be down to 16 before no syncthreads

	if(tix >= 16) return;
	if(locBloc[tix+16] > locBloc[tix]) { locBloc[tix] = locBloc[tix+16]; }

	if(tix >= 8) return;
	if(locBloc[tix+8] > locBloc[tix]) {  locBloc[tix] = locBloc[tix+8];  }

	if(tix >= 4) return;
	if(locBloc[tix+4] > locBloc[tix]) {  locBloc[tix] = locBloc[tix+4];  }

	if(tix >= 2) return;
	if(locBloc[tix+2] > locBloc[tix]) {  locBloc[tix] = locBloc[tix+2];  }

	if(tix == 0) {
		if(locBloc[1] > locBloc[0]) {  locBloc[0] = locBloc[1];  }
		freeze[blockIdx.x + gridDim.x*blockIdx.y] = locBloc[0];
	}


}
