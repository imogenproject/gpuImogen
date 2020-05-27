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

#include "nvToolsExt.h"

#include "cudaCommon.h"
#include "cudaFluidStep.h"

#include "cflTimestep.h"

#include "fluidMethod.h"

/* cflTimestep.cu responds to the following -D options:
 * USE_STATIC_CHALLOC: If defined, allocates a static array of page locked gpu-accessible memory 
 *                     used in max() reductions. This array is grown on demand, but is small.
 * SYNCMEX: If set, calls MGA_sledgehammerSequentialize before exit
 */

/* THIS FUNCTION:
   directionalMaxFinder has three different behaviors depending on how it is called.
   m = directionalMaxFinder(array) will calculate the global maximum of array
   c = directionalMaxFinder(a1, a2, direct) will find the max of |a1(r)+a2(r)| in the
      'direct' direction (1=X, 2=Y, 3=Z)
   c = directionalMaxFinder(rho, c_s, px, py, pz) will specifically calculate the x direction
       CFL limiting speed, max(|px/rho| + c_s)
    */

template <int simulationDimension, geometryType_t shape, FluidMethods algo>
__global__ void cukern_CFLtimestep(double *fluid, double *cs, double *out, int nx, int ntotal, int64_t slabpitch);

#define BLOCKDIM 8
#define GLOBAL_BLOCKDIM 128

__constant__ __device__ double geoParams[5];
#define GEO_DX geoParams[0]
#define GEO_DY geoParams[1]
#define GEO_DZ geoParams[2]
#define GEO_RIN geoParams[3]

// The process of acquiring page locked memory on the host is nauseatingly slow
// If this is set, grab these pointers once and never look again
#ifdef USE_STATIC_CHALLOC
static double *hostptrs[MAX_GPUS_USED];
// amount alloc'd in # doubles
static int hostptrLength[MAX_GPUS_USED];
#endif

#ifdef STANDALONE_MEX_FUNCTION
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// Form of call: tau = cflTimestep(FluidManager, soundspeed gpu array, GeometryManager)

	MGArray fluid[5];
	GeometryParams geom;
	MGArray sndspeed;

	// At least 2 arguments expected
	// Input and result
	if((nlhs != 1) || (nrhs != 4))
		mexErrMsgTxt("Call must be tau = cflTimestep(FluidManager, soundspeed gpu array, GeometryManager, cfd_method);");

#ifdef USE_NVTX
		nvtxRangePush("Entering cflTimestep from mex");
#endif
	CHECK_CUDA_ERROR("entering cflTimestep");

	int worked;

	worked = MGA_accessFluidCanister(prhs[0], 0, &fluid[0]);
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) { mexErrMsgTxt("Dumping"); }
	worked = MGA_accessMatlabArrays(prhs, 1, 1, &sndspeed);
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) { mexErrMsgTxt("Dumping"); }

	geom = accessMatlabGeometryClass(prhs[2]);
	int meth = (int)*mxGetPr(prhs[3]);

	const mxArray *gdr = mxGetProperty(prhs[2], 0, "globalDomainRez");
	double *globrez = mxGetPr(gdr);

    mwSize outputDims[2];
    outputDims[0] = 1;
    outputDims[1] = 1;
    plhs[0] = mxCreateNumericArray (2, outputDims, mxDOUBLE_CLASS, mxREAL);

    double *timeStep = mxGetPr(plhs[0]);

	worked = computeLocalCFLTimestep(&fluid[0], &sndspeed, &geom, meth, globrez, timeStep);

}
#endif


int computeLocalCFLTimestep(MGArray *fluid, MGArray *csound, GeometryParams *geom, int method, double *globalResolution, double *tstep)
{
	int i;
	int sub[6];
	int worked;

	double geoarray[5];
	geoarray[0] = geom->h[0];
	geoarray[1] = geom->h[1];
	geoarray[2] = geom->h[2];
	geoarray[3] = geom->Rinner;

    dim3 blocksize, gridsize;
    blocksize.x = GLOBAL_BLOCKDIM; blocksize.y = blocksize.z = 1;

    // Launches enough blocks to fully occupy the GPU
    gridsize.x = 128;
    gridsize.y = gridsize.z =1;

    // Allocate enough pinned memory to hold results
    double *blkA[fluid->nGPUs];
    int hblockElements = gridsize.x;

    int spacedim = 0;
    if(globalResolution[1] > 1) spacedim = 1;
    if(globalResolution[2] > 1) spacedim = 2;

    int gt = 0;
    if(geom->shape == CYLINDRICAL) gt = 1;
    int ctype = spacedim + 3*(gt + 2*(method-1)); // value in 0..17

    int numBlocks[fluid->nGPUs];

    for(i = 0; i < fluid->nGPUs; i++) {
    	cudaSetDevice(fluid->deviceID[i]);
    	worked = CHECK_CUDA_ERROR("cudaSetDevice()");
    	if(worked != SUCCESSFUL) break;
    	// This DOES grow the allocation "dumbly" however it is extremely unlikely that reallocation will be necessary more than once, if that
#ifdef USE_STATIC_CHALLOC
    	if(hblockElements > hostptrLength[i]) {
    		if(hostptrLength[i] > 0) {
    			cudaFreeHost(hostptrs[i]);
    			CHECK_CUDA_ERROR("cudaFreeHost()");
    		}
    		cudaMallocHost((void **)&hostptrs[i], hblockElements * sizeof(double));
    		hostptrLength[i] = hblockElements;
    		worked = CHECK_CUDA_ERROR("CFL malloc doubles");
    		if(worked != SUCCESSFUL) break;
    	}
    	blkA[i] = hostptrs[i];
#else
    	cudaMallocHost((void **)&blkA[i], hblockElements * sizeof(double));
    	worked = CHECK_CUDA_ERROR("CFL malloc doubles");
    	if(worked != SUCCESSFUL) break;
#endif

    	cudaMemcpyToSymbol(geoParams, &geoarray[0], 5*sizeof(double), 0, cudaMemcpyHostToDevice);
    	worked = CHECK_CUDA_ERROR("cfl const memcpy");
    	if(worked != SUCCESSFUL) break;
    	calcPartitionExtent(&fluid[0], i, &sub[0]);

    	gridsize.x = ROUNDUPTO(fluid[0].partNumel[i], blocksize.x) / blocksize.x;
    	if(gridsize.x > 128) gridsize.x = 128;

    	numBlocks[i] = gridsize.x;
    	double *fpi = fluid[0].devicePtr[i];
    	double *spi = csound->devicePtr[i];

    	switch(ctype) {
    	case 0:  cukern_CFLtimestep<1, SQUARE,      METHOD_HLL   ><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 1:  cukern_CFLtimestep<2, SQUARE,      METHOD_HLL   ><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 2:  cukern_CFLtimestep<3, SQUARE,      METHOD_HLL   ><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 3:  cukern_CFLtimestep<1, CYLINDRICAL, METHOD_HLL   ><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 4:  cukern_CFLtimestep<2, CYLINDRICAL, METHOD_HLL   ><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 5:  cukern_CFLtimestep<3, CYLINDRICAL, METHOD_HLL   ><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 6:  cukern_CFLtimestep<1, SQUARE,      METHOD_HLL   ><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 7:  cukern_CFLtimestep<2, SQUARE,      METHOD_HLLC  ><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 8:  cukern_CFLtimestep<3, SQUARE,      METHOD_HLLC  ><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 9:  cukern_CFLtimestep<1, CYLINDRICAL, METHOD_HLLC  ><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 10: cukern_CFLtimestep<2, CYLINDRICAL, METHOD_HLLC  ><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 11: cukern_CFLtimestep<3, CYLINDRICAL, METHOD_HLLC  ><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 12: cukern_CFLtimestep<1, SQUARE,      METHOD_HLLC  ><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 13: cukern_CFLtimestep<2, SQUARE,      METHOD_XINJIN><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 14: cukern_CFLtimestep<3, SQUARE,      METHOD_XINJIN><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 15: cukern_CFLtimestep<1, CYLINDRICAL, METHOD_XINJIN><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 16: cukern_CFLtimestep<2, CYLINDRICAL, METHOD_XINJIN><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	case 17: cukern_CFLtimestep<3, CYLINDRICAL, METHOD_XINJIN><<<gridsize, blocksize>>>(fpi, spi, blkA[i], sub[3], fluid[0].partNumel[i], fluid[0].slabPitch[i] / 8); break;
    	default:
    		PRINT_FAULT_HEADER;
    		printf("ctype ended up being %i, which is invalid because ctype must be int in [0,17].\n", ctype);
    		PRINT_FAULT_FOOTER;
    		worked = ERROR_INVALID_ARGS;
    		break;
    	}
    	worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, &fluid[0], i, "CFL max finder for Riemann solvers");
    	if(worked != SUCCESSFUL) break;
    }
    if(worked != SUCCESSFUL) { return worked; }

    double tmin = 1e38;

    int j;
    for(i = 0; i < fluid->nGPUs; i++) {
    	// They were all fired off at once, but we have to make sure each is done before
    	// reading the results of course
    	worked = cudaSetDevice(fluid->deviceID[i]);
    	if(worked != SUCCESSFUL) { break; }
    	worked = cudaDeviceSynchronize();
    	if(worked != SUCCESSFUL) { break; }

    	for(j = 0; j < numBlocks[i]; j++) {
    		tmin = (tmin < blkA[i][j]) ? tmin : blkA[i][j];
    	}
#ifdef USE_STATIC_CHALLOC

#else
    	cudaFreeHost(blkA[i]);
    	worked = CHECK_CUDA_ERROR("cudaFreeHost");
#endif
    }
    if(worked != SUCCESSFUL) { return worked; }

    double trueMin;
    MPI_Allreduce((void *)&tmin, (void *)&trueMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    if((method == 1) || (method == 2)) { // HLL or HLLC
    	trueMin = trueMin / 2;
    	// Waves cannot be let to go more than HALF a cell: otherwise in principle two waves could collide,
    	// and emit a fast-moving signal that might propagate back to the fluxing interface.
    	// The method is, however, likely to be stable up to a twice this timestep. But likely is NOT good enough.
    }
#ifndef USE_SSPRK
    trueMin = trueMin / 2;
    // If we are using explicit midpoint, the timestep must be halved again to remain TVD
#endif
    tstep[0] = trueMin;

#ifdef SYNCMEX
    MGA_sledgehammerSequentialize(&fluid[0]);
#endif
#ifdef USE_NVTX
    nvtxRangePop();
#endif

    return worked;
}

template <int dimension>
__device__ __inline__ double getMagnitudeMomentum(double *base, int64_t pitch)
{
	double p, q;
	p = base[2*pitch]; // p_x or p_r
	if(dimension == 1) {
		return fabs(p);
	} else {

		p=p*p;
		q = base[3*pitch];
		p=p+q*q;

		if(dimension > 2) {
			q = base[4*pitch];
			p=p+q*q;
		}
		return sqrt(p);
	}
}

// 3 dims x 2 shapes x 3 algorithms = 18 kernels total
template <int simulationDimension, geometryType_t shape, FluidMethods algo>
__global__ void cukern_CFLtimestep(double *fluid, double *cs, double *out, int nx, int ntotal, int64_t slabpitch)
{
	unsigned int tix = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tix; // address
	int blockhop = blockDim.x * gridDim.x;         // stepsize

	__shared__ double dtLimit[GLOBAL_BLOCKDIM];

	double u, v, w;
	double localTmin = 1e37;

	dtLimit[tix] = 1e37;

	if(x >= ntotal) return; // This is unlikely but we may get a stupid-small resolution

	fluid += x; // compute base offset
	cs += x;

	if((algo == METHOD_HLL) || (algo == METHOD_HLLC)) {
		if(shape == SQUARE) { // compute h once
			v = GEO_DX;
			if(simulationDimension > 1) { if(GEO_DY < v) v = GEO_DY; }
			if(simulationDimension > 2) { if(GEO_DZ < v) v = GEO_DZ; }
		}
		if(shape == CYLINDRICAL) { // Compute what we can compute just once
			v = GEO_DX;
			if(simulationDimension == 3) v = (v < GEO_DZ) ? v : GEO_DZ;
		}
	}

	while(x < ntotal) {
		if((algo == METHOD_HLL) || (algo == METHOD_HLLC)) {
			// get max signal speed
			u = getMagnitudeMomentum<simulationDimension>(fluid, slabpitch) / fluid[0] + cs[0]; // |v| + c

			// Identify local constraint on dt < dx / c_signal
			if(shape == SQUARE) {
				u = v / u;
			}
			if(shape == CYLINDRICAL) {
				w = (GEO_RIN + (x % nx) *GEO_DX)*GEO_DY; // r dtheta changes with r...
				w = (w < v) ? w : v;
				u = w / u;
			}
		}
		if(algo == METHOD_XINJIN) {
			double rho = fluid[0];
			// get max signal speed
			u = GEO_DX / ( fabs(fluid[2*slabpitch])/rho + cs[0] );

			if(simulationDimension > 1) {
				if(shape == SQUARE) {
					v = GEO_DY / ( fabs(fluid[3*slabpitch])/rho + cs[0] );
				}
				if(shape == CYLINDRICAL){
					v = (GEO_RIN + (x % nx)*GEO_DX)*GEO_DY / ( fabs(fluid[3*slabpitch])/rho + cs[0] );
				}
				u = (u < v) ? u : v;
			}
			if(simulationDimension > 2) {
				v = GEO_DZ / ( fabs(fluid[4*slabpitch])/rho + cs[0] );
				u = (u < v) ? u : v;
			}
		}

		// Each thread keeps running track of minimum dt
		localTmin = (u < localTmin) ? u : localTmin;

		fluid += blockhop;
		cs    += blockhop;
		x += blockhop; // skip the first block since we've already done it.
	}

	dtLimit[tix] = localTmin;

	__syncthreads();

	x = GLOBAL_BLOCKDIM / 2;
	while(x > 16) {
		if(tix >= x) return;
		__syncthreads();
		if(dtLimit[tix+x] < dtLimit[tix]) { dtLimit[tix] = dtLimit[tix+x]; }
		x=x/2;
	}

	if(tix >= 16) return;

	// We have one halfwarp (16 threads) remaining, proceed synchronously
	// cuda-memcheck --racecheck whines bitterly about this but because of warp synchronicity
	// there is no RAW problem.
	if(dtLimit[tix+16] < dtLimit[tix]) { dtLimit[tix] = dtLimit[tix+16]; } if(tix >= 8) return;
	//__syncthreads();
	if(dtLimit[tix+8] < dtLimit[tix])  { dtLimit[tix] = dtLimit[tix+8 ]; } if(tix >= 4) return;
	//__syncthreads();
	if(dtLimit[tix+4] < dtLimit[tix])  { dtLimit[tix] = dtLimit[tix+4 ]; } if(tix >= 2) return;
	//__syncthreads();
	if(dtLimit[tix+2] < dtLimit[tix])  { dtLimit[tix] = dtLimit[tix+2 ]; } if(tix) return;
	//__syncthreads();
	out[blockIdx.x] = (dtLimit[1] < dtLimit[0]) ? dtLimit[1] : dtLimit[0];

}
