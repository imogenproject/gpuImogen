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
#include "cudaSoundspeed.h"

/* THIS FUNCTION:

   When not passed a magnetic field, calculates the local adiabatic sound speed of the fluid at 
   all points,

   c_s^2 = gamma*P/rho

   where c_s is the adiabatic sound speed, gamma is the adiabatic index (1 < gamma <= 5/3), P
   is the thermal pressure (gamma-1)*(Etotal - rho v^2/2) and rho is the matter density.

   When passed a magnetic field, calculates the maximal (field-aligned) magnetosonic velocity,
   C_fast^2 = C_s^2 + C_a^2,

   where C_s is the thermal sound speed above (Except subtracting magnetic energy density from
   the total energy as well) and C_a is the Alfven speed,

   C_a^2 = (B^2)/rho.
*/

__global__ void cukern_Soundspeed_mhd(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *dout, int n);
__global__ void cukern_Soundspeed_hd(double *rho, double *E, double *px, double *py, double *pz, double *dout, int n);

#define BLOCKDIM 256
#define GRIDDIM  64

// FIXME: Not quite clear why this is 6 elements and not 2...
__device__ __constant__ double pressParams[6];
#define MHD_CS_B pressParams[0]
#define GG1 pressParams[1]

#ifdef STANDALONE_MEX_FUNCTION
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// Determine appropriate number of arguments for RHS
	if( (nlhs != 1) || ( (nrhs != 9) && (nrhs != 6) ))
		mexErrMsgTxt("calling form for cudaSoundspeed is c_s = cudaSoundspeed(mass, ener, momx, momy, momz, [bx, by, bz,] gamma);");

	CHECK_CUDA_ERROR("entering cudaSoundspeed");

	// Select the appropriate kernel to invoke
	int pureHydro = (nrhs == 6);

	double gamma;
	MGArray fluid[5];
	MGArray mag[3];
	int worked;

	if(pureHydro == 1) {
		gamma    = *mxGetPr(prhs[5]);
		worked = MGA_accessMatlabArrays(prhs, 0, 4, &fluid[0]);
		MGArray *dest = MGA_createReturnedArrays(plhs, 1, &fluid[0]);

		calculateSoundspeed(&fluid[0], NULL, dest, gamma);
		free(dest);
	} else {
		gamma = *mxGetPr(prhs[8]);
		worked = MGA_accessMatlabArrays(prhs, 0, 4, &fluid[0]);
		worked = MGA_accessMatlabArrays(prhs, 5, 7, &mag[0]);
		MGArray *dest = MGA_createReturnedArrays(plhs, 1, &fluid[0]);

		calculateSoundspeed(&fluid[0], &mag[0], dest, gamma);
		free(dest);
	}

	return;
}
#endif

int calculateSoundspeed(MGArray *fluid, MGArray *mag, MGArray *cs, double gamma)
{
	double gg1 = gamma*(gamma-1);
	double hostParams[6];

	hostParams[0] = ALFVEN_CSQ_FACTOR - .5*gg1;
	hostParams[1] = gg1;

	dim3 blocksize; blocksize.x = BLOCKDIM; blocksize.y = blocksize.z = 1;
	dim3 gridsize;  gridsize.x = GRIDDIM;   gridsize.y = gridsize.z = 1;

	int pureHydro = (mag == NULL);

	int fail = SUCCESSFUL;

	int i, j;
	int sub[6];
	double *srcs[8];
	for(i = 0; i < fluid[0].nGPUs; i++) {
		calcPartitionExtent(&fluid[0], i, &sub[0]);
		cudaSetDevice(fluid[0].deviceID[i]);
		CHECK_CUDA_ERROR("Setting device.");
		cudaMemcpyToSymbol((const void *)pressParams, &hostParams[0], 6*sizeof(double), 0, cudaMemcpyHostToDevice);
		CHECK_CUDA_ERROR("cudaSoundspeed memcpy to constants.");

		for(j = 0; j < 5; j++) { srcs[j] = fluid[j].devicePtr[i]; }

		if(pureHydro == 1) {
			cukern_Soundspeed_hd<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], cs->devicePtr[i], fluid[0].partNumel[i]);
			fail = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "cuda hydro soundspeed");
		} else {
			for(j = 0; j < 3; j++) { srcs[j+5] = mag[j].devicePtr[i]; }

			cukern_Soundspeed_mhd<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], srcs[6], srcs[7], cs->devicePtr[i], fluid[0].partNumel[i]);
			fail = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, fluid, i, "cuda mhd soundspeed");
		}
	}

	return 0;
}

// THIS KERNEL CALCULATES SOUNDSPEED IN THE MHD CASE, TAKEN AS THE FAST MA SPEED
// We increase the Alfven contribution to stabilize the code
__global__ void cukern_Soundspeed_mhd(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *dout, int n)
{

	int x = threadIdx.x + blockIdx.x * BLOCKDIM;
	int dx = blockDim.x * gridDim.x;
	double csq, T, Bsq;
	double invrho;

	while(x < n) {
		invrho = 1.0 / rho[x];
		T = .5*(px[x]*px[x] + py[x]*py[x] + pz[x]*pz[x])*invrho;
		Bsq = bx[x]*bx[x] + by[x]*by[x] + bz[x]*bz[x];

		// MHD_CS_B is (alfven constant A) - .5(gamma)(gamma-1), where A is physically 1
		// but may be increased beyond 1 to stabilize simulations where low-beta conditions occur
		csq = (GG1*(E[x] - T) + MHD_CS_B * Bsq ) * invrho ;
		if(csq < 0.0) csq = 0.0;
		dout[x] = sqrt(csq);
		x += dx;
	}

}

// THIS KERNEL CALCULATES SOUNDSPEED IN THE HYDRODYNAMIC CASE
__global__ void cukern_Soundspeed_hd(double *rho, double *E, double *px, double *py, double *pz, double *dout, int n)
{
	int x = threadIdx.x + blockIdx.x * BLOCKDIM;
	int dx = blockDim.x * gridDim.x;
	double csq, rhoinv;

	while(x < n) {
		rhoinv = 1/rho[x];
		csq = GG1*(E[x] - .5*(px[x]*px[x] + py[x]*py[x] + pz[x]*pz[x])*rhoinv)*rhoinv;
		// Imogen's energy flux is unfortunately not positivity preserving
		if(csq < 0.0) csq = 0.0;
		dout[x] = sqrt(csq);
		x += dx;
	}

}


