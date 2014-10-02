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

/* THIS FUNCTION
   We're trying to remove all these to proper functions and get rid of this routine.
   */

#define OP_SOUNDSPEED 1
#define OP_GASPRESSURE 2
#define OP_TOTALPRESSURE 3
#define OP_MAGPRESSURE 4
#define OP_TOTALANDSND 5
#define OP_WARRAYS 6
#define OP_RELAXINGFLUX 7
#define OP_SEPERATELRFLUX 8
__global__ void cukern_Soundspeed(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *dout, double gam, int n);
__global__ void cukern_GasPressure(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *dout, double gam, int n);
__global__ void cukern_TotalPressure(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *dout, double gam, int n);
__global__ void cukern_MagneticPressure(double *bx, double *by, double *bz, double *dout, int n);
__global__ void cukern_TotalAndSound(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *total, double *sound, double gam, int n);
__global__ void cukern_CalcWArrays(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *P, double *Cfreeze, double *rhoW, double *enerW, double *pxW, double *pyW, double *pzW, int dir, int n);

__global__ void cukern_SeperateLRFlux(double *arr, double *wArr, double *left, double *right, int n);
__global__ void cukern_PerformFlux(double *array0, double *Cfreeze, double *fluxRa, double *fluxRb, double *fluxLa, double *fluxLb, double *out, double lambda, int n);

#define BLOCKWIDTH 256
#define THREADLOOPS 1

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Determine appropriate number of arguments for RHS
  if (nrhs < 2) mexErrMsgTxt("Require at least (computation type, input argument)");
  int operation = (int)*mxGetPr(prhs[0]);

  dim3 blocksize; blocksize.x = BLOCKWIDTH; blocksize.y = blocksize.z = 1;
  dim3 gridsize;

  // Select the appropriate kernel to invoke
  if((operation == OP_SOUNDSPEED) || (operation == OP_GASPRESSURE) || (operation == OP_TOTALPRESSURE)) {
    if( (nlhs != 1) || (nrhs != 10)) { mexErrMsgTxt("Soundspeed operator is Cs = cudaMHDKernels(1, rho, E, px, py, pz, bx, by, bz, gamma)"); }
    double gam = *mxGetPr(prhs[9]);

    MGArray fluid[8];
    int worked   = accessMGArrays(prhs, 1, 8, fluid);
    MGArray *dst = createMGArrays(plhs, 1, fluid);

    gridsize.x = fluid->numel / (BLOCKWIDTH*THREADLOOPS); if(gridsize.x * (BLOCKWIDTH*THREADLOOPS) < fluid->numel) gridsize.x++;
    gridsize.y = gridsize.z =1;

    double *srcs[8]; pullMGAPointers(fluid, 8, 0, srcs);

//printf("%i %i %i %i %i %i\n", blocksize.x, blocksize.y, blocksize.z, gridsize.x, gridsize.y, gridsize.z);
    switch(operation) {
      case OP_SOUNDSPEED:       cukern_Soundspeed<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], srcs[6], srcs[7], dst->devicePtr[0], gam, fluid->numel); break;
      case OP_GASPRESSURE:     cukern_GasPressure<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], srcs[6], srcs[7], dst->devicePtr[0], gam, fluid->numel); break;
      case OP_TOTALPRESSURE: cukern_TotalPressure<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], srcs[6], srcs[7], dst->devicePtr[0], gam, fluid->numel); break;
    }

    free(dst);

  } else if((operation == OP_MAGPRESSURE)) {
    if( (nlhs != 1) || (nrhs != 4)) { mexErrMsgTxt("Magnetic pressure operator is Pm = cudaMHDKernels(4, bx, by, bz)"); }
    MGArray mag[3];
    int worked = accessMGArrays(prhs, 1, 3, mag);
    MGArray *Pmag = createMGArrays(plhs, 1, mag);

    gridsize.x = mag->numel / (BLOCKWIDTH*THREADLOOPS); if(gridsize.x * (BLOCKWIDTH*THREADLOOPS) < mag->numel) gridsize.x++;
    gridsize.y = gridsize.z =1;

    cukern_MagneticPressure<<<gridsize, blocksize>>>(mag[0].devicePtr[0], mag[1].devicePtr[0], mag[2].devicePtr[0], Pmag->devicePtr[0], mag->numel);

    free(Pmag);

  } else if((operation == OP_TOTALANDSND)) {
    if( (nlhs != 2) || (nrhs != 10)) { mexErrMsgTxt("Soundspeed operator is [Ptot Cs] = cudaMHDKernels(5, rho, E, px, py, pz, bx, by, bz, gamma)"); }
    double gam = *mxGetPr(prhs[9]);
    MGArray fluid[8];
    int worked = accessMGArrays(prhs, 1, 8, fluid);

    gridsize.x = fluid->numel / (BLOCKWIDTH*THREADLOOPS); if(gridsize.x * (BLOCKWIDTH*THREADLOOPS) < fluid->numel) gridsize.x++;
    gridsize.y = gridsize.z = 1;
    MGArray *out = createMGArrays(plhs, 2, fluid);

    double *srcs[8]; pullMGAPointers(fluid, 8, 0, srcs);
    cukern_TotalAndSound<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], srcs[6], srcs[7], out[0].devicePtr[0], out[1].devicePtr[0], gam, fluid->numel);
    free(out);

  } else if ((operation == OP_WARRAYS)) {
    if( (nlhs != 5) || (nrhs != 12)) { mexErrMsgTxt("solving W operator is [rhoW enerW pxW pyW pzW] = cudaMHDKernels(6, rho, E, px, py, pz, bx, by, bz, P, cFreeze, direction)"); }
    int dir = (int)*mxGetPr(prhs[11]);
    MGArray fluid[10];
    int worked = accessMGArrays(prhs, 1, 10, fluid);
    MGArray *Wout = createMGArrays(plhs, 5, fluid);

    gridsize.x = fluid->numel / (BLOCKWIDTH*THREADLOOPS); if(gridsize.x * (BLOCKWIDTH*THREADLOOPS) < fluid->numel) gridsize.x++;
    gridsize.y = gridsize.z =1;

    double *srcs[10]; pullMGAPointers(fluid, 10, 0, srcs);
    double *dst[5]; pullMGAPointers(Wout, 5, 0, dst);
    cukern_CalcWArrays<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], srcs[6], srcs[7], srcs[8], srcs[9], dst[0], dst[1], dst[2], dst[3], dst[4], dir, fluid->numel);

    free(Wout);
  } else if ((operation == OP_RELAXINGFLUX)) {
    if( (nlhs != 1) || (nrhs != 8)) { mexErrMsgTxt("relaxing flux operator is fluxed = cudaMHDKernels(7, old, tempfreeze, right, right_shifted, left, left_shifted, lambda)"); }
    double lambda = *mxGetPr(prhs[7]);
    MGArray fluid[6];
    int worked = accessMGArrays(prhs, 1, 6, fluid);
    MGArray *dst = createMGArrays(plhs, 1, fluid);

    gridsize.x = fluid->numel / (BLOCKWIDTH*THREADLOOPS); if(gridsize.x * (BLOCKWIDTH*THREADLOOPS) < fluid->numel) gridsize.x++;
    gridsize.y = gridsize.z =1;

    double *srcs[6]; pullMGAPointers(fluid, 6, 0, srcs);

    cukern_PerformFlux<<<gridsize, blocksize>>>(srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], dst->devicePtr[0], lambda, fluid->numel);
    free(dst);

  } else if ((operation == OP_SEPERATELRFLUX)) {
    if ((nlhs != 2) || (nrhs != 3)) { mexErrMsgTxt("flux seperation operator is [Fl Fr] = cudaMHDKernels(8, array, wArray)"); }
    MGArray in[2];
    int worked = accessMGArrays(prhs, 1, 2, in);
    MGArray *out = createMGArrays(plhs, 2, in);

    gridsize.x = in->numel / (BLOCKWIDTH*THREADLOOPS); if(gridsize.x * (BLOCKWIDTH*THREADLOOPS) < in->numel) gridsize.x++;
    gridsize.y = gridsize.z =1;

    cukern_SeperateLRFlux<<<gridsize, blocksize>>>(in[0].devicePtr[0], in[1].devicePtr[0], out[0].devicePtr[0], out[1].devicePtr[0], in->numel);
    free(out);

  }

}

//#define KERNEL_PREAMBLE int x = THREADLOOPS*(threadIdx.x + blockDim.x*blockIdx.x); if (x >= n) {return;} int imax; ((x+THREADLOOPS) > n) ? imax = n : imax = x + THREADLOOPS; for(; x < imax; x++)
#define KERNEL_PREAMBLE int x = threadIdx.x + blockDim.x*blockIdx.x; if (x >= n) { return; }

// THIS KERNEL CALCULATES SOUNDSPEED 
__global__ void cukern_Soundspeed(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *dout, double gam, int n)
{
double gg1 = gam*(gam-1.0);

KERNEL_PREAMBLE
gg1 = ( (gg1*(E[x] - .5*(px[x]*px[x] + py[x]*py[x] + pz[x]*pz[x])/rho[x]) + (2.0 -.5*gg1)*(bx[x]*bx[x] + by[x]*by[x] + bz[x]*bz[x]))/rho[x] );
if(gg1 < 0.0) gg1 = 0.0;
dout[x] = sqrt(gg1);
}

// THIS KERNEL CALCULATES GAS PRESSURE
__global__ void cukern_GasPressure(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *dout, double gam, int n)
{
double pres;
KERNEL_PREAMBLE
pres = (gam-1.0)*(E[x] - .5*((px[x]*px[x]+py[x]*py[x]+pz[x]*pz[x])/rho[x] + bx[x]*bx[x]+by[x]*by[x]+bz[x]*bz[x]));
if(pres < 0.0) pres = 0.0; // Deny existence of negative presure
dout[x] = pres;

}

// THIS KERNEL CALCULATES TOTAL PRESSURE
__global__ void cukern_TotalPressure(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *dout, double gam, int n)
{
double pres;
KERNEL_PREAMBLE
pres = (gam-1.0)*(E[x] - .5*((px[x]*px[x]+py[x]*py[x]+pz[x]*pz[x])/rho[x])) + .5*(2.0-gam)*(bx[x]*bx[x]+by[x]*by[x]+bz[x]*bz[x]);
if(pres < 0.0) pres = 0.0;
dout[x] = pres;
}

// THIS KERNEL CALCULATES MAGNETIC PRESSURE
__global__ void cukern_MagneticPressure(double *bx, double *by, double *bz, double *dout, int n)
{
KERNEL_PREAMBLE
dout[x] = .5*(bx[x]*bx[x]+by[x]*by[x]+bz[x]*bz[x]);
}

// THIS KERNEL CALCULATE TOTAL PRESSURE AND SOUNDSPEED
__global__ void cukern_TotalAndSound(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *total, double *sound, double gam, int n)
{
double gg1 = gam*(gam-1.0);
double psqhf, bsqhf;
double p0;

KERNEL_PREAMBLE {
	psqhf = .5*(px[x]*px[x]+py[x]*py[x]+pz[x]*pz[x]);
	bsqhf = .5*(bx[x]*bx[x]+by[x]*by[x]+bz[x]*bz[x]);
	
	p0 = (gam-1.0)*(E[x] - psqhf/rho[x]) + (2.0-gam)*bsqhf;
        if(p0 < 0.0) p0 = 0.0;
        total[x] = p0;
	p0   = ( (gg1*(E[x] - psqhf/rho[x]) + (4.0 - gg1)*bsqhf)/rho[x] );
        if(p0 < 0.0) p0 = 0.0;
        sound[x] = sqrt(p0);
	}
}

__global__ void cukern_CalcWArrays(double *rho, double *E, double *px, double *py, double *pz, double *bx, double *by, double *bz, double *P, double *Cfreeze, double *rhoW, double *enerW, double *pxW, double *pyW, double *pzW, int dir, int n)
{
double Cinv, rhoinv;

KERNEL_PREAMBLE {

Cinv = 1.0/Cfreeze[x];
rhoinv = 1.0/rho[x];

switch(dir) {
  case 1:
    rhoW[x]  = px[x] * Cinv;
    enerW[x] = (px[x] * (E[x] + P[x]) - bx[x]*(px[x]*bx[x]+py[x]*by[x]+pz[x]*bz[x]) ) * (rhoinv*Cinv);
    pxW[x]   = (px[x]*px[x]*rhoinv + P[x] - bx[x]*bx[x])*Cinv;
    pyW[x]   = (px[x]*py[x]*rhoinv        - bx[x]*by[x])*Cinv;
    pzW[x]   = (px[x]*pz[x]*rhoinv        - bx[x]*bz[x])*Cinv;
    break;
  case 2:
    rhoW[x]  = py[x] * Cinv;
    enerW[x] = (py[x] * (E[x] + P[x]) - by[x]*(px[x]*bx[x]+py[x]*by[x]+pz[x]*bz[x]) ) * (rhoinv*Cinv);
    pxW[x]   = (py[x]*px[x]*rhoinv        - by[x]*bx[x])*Cinv;
    pyW[x]   = (py[x]*py[x]*rhoinv + P[x] - by[x]*by[x])*Cinv;
    pzW[x]   = (py[x]*pz[x]*rhoinv        - by[x]*bz[x])*Cinv;
    break;
  case 3:
    rhoW[x]  = pz[x] * Cinv;
    enerW[x] = (pz[x] * (E[x] + P[x]) - bz[x]*(px[x]*bx[x]+py[x]*by[x]+pz[x]*bz[x]) ) * (rhoinv*Cinv);
    pxW[x]   = (pz[x]*px[x]*rhoinv        - bz[x]*bx[x])*Cinv;
    pyW[x]   = (pz[x]*py[x]*rhoinv        - bz[x]*by[x])*Cinv;
    pzW[x]   = (pz[x]*pz[x]*rhoinv + P[x] - bz[x]*bz[x])*Cinv;
    break;
  }

}
/*mass.wArray    = mom(X).array ./ freezeSpd.array;

    %--- ENERGY DENSITY ---%
    ener.wArray    = velocity .* (ener.array + press) - mag(X).cellMag.array .* ...
                        ( mag(1).cellMag.array .* mom(1).array ...
                        + mag(2).cellMag.array .* mom(2).array ...
                        + mag(3).cellMag.array .* mom(3).array) ./ mass.array;
    ener.wArray    = ener.wArray ./ freezeSpd.array;

    %--- MOMENTUM DENSITY ---%
    for i=1:3
        mom(i).wArray    = (velocity .* mom(i).array + press*dirVec(i)...
                             - mag(X).cellMag.array .* mag(i).cellMag.array) ./ freezeSpd.array;
    end*/

}

__global__ void cukern_PerformFlux(double *array0, double *Cfreeze, double *fluxRa, double *fluxRb, double *fluxLa, double *fluxLb, double *out, double lambda, int n)
{
KERNEL_PREAMBLE 
out[x] = array0[x] - lambda*Cfreeze[x]*(fluxRa[x] - fluxRb[x] + fluxLa[x] - fluxLb[x]);

//v(i).store.array = v(i).array - 0.5*fluxFactor .* tempFreeze .* ...
//                        ( v(i).store.fluxR.array - v(i).store.fluxR.shift(X,-1) ...
//                        + v(i).store.fluxL.array - v(i).store.fluxL.shift(X,1) );
}

__global__ void cukern_SeperateLRFlux(double *arr, double *wArr, double *left, double *right, int n)
{
KERNEL_PREAMBLE {
	left[x]  = .5*(arr[x] - wArr[x]);
	right[x] = .5*(arr[x] + wArr[x]);
	}

}


