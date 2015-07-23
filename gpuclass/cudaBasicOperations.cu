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

/* Simply set a block size for these silly cookie-cutter kernels */
#define BLEN 256

#define KERNPREAMBLE int addr = BLEN*blockIdx.x + threadIdx.x;\
if(addr >= n) { return; }

__global__ void cukern_add(double *a, double *b, double *y, int n) { KERNPREAMBLE y[addr] = a[addr] + b[addr]; }
__global__ void cukern_sub(double *a, double *b, double *y, int n) { KERNPREAMBLE y[addr] = a[addr] - b[addr]; }
__global__ void cukern_mul(double *a, double *b, double *y, int n) { KERNPREAMBLE y[addr] = a[addr] * b[addr]; }
__global__ void cukern_div(double *a, double *b, double *y, int n) { KERNPREAMBLE y[addr] = a[addr] / b[addr]; }

__global__ void cukern_addsc(double *a, double b, double *y, int n) { KERNPREAMBLE y[addr] = a[addr] + b; }
__global__ void cukern_subsc(double *a, double b, double *y, int n) { KERNPREAMBLE y[addr] = a[addr] - b; }
__global__ void cukern_scsub(double a, double *b, double *y, int n) { KERNPREAMBLE y[addr] = a - b[addr]; }
__global__ void cukern_mulsc(double *a, double b, double *y, int n) { KERNPREAMBLE y[addr] = a[addr] * b; }
__global__ void cukern_divsc(double *a, double b, double *y, int n) { KERNPREAMBLE y[addr] = a[addr] / b; }
__global__ void cukern_scdiv(double a, double *b, double *y, int n) { KERNPREAMBLE y[addr] = a / b[addr]; }

__global__ void cukern_sqrt(double *a, double *y, int n) { KERNPREAMBLE y[addr] = sqrt(a[addr]); }
__global__ void cukern_log(double *a, double *y, int n) { KERNPREAMBLE y[addr] = log(a[addr]);  }
__global__ void cukern_exp(double *a, double *y, int n) { KERNPREAMBLE y[addr] = exp(a[addr]); }
__global__ void cukern_sin(double *a, double *y, int n) { KERNPREAMBLE y[addr] = sin(a[addr]); }
__global__ void cukern_cos(double *a, double *y, int n) { KERNPREAMBLE y[addr] = cos(a[addr]); }
__global__ void cukern_tan(double *a, double *y, int n) { KERNPREAMBLE y[addr] = tan(a[addr]); }
__global__ void cukern_asin(double *a, double *y, int n) { KERNPREAMBLE y[addr] = asin(a[addr]); }
__global__ void cukern_acos(double *a, double *y, int n) { KERNPREAMBLE y[addr] = acos(a[addr]); }
__global__ void cukern_atan(double *a, double *y, int n) { KERNPREAMBLE y[addr] = atan(a[addr]); }
__global__ void cukern_sinh(double *a, double *y, int n) { KERNPREAMBLE y[addr] = sinh(a[addr]); }
__global__ void cukern_cosh(double *a, double *y, int n) { KERNPREAMBLE y[addr] = cosh(a[addr]); }
__global__ void cukern_tanh(double *a, double *y, int n) { KERNPREAMBLE y[addr] = tanh(a[addr]); }
__global__ void cukern_asinh(double *a, double *y, int n) { KERNPREAMBLE y[addr] = asinh(a[addr]); }
__global__ void cukern_acosh(double *a, double *y, int n) { KERNPREAMBLE y[addr] = acosh(a[addr]); }
__global__ void cukern_atanh(double *a, double *y, int n) { KERNPREAMBLE y[addr] = atanh(a[addr]); }

__global__ void cukern_max(double *a, double *b, double *y, int n)
    { KERNPREAMBLE if(b[addr] > a[addr]) { y[addr] = b[addr]; } else { y[addr] = a[addr]; } }
__global__ void cukern_maxsc(double *a, double b, double *y, int n)
    { KERNPREAMBLE if(b > a[addr]) { y[addr] = b; } else { y[addr] = a[addr]; } }

__global__ void cukern_min(double *a, double *b, double *y, int n)
    { KERNPREAMBLE if(b[addr] > a[addr]) { y[addr] = a[addr]; } else { y[addr] = b[addr]; } }
__global__ void cukern_minsc(double *a, double b, double *y, int n)
    { KERNPREAMBLE if(b > a[addr]) { y[addr] = a[addr]; } else { y[addr] = b; } }

__global__ void cukern_harmonicmean(double *a, double *b, double *y, int n) { KERNPREAMBLE
    double al = a[addr];
    double bl = b[addr];

    double p = al*bl;
    if(p <= 0.0) { y[addr] = 0.0; } else { y[addr] = .5*p/(al+bl); }
    }


dim3 setLaunchParams(int *arrdim)
{
int numel = arrdim[0]*arrdim[1]*arrdim[2];

dim3 griddim;

griddim.x = numel/BLEN;
if(BLEN*griddim.x < numel) griddim.x++;

griddim.y = griddim.z = 1;

return griddim;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  dim3 blocksize; blocksize.x = BLEN; blocksize.y = blocksize.z = 1;
  dim3 gridsize;

  if((nlhs == 1) && (nrhs == 2)) {
    // a = f(b) operators

    void *ktable[15] = { (void *)cukern_sqrt, (void *)cukern_log, (void *)cukern_exp, (void *)cukern_sin, (void *)cukern_cos, (void *)cukern_tan, (void *)cukern_asin, (void *)cukern_acos, (void *)cukern_atan, (void *)cukern_sinh, (void *)cukern_cosh, (void *)cukern_tanh, (void *)cukern_asinh, (void *)cukern_acosh, (void *)cukern_atanh};

    int op = (int)*mxGetPr(prhs[1]);
    if((op < 1) || (op > 15)) mexErrMsgTxt("cudaBasicOperations: fatal, y=f(a) operation code invalid");

    void (* baskern)(double *, double *, int) = (void (*)(double *, double *, int))ktable[op-1];

    MGArray srcArray;
    MGArray *dstArray;
    int worked = MGA_accessMatlabArrays(prhs, 0, 0, &srcArray);
    dstArray = MGA_createReturnedArrays(plhs, 1, &srcArray);

    int j;
    int sub[6];
    for(j = 0; j < srcArray.nGPUs; j++) {
        calcPartitionExtent(&srcArray, j, &sub[0]);
        gridsize = setLaunchParams(sub+3);
        cudaSetDevice(srcArray.deviceID[j]);
        baskern<<<gridsize, blocksize>>>(srcArray.devicePtr[j], dstArray->devicePtr[j], srcArray.partNumel[j]);
    }

    free(dstArray);
    return;
  }

  if((nlhs == 1) && (nrhs == 3)) {
    // y = f(a, b) operators
    MGArray srcA, srcB;
    MGArray *dest;
    int optype = 0;
    int worked = 0;
    double n;

    if(mxGetClassID(prhs[1]) == mxDOUBLE_CLASS) {
      int64_t ne = mxGetNumberOfElements(prhs[1]);
      if(ne != 1) mexErrMsgTxt("cudaBasicOperations: fatal, y = f(a, scalar): \"scalar\" is not scalar.");

      n = *mxGetPr(prhs[1]);
      worked = MGA_accessMatlabArrays(prhs, 0, 0, &srcA);
      optype = 1;
    } else if(mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
      int64_t ne = mxGetNumberOfElements(prhs[0]);
      if(ne != 1) mexErrMsgTxt("cudaBasicOperations: fatal, y = f(scalar, a): \"scalar\" is not scalar.");

      n = *mxGetPr(prhs[0]);
      worked = MGA_accessMatlabArrays(prhs, 1, 1, &srcA);
      optype = 2;
    } else {
      worked = MGA_accessMatlabArrays(prhs, 0, 0, &srcA);
      worked = MGA_accessMatlabArrays(prhs, 1, 1, &srcB);
      optype = 3;
    }
    
    dest   = MGA_createReturnedArrays(plhs, 1, &srcA);

    int i;
    int sub[6];
    int op = (int)*mxGetPr(prhs[2]);

    for(i = 0; i < srcA.nGPUs; i++) {
      calcPartitionExtent(&srcA, i, &sub[0]);
      gridsize = setLaunchParams(sub+3);

      cudaSetDevice(srcA.deviceID[i]);

      switch(optype) {
        case 1: switch(op) { // gpu + scalar
          case 1: cukern_addsc<<<gridsize, blocksize>>>(srcA.devicePtr[i], n, dest->devicePtr[i], srcA.partNumel[i]); break;
          case 2: cukern_subsc<<<gridsize, blocksize>>>(srcA.devicePtr[i], n, dest->devicePtr[i], srcA.partNumel[i]); break;
          case 3: cukern_mulsc<<<gridsize, blocksize>>>(srcA.devicePtr[i], n, dest->devicePtr[i], srcA.partNumel[i]); break;
          case 4: cukern_mulsc<<<gridsize, blocksize>>>(srcA.devicePtr[i], 1.0/n, dest->devicePtr[i], srcA.partNumel[i]); break;
          case 5: cukern_minsc<<<gridsize, blocksize>>>(srcA.devicePtr[i], n, dest->devicePtr[i], srcA.partNumel[i]); break;
          case 6: cukern_maxsc<<<gridsize, blocksize>>>(srcA.devicePtr[i], n, dest->devicePtr[i], srcA.partNumel[i]); break;
          default: mexErrMsgTxt("cudaBasicOperations: fatal, y=f(a,scalar operation code invalid"); break;
        } break;
      case 2: switch(op) { // scalar + gpu
          case 1: cukern_addsc<<<gridsize, blocksize>>>(srcA.devicePtr[i], n, dest->devicePtr[i], srcA.partNumel[i]); break;
          case 2: cukern_scsub<<<gridsize, blocksize>>>(n, srcA.devicePtr[i], dest->devicePtr[i], srcA.partNumel[i]); break;
          case 3: cukern_mulsc<<<gridsize, blocksize>>>(srcA.devicePtr[i], n, dest->devicePtr[i], srcA.partNumel[i]); break;
          case 4: cukern_scdiv<<<gridsize, blocksize>>>(n, srcA.devicePtr[i], dest->devicePtr[i], srcA.partNumel[i]); break;
          case 5: cukern_minsc<<<gridsize, blocksize>>>(srcA.devicePtr[i], n, dest->devicePtr[i], srcA.partNumel[i]); break;
          case 6: cukern_maxsc<<<gridsize, blocksize>>>(srcA.devicePtr[i], n, dest->devicePtr[i], srcA.partNumel[i]); break;
          default: mexErrMsgTxt("cudaBasicOperations: fatal, y=f(scalar,a) operation code invalid"); break;
        } break;
      case 3: switch(op) { // gpu + gpu
          case 1: cukern_add<<<gridsize, blocksize>>>(srcA.devicePtr[i], srcB.devicePtr[i], dest->devicePtr[i], srcA.partNumel[i]); break;
          case 2: cukern_sub<<<gridsize, blocksize>>>(srcA.devicePtr[i], srcB.devicePtr[i], dest->devicePtr[i], srcA.partNumel[i]); break;
          case 3: cukern_mul<<<gridsize, blocksize>>>(srcA.devicePtr[i], srcB.devicePtr[i], dest->devicePtr[i], srcA.partNumel[i]); break;
          case 4: cukern_div<<<gridsize, blocksize>>>(srcA.devicePtr[i], srcB.devicePtr[i], dest->devicePtr[i], srcA.partNumel[i]); break;
          case 5: cukern_min<<<gridsize, blocksize>>>(srcA.devicePtr[i], srcB.devicePtr[i], dest->devicePtr[i], srcA.partNumel[i]); break;
          case 6: cukern_max<<<gridsize, blocksize>>>(srcA.devicePtr[i], srcB.devicePtr[i], dest->devicePtr[i], srcA.partNumel[i]); break;
          case 7: cukern_harmonicmean<<<gridsize, blocksize>>>(srcA.devicePtr[i], srcB.devicePtr[i], dest->devicePtr[i], srcA.partNumel[i]); break;
          default: mexErrMsgTxt("cudaBasicOperations: fatal, y=f(a,b) operation code invalid"); break;
        } break;
      }

    }
  free(dest);
  }

}


