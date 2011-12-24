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

#define BLEN 128

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

    double s = al + bl;
    if(abs(s) < 1e-14) { y[addr] = 0.0; } else { al = al*bl; if(al > 0) { y[addr] = al / s; } else { y[addr] = 0; } } 
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
  ArrayMetadata amd;

  if((nlhs == 1) && (nrhs == 2)) {
    // a = f(b) operators
    double **srcs = getGPUSourcePointers(prhs, &amd, 0, 0);
    double **dest = makeGPUDestinationArrays((int64_t *)mxGetData(prhs[0]), plhs, 1);

    gridsize = setLaunchParams(&amd.dim[0]);

    int op = (int)*mxGetPr(prhs[1]);
    switch(op) {
      case 1: cukern_sqrt<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 2: cukern_log<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 3: cukern_exp<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 4: cukern_sin<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 5: cukern_cos<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 6: cukern_tan<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 7: cukern_asin<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 8: cukern_acos<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 9: cukern_atan<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 10: cukern_sinh<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 11: cukern_cosh<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 12: cukern_tanh<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 13: cukern_asinh<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 14: cukern_acosh<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;
      case 15: cukern_atanh<<<gridsize, blocksize>>>(srcs[0], dest[0], amd.numel); break;

      default: mexErrMsgTxt("cudaBasicOperations: fatal, y=f(a) operation code invalid");
      }
     return;
  }

  if((nlhs == 1) && (nrhs == 3)) {
    // y = f(a, b) operators
    double **srcs;
    double **dest;
    double n;
    int optype = 0;

    if((mxGetClassID(prhs[0]) == mxINT64_CLASS) && (mxGetClassID(prhs[1]) == mxDOUBLE_CLASS)) {
      n = *mxGetPr(prhs[1]);
      srcs = getGPUSourcePointers(prhs, &amd, 0, 0);
      dest = makeGPUDestinationArrays((int64_t *)mxGetData(prhs[0]), plhs, 1);
      optype = 1;
      }

    if((mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) && (mxGetClassID(prhs[1]) == mxINT64_CLASS)) {
      n = *mxGetPr(prhs[0]);
      srcs = getGPUSourcePointers(prhs, &amd, 1,1);
      dest = makeGPUDestinationArrays((int64_t *)mxGetData(prhs[1]), plhs, 1);
      optype = 2;
      }

    if((mxGetClassID(prhs[0]) == mxINT64_CLASS) && (mxGetClassID(prhs[1]) == mxINT64_CLASS)) {
      srcs = getGPUSourcePointers(prhs, &amd, 0, 1);
      dest = makeGPUDestinationArrays((int64_t *)mxGetData(prhs[0]), plhs, 1);
      optype = 3;
      }

    gridsize = setLaunchParams(&amd.dim[0]);
    int op = (int)*mxGetPr(prhs[2]);

    switch(optype) {
      case 1: switch(op) { // gpu + scalar
        case 1: cukern_addsc<<<gridsize, blocksize>>>(srcs[0], n, dest[0], amd.numel); break;
        case 2: cukern_subsc<<<gridsize, blocksize>>>(srcs[0], n, dest[0], amd.numel); break;
        case 3: cukern_mulsc<<<gridsize, blocksize>>>(srcs[0], n, dest[0], amd.numel); break;
        case 4: cukern_divsc<<<gridsize, blocksize>>>(srcs[0], n, dest[0], amd.numel); break;
        case 5: cukern_minsc<<<gridsize, blocksize>>>(srcs[0], n, dest[0], amd.numel); break;
        case 6: cukern_maxsc<<<gridsize, blocksize>>>(srcs[0], n, dest[0], amd.numel); break;
        } break;
      case 2: switch(op) { // scalar + gpu
        case 1: cukern_addsc<<<gridsize, blocksize>>>(srcs[0], n, dest[0], amd.numel); break;
        case 2: cukern_scsub<<<gridsize, blocksize>>>(n, srcs[0], dest[0], amd.numel); break;
        case 3: cukern_mulsc<<<gridsize, blocksize>>>(srcs[0], n, dest[0], amd.numel); break;
        case 4: cukern_scdiv<<<gridsize, blocksize>>>(n, srcs[0], dest[0], amd.numel); break;
        case 5: cukern_minsc<<<gridsize, blocksize>>>(srcs[0], n, dest[0], amd.numel); break;
        case 6: cukern_maxsc<<<gridsize, blocksize>>>(srcs[0], n, dest[0], amd.numel); break;
        } break;
      case 3: switch(op) { // gpu + gpu
        case 1: cukern_add<<<gridsize, blocksize>>>(srcs[0], srcs[1], dest[0], amd.numel); break;
        case 2: cukern_sub<<<gridsize, blocksize>>>(srcs[0], srcs[1], dest[0], amd.numel); break;
        case 3: cukern_mul<<<gridsize, blocksize>>>(srcs[0], srcs[1], dest[0], amd.numel); break;
        case 4: cukern_div<<<gridsize, blocksize>>>(srcs[0], srcs[1], dest[0], amd.numel); break;
        case 5: cukern_min<<<gridsize, blocksize>>>(srcs[0], srcs[1], dest[0], amd.numel); break;
        case 6: cukern_max<<<gridsize, blocksize>>>(srcs[0], srcs[1], dest[0], amd.numel); break;
        case 7: cukern_harmonicmean<<<gridsize, blocksize>>>(srcs[0], srcs[1], dest[0], amd.numel); break;
        default: mexErrMsgTxt("cudaBasicOperations: fatal, y=f(a,b) operation code invalid");
        } break;
      }

  }

}


