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
#include "GPUmat.hh"

// static paramaters
static int init = 0;
static GPUmat *gm;

double **getSourcePointers(const mxArray *prhs[], int num, int *retNumel);
double **makeDestinationArrays(GPUtype src, mxArray *retArray[], int howmany);

__global__ void cukern_DumbSubsagn(double *array, double *ind, double *val, int nval);

#define BLOCKDIM 16

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (init == 0) {
    // Initialize function
    // mexLock();
    // load GPUmat
    gm = gmGetGPUmat();
    init = 1;
  }

  dim3 blocksize; blocksize.x = BLOCKDIM; blocksize.y = blocksize.z = 1;
  int numel; dim3 gridsize;

  if( (nlhs != 0) || (nrhs != 3)) { mexErrMsgTxt("dumb linear subset assign operator is cudaLinearSubsagn(orig, statIndex, statVal)"); }
  double **mainarray = getSourcePointers(prhs, 1, &numel);
  double **statarrs  = getSourcePointers(&prhs[1], 2, &numel);

  gridsize.x = numel / BLOCKDIM; if(gridsize.x * BLOCKDIM < numel) gridsize.x++;
  gridsize.y = gridsize.z = 1;

  cukern_DumbSubsagn<<<gridsize, blocksize>>>(mainarray[0], statarrs[0], statarrs[1], numel);
  free(mainarray);
  free(statarrs);

}

// Given the RHS and how many cuda arrays we expect, extracts a set of pointers to GPU memory for us
// Also conveniently checked for equal array extent and returns it for us
double **getSourcePointers(const mxArray *prhs[], int num, int *retNumel)
{
  GPUtype src;
  double **gpuPointers = (double **)malloc(num * sizeof(double *));
  int iter;
  int numel = gm->gputype.getNumel(gm->gputype.getGPUtype(prhs[0]));
  for(iter = 0; iter < num; iter++) {
    src = gm->gputype.getGPUtype(prhs[iter]);
    if (gm->gputype.getNumel(src) != numel) { free(gpuPointers); mexErrMsgTxt("Fatal: Arrays contain nonequal number of elements."); }
    gpuPointers[iter] = (double *)gm->gputype.getGPUptr(src);
  }

retNumel[0] = numel;
return gpuPointers;
}

__global__ void cukern_DumbSubsagn(double *array, double *ind, double *val, int nval)
{
int idx = threadIdx.x + blockDim.x * blockIdx.x;

if(idx >= nval) return;

double lcopy[BLOCKDIM];
lcopy[threadIdx.x] = val[idx];

array[(int)ind[idx]] = lcopy[threadIdx.x];
}
