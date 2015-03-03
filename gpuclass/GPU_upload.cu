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

// GPU_Tag = GPU_upload(host_array[double], device IDs[integers], [integer halo dim, integer partition direction])

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // At least 2 arguments expected
  // Input and result
  if((nlhs != 1) || (nrhs < 1)) { mexErrMsgTxt("Form: result_tag = GPU_upload(host array [, device list [, (halo [,partition direct])]])"); }

  CHECK_CUDA_ERROR("entering GPU_upload");

  MGArray m;
  
  // Default to no halo, X partition
  m.haloSize = 0;
  m.partitionDir = PARTITION_X;

  if(nrhs >= 3) {
    int a = mxGetNumberOfElements(prhs[2]);
    double *d = mxGetPr(prhs[2]);

    if(a >= 1) {
      m.haloSize = (int)*d;
      if(m.haloSize < 0) m.haloSize = 0;
    }
    if(a >= 2) {
      m.partitionDir = (int)d[1];
      if((m.partitionDir < 1) || (m.partitionDir > 3)) m.partitionDir = PARTITION_X;
    }
  }

  // Default to entire array on current device
  m.nGPUs = 1;
  cudaGetDevice(&m.deviceID[0]);

  // But of course we may partition it otherwise
  if(nrhs >= 2) {
    int j;
    double *g = mxGetPr(prhs[1]);
    m.nGPUs = mxGetNumberOfElements(prhs[1]);
    for(j = 0; j < m.nGPUs; j++) { 
      m.deviceID[j] = (int)g[j];
      m.devicePtr[j] = 0x0;
    }
  }



  double *hmem = mxGetPr(prhs[0]);
  int nd = mxGetNumberOfDimensions(prhs[0]);
  if(nd > 3) mexErrMsgTxt("Array dimensionality > 3 unsupported.");
  const mwSize *idims = mxGetDimensions(prhs[0]);
  int i;
  for(i = 0; i < nd; i++) { m.dim[i] = idims[i]; }
  for(;      i < 3; i++) { m.dim[i] = 1; }

  // If the size in the partition direction is 1, clone it instead
  // calcPartitionExtent will take care of this for us
  if(m.dim[m.partitionDir-1] == 1) m.haloSize = PARTITION_CLONED;

  m.numel = m.dim[0]*m.dim[1]*m.dim[2];
  int sub[6];
  for(i = 0; i < m.nGPUs; i++) {
    calcPartitionExtent(&m, i, &sub[0]);
    m.partNumel[i] = sub[3]*sub[4]*sub[5];
  }
  m.numSlabs = 1;

  MGArray *dest = createMGArrays(plhs, 1, &m);

  double *gmem = NULL;
 
  int u, v, w;
  int64_t iT, iS;

  for(i = 0; i < m.nGPUs; i++) {
    calcPartitionExtent(dest, i, &sub[0]);
    gmem = (double *)realloc((void *)gmem, m.partNumel[i]*sizeof(double));
    
    // Copy the partition out of the full host array
    for(w = sub[2]; w < sub[2]+sub[5]; w++)
      for(v = sub[1]; v < sub[1]+sub[4]; v++)
        for(u = sub[0]; u < sub[0] + sub[3]; u++) {
          iT = (u-sub[0]) + sub[3]*(v - sub[1]  + sub[4]*(w-sub[2]));
          iS = u+m.dim[0]*(v+m.dim[1]*w);

          gmem[iT] = hmem[iS];
        }

    cudaSetDevice(m.deviceID[i]);
    cudaError_t fail;// = cudaMalloc((void **)&m.devicePtr[i],  sub[3]*sub[4]*sub[5]*sizeof(double));
  //  if(fail == cudaErrorMemoryAllocation) mexErrMsgTxt("GPU_upload: H2D, error attempting to allocate memory (cudaErrorMemoryAllocation). We've been had.");

    fail = cudaMemcpy((void *)dest->devicePtr[i], (void *)gmem, m.partNumel[i]*sizeof(double), cudaMemcpyHostToDevice);
    if(fail != cudaSuccess) mexErrMsgTxt("GPU_upload: H2D, Copy to device failed.");
  }



//  mwSize dims[2]; dims[0] = 6+2*m.nGPUs; dims[1] = 1;
//  plhs[0] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);
//  int64_t *tag = (int64_t *)mxGetData(plhs[0]);

//  serializeMGArrayToTag(&m, tag);

  return;
}
