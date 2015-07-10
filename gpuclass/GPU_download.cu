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

// host_array = GPU_download(gpu type)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // At least 2 arguments expected
  // Input and result
  if((nlhs != 1) || (nrhs != 1)) { mexErrMsgTxt("Form: host_array = GPU_download(GPU array)"); }

  CHECK_CUDA_ERROR("entering GPU_download");

  MGArray m;
  
  accessMGArrays(prhs, 0, 0, &m);

  int nd = 3;
  if(m.dim[2] == 1) {
    nd = 2;
    if(m.dim[1] == 1) {
      nd = 1;
    }
  }
  mwSize odims[3];
  odims[0] = m.dim[0];
  odims[1] = m.dim[1];
  odims[2] = m.dim[2];

  // Create output numeric array
  plhs[0] = mxCreateNumericArray(nd, odims, mxDOUBLE_CLASS, mxREAL);

  double *result = mxGetPr(plhs[0]);

  int sub[6];
  int htrim[6];

  int u, v, w, i;
  int64_t iT, iS;
  double *gmem[m.nGPUs];

  if(m.haloSize == PARTITION_CLONED) {
    cudaSetDevice(m.deviceID[0]);
    CHECK_CUDA_ERROR("cudaSetDevice()");
    cudaError_t fail = cudaMemcpy((void *)result, (void *)m.devicePtr[0], m.numel*sizeof(double), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR("GPU_download to host.");
    return;
  }

  for(i = 0; i < m.nGPUs; i++) {
    // allocate CPU memory to dump it back
    gmem[i] = (double *)malloc(m.partNumel[i]*sizeof(double));

    // Download the whole thing from the GPU to CPU
    //cudaSetDevice(m.deviceID[i]);
    CHECK_CUDA_ERROR("cudaSetDevice()");
    cudaError_t fail = cudaMemcpy((void *)gmem[i], (void *)m.devicePtr[i], m.partNumel[i]*sizeof(double), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR("GPU_download to host.");
  }


  double *currentTarget;
  for(i = 0; i < m.nGPUs; i++) {
//   cudaSetDevice(i);
//   cudaDeviceSynchronize();
    // Get this partition's extent
    calcPartitionExtent(&m, i, &sub[0]);
    // Trim the halo away when copying back to CPU
    for(u = 0; u < 6; u++) { htrim[u] = sub[u]; }
    if(i < (m.nGPUs-1)) { htrim[3+m.partitionDir-1] -= m.haloSize; }
    if(i > 0)           { htrim[m.partitionDir-1] += m.haloSize; htrim[3+m.partitionDir-1] -= m.haloSize; }

    currentTarget = gmem[i];

    // Copy into the output array
    #pragma omp parallel for private(u, v, w, iT, iS) default shared
    for(w = htrim[2]; w < htrim[2]+htrim[5]; w++)
      for(v = htrim[1]; v < htrim[1]+htrim[4]; v++)
        for(u = htrim[0]; u < htrim[0] + htrim[3]; u++) {
          iT = (u-sub[0]) + sub[3]*(v - sub[1]  + sub[4]*(w-sub[2]));
          iS = u+m.dim[0]*(v+m.dim[1]*w);

          result[iS] = currentTarget[iT];
        }
    free(currentTarget);

  }

  return;
}
