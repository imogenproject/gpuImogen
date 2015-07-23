#warning "WARNING: COMPILING cudaFluidStep() WITH DEBUG ENABLED. cudaFluidStep will require an output argument to dump to!"

// If defined, the code runs the Euler prediction step and copies wStepValues back to the Matlab fluid data arrays
// If not defined, it runs the RK2 predictor/corrector step
#define DBG_FIRSTORDER

// If not debugging the 1st order step, flips on debugging of the 2nd order step
#ifndef DBG_FIRSTORDER
#define DBG_SECONDORDER
#else
#warning "WARNING: Compiling cudaFluidStep to take 1st order time steps [dump wStep array straight to output]"
#endif

#define DBG_NUMARRAYS 6

#ifdef DBG_FIRSTORDER
#define DBGSAVE(n, x) if(thisThreadDelivers) { Qout[((n)+6)*DEV_SLABSIZE] = (x); }
#else
#define DBGSAVE(n, x) if(thisThreadDelivers) {  Qin[((n)+6)*DEV_SLABSIZE] = (x); }
#endif

// Assuming debug has been put on the wStepValues array, download it to a Matlab array
void returnDebugArray(MGArray *ref, int x, double **wStepValues, mxArray *plhs[])
{
  CHECK_CUDA_ERROR("entering returnDebugArray");

  MGArray m = *ref;
  
  int nd = 3;
  if(m.dim[2] == 1) {
    nd = 2;
    if(m.dim[1] == 1) {
      nd = 1;
    }
  }
  nd = 4;
  mwSize odims[4];
  odims[0] = m.dim[0];
  odims[1] = m.dim[1];
  odims[2] = m.dim[2];
  odims[3] = x;

  // Create output numeric array
  plhs[0] = mxCreateNumericArray(nd, odims, mxDOUBLE_CLASS, mxREAL);

  double *result = mxGetPr(plhs[0]);

  int sub[6];
  int htrim[6];

  int u, v, w, i;
  int64_t iT, iS;
  double *gmem = NULL;

  if(m.haloSize == PARTITION_CLONED) {
    printf("Dude, can't return debug stuff from cloned partitions. How did this even happen? cudaFluidStep is the only function that's such a douchebag this is needed\n");
    return;
  }

int y;
for(y = 0; y < x; y++) {
  for(i = 0; i < m.nGPUs; i++) {
    // Get this partition's extent
    calcPartitionExtent(&m, i, &sub[0]);
    // allocate CPU memory to dump it back
    gmem = (double *)realloc((void *)gmem, m.partNumel[i]*sizeof(double));

    // Trim the halo away when copying back to CPU
    for(u = 0; u < 6; u++) { htrim[u] = sub[u]; }
    if(i < (m.nGPUs-1)) { htrim[3+m.partitionDir-1] -= m.haloSize; }
    if(i > 0)           { htrim[m.partitionDir-1] += m.haloSize; htrim[3+m.partitionDir-1] -= m.haloSize; }

    // Download the whole thing from the GPU to CPU
    //cudaSetDevice(m.deviceID[i]);
    CHECK_CUDA_ERROR("cudaSetDevice()");
    cudaError_t fail = cudaMemcpy((void *)gmem, (void *)(wStepValues[i] + (y+6)*m.slabPitch[i]/sizeof(double)), m.partNumel[i]*sizeof(double), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR("GPU_download to host.");

    // Copy into the output array
    for(w = htrim[2]; w < htrim[2]+htrim[5]; w++)
      for(v = htrim[1]; v < htrim[1]+htrim[4]; v++)
        for(u = htrim[0]; u < htrim[0] + htrim[3]; u++) {
          iT = (u-sub[0]) + sub[3]*(v - sub[1]  + sub[4]*(w-sub[2]));
          iS = u+m.dim[0]*(v+m.dim[1]*w);

          result[iS] = gmem[iT];
        }

  }
result += m.dim[0]*m.dim[1]*m.dim[2];
}
  free(gmem);

  return;
}
