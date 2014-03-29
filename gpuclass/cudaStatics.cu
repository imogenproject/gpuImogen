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

/* X DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS FOR MIRROR BCS */
/* Assume a block size of [3 A B] */
__global__ void cukern_xminusSymmetrize(double *phi, int nx, int ny, int nz);
__global__ void cukern_xminusAntisymmetrize(double *phi, int nx, int ny, int nz);
__global__ void cukern_xplusSymmetrize(double *phi, int nx, int ny, int nz);
__global__ void cukern_xplusAntisymmetrize(double *phi, int nx, int ny, int nz);
/* Y DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS */
/* assume a block size of [N 1 M] */
__global__ void cukern_yminusSymmetrize(double *phi, int nx, int ny, int nz);
__global__ void cukern_yminusAntisymmetrize(double *phi, int nx, int ny, int nz);
__global__ void cukern_yplusSymmetrize(double *phi, int nx, int ny, int nz);
__global__ void cukern_yplusAntisymmetrize(double *phi, int nx, int ny, int nz);
/* Z DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS */
/* Assume launch with size [U V 1] */
__global__ void cukern_zminusSymmetrize(double *Phi, int nx, int ny, int nz);
__global__ void cukern_zminusAntisymmetrize(double *Phi, int nx, int ny, int nz);
__global__ void cukern_zplusSymmetrize(double *Phi, int nx, int ny, int nz);
__global__ void cukern_zplusAntisymmetrize(double *Phi, int nx, int ny, int nz);

__global__ void cukern_applySpecial_fade(double *phi, double *statics, int nSpecials, int blkOffset);

void setBoundarySAS(double *gpuarray, ArrayMetadata *amd, int side, int direction, int sas);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if( (nlhs != 0) || (nrhs != 3)) { mexErrMsgTxt("cudaStatics operator is cudaStatics(ImogenArray, blockdim, direction)"); }

  CHECK_CUDA_ERROR("entering cudaStatics");

  ArrayMetadata ama, amf;

  /* This will force an error exit if invalid */
  double **array = getGPUSourcePointers(prhs, &ama, 0, 0);

  /* Grabs the whole boundaryData struct from the ImogenArray class */
  mxArray *boundaryData = mxGetProperty(prhs[0], 0, "boundaryData");
  if(boundaryData == NULL) mexErrMsgTxt("FATAL: field 'boundaryData' D.N.E. in class. Not a class? Not an ImogenArray?\n");

  /* The statics describe "solid" structures which we force the grid to have */
  mxArray *gpuStatics = mxGetField(boundaryData, 0, "staticsData");
  if(gpuStatics == NULL) mexErrMsgTxt("FATAL: field 'staticsData' D.N.E. in boundaryData struct. Statics weren't compiled?\n");
  double **statics = getGPUSourcePointers((const mxArray **)(&gpuStatics), &amf, 0, 0);

  /* The indexPermute property tells us how the array's indices are currently oriented. */
  mxArray *permArray =  mxGetProperty(prhs[0], 0, "indexPermute");
  if(permArray == NULL) mexErrMsgTxt("FATAL: field 'indexPermute' D.N.E. in class. Not an ImogenArray?\n");
  double *perm = mxGetPr(permArray);
  int offsetidx = 2*(perm[0]-1) + 1*(perm[1] > perm[2]);

  /* The offset array describes the index offsets for the data in the gpuStatics array */
  mxArray *offsets    = mxGetField(boundaryData, 0, "compOffset");
  if(offsets == NULL) mexErrMsgTxt("FATAL: field 'compOffset' D.N.E. in boundaryData. Not an ImogenArray? Statics not compiled?\n");
  double *offsetcount = mxGetPr(offsets);
  long int staticsOffset = (long int)offsetcount[2*offsetidx];
  int staticsNumel  = (int)offsetcount[2*offsetidx+1];

  /* Parameter describes what block size to launch with... */
  int blockdim = (int)*mxGetPr(prhs[1]);

  dim3 griddim; griddim.x = staticsNumel / blockdim + 1;
  if(griddim.x > 32768) {
    griddim.x = 32768;
    griddim.y = staticsNumel/(blockdim*griddim.x) + 1;
    }

  /* Every call results in applying specials */
  cukern_applySpecial_fade<<<griddim, blockdim>>>(array[0], statics[0] + staticsOffset, staticsNumel, amf.dim[0]);

  CHECK_CUDA_LAUNCH_ERROR(blockdim, griddim, &ama, 0, "cuda statics application");

  /* Indicates which part of a 3-vector this array is (0 = scalar, 123=XYZ) */
  int vectorComponent = (int)(*mxGetPr(mxGetProperty(prhs[0], 0, "component")) );
  

  /* BEGIN DETERMINATION OF ANALYTIC BOUNDARY CONDITIONS */
  int numDirections = mxGetNumberOfElements(prhs[2]);
  if(numDirections > 3) {
    mexErrMsgTxt("More than 3 directions specified to apply boundary conditions to. We only have 3...?\n");
    }
  double *directionToSet = mxGetPr(prhs[2]);

  mxArray *bcModes = mxGetField(boundaryData, 0, "bcModes");
  if(bcModes == NULL) mexErrMsgTxt("FATAL: bcModes structure not present. defective class detected.\n");

  int j;
  for(j = 0; j < numDirections; j++) {
    if((int)directionToSet[j] == 0) continue; /* Skips edge BCs if desired. */
    int trueDirect = (int)perm[(int)directionToSet[j]-1];

    /* So this is kinda dain-bramaged, but the boundary condition modes are stored in the form
       { 'type minus x', 'type minus y', 'type minus z';
         'type plus  x', 'type plus y',  'type plus z'};
       Yes, strings in a cell array. But hey, you can totally read that off by eye if you're
       in Matlab debug mode and the desire to print it out strikes you. */

    mxArray *bcstr; char *bs;
    
    int d; for(d = 0; d < 2; d++) {
      bcstr = mxGetCell(bcModes, 2*(trueDirect-1) + d);
      bs = (char *)malloc(sizeof(char) * (mxGetNumberOfElements(bcstr)+1));
      mxGetString(bcstr, bs, mxGetNumberOfElements(bcstr)+1);

      if(strcmp(bs, "mirror") == 0)
        setBoundarySAS(array[0], &ama, d, (int)directionToSet[j], vectorComponent == trueDirect);
       
      if(strcmp(bs, "const") == 0) {
//      ...
      }
      if(strcmp(bs, "linear") == 0) {
//      ...
      }
      
    }
  }

  free(array);
  free(statics);

}

/* Sets the given array+AMD's boundary in the following manner:
   side      -> 0 = negative edge  1 = positive edge
   direction -> 1 = X              2 = Y               3 = Z*
   sas       -> 0 = symmetric      1 => antisymmetric

   *: As passed, assuming ImogenArray's indexPermute has been handled for us.
   */

void setBoundarySAS(double *gpuarray, ArrayMetadata *amd, int side, int direction, int sas)
{
dim3 blockdim, griddim;

void *kerntable[12] = {(void *)&cukern_xminusSymmetrize, \
                       (void *)&cukern_xminusAntisymmetrize, \
		       (void *)&cukern_xplusSymmetrize, \
		       (void *)&cukern_xplusAntisymmetrize,
                       (void *)&cukern_yminusSymmetrize, \
                       (void *)&cukern_yminusAntisymmetrize, \
                       (void *)&cukern_yplusSymmetrize, \
                       (void *)&cukern_yplusAntisymmetrize,
                       (void *)&cukern_zminusSymmetrize, \
                       (void *)&cukern_zminusAntisymmetrize, \
                       (void *)&cukern_zplusSymmetrize, \
                       (void *)&cukern_zplusAntisymmetrize };

void (* bckernel)(double *, int, int, int) = (void (*)(double *, int, int, int))kerntable[sas + 2*side + 4*(direction-1)];

switch(direction) {
  case 1: {
    blockdim.x = 3;
    blockdim.y = 16;
    blockdim.z = 16;
    griddim.x = amd->dim[1] / blockdim.y; griddim.x += (griddim.x*blockdim.y < amd->dim[1]);
    griddim.y = amd->dim[2] / blockdim.z; griddim.y += (griddim.y*blockdim.z < amd->dim[2]);
    }; break;
  case 2: {
    blockdim.x = 16;
    blockdim.y = 1;
    blockdim.z = 16;
    griddim.x = amd->dim[0] / blockdim.x; griddim.x += (griddim.x*blockdim.x < amd->dim[0]);
    griddim.y = amd->dim[2] / blockdim.z; griddim.y += (griddim.y*blockdim.z < amd->dim[2]);
    } break;
  case 3: {
    blockdim.x = 16;
    blockdim.y = 16;
    blockdim.z = 1;
    griddim.x = amd->dim[0] / blockdim.x; griddim.x += (griddim.x*blockdim.x < amd->dim[0]);
    griddim.y = amd->dim[1] / blockdim.y; griddim.y += (griddim.y*blockdim.y < amd->dim[1]);
    } break;
  }

bckernel<<<griddim, blockdim>>>(gpuarray, amd->dim[0], amd->dim[1], amd->dim[2]);

CHECK_CUDA_LAUNCH_ERROR(blockdim, griddim, amd, sas + 2*side + 4*direction, "In setBoundarySAS; integer -> cukern table index");

return;
}

__global__ void cukern_applySpecial_fade(double *phi, double *statics, int nSpecials, int blkOffset)
{
int myAddr = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x*blockIdx.y);
if(myAddr >= nSpecials) return;

long int xaddr = (long int)statics[myAddr];
double f0      =           statics[myAddr + blkOffset];
double c       =           statics[myAddr + blkOffset*2];

phi[xaddr] = f0*c + (1.0-c)*phi[xaddr];

}

/* X DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS FOR MIRROR BCS */
/* Assume a block size of [3 A B] with grid dimensions [M N 1] s.t. AM >= ny, BN >= nz*/
/* Define the preamble common to all of these kernels: */
#define XSASKERN_PREAMBLE \
int stridey = nx; int stridez = nx*ny; \
int yidx = threadIdx.y + blockIdx.x*blockDim.y; \
int zidx = threadIdx.z + blockIdx.y*blockDim.z; \
if(yidx >= ny) return; if(zidx >= nz) return;


__global__ void cukern_xminusSymmetrize(double *phi, int nx, int ny, int nz)
{
XSASKERN_PREAMBLE

phi += stridey*yidx + stridez*zidx;
phi[2-threadIdx.x] = phi[4+threadIdx.x];
}

__global__ void cukern_xminusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
XSASKERN_PREAMBLE

phi += stridey*yidx + stridez*zidx;
phi[2-threadIdx.x] = -phi[4+threadIdx.x];
}

__global__ void cukern_xplusSymmetrize(double *phi, int nx, int ny, int nz)
{
XSASKERN_PREAMBLE

phi += stridey*yidx + stridez*zidx + nx - 7;
phi[4+threadIdx.x] = phi[2-threadIdx.x];
}

__global__ void cukern_xplusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
XSASKERN_PREAMBLE

phi += stridey*yidx + stridez*zidx + nx - 7;
phi[4+threadIdx.x] = -phi[2-threadIdx.x];
}

/* Y DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS */
/* assume a block size of [A 1 B] with grid dimensions [M N 1] s.t. AM >= nx, BN >=nz */
#define YSASKERN_PREAMBLE \
int xidx = threadIdx.x + blockIdx.x*blockDim.x; \
int zidx = threadIdx.z + blockIdx.y*blockDim.y; \
if(xidx >= nx) return; if(zidx >= nz) return;   \
phi += nx*ny*zidx; 

__global__ void cukern_yminusSymmetrize(double *phi, int nx, int ny, int nz)
{
YSASKERN_PREAMBLE
int q;
for(q = 0; q < 3; q++) { phi[xidx+nx*q] = phi[xidx+nx*(6-q)]; }
}

__global__ void cukern_yminusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
YSASKERN_PREAMBLE
int q;
for(q = 0; q < 3; q++) { phi[xidx+nx*q] = -phi[xidx+nx*(6-q)]; }
}

__global__ void cukern_yplusSymmetrize(double *phi, int nx, int ny, int nz)
{
YSASKERN_PREAMBLE
int q;
for(q = 0; q < 3; q++) { phi[xidx-nx*q] = phi[xidx+nx*(q-6)]; }
}

__global__ void cukern_yplusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
YSASKERN_PREAMBLE
int q;
for(q = 0; q < 3; q++) { phi[xidx-nx*q] = -phi[xidx+nx*(q-6)]; }
}

/* Z DIRECTION SYMMETRIC/ANTISYMMETRIC BC KERNELS */
/* Assume launch with size [A B 1] and grid of size [M N 1] s.t. AM >= nx, BN >= ny*/
#define ZSASKERN_PREAMBLE \
int xidx = threadIdx.x + blockIdx.x * blockDim.x; \
int yidx = threadIdx.y + blockIdx.y * blockDim.y; \
if(xidx >= nx) return; if(yidx >= ny) return; \
phi += xidx + nx*yidx;

__global__ void cukern_zminusSymmetrize(double *phi, int nx, int ny, int nz)
{
ZSASKERN_PREAMBLE

double p[3];
int stride = nx*ny;

p[0] = phi[4*stride];
p[1] = phi[5*stride];
p[2] = phi[6*stride];

phi[  0     ] = p[2];
phi[  stride] = p[1];
phi[2*stride] = p[0];
}

__global__ void cukern_zminusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
ZSASKERN_PREAMBLE

double p[3];
int stride = nx*ny;

p[0] = phi[4*stride];
p[1] = phi[5*stride];
p[2] = phi[6*stride];

phi[  0     ] = -p[2];
phi[  stride] = -p[1];
phi[2*stride] = -p[0];
}

__global__ void cukern_zplusSymmetrize(double *phi, int nx, int ny, int nz)
{
ZSASKERN_PREAMBLE

double p[3];
int stride = nx*ny;

p[0] = phi[0];
p[1] = phi[stride];
p[2] = phi[2*stride];

phi[4*stride] = p[2];
phi[5*stride] = p[1];
phi[6*stride] = p[0];
}

__global__ void cukern_zplusAntisymmetrize(double *phi, int nx, int ny, int nz)
{
ZSASKERN_PREAMBLE

double p[3];
int stride = nx*ny;

p[0] = phi[0];
p[1] = phi[stride];
p[2] = phi[2*stride];

phi[4*stride] = -p[2];
phi[5*stride] = -p[1];
phi[6*stride] = -p[0];

}


