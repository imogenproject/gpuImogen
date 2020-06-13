#include "cuda.h"


/* Simple kernel:
 * Given in[0 ... (nodeN-1)], copies the segment in[partX0 ... (partX0 + partNX -1)] to out[0 ... (partNX-1)]
 * and helpfully wraps addresses circularly
 * invoke with gridDim.x * blockDim.x >= partNX
 */
__global__ void cukern_FetchPartitionSubset1D(double *in, int nodeN, double *out, int partX0, int partNX)
{
// calculate output address
int addrOut = threadIdx.x + blockDim.x * blockIdx.x;
if(addrOut >= partNX) return;

// Affine map back to input address
int addrIn = addrOut + partX0;
if(addrIn < 0) addrIn += partNX;

out[addrOut] = in[addrIn];
}

// Needed with the gradient calculators in 2D because they leave the empty directions uninitialized
// Vomits the value f into array x, from x[0] to x[numel-1]
__global__ void writeScalarToVector(double *x, long numel, double f)
{
	long a = threadIdx.x + blockDim.x*blockIdx.x;

	for(; a < numel; a+= blockDim.x*gridDim.x) {
		x[a] = f;
	}
}
