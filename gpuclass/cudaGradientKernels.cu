
#ifdef NOMATLAB
#include "stdio.h"
#endif


#include "cuda.h"
#include "cudaCommon.h"
#include "cudaUtilities.h"
#include "cudaGradientKernels.h"

__constant__ double devLambda[12];

// compute grad(phi) in XYZ or R-Theta-Z with 2nd or 4th order accuracy
template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient3D_h2(double *phi, double *f_x, double *f_y, double *f_z, int3 arraysize);
template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient3D_h4_partone(double *phi, double *fx, double *fy, int3 arraysize);
__global__ void  cukern_computeScalarGradient3D_h4_parttwo(double *phi, double *fz, int3 arraysize);

// compute grad(phi) in X-Y or R-Theta with 2nd or 4th order accuracy
template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient2D_h2(double *phi, double *fx, double *fy, int3 arraysize);
template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient2D_h4(double *phi, double *fx, double *fy, int3 arraysize);

// Compute grad(phi) in X-Z or R-Z with 2nd or 4th order accuracy
__global__ void  cukern_computeScalarGradientRZ_h2(double *phi, double *fx, double *fz, int3 arraysize);
__global__ void  cukern_computeScalarGradientRZ_h4(double *phi, double *fx, double *fz, int3 arraysize);

#define GRADBLOCKX 18
#define GRADBLOCKY 18

// scalingParameter / 2h or /12h depending on spatial order of scheme
#define LAMX devLambda[0]
#define LAMY devLambda[1]
#define LAMZ devLambda[2]

#define RINNER devLambda[7]
#define DELTAR devLambda[8]


/* Given a scalar MGArray *phi and a three-slab MGArray *gradient, uses geometry info to compute the
 * gradient of phi into gradient using centered differences of order zero (return zeros), 2 (2-pt central difference)
 * or 4 (4pt central difference), multiplied by arbitrary scalar scalingParameter.
 */
int computeCentralGradient(MGArray *phi, MGArray *gradient, GeometryParams geom, int spaceOrder, double scalingParameter)
{
	dim3 gridsize, blocksize;
	//int3 arraysize;

	double lambda[11];

	int i;
	int worked;
	int sub[6];

	double *dx = &geom.h[0];
	if(spaceOrder == 4) {
		lambda[0] = scalingParameter/(12.0*dx[0]);
		lambda[1] = scalingParameter/(12.0*dx[1]);
		lambda[2] = scalingParameter/(12.0*dx[2]);
	} else {
		lambda[0] = scalingParameter/(2.0*dx[0]);
		lambda[1] = scalingParameter/(2.0*dx[1]);
		lambda[2] = scalingParameter/(2.0*dx[2]);
	}

	lambda[7] = geom.Rinner; // This is actually overwritten per partition below
	lambda[8] = dx[1];

	int isThreeD = (phi->dim[2] > 1);
	int isRZ = (phi->dim[2] > 1) & (phi->dim[1] == 1);

	for(i = 0; i < phi->nGPUs; i++) {
		cudaSetDevice(phi->deviceID[i]);
		calcPartitionExtent(phi, i, &sub[0]);

		lambda[7] = geom.Rinner + dx[0] * sub[0]; // Innermost cell coord may change per-partition

		cudaMemcpyToSymbol((const void *)devLambda, lambda, 11*sizeof(double), 0, cudaMemcpyHostToDevice);
		worked = CHECK_CUDA_ERROR("cudaMemcpyToSymbol");
		if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) break;

		//cudaMemcpyToSymbol((const void *)devIntParams, &sub[3], 3*sizeof(int), 0, cudaMemcpyHostToDevice);
		//worked = CHECK_CUDA_ERROR("memcpy to symbol");
		//if(worked != SUCCESSFUL) break;
	}

	if(worked != SUCCESSFUL) return worked;

	double *phiPtr;
	double *gradPtr;
	long slabsize;

	// Iterate over all partitions, and here we GO!
	for(i = 0; i < phi->nGPUs; i++) {
		cudaSetDevice(phi->deviceID[i]);
		worked = CHECK_CUDA_ERROR("cudaSetDevice");
		if(worked != SUCCESSFUL) break;

		calcPartitionExtent(phi, i, sub);

		int3 arraysize; arraysize.x = sub[3]; arraysize.y = sub[4]; arraysize.z = sub[5];
		dim3 blocksize(GRADBLOCKX, GRADBLOCKY, 1);
		gridsize.x = arraysize.x / (blocksize.x - spaceOrder);
		gridsize.x += ((blocksize.x-spaceOrder) * gridsize.x < arraysize.x);
		if(isRZ) {
			gridsize.y = arraysize.z / (blocksize.y - spaceOrder); gridsize.y += ((blocksize.y-spaceOrder) * gridsize.y < arraysize.z);
		} else {
			gridsize.y = arraysize.y / (blocksize.y - spaceOrder); gridsize.y += ((blocksize.y-spaceOrder) * gridsize.y < arraysize.y);
		}
		gridsize.z = 1;

		phiPtr = phi->devicePtr[i]; // WARNING: this could be garbage if spaceOrder == 0 and we rx'd no potential array
		gradPtr = gradient->devicePtr[i];
		slabsize = gradient->slabPitch[i] / 8;

		switch(spaceOrder) {
		case 0:
			// dump zeros so as to have a technically-valid result and not cause reads of uninitialized memory
			writeScalarToVector<<<32, 256>>>(gradPtr + 0 * slabsize, phi->partNumel[i], 0.0);
			writeScalarToVector<<<32, 256>>>(gradPtr + 1 * slabsize, phi->partNumel[i], 0.0);
			writeScalarToVector<<<32, 256>>>(gradPtr + 2 * slabsize, phi->partNumel[i], 0.0);
			break;
		case 2:
			if(isThreeD) {
				if(isRZ) {
					cukern_computeScalarGradientRZ_h2<<<gridsize, blocksize>>>(phiPtr, gradPtr, gradPtr+2*slabsize, arraysize);
					writeScalarToVector<<<32, 256>>>(gradPtr + slabsize, phi->partNumel[i], 0.0);
				} else {
					if(geom.shape == SQUARE) {
						cukern_computeScalarGradient3D_h2<SQUARE><<<gridsize, blocksize>>> (phiPtr, gradPtr, gradPtr + slabsize, gradPtr + slabsize*2, arraysize); }
					if(geom.shape == CYLINDRICAL) {
						cukern_computeScalarGradient3D_h2<CYLINDRICAL><<<gridsize, blocksize>>> (phiPtr, gradPtr, gradPtr + slabsize, gradPtr+ slabsize*2, arraysize); }
				}
			} else {
				if(geom.shape == SQUARE) {
					cukern_computeScalarGradient2D_h2<SQUARE><<<gridsize, blocksize>>>(phiPtr, gradPtr, gradPtr + slabsize, arraysize); }
				if(geom.shape == CYLINDRICAL) {
					cukern_computeScalarGradient2D_h2<CYLINDRICAL><<<gridsize, blocksize>>>(phiPtr, gradPtr, gradPtr+ slabsize, arraysize); }

				writeScalarToVector<<<32, 256>>>(gradPtr+2*slabsize, phi->partNumel[i], 0.0);
			}
			break;
		case 4:
			if(isThreeD) {
				if(isRZ) {
					cukern_computeScalarGradientRZ_h4<<<gridsize, blocksize>>>(phiPtr, gradPtr, gradPtr + 2*phi->partNumel[i],  arraysize);
					writeScalarToVector<<<32, 256>>>(gradPtr + slabsize, phi->partNumel[i], 0.0);
				} else {
					if(geom.shape == SQUARE) {
						cukern_computeScalarGradient3D_h4_partone<SQUARE><<<gridsize, blocksize>>>(phiPtr, gradPtr, gradPtr+ slabsize, arraysize);
						cukern_computeScalarGradient3D_h4_parttwo<<<gridsize, blocksize>>>(phiPtr, gradPtr+ slabsize*2, arraysize);
					}
					if(geom.shape == CYLINDRICAL) {
						cukern_computeScalarGradient3D_h4_partone<CYLINDRICAL><<<gridsize, blocksize>>>(phiPtr, gradPtr, gradPtr+ slabsize, arraysize);
						cukern_computeScalarGradient3D_h4_parttwo<<<gridsize, blocksize>>>(phiPtr, gradPtr+ slabsize*2, arraysize);
					}
				}
			} else {
				if(geom.shape == SQUARE) {
					cukern_computeScalarGradient2D_h4<SQUARE><<<gridsize, blocksize>>>(phiPtr, gradPtr, gradPtr+ slabsize, arraysize); }
				if(geom.shape == CYLINDRICAL) {
					cukern_computeScalarGradient2D_h4<CYLINDRICAL><<<gridsize, blocksize>>>(phiPtr, gradPtr, gradPtr+ slabsize, arraysize); }

				writeScalarToVector<<<32, 256>>>(gradPtr+2*phi->partNumel[i], phi->partNumel[i], 0.0);

			}

			break;
		default:
			PRINT_FAULT_HEADER;
			printf("Was passed spatial order parameter of %i, must be passed 0 (off), 2 (2nd order), or 4 (4th order)\n", spaceOrder);
			PRINT_FAULT_FOOTER;
			return ERROR_INVALID_ARGS;
		}

		worked = CHECK_CUDA_LAUNCH_ERROR(blocksize, gridsize, phi, i, "cukern_computeScalarGradient");
		if(worked != SUCCESSFUL) break;
	}

	if(worked != SUCCESSFUL) return worked;

	// FIXME this needs to either understand slabs, or we need to fetch 3 slab ptrs into an array & pass it instead
	//    worked = MGA_exchangeLocalHalos(gradient, 5); // need to?
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) return worked;

	return CHECK_IMOGEN_ERROR(worked);

}

/* Second order methods compute dU/dx using the 2-point central derivative,
 *     dU/dx = [ -f(x-h) + f(x+h) ] / 2h + O(h^2)
 * Fourth order methods compute dU/dx using the 4-point central derivative,
 *     dU/dx = [ f(x-2h) - 8 f(x-h) + 8 f(x+h) - f(x+2h) ] / 12h + O(h^4)
 * applied independently to the directions of interest.
 * Phi-direction derivatives in cylindrical geometry acquire an additional factor of 1/r
 * because lambda computes dU/dtheta in this case, not (grad U).(theta-hat).
 */

/* Computes the gradient of 3d array phi using the 2-point centered derivative,
 * and stores phi_x in fx, phi_y in fy, phi_z in fz.
 * All arrays (rho, phi, fx, fy, fz) must be of size arraysize.
 * In cylindrical geometry, f_x -> f_r,
 *                          f_y -> f_phi
 */
template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient3D_h2(double *phi, double *fx, double *fy, double *fz, int3 arraysize)
{
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	int myX = threadIdx.x + (GRADBLOCKX-2)*blockIdx.x - 1;
	int myY = threadIdx.y + (GRADBLOCKY-2)*blockIdx.y - 1;

	if((myX > arraysize.x) || (myY > arraysize.y)) return;

	bool IWrite = (threadIdx.x > 0) && (threadIdx.x < (GRADBLOCKX-1)) && (threadIdx.y > 0) && (threadIdx.y < (GRADBLOCKY-1));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.y);

	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.y) % arraysize.y;

	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction

	__shared__ double phiA[GRADBLOCKX*GRADBLOCKY];
	__shared__ double phiB[GRADBLOCKX*GRADBLOCKY];
	__shared__ double phiC[GRADBLOCKX*GRADBLOCKY];

	double *U; double *V; double *W;
	double *temp;

	U = phiA; V = phiB; W = phiC;

	// Preload lower and middle planes
	U[myLocAddr] = phi[globAddr + arraysize.x*arraysize.y*(arraysize.z-1)];
	V[myLocAddr] = phi[globAddr];

	__syncthreads();

	int z;
	int deltaz = arraysize.x*arraysize.y;
	for(z = 0; z < arraysize.z; z++) {
		if(z >= arraysize.z - 1) deltaz = - arraysize.x*arraysize.y*(arraysize.z-1);

		if(IWrite) {
			deltaphi         = LAMX*(V[myLocAddr+1]-V[myLocAddr-1]);
			fx[globAddr]     = deltaphi; // store px <- px - dt * rho dphi/dx;
		}

		if(IWrite) {
			if(coords == SQUARE) {
				deltaphi         = LAMY*(V[myLocAddr+GRADBLOCKX]-V[myLocAddr-GRADBLOCKX]);
			}
			if(coords == CYLINDRICAL) {
				// In cylindrical coords, use dt/dphi * (delta-phi) / r to get d/dy
				deltaphi         = LAMY*(V[myLocAddr+GRADBLOCKX]-V[myLocAddr-GRADBLOCKX]) / (RINNER + DELTAR*myX);
			}
			fy[globAddr]     = deltaphi;
		}

		W[myLocAddr]       = phi[globAddr + deltaz]; // load phi(z+1) -> phiC
		__syncthreads();
		deltaphi           = LAMZ*(W[myLocAddr] - U[myLocAddr]);

		if(IWrite) {
			fz[globAddr]     = deltaphi;
		}

		temp = U; U = V; V = W; W = temp; // cyclically shift them back
		globAddr += arraysize.x * arraysize.y;

		__syncthreads();

	}

}

/* Computes the gradient of 3d array phi using the 4-point centered derivative and
 * stores phi_x in fx, phi_y in fy, phi_z in fz.
 * All arrays (rho, phi, fx, fy, fz) must be of size arraysize.
 * In cylindrical geometry, f_x -> f_r,
 *                          f_y -> f_phi
 * This call must be invoked in two parts:
 * cukern_computeScalarGradient3D_h4_partone computes the X and Y (or r/theta) derivatives,
 * cukern_computeScalarGradient3D_h4_parttwo computes the Z derivative.
 */
template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient3D_h4_partone(double *phi, double *fx, double *fy, int3 arraysize)
{
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	int myX = threadIdx.x + (GRADBLOCKX-4)*blockIdx.x - 2;
	int myY = threadIdx.y + (GRADBLOCKY-4)*blockIdx.y - 2;

	if((myX > (arraysize.x+1)) || (myY > (arraysize.y+1))) return;

	bool IWrite = (threadIdx.x > 1) && (threadIdx.x < (GRADBLOCKX-2)) && (threadIdx.y > 1) && (threadIdx.y < (GRADBLOCKY-2));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.y);

	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.y) % arraysize.y;

	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction

	__shared__ double phishm[GRADBLOCKX*GRADBLOCKY];

	__syncthreads();

	int z;
	int deltaz = arraysize.x*arraysize.y;
	for(z = 0; z < arraysize.z; z++) {
		phishm[myLocAddr] = phi[globAddr];

		__syncthreads();

		if(IWrite) {
			deltaphi         = LAMX*(-phishm[myLocAddr+2]+8.0*phishm[myLocAddr+1]-8.0*phishm[myLocAddr-1]+phishm[myLocAddr-2]);
			fx[globAddr]     = deltaphi; // store px <- px - dt * rho dphi/dx;

			if(coords == SQUARE) {
				deltaphi         = LAMY*(-phishm[myLocAddr+2*GRADBLOCKX]+8*phishm[myLocAddr+GRADBLOCKX]-8*phishm[myLocAddr-GRADBLOCKX]+phishm[myLocAddr-2*GRADBLOCKX]);
			}
			if(coords == CYLINDRICAL) {
				// In cylindrical coords, use dt/dphi * (delta-phi) / r to get d/dy
				deltaphi         = LAMY*(-phishm[myLocAddr+2*GRADBLOCKX]+8*phishm[myLocAddr+GRADBLOCKX]-8*phishm[myLocAddr-GRADBLOCKX]+phishm[myLocAddr-2*GRADBLOCKX]) / (RINNER + DELTAR*myX);
			}
			fy[globAddr]     = deltaphi;
		}

		globAddr += deltaz;
	}
}

/* 2nd part of 4th order 3D spatial gradient computes d/dz (same in cart & cyl coords so no template */
__global__ void  cukern_computeScalarGradient3D_h4_parttwo(double *phi, double *fz, int3 arraysize)
{
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	int myX = threadIdx.x + (GRADBLOCKX-4)*blockIdx.x - 2;
	int myZ = threadIdx.y + (GRADBLOCKY-4)*blockIdx.y - 2;

	if((myX > (arraysize.x+1)) || (myZ > (arraysize.z+1))) return;

	bool IWrite = (threadIdx.x > 1) && (threadIdx.x < (GRADBLOCKX-2)) && (threadIdx.y > 1) && (threadIdx.y < (GRADBLOCKY-2));
	IWrite = IWrite && (myX < arraysize.x) && (myZ < arraysize.z);

	myX = (myX + arraysize.x) % arraysize.x;
	myZ = (myZ + arraysize.z) % arraysize.z;

	int delta = arraysize.x*arraysize.y;

	int globAddr = myX + delta*myZ;

	double deltaphi; // Store derivative of phi in one direction

	__shared__ double phishm[GRADBLOCKX*GRADBLOCKY];

	__syncthreads();

	int y;
	for(y = 0; y < arraysize.y; y++) {
		phishm[myLocAddr] = phi[globAddr];

		__syncthreads();

		if(IWrite) {
			deltaphi         = LAMZ*(-phishm[myLocAddr+2*GRADBLOCKX]+8*phishm[myLocAddr+GRADBLOCKX]-8*phishm[myLocAddr-GRADBLOCKX]+phishm[myLocAddr-2*GRADBLOCKX]);
			fz[globAddr]     = deltaphi;
		}
		globAddr += arraysize.x;
	}
}

/* Compute the gradient of 2d array phi with 2nd order accuracy; store the results in f_x, f_y
 *    In cylindrical geometry, f_x -> f_r,
 *                             f_y -> f_phi
 */
template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient2D_h2(double *phi, double *fx, double *fy, int3 arraysize)
{
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	int myX = threadIdx.x + (GRADBLOCKX-2)*blockIdx.x - 1;
	int myY = threadIdx.y + (GRADBLOCKY-2)*blockIdx.y - 1;

	if((myX > arraysize.x) || (myY > arraysize.y)) return;

	bool IWrite = (threadIdx.x > 0) && (threadIdx.x < (GRADBLOCKX-1)) && (threadIdx.y > 0) && (threadIdx.y < (GRADBLOCKY-1));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.y);

	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.y) % arraysize.y;

	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction
	__shared__ double phiLoc[GRADBLOCKX*GRADBLOCKY];

	phiLoc[myLocAddr] = phi[globAddr];

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		// compute dt * (dphi/dx)
		deltaphi         = LAMX*(phiLoc[myLocAddr+1]-phiLoc[myLocAddr-1]);
		fx[globAddr] = deltaphi;

		// Calculate dt*(dphi/dy)
		if(coords == SQUARE) {
			deltaphi         = LAMY*(phiLoc[myLocAddr+GRADBLOCKX]-phiLoc[myLocAddr-GRADBLOCKX]);
		}
		if(coords == CYLINDRICAL) {
			// Converts d/dphi into physical distance based on R
			deltaphi         = LAMY*(phiLoc[myLocAddr+GRADBLOCKX]-phiLoc[myLocAddr-GRADBLOCKX]) / (RINNER + myX*DELTAR);
		}
		fy[globAddr]     = deltaphi;
	}

}

/* Compute the gradient of 2d array phi with 4th order accuracy; store the results in f_x, f_y
 *    In cylindrical geometry, f_x -> f_r,
 *                             f_y -> f_phi
 */
template <geometryType_t coords>
__global__ void  cukern_computeScalarGradient2D_h4(double *phi, double *fx, double *fy, int3 arraysize)
{
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	// decrement two
	int myX = threadIdx.x + (GRADBLOCKX-4)*blockIdx.x - 2;
	int myY = threadIdx.y + (GRADBLOCKY-4)*blockIdx.y - 2;

	// keep two past
	if((myX > (arraysize.x+1)) || (myY > (arraysize.y+1))) return;

	// net result: this tile has a buffer of two on every side;
	// mask out two cells around the edges
	bool IWrite = (threadIdx.x > 1) && (threadIdx.x < (GRADBLOCKX-2)) && (threadIdx.y > 1) && (threadIdx.y < (GRADBLOCKY-2));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.y);

	// wrap circularly & translate to global address
	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.y) % arraysize.y;
	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction
	__shared__ double phiLoc[GRADBLOCKX*GRADBLOCKY];

	phiLoc[myLocAddr] = phi[globAddr];

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		// compute dt * (dphi/dx)
		deltaphi         = LAMX*(-phiLoc[myLocAddr+2] + 8*phiLoc[myLocAddr+1] - 8*phiLoc[myLocAddr-1] + phiLoc[myLocAddr-2]);
		fx[globAddr] = deltaphi;

		// Calculate dt*(dphi/dy)
		if(coords == SQUARE) {
			deltaphi         = LAMY*(-phiLoc[myLocAddr+2*GRADBLOCKX] + 8*phiLoc[myLocAddr+1*GRADBLOCKX] - 8*phiLoc[myLocAddr-1*GRADBLOCKX] + phiLoc[myLocAddr-2*GRADBLOCKX]);
		}
		if(coords == CYLINDRICAL) {
			// Converts d/dphi into physical distance based on R
			deltaphi         = LAMY*(-phiLoc[myLocAddr+2*GRADBLOCKX] + 8*phiLoc[myLocAddr+1*GRADBLOCKX] - 8*phiLoc[myLocAddr-1*GRADBLOCKX] + phiLoc[myLocAddr-2*GRADBLOCKX])/(RINNER + myX*DELTAR);
		}
		fy[globAddr]     = deltaphi;
	}

}

/* Compute the gradient of R-Z array phi with 2nd order accuracy; store the results in f_x, f_z
 *    In cylindrical geometry, f_x -> f_r
 */
__global__ void  cukern_computeScalarGradientRZ_h2(double *phi, double *fx, double *fz, int3 arraysize)
{
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	int myX = threadIdx.x + (GRADBLOCKX-2)*blockIdx.x - 1;
	int myY = threadIdx.y + (GRADBLOCKY-2)*blockIdx.y - 1;

	if((myX > arraysize.x) || (myY > arraysize.z)) return;

	bool IWrite = (threadIdx.x > 0) && (threadIdx.x < (GRADBLOCKX-1)) && (threadIdx.y > 0) && (threadIdx.y < (GRADBLOCKY-1));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.z);

	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.z) % arraysize.z;

	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction
	__shared__ double phiLoc[GRADBLOCKX*GRADBLOCKY];

	phiLoc[myLocAddr] = phi[globAddr];

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		// compute dt * (dphi/dx)
		deltaphi         = LAMX*(phiLoc[myLocAddr+1]-phiLoc[myLocAddr-1]);
		fx[globAddr]     = deltaphi;

		// Calculate dt*(dphi/dz)
		deltaphi         = LAMZ*(phiLoc[myLocAddr+GRADBLOCKX]-phiLoc[myLocAddr-GRADBLOCKX]);
		fz[globAddr]     = deltaphi;
	}

}

/* Compute the gradient of RZ array phi with 4th order accuracy; store the results in f_x, f_y
 *    In cylindrical geometry, f_x -> f_r,
 */
__global__ void  cukern_computeScalarGradientRZ_h4(double *phi, double *fx, double *fz, int3 arraysize)
{
	int myLocAddr = threadIdx.x + GRADBLOCKX*threadIdx.y;

	// move two left
	int myX = threadIdx.x + (GRADBLOCKX-4)*blockIdx.x - 2;
	int myY = threadIdx.y + (GRADBLOCKY-4)*blockIdx.y - 2;

	// keep two past
	if((myX > (arraysize.x+1)) || (myY > (arraysize.z+1))) return;

	// mask out two edge cells in all directions
	bool IWrite = (threadIdx.x > 1) && (threadIdx.x < (GRADBLOCKX-2)) && (threadIdx.y > 1) && (threadIdx.y < (GRADBLOCKY-2));
	IWrite = IWrite && (myX < arraysize.x) && (myY < arraysize.z);

	// circularly wrap & compute global translation
	myX = (myX + arraysize.x) % arraysize.x;
	myY = (myY + arraysize.z) % arraysize.z;
	int globAddr = myX + arraysize.x*myY;

	double deltaphi; // Store derivative of phi in one direction
	__shared__ double phiLoc[GRADBLOCKX*GRADBLOCKY];

	phiLoc[myLocAddr] = phi[globAddr];

	__syncthreads(); // Make sure loaded phi is visible

	// coupling is exactly zero if rho <= rhomin
	if(IWrite) {
		// compute dt * (dphi/dx)
		deltaphi         = LAMX*(-phiLoc[myLocAddr+2] + 8*phiLoc[myLocAddr+1] - 8*phiLoc[myLocAddr-1] + phiLoc[myLocAddr-2]);
		fx[globAddr]     = deltaphi;

		// Calculate dt*(dphi/dz)
		deltaphi         = LAMZ*(-phiLoc[myLocAddr+2*GRADBLOCKX] + 8*phiLoc[myLocAddr+1*GRADBLOCKX] - 8*phiLoc[myLocAddr-1*GRADBLOCKX] + phiLoc[myLocAddr-2*GRADBLOCKX]);
		fz[globAddr]     = deltaphi;
	}

}

