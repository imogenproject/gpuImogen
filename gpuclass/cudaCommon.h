#ifndef CUDACOMMONH_
#define CUDACOMMONH_

#include "mpi_common.h" 
// These define Imogen's equation of state, giving total pressure and square of soundspeed as functions of the other variables
// These parameters would normally be stored in a __device__ __constant__ array:
// gm1fact = gamma - 1
// alfact = (1-gamma/2)
// gg1fact = gamma*(gamma-1)
// alcoef = 1-.5*gamma*(gamma-1)

// This modifies the freezing speed calculation and can stabilize low-beta conditions
#define ALFVEN_CSQ_FACTOR 1

// gasdynamic pressure = (gamma-1)*eint = (gamma-1)*(Etot - T) = (gamma-1)*(Etot - .5*p.p/rho)
#define PRESS_HD(E, Psq, rho, gm1fact) ( (gm1fact)*((E)-.5*(Psq)/(rho)) )
// mhd total pressure  = (gamma-1)*eint + B^2/2 = (gamma-1)*(E-T-B^2/2) + B^2/2
//                                              = (gamma-1)*(E-p^2/2rho) + (2-gamma) B^2/2
#define PRESS_MHD(E, Psq, rho, Bsq, gm1fact, alfact) (  (gm1fact)*((E)-.5*((Psq)/(rho))) + alfact*(bsq)  )
// gasdynamic csq = gamma*P/rho
#define CSQ_HD(E, Psq, rho, gg1fact) ( (gg1fact)*((E) - .5*((Psq)/(rho)) ) / (rho) )
// mhd maximal c_fast^2 = v_s^2 + v_a^2 = g*P/rho + B^2/rho
//                      = g*[(g-1)*(E-T-B^2/2)]/rho + B^2/rho
//                      = [g*(g-1)*(E-T) + (1-.5*g*(g-1))B^2] / rho
#define CSQ_MHD(E, Psq, rho, Bsq, gg1fact, alcoef)   ( (gg1fact)*((E)-.5*((Psq)/(rho)) + ALFVEN_CSQ_FACTOR*(alcoef)*(bsq) )/(rho)  )

enum geometryType_t { SQUARE, CYLINDRICAL };

typedef struct __GeometryParams {
	geometryType_t shape;
	double h[3]; // dx dy dz, or dr dphi dz
	double x0, y0, z0;
	double Rinner; // Only for cylindrical geometry

	// TODO: add allocatable vectors here for variable spacing in the future
} GeometryParams;

typedef struct {
        double *fluidIn[5];
        double *fluidOut[5];

        double *B[3];

        double *Ptotal;
        double *cFreeze;
        } fluidVarPtrs;

#define ROUNDUPTO(x, y) ( x + ((x % y == 0) ? 0 : (y - x % y)) )

// warning: reduceClonedMGArray is hardcoded with the reduction algo for a max of 4 devices because lazy
#define MAX_GPUS_USED 4
#define PARTITION_X 1
#define PARTITION_Y 2
#define PARTITION_Z 3

// Never ever don't use these
        // NOTE: If this length is changed you MUST change
#define GPU_TAG_LENGTH 10
#define GPU_TAG_DIM0 0
#define GPU_TAG_DIM1 1
#define GPU_TAG_DIM2 2
#define GPU_TAG_DIMSLAB 3
#define GPU_TAG_HALO 4
#define GPU_TAG_PARTDIR 5
#define GPU_TAG_NGPUS 6
#define GPU_TAG_EXTERIORHALO 7
#define GPU_TAG_DIMPERMUTATION 8
#define GPU_TAG_CIRCULARBITS 9

        // These mask the bits in the MGArray.circularBoundaryBits field
#define MGA_BOUNDARY_XMINUS 1
#define MGA_BOUNDARY_XPLUS 2
#define MGA_BOUNDARY_YMINUS 4
#define MGA_BOUNDARY_YPLUS 8
#define MGA_BOUNDARY_ZMINUS 16
#define MGA_BOUNDARY_ZPLUS 32

/* These enumerate the reduction operators which we accelerate on the GPU */
enum MGAReductionOperator { MGA_OP_SUM, MGA_OP_PROD, MGA_OP_MAX, MGA_OP_MIN };

/* The MGArray is the basic data unit for GPU arrays
 * This is interfaced with Matlab through the serializer/deserializer functions
 * that return/decipher opaque blobs of uint64_t's.
 */
typedef struct {
    int dim[3];    // The size of the array that the CPU downloader will return
                   // This does not include halos of any kind
    int64_t numel; // = nx ny nz of the global array, not # allocated
    int numSlabs; // = dim[4] if positive, identifies which zero-indexed slab # if <= 0
    int64_t slabPitch[MAX_GPUS_USED]; // numel[partition]*sizeof(double) (BYTES ALLOCATED PER SLAB), rounded up to nearest 256

    int haloSize; // Gives the depth of the halo region on interfacial boundaries
                  // NOTE that this applies only to inter-gpu partitions created
                  // by MGA partitioning onto multiple devices: The MPI-level scheme
                  // can and usually does have inter-node halos
    int partitionDir;
                  // Indicates which direction data is split across GPUs in
                  // Should equal one of PARTITION_X, PARTITION_Y, PARTITION_Z

    /* MGArrays don't concern themselves with the cluster-wide partitioning scheme. While we always
     * have to add halo cells to the interior interfaces for nGPUs > 1, it may or may not be necessary
     * at the exterior interfaces:
     *
     * When there is only 1 node in our partitioning direction, GIS will NOT supply halo cells and it is
     * necessary that we do. When there is more than 1, the higher-level partitioning will supply
     * halo cells at the outside boundaries so we should not add them.
     * Ergo, this value is TRUE iff there is exactly one rank in the partition direction.
     */
    int addExteriorHalo;
    // As mentioned at haloSize: This indicates whether MGA needs to add an exterior
    // halo if true (typical of mpi-serial operation) or not (typical of mpi-parallel
    // where the MPI partitioning has already added it)

    int permtag;
    // Unique integer that compactly represents the 6 possible tuple->linear mappings
    // of a 3D array
    int currentPermutation[3];
    // Marks which memory stride is associated with each physical direction

    int nGPUs; // Number of devices this MGA lives on: \elem [1, MAX_GPUS_USED].
    int deviceID[MAX_GPUS_USED]; // for indexes [0, ..., MAX_GPUS_USED-1], appropriate value for cudaSetDevice() etc
    double *devicePtr[MAX_GPUS_USED]; // for indices [0, ..., MAX_GPUS_USED-1], device data pointers allocated on the matching device.
    int partNumel[MAX_GPUS_USED]; // Number of elements allocated per devicePtr (i.e. includes halo).

    // Use MGA_BOUNDARY_{XYZ}{MINUS | PLUS} defines to select
    int circularBoundaryBits;

    // Continuing my long tradition of bad boundary condition handling,
    // attach the original mxArray pointer so that cudaStatics can be re-used w/o substantial alteration.
    const mxArray *matlabClassHandle;
    int mlClassHandleIndex;
    } MGArray;

/* An in-place overwrite of an MGArray is a nontrivial thing. The following explains how to do it successfully:
 *
 * foo(MGArray *phi) {
 * MGArray tmp = *phi; MGArray *B;
 *   setNewParameters(&tmp);
 *   B = MGA_allocArrays(1, &tmp);
 *   doThingsToB(B);
 *
 *   phi[0] = B[0];
 *
 *   // Matters for slabs: MGA_allocArrays clobbers the original numSlabs param
 *   phi->numSlabs = tmp.numSlabs;
 *   // And reset original pointers
 *   phi->devicePtr[*] = tmp.devicePtr[*]; //
 *   cudaMemcpy(phi->devicePtr[*], B->devicePtr[*], ...);
 *
 *   MGA_delete(B);
 * }
 *
 */

/* getGPUTypeTag(mxArray *gputype, int64_t **tagpointer)
 *     gputype must be a Matlab array that is one of
 *       - an Nx1 vector of uint64_ts as created by MGA
 *       - a Matlab GPU_Type class
 *       - an Imogen fluid array which has a public .gpuptr property
 */
int getGPUTypeTag (const mxArray *gputype, int64_t **tagPointer);
/* getGPUTypeTagIndexed(mxArray *gputype, int64_t **tagPointer, int idx)
 *     behaves as getGPUTypeTag, but accepts an index as well such that given
 *     matlabFoo({tag0, tag1, ..., tagN})
 *     getGPUTypeTagIndexed(..., M) fetches the Mth element of the cell array
 */
int getGPUTypeTagIndexed(const mxArray *gputype, int64_t **tagPointer, int mxarrayIndex);
cudaStream_t *getGPUTypeStreams(const mxArray *gputype);

/* Not meant for user calls: Checks basic properties of the input
 * tag; Returns FALSE if they clearly violate basic properties a
 * valid gpu tag must have */
bool     sanityCheckTag(const mxArray *tag);     // sanity

void     calcPartitionExtent(MGArray *m, int P, int *sub);

void MGA_permtagToNums(int permtag, int *p);
int MGA_numsToPermtag(int *nums);

int      deserializeTagToMGArray(int64_t *tag, MGArray *mg); // array -> struct
void     serializeMGArrayToTag(MGArray *mg, int64_t *tag);   // struct -> array

/* MultiGPU Array allocation stuff */
int      MGA_accessMatlabArrays(const mxArray *prhs[], int idxFrom, int idxTo, MGArray *mg); // Extracts a series of Matlab handles into MGArrays
int      MGA_accessMatlabArrayVector(const mxArray *m, int idxFrom, int idxTo, MGArray *mg);
MGArray *MGA_allocArrays(int N, MGArray *skeleton);
int      MGA_duplicateArray(MGArray **dst, MGArray *src);
MGArray *MGA_createReturnedArrays(mxArray *plhs[], int N, MGArray *skeleton); // clone existing MG array'
void     MGA_returnOneArray(mxArray *plhs[], MGArray *m);
int     MGA_delete(MGArray *victim);

void MGA_sledgehammerSequentialize(MGArray *q);

/* MultiGPU reduction calculators */

int MGA_partitionReduceDimension(MGArray *in, MGArray *out, MGAReductionOperator operate, int dir, int partition);
int  MGA_reduceAcrossDevices(MGArray *a, MGAReductionOperator operate, int redistribute);

int MGA_localReduceDimension(MGArray *in, MGArray **out, MGAReductionOperator operate, int dir, int partitionOnto, int redistribute);
int MGA_globalReduceDimension(MGArray *in, MGArray **out, MGAReductionOperator operate, int dir, int partitionOnto, int redistribute, ParallelTopology * topology);

int MGA_localReduceScalar(MGArray *in, double *scalar, MGAReductionOperator operate);
int MGA_globalReduceScalar(MGArray *in, double *scalar, MGAReductionOperator operate, ParallelTopology * topology);

// Drops m[0...N].devicePtr[i] into dst[0...N] to avoid hueg sets of fluid[n].devicePtr[i] in calls:
void pullMGAPointers( MGArray *m, int N, int i, double **dst);

/* MultiGPU Array I/O */
int  MGA_downloadArrayToCPU(MGArray *g, double **p, int partitionFrom);
int  MGA_uploadMatlabArrayToGPU(const mxArray *m, MGArray *g, int partitionTo);

int  MGA_uploadArrayToGPU(double *p, MGArray *g, int partitionTo);
int  MGA_distributeArrayClones(MGArray *cloned, int partitionFrom);

MGAReductionOperator MGAReductionOperator_mpi2mga(MPI_Op mo);
MPI_Op MGAReductionOperator_mga2mpi(MGAReductionOperator op);

/* Functions for managing halos of MGA partitioned data */
int MGA_exchangeLocalHalos(MGArray *a, int n);
int MGA_partitionHaloNumel(MGArray *a, int partition, int direction, int h);
int MGA_wholeFaceHaloNumel(MGArray *a, int direction, int h);

int MGA_partitionHaloToLinear(MGArray *a, int partition, int direction, int right, int toHalo, int h, double **linear);
int MGA_wholeFaceToLinear(MGArray *a, int direction, int rightside, int writehalo, int h, double **linear);

// Error catchers, handlers, etc
int checkCudaError(const char *where, const char *fname, int lname);
int checkCudaLaunchError(cudaError_t E, dim3 blockdim, dim3 griddim, MGArray *a, int i, const char *srcname, const char *fname, int lname);
int checkImogenError(int errtype, const char *infile, const char *infunc, int atline);

void MGA_debugPrintAboutArray(MGArray *x);

#define PRINT_FAULT_HEADER printf("========== FAULT IN COMPILED CODE: function %s (%s:%i)\n", __func__, __FILE__, __LINE__)
#define PRINT_FAULT_FOOTER printf("========== COMPILED CODE STACK TRACE SHOULD FOLLOW: ============================\n")

#define CHECK_IMOGEN_ERROR(errtype) checkImogenError(errtype, __FILE__, __func__, __LINE__)
#define CHECK_CUDA_LAUNCH_ERROR(bsize, gsize, mg_ptr, direction, string) \
checkCudaLaunchError(cudaGetLastError(), bsize, gsize, mg_ptr, direction, string, __FILE__, __LINE__)
#define CHECK_CUDA_ERROR(astring) CHECK_IMOGEN_ERROR(checkCudaError(astring, __FILE__, __LINE__))
#define BAIL_ON_FAIL(errtype) if( CHECK_IMOGEN_ERROR(errtype) != SUCCESSFUL) { return errtype; }

void dropMexError(const char *excuse, const char *infile, int atline);
#define DROP_MEX_ERROR(dangit) dropMexError(dangit, __FILE__, __LINE__)

#define SUCCESSFUL 0
#define ERROR_INVALID_ARGS -100
#define ERROR_CRASH -101
#define ERROR_NULL_POINTER -102
#define ERROR_GET_GPUTAG_FAILED -103
#define ERROR_DESERIALIZE_GPUTAG_FAILED -104

#define ERROR_CUDA_BLEW_UP -1000

#define PAR_WARN(x) if(x.nGPUs > 1) { printf("In %s:\n", __FILE__); mexWarnMsgTxt("WARNING: This function is shimmed but multi-GPU operation WILL NOT WORK"); }

#define FATAL_NOT_IMPLEMENTED mexErrMsgTxt("Fatal: Encountered required but completely non-implemented code branch.");

typedef struct {
    int ndims;
    int dim[3];
    int numel;
    } ArrayMetadata;

void arrayMetadataToTag(ArrayMetadata *meta, int64_t *tag); 
void getTagFromGPUType(const mxArray *gputype, int64_t *tag);

double **getGPUSourcePointers(const mxArray *prhs[], ArrayMetadata *metaReturn, int fromarg, int toarg);
double **makeGPUDestinationArrays(ArrayMetadata *amdRef, mxArray *retArray[], int howmany);
double *replaceGPUArray(const mxArray *prhs[], int target, int *newdims);

int3 makeInt3(int x, int y, int z);
int3 makeInt3(int *b);
dim3 makeDim3(unsigned int x, unsigned int y, unsigned int z);
dim3 makeDim3(unsigned int *b);
dim3 makeDim3(int *b);

int MGA_localElmentwiseReduce(MGArray *in, int dir, int partitionOnto, int redistribute);
int MGA_localPancakeReduce(MGArray *in, MGArray *out, int dir, int partitionOnto, int redistribute);
int MGA_globalElementwiseReduce(MGArray *in, int dir, int partitionOnto, int redistribute, const mxArray *topo);
int MGA_globalPancakeReduce(MGArray *in, MGArray *out, int dir, int partitionOnto, int redistribute, const mxArray *topo);
int MGA_distributeArrayClones(MGArray *cloned, int partitionFrom);

// FIXME: This should go in a different file because it has nothing to do with CUDA per se...
int MGA_accessFluidCanister(const mxArray *canister, int fluidIdx, MGArray *fluid);
GeometryParams accessMatlabGeometryClass(const mxArray *geoclass);

mxArray *derefXdotAdotB(const mxArray *in, const char *fieldA, const char *fieldB);
double derefXdotAdotB_scalar(const mxArray *in, const char *fieldA, const char *fieldB);
void derefXdotAdotB_vector(const mxArray *in, const char *fieldA, const char *fieldB, double *x, int N);

void getTiledLaunchDims(int *dims, dim3 *tileDim, dim3 *halo, dim3 *blockdim, dim3 *griddim);

const char *errorName(cudaError_t E);

void printdim3(char *name, dim3 dim);
void printgputag(char *name, int64_t *tag);

__device__ __inline__ double fluxLimiter_VanLeer(double derivL, double derivR)
{
double r;

r = 2.0 * derivL * derivR;
if(r > 0.0) { return r /(derivL+derivR); }

return 0;
}

__device__ __inline__ double fluxLimiter_minmod(double derivL, double derivR)
{
if(derivL * derivR < 0) return 0.0;

if(fabs(derivL) > fabs(derivR)) { return derivR; } else { return derivL; }
}

__device__ __inline__ double fluxLimiter_superbee(double derivL, double derivR)
{
if(derivL * derivR < 0) return 0.0;

if(derivR < derivL) return fluxLimiter_minmod(derivL, 2*derivR);
return fluxLimiter_minmod(2*derivL, derivR);
}

__device__ __inline__ double fluxLimiter_Osher(double A, double B)
{
double r = A*B;
if(r <= 0.0) return 0.0;

return 1.5*r*(A+B)/(A*A+r+B*B);
}

__device__ __inline__ double fluxLimiter_Zero(double A, double B) { return 0.0; }

/* These differ in that they return _HALF_ of the (limited) difference,
 * i.e. the projection from i to i+1/2 assuming uniform widths of cells i and i+1
 */
__device__ __inline__ double slopeLimiter_VanLeer(double derivL, double derivR)
{
double r;

r = derivL * derivR;
if(r > 0.0) { return r /(derivL+derivR); }

return 0;
}

#ifdef FLOATFLUX
__device__ __inline__ float slopeLimiter_minmod(float derivL, float derivR)
{
if(derivL * derivR < 0) return 0.0;

if(fabsf(derivL) > fabsf(derivR)) { return .5*derivR; } else { return .5*derivL; }
}
#else
__device__ __inline__ float slopeLimiter_minmod(double derivL, double derivR)
{
if(derivL * derivR < 0) return 0.0;

if(fabs(derivL) > fabs(derivR)) { return .5*derivR; } else { return .5*derivL; }
}
#endif

__device__ __inline__ double slopeLimiter_superbee(double derivL, double derivR)
{
if(derivL * derivR < 0) return 0.0;

if(derivR < derivL) return fluxLimiter_minmod(derivL, 2*derivR);
return .5*fluxLimiter_minmod(2*derivL, derivR);
}

__device__ __inline__ double slopeLimiter_Osher(double A, double B)
{
double R = A*B;
if(R > 0) {
	double S = A+B;
	return .75*R*S/(S*S-R);
	}
return 0.0;
}

__device__ __inline__ double slopeLimiter_Zero(double A, double B) { return 0.0; }


#define FINITEDIFFX_PREAMBLE \
/* Our assumption implicitly is that differencing occurs in the X direction in the local tile */\
int addrX = (threadIdx.x - DIFFEDGE) + blockIdx.x * (TILEDIM_X - 2*DIFFEDGE);\
int addrY = threadIdx.y + blockIdx.y * TILEDIM_Y;\
\
addrX += (addrX < 0)*FD_DIMENSION;\
\
/* Nuke the threads hanging out past the end of the X extent of the array */\
/* addrX is zero indexed, mind */\
if(addrX >= FD_DIMENSION - 1 + DIFFEDGE) return;\
if(addrY >= OTHER_DIMENSION) return;\
\
/* Mask out threads who are near the edges to prevent seg violation upon differencing */\
bool ITakeDerivative = (threadIdx.x >= DIFFEDGE) && (threadIdx.x < (TILEDIM_X - DIFFEDGE)) && (addrX < FD_DIMENSION);\
\
addrX %= FD_DIMENSION; /* Wraparound (circular boundary conditions) */\
\
/* NOTE: This chooses which direction we "actually" take derivatives in\
 *          along with the conditional add a few lines up */\
int globAddr = FD_MEMSTEP * addrX + OTHER_MEMSTEP * addrY;\
\
int tileAddr = threadIdx.x + TILEDIM_X * threadIdx.y + 1;\

#define FINITEDIFFY_PREAMBLE \
/* Our assumption implicitly is that differencing occurs in the X direction in the local tile */\
int addrX = threadIdx.x + blockIdx.x * TILEDIM_X;\
int addrY = (threadIdx.y - DIFFEDGE) + blockIdx.y * (TILEDIM_Y - 2*DIFFEDGE);\
\
addrY += (addrY < 0)*FD_DIMENSION;\
\
/* Nuke the threads hanging out past the end of the X extent of the array */\
/* addrX is zero indexed, mind */\
if(addrY >= FD_DIMENSION - 1 + DIFFEDGE) return;\
if(addrX >= OTHER_DIMENSION) return;\
\
/* Mask out threads who are near the edges to prevent seg violation upon differencing */\
bool ITakeDerivative = (threadIdx.y >= DIFFEDGE) && (threadIdx.y < (TILEDIM_Y - DIFFEDGE)) && (addrY < FD_DIMENSION);\
\
addrY %= FD_DIMENSION; /* Wraparound (circular boundary conditions) */\
\
/* NOTE: This chooses which direction we "actually" take derivatives in\
 *          along with the conditional add a few lines up */\
int globAddr = FD_MEMSTEP * addrY + OTHER_MEMSTEP * addrX;\
\
int tileAddr = threadIdx.y + TILEDIM_Y * threadIdx.x + 1;\

#endif
