#include "parallel_halo_arrays.h"

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

// Generic error catcher routines
#define CHECK_CUDA_LAUNCH_ERROR(bsize, gsize, mg_ptr, direction, string) \
checkCudaLaunchError(cudaGetLastError(), bsize, gsize, mg_ptr, direction, string, __FILE__, __LINE__)
#define CHECK_CUDA_ERROR(astring) checkCudaError(astring, __FILE__, __LINE__)
void dropMexError(char *excuse, char *infile, int atline);
#define DROP_MEX_ERROR(dangit) dropMexError(dangit, __FILE__, __LINE__)

#define PAR_WARN(x) if(x.nGPUs > 1) { printf("In %s:\n", __FILE__); mexWarnMsgTxt("WARNING: This function is shimmed but parallel multi-GPU operation WILL NOT WORK"); }

#define FATAL_NOT_IMPLEMENTED mexErrMsgTxt("Fatal: Encountered required but completely non-implemented code branch.");

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
#define GPU_TAG_LENGTH 8
#define GPU_TAG_DIM0 0
#define GPU_TAG_DIM1 1
#define GPU_TAG_DIM2 2
#define GPU_TAG_DIMSLAB 3
#define GPU_TAG_HALO 4
#define GPU_TAG_PARTDIR 5
#define GPU_TAG_NGPUS 6
#define GPU_TAG_EXTERIORHALO 7

// Templates seem to dislike MPI_Op? Switches definitely do.
typedef enum { OP_SUM, OP_PROD, OP_MAX, OP_MIN } MGAReductionOperator;

// If the haloSize parameter is set to this,
// then each gpu has a *.dims[] size array
// This occurs if e.g. a reduction occurs along the partition direction
#define PARTITION_CLONED -1

typedef struct {
    int dim[3];
    int64_t numel; // = nx ny nz of the global array, not # allocated
    int numSlabs; // = dim[4] if positive, which zero-indexed slab # if <= 0
    int64_t slabPitch[MAX_GPUS_USED]; // numel[partition] rounded up to nearest 256

    int haloSize;
    int partitionDir;

    // For mpi-serial runs, GIS will NOT supply halo cells at the edge of the array and
    // GIS must add them: addExteriorHalo is true

    // For mpi-parallel runs with >1 nodes in partition direct, GIS adds the parallel halo
    // so we do NOT need to.
    int addExteriorHalo;

    int nGPUs;
    int deviceID[MAX_GPUS_USED];
    double *devicePtr[MAX_GPUS_USED];
    int partNumel[MAX_GPUS_USED];
    } MGArray; 

int64_t *getGPUTypeTag (const mxArray *gputype); // matlab -> c
cudaStream_t *getGPUTypeStreams(const mxArray *gputype);
bool     sanityCheckTag(const mxArray *tag);     // sanity

void     calcPartitionExtent(MGArray *m, int P, int *sub);

void     deserializeTagToMGArray(int64_t *tag, MGArray *mg); // array -> struct
void     serializeMGArrayToTag(MGArray *mg, int64_t *tag);   // struct -> array

/* MultiGPU Array allocation stuff */
int      MGA_accessMatlabArrays(const mxArray *prhs[], int idxFrom, int idxTo, MGArray *mg); // autoloop ML packed arrays -> MGArrays
MGArray *MGA_allocArrays(int N, MGArray *skeleton);
MGArray *MGA_createReturnedArrays(mxArray *plhs[], int N, MGArray *skeleton); // clone existing MG array'
void     MGA_returnOneArray(mxArray *plhs[], MGArray *m);
void     MGA_delete(MGArray *victim);

/* MultiGPU Array I/O */
int MGA_downloadArrayToCPU(MGArray *g, double **p, int partitionFrom);
int MGA_uploadArrayToGPU(double *p, MGArray *g, int partitionTo);
int MGA_distributeArrayClones(MGArray *cloned, int partitionFrom);

/* MultiGPU "extensions" for mpi reduce type stuff */
int MGA_localPancakeReduce(MGArray *in, MGArray *out, MPI_Op operate, int dir, int partitionOnto, int redistribute);
int MGA_localElementwiseReduce(MGArray *in, MPI_Op operate, int dir, int partitionOnto, int redistribute);

int MGA_globalPancakeReduce(MGArray *in, MGArray *out, MPI_Op operate, int dir, int partitionOnto, int redistribute, const mxArray *topo);
int MGA_globalElementwiseReduce(MGArray *in, MPI_Op operate, int dir, int partitionOnto, int redistribute, const mxArray *topo);\

// Drops m[0...N].devicePtr[i] into dst[0...N] to avoid hueg sets of fluid[n].devicePtr[i] in calls:
void pullMGAPointers( MGArray *m, int N, int i, double **dst);

int MGA_reduceClonedArray(MGArray *a, MPI_Op operate, int redistribute);

/* Functions for managing halos of MGA partitioned data */
void MGA_exchangeLocalHalos(MGArray *a, int n);
void MGA_partitionHaloToLinear(MGArray *a, int partition, int direction, int right, int toHalo, int h, double **linear);

__global__ void cudaMGHaloSyncX_p2p(double *L, double *R, int nxL, int nxR, int ny, int nz, int h);
__global__ void cudaMGHaloSyncY_p2p(double *L, double *R, int nx, int nyL, int nyR, int nz, int h);

template<int lr_rw>
__global__ void cudaMGA_haloXrw(double *phi, double *linear, int nx, int ny, int nz, int h);

template<int lr_rw>
__global__ void cudaMGA_haloYrw(double *phi, double *linear, int nx, int ny, int nz, int h);

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

pParallelTopology topoStructureToC(const mxArray *prhs);
int MGA_localElmentwiseReduce(MGArray *in, int dir, int partitionOnto, int redistribute);
int MGA_localPancakeReduce(MGArray *in, MGArray *out, int dir, int partitionOnto, int redistribute);
int MGA_globalElementwiseReduce(MGArray *in, int dir, int partitionOnto, int redistribute, const mxArray *topo);
int MGA_globalPancakeReduce(MGArray *in, MGArray *out, int dir, int partitionOnto, int redistribute, const mxArray *topo);
int MGA_distributeArrayClones(MGArray *cloned, int partitionFrom);


mxArray *derefXdotAdotB(const mxArray *in, char *fieldA, char *fieldB);
double derefXdotAdotB_scalar(const mxArray *in, char *fieldA, char *fieldB);
void derefXdotAdotB_vector(const mxArray *in, char *fieldA, char *fieldB, double *x, int N);

void getTiledLaunchDims(int *dims, dim3 *tileDim, dim3 *halo, dim3 *blockdim, dim3 *griddim);

void checkCudaError(char *where, char *fname, int lname);
void checkCudaLaunchError(cudaError_t E, dim3 blockdim, dim3 griddim, MGArray *a, int i, char *srcname, char *fname, int lname);
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

__device__ __inline__ double slopeLimiter_minmod(double derivL, double derivR)
{
if(derivL * derivR < 0) return 0.0;

if(fabs(derivL) > fabs(derivR)) { return .5*derivR; } else { return .5*derivL; }
}

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


