#ifndef CUDACOMMONH_
#define CUDACOMMONH_

#ifndef NOMATLAB
#include "mex.h"
#endif

#include "driver_types.h"
#include "cuda_runtime_api.h"

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

/* The enumerated types of boundary conditions for the BCSettings structure
 * circular   - periodic boundary condition with opposing side
 * mirror     - all quantities have F(-x) = parity * F(x), where parity for normal momentum is odd & otherwise even
 * wall       - scalar and transverse vectors behave as mirror, normal vectors are zero
 * stationary - boundary cells are overwritten with fixed data
 * extrapConstant - F(-x) = F(0)
 * extrapLinear   - F(-x) = F(0) - x(F(1)-F(0)) [ WARNING this method is unstable except for supersonic inflow ]
 * outflow        - Scalar and transverse properties are constant extrap. Normal momentum is constant (outward) or zero (inward)
 * freebalance    - designed for the disk simulations, attempts to solve vertical or radial balance equations */
enum BCModeTypes { circular, mirror, wall, stationary, extrapConstant, extrapLinear, outflow, freebalance };

typedef struct __BCSettings {
	BCModeTypes mode[6]; // [-X, +X, -Y, +Y, -Z, +Z]
	void *externalData;
	int extIndex; // c.f. mlClassHandleIndex
} BCSettings;

#define MAX_GPUS_USED 4
/* The MGArray is the basic data unit for GPU arrays
 * This is interfaced with Matlab through the serializer/deserializer functions
 * that return/decipher opaque blobs of uint64_t's.
 */
typedef struct __MGArray {
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
     * When there is only 1 node in our partitioning direction, GeometryManager will NOT supply halo cells and it is
     * necessary that we do. When there is more than 1, the higher-level partitioning will supply
     * halo cells at the outside boundaries so we should not add them.
     *
     * Ergo, THIS SHOULD BE TRUE IFF THERE IS EXACTLY ONE RANK IN THE PARTITION DIRECTION AND NGPUS > 1
     * As mentioned at haloSize: This indicates whether MGA needs to add an exterior
     * halo if true (typical of mpi-serial operation) or not (typical of mpi-parallel
     * where the MPI partitioning has already added it)*/
    int addExteriorHalo;

    // Unique integer that compactly represents the 6 possible tuple->linear mappings
    // of a 3D array
    int permtag;

    // Marks which memory stride is associated with each physical direction
    int currentPermutation[3];

    int nGPUs; // Number of devices this MGA lives on: \elem [1, MAX_GPUS_USED].
    int deviceID[MAX_GPUS_USED]; // for indexes [0, ..., MAX_GPUS_USED-1], appropriate value for cudaSetDevice() etc
    double *devicePtr[MAX_GPUS_USED]; // for indices [0, ..., MAX_GPUS_USED-1], device data pointers allocated on the matching device.
    int partNumel[MAX_GPUS_USED]; // Number of elements allocated per devicePtr (i.e. includes halo).

    // Use MGA_BOUNDARY_{XYZ}{MINUS | PLUS} defines to select
    int mpiCircularBoundaryBits;
    BCSettings boundaryConditions;

    // Continuing my long tradition of bad boundary condition handling,
    // attach the original mxArray pointer so that cudaStatics can be re-used w/o substantial alteration.
    //const mxArray *matlabClassHandle;
    //int mlClassHandleIndex;

    // zero = scalar, 1/2/3 = x/y/z or r/theta/z: Used in BCs mainly
    int vectorComponent;
} MGArray;

enum geometryType_t { SQUARE, CYLINDRICAL, RZSQUARE, RZCYLINDRICAL };

typedef struct __GeometryParams {
	geometryType_t shape; /* Describes geometry in play:
			       *    SQUARE - two-dimensional XY, or three-dimensional XYZ cartesian
			       *    CYLINRICAL - two-dimensional R-theta, or three-dimensional R-theta-z coordinates
			       *    RZSQUARE - two-dimensional cartesian coordinates of size [NX 1 NZ]
			       *    RZCYLINDRICAL - two-dimensional cylindrical coordinates of size [NR 1 NZ] */
	double h[3]; // [dx dy dz] or [dr dphi dz]
	double x0, y0, z0; // Physical coordinates: The center of the cell with global index (0,0,0).

	double Rinner; // Only for cylindrical geometry: The radial coordinate of the innermost cell

	double frameRotateCenter[3]; // The [0] [1] components identify the physical coordinate the frame rotates about
	double frameOmega; // the angular rotation speed of the frame. In a right handed coordinate system,
	//w > 0 means inertially stationary fluid will appear to rotate clockwise

	int globalRez[3];  // The global grid resolution (halos excluded)

	int localRez[3];   // Only used in imogenCore: The local grid resolution (halos included)
	int gridAffine[3]; // Used only in imogenCore: The offset of this node's array(0,0,0) from the global array(0,0,0). Note that (0,0,0) includes ghost cells.

	MGArray *XYVector;
	// TODO: add allocatable vectors here for variable spacing in the future
} GeometryParams;

typedef struct __ThermoDetails {
	double gamma; // (fixed) adiabatic index
	double m; // particle mass
	double Cisothermal; // if negative one, ignore.

	double kBolt; // Boltzmann constant referred to our units of length/mass/time (T = Kelvin);

	double mu0; // dynamic viscosity at reference temperature
	double muTindex; // temperature exponent for viscosity; mu = mu0 (T/Tref)^muTindex

	double sigma0; // kinetic cross section at reference temperature (~ pi * diameter^2)
	double sigmaTindex; // temperature exp for cross section; sigma = sigma0 (T/Tref)^sigmaTindex
} ThermoDetails;

typedef struct {
        double *fluidIn[5];
        double *fluidOut[5];

        double *B[3];

        double *Ptotal;
        double *cFreeze;
        } fluidVarPtrs;

#define ROUNDUPTO(x, y) ( x + ((x % y == 0) ? 0 : (y - x % y)) )

// warning: reduceClonedMGArray is hardcoded with the reduction algo for a max of 4 devices because lazy
#define PARTITION_X 1
#define PARTITION_Y 2
#define PARTITION_Z 3

// Never ever don't use these
        // NOTE: If this length is changed you MUST also update:
        // sanityCheckTag
        // deserializeTagToMGArray
        // serializeMGArrayToTag
        // gpuclass/GPU_tag2struct.m utility function

#define GPU_TAG_LENGTH 11
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
#define GPU_TAG_VECTOR_COMPONENT 10
#define GPU_TAG_MAXLEN (GPU_TAG_LENGTH + 2*MAX_GPUS_USED)

// These mask the bits in the MGArray.mpiCircularBoundaryBits field
// These bits apply to MPI halo exchanges only:
//   All inter-partition boundaries are always circular (MGA_exchangeHalos())
//   All node-to-node partition edges interior to the global grid are always circular
//     (These are masked so by BCManager.m:134)
//   The only cases which are not marked circular are:
//     Global grid boundaries, with simulation BCs other than circular
//     Global grid boundaries, in directions with only one MPI rank in that direction
#define MGA_BOUNDARY_XMINUS 1
#define MGA_BOUNDARY_XPLUS 2
#define MGA_BOUNDARY_YMINUS 4
#define MGA_BOUNDARY_YPLUS 8
#define MGA_BOUNDARY_ZMINUS 16
#define MGA_BOUNDARY_ZPLUS 32

/* These enumerate the reduction operators which we accelerate on the GPU */
enum MGAReductionOperator { MGA_OP_SUM, MGA_OP_PROD, MGA_OP_MAX, MGA_OP_MIN };

typedef struct __GravityData {
	MGArray *phi; // gravity potential array

	int spaceOrder, timeOrder; // spaceOrder for computing gradPhi, timeOrder for the grav or composite solvers
	// NOTE: other data pertaining to solving for phi, if that is ever to be done, would go here
	// NOTE: e.g. the gravity constant G for solving del2(phi) = 4 pi G rho
} GravityData;

typedef struct __GridFluid {
	// This is intended to be the "root" of the slab that holds the alloc so we don't need special behaviors for the below data[] slab pointers
	MGArray DataHolder;

	// [rho E px py pz] data pointers as from MGA_AccessFluidCanister(prhs[0], 0, &self.data[0], ...)
	MGArray data[5];
	ThermoDetails thermo;

	BCSettings bcond[5];
	double rhoMin;
} GridFluid;

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

void     calcPartitionExtent(MGArray *m, int P, int *sub);

int MGA_dir2memdir(int *perm, int dir);
void MGA_permtagToNums(int permtag, int *p);
int MGA_numsToPermtag(int *nums);

int      deserializeTagToMGArray(int64_t *tag, MGArray *mg); // array -> struct
void     serializeMGArrayToTag(MGArray *mg, int64_t *tag);   // struct -> array


// FIXME this should return a status & take a ** to the value to return..
int      MGA_allocArrays(MGArray **ret, int N, MGArray *skeleton);
int      MGA_allocSlab(MGArray *skeleton, MGArray *nu, int Nslabs);
int      MGA_duplicateArray(MGArray **dst, MGArray *src);
int     MGA_delete(MGArray *victim);

void MGA_sledgehammerSequentialize(MGArray *q);
void MGA_parallelSledgehammerSequentialize(MGArray *q);

/* MultiGPU reduction calculators */
int MGA_partitionReduceDimension(MGArray *in, MGArray *out, MGAReductionOperator operate, int dir, int partition);
int MGA_reduceAcrossDevices(MGArray *a, MGAReductionOperator operate, int redistribute);

int MGA_localReduceDimension(MGArray *in, MGArray **out, MGAReductionOperator operate, int dir, int partitionOnto, int redistribute);
int MGA_globalReduceDimension(MGArray *in, MGArray **out, MGAReductionOperator operate, int dir, int partitionOnto, int redistribute, ParallelTopology * topology);

int MGA_localReduceScalar(MGArray *in, double *scalar, MGAReductionOperator operate);
int MGA_globalReduceScalar(MGArray *in, double *scalar, MGAReductionOperator operate, ParallelTopology * topology);

// Drops m[0...N].devicePtr[i] into dst[0...N] to avoid hueg sets of fluid[n].devicePtr[i] in calls:
void pullMGAPointers( MGArray *m, int N, int i, double **dst);

/* MultiGPU Array I/O */
int  MGA_downloadArrayToCPU(MGArray *g, double **p, int partitionFrom);

int  MGA_uploadArrayToGPU(double *p, MGArray *g, int partitionTo);
int  MGA_distributeArrayClones(MGArray *cloned, int partitionFrom);

MGAReductionOperator MGAReductionOperator_mpi2mga(MPI_Op mo);
MPI_Op MGAReductionOperator_mga2mpi(MGAReductionOperator op);

int MGA_arraysAreIdenticallyShaped(MGArray *a, MGArray *b);

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
int parCheckImogenError(int errtype, const char *infile, const char *infunc, int atline, MPI_Comm whom);

//
void MGA_debugPrintAboutArray(MGArray *x);
void MGA_debugPrintAboutArrayBrief(MGArray *x);

int dbgfcn_CheckArrayVals(MGArray *x, int maxslab, int crashit);
int dbgfcn_CheckArrayVals(MGArray *x, int crashit);
int dbgfcn_CheckFluidVals(MGArray *fluid, int crashit);

#include "imogenChecks.h"

#define PRINT_FAULT_HEADER printf("========== FAULT IN COMPILED CODE: function %s (%s:%i)\n", __func__, __FILE__, __LINE__)
#define PRINT_FAULT_FOOTER printf("========== COMPILED CODE STACK TRACE SHOULD FOLLOW: ============================\n")

#define PRINT_SIMPLE_FAULT(x) PRINT_FAULT_HEADER; printf(x); PRINT_FAULT_FOOTER;

#define CHECK_IMOGEN_ERROR(errtype) checkImogenError(errtype, __FILE__, __func__, __LINE__)
#define PAR_CHECK_IMOGEN_ERROR(errtype) parCheckImogenError(errtype, __FILE__, __func__, __LINE__, MPI_COMM_WORLD)

#define CHECK_CUDA_LAUNCH_ERROR(bsize, gsize, mg_ptr, direction, string) \
     checkCudaLaunchError(cudaGetLastError(), bsize, gsize, mg_ptr, direction, string, __FILE__, __LINE__)
#define CHECK_CUDA_ERROR(astring) CHECK_IMOGEN_ERROR(checkCudaError(astring, __FILE__, __LINE__))
#define BAIL_ON_FAIL(errtype) if( CHECK_IMOGEN_ERROR(errtype) != SUCCESSFUL) { return errtype; }

void dropMexError(const char *excuse, const char *infile, int atline);
#define DROP_MEX_ERROR(dangit) dropMexError(dangit, __FILE__, __LINE__)

#define PAR_WARN(x) if(x.nGPUs > 1) { printf("In %s:\n", __FILE__); mexWarnMsgTxt("WARNING: This function is shimmed but multi-GPU operation WILL NOT WORK"); }

#define FATAL_NOT_IMPLEMENTED mexErrMsgTxt("Fatal: Encountered required but completely non-implemented code branch.");

typedef struct {
    int ndims;
    int dim[3];
    int numel;
    } ArrayMetadata;

void arrayMetadataToTag(ArrayMetadata *meta, int64_t *tag); 

int3 makeInt3(int x, int y, int z);
int3 makeInt3(int *b);
dim3 makeDim3(unsigned int x, unsigned int y, unsigned int z);
dim3 makeDim3(unsigned int *b);
dim3 makeDim3(int *b);

int MGA_distributeArrayClones(MGArray *cloned, int partitionFrom);

void getTiledLaunchDims(int *dims, dim3 *tileDim, dim3 *halo, dim3 *blockdim, dim3 *griddim);

const char *errorName(cudaError_t E);

void printdim3(char *name, dim3 dim);
void printgputag(char *name, int64_t *tag);

#ifndef NOMATLAB
#include "cudaCommonML.h"
#endif

#endif
