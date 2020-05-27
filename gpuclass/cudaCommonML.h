

/* Not meant for user calls: Checks basic properties of the input
 * tag; Returns FALSE if they clearly violate basic properties a
 * valid gpu tag must have */
bool     sanityCheckTag(const mxArray *tag);     // sanity

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
int getGPUTypeStreams(const mxArray *fluidarray, cudaStream_t **streams, int *numel);

/* MultiGPU Array allocation stuff */
int      MGA_accessMatlabArrays(const mxArray *prhs[], int idxFrom, int idxTo, MGArray *mg); // Extracts a series of Matlab handles into MGArrays
int      MGA_accessMatlabArrayVector(const mxArray *m, int idxFrom, int idxTo, MGArray *mg);

MGArray *MGA_createReturnedArrays(mxArray *plhs[], int N, MGArray *skeleton); // clone existing MG array'
void     MGA_returnOneArray(mxArray *plhs[], MGArray *m);

int  MGA_uploadMatlabArrayToGPU(const mxArray *m, MGArray *g, int partitionTo);

void getTagFromGPUType(const mxArray *gputype, int64_t *tag);

double **getGPUSourcePointers(const mxArray *prhs[], ArrayMetadata *metaReturn, int fromarg, int toarg);
double **makeGPUDestinationArrays(ArrayMetadata *amdRef, mxArray *retArray[], int howmany);
double *replaceGPUArray(const mxArray *prhs[], int target, int *newdims);

int MGA_accessFluidCanister(const mxArray *canister, int fluidIdx, MGArray *fluid);
GeometryParams accessMatlabGeometryClass(const mxArray *geoclass);

ThermoDetails accessMatlabThermoDetails(const mxArray *thermstruct);

mxArray *derefXatNdotAdotB(const mxArray *in, int idx, const char *fieldA, const char *fieldB);
mxArray *derefXdotAdotB(const mxArray *in, const char *fieldA, const char *fieldB);
double derefXdotAdotB_scalar(const mxArray *in, const char *fieldA, const char *fieldB);
void derefXdotAdotB_vector(const mxArray *in, const char *fieldA, const char *fieldB, double *x, int N);

