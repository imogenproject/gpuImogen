#include "cudaCommon.h"

/* cudaCommonML.cu extracts away from the "normal" cudaCommon.cu the functions that exclusively talk
 * to Matlab's API. This represents a first step in rendering the gpuImogen core separable from Matlab.
 */

/* Given an mxArray *X, it searches "all the places you'd expect" any function in Imogen to have
 * stored the uint64_t pointer to the tag itself. Specifically, if X is a:
 *   uint64_t class: returns mxGetData(X)
 *   GPU_Type class: returns mxGetData(mxGetProperty(X, 0, "GPU_MemPtr"));
 *   FluidArray    : returns mxGetData(mxGetProperty(X, 0, "gputag"));
 */
int getGPUTypeTag(const mxArray *gputype, int64_t **tagPointer)
{
	return getGPUTypeTagIndexed(gputype, tagPointer, 0);
}

/* Behaves as getGPUTypeTag, but can fetch outside of index zero. */
int getGPUTypeTagIndexed(const mxArray *gputype, int64_t **tagPointer, int mxarrayIndex)
{
	static int64_t locptr[GPU_TAG_MAXLEN];

	if(tagPointer == NULL) {
		PRINT_FAULT_HEADER;
		printf("input tag pointer was null!\n");
		PRINT_FAULT_FOOTER;
		return ERROR_NULL_POINTER;
	}
	tagPointer[0] = NULL;

	mxClassID dtype = mxGetClassID(gputype);

	/* Handle gpu tags straight off */
	if(dtype == mxINT64_CLASS) {
		int64_t *q = (int64_t *)mxGetData(gputype);
		bool sanity = sanityCheckTag(gputype);
				if(sanity == false) {
					PRINT_FAULT_HEADER;
					printf((const char *)"Failure to access GPU tag: Sanity check failed.\n");
					PRINT_FAULT_FOOTER;
					return ERROR_GET_GPUTAG_FAILED;
				}
		int ctr;
		for(ctr = 0; ctr < mxGetNumberOfElements(gputype); ctr++) {
			locptr[ctr] = q[ctr];
		}
		tagPointer[0] = &locptr[0];
		return SUCCESSFUL;
	}

	mxArray *tag;
	const char *cname = mxGetClassName(gputype);

	/* If we were passed a GPU_Type, retreive the GPU_MemPtr element */
	if(strcmp(cname, (const char *)"GPU_Type") == 0) {
		tag = mxGetProperty(gputype, mxarrayIndex, (const char *)"GPU_MemPtr");
	} else { /* Assume it's an ImogenArray or descendant and retrieve the gputag property */
		tag = mxGetProperty(gputype, mxarrayIndex, (const char *)"gputag");
	}

	/* We have done all that duty required, there is no dishonor in surrendering */
	if(tag == NULL) {
		PRINT_FAULT_HEADER;
		printf((const char *)"getGPUTypeTag was called with something that is not a gpu tag, or GPU_Type class, or ImogenArray class\nArgument order wrong?\n");
		PRINT_FAULT_FOOTER;
		return ERROR_CRASH;
	}

	int64_t *q = (int64_t *)mxGetData(tag);
	bool sanity = sanityCheckTag(tag);
	if(sanity == false) {
		PRINT_FAULT_HEADER;
		printf((const char *)"Failure to access GPU tag: Sanity check failed.\n");
		PRINT_FAULT_FOOTER;
		return ERROR_GET_GPUTAG_FAILED;
	}

	int ctr;
	for(ctr = 0; ctr < mxGetNumberOfElements(tag); ctr++) {
		locptr[ctr] = q[ctr];
	}
	tagPointer[0] = &locptr[0];
	mxDestroyArray(tag);
	//tagPointer[0] = (int64_t *)mxGetData(tag);

	return SUCCESSFUL;
}

int getGPUTypeStreams(void *inputArray, cudaStream_t **streams, int *numel)
{
	// This is so we can have the same function in the standalone that doesn't require mxArray
	const mxArray *fluidarray = (const mxArray *)inputArray;
	mxArray *streamptr  = mxGetProperty(fluidarray, 0, (const char *)"streamptr");

	if(streamptr != NULL) {
		*numel = (int)mxGetNumberOfElements(streamptr);
		streams[0] = (cudaStream_t *)mxGetData(streamptr);
		return 0;
	} else {
		*numel = 0;
		return 0;
	}
}

/* Given an mxArray* that points to a GPU tag (specifically the uint64_t array, not the more
 * general types tolerated by the higher-level functions), checks whether it can pass
 * muster as a MGArray (i.e. one packed by serializeMGArrayToTag).
 */
bool sanityCheckTag(const mxArray *tag)
{
	int64_t *x = (int64_t *)mxGetData(tag);

	int tagsize = mxGetNumberOfElements(tag);

	// This cannot possibly be valid
	if(tagsize < GPU_TAG_LENGTH) {
		printf("Tag length is %i < min possible valid length of %i. Dumping.\n", tagsize, GPU_TAG_LENGTH);
		return false;
	}

	int nx = x[GPU_TAG_DIM0];
	int ny = x[GPU_TAG_DIM1];
	int nz = x[GPU_TAG_DIM2];

	// Null array OK
	if((nx == 0) && (ny == 0) && (nz == 0) && (tagsize == GPU_TAG_LENGTH)) return true;

	if((nx < 0) || (ny < 0) || (nz < 0)) {
		printf("One or more indices was of negative size. Dumping.\n");
		return false;
	}

	int halo         = x[GPU_TAG_HALO];
	int partitionDir = x[GPU_TAG_PARTDIR];
	int nDevs        = x[GPU_TAG_NGPUS];

	int permtag      = x[GPU_TAG_DIMPERMUTATION];

	int circlebits   = x[GPU_TAG_CIRCULARBITS];

	int vecpart      = x[GPU_TAG_VECTOR_COMPONENT];

	// Some basic does-this-make-sense
	if(nDevs < 1) {
		printf((const char *)"Tag indicates less than one GPU in use.\n");
		return false;
	}
	if(nDevs > MAX_GPUS_USED) {
		printf((const char *)"Tag indicates %i GPUs in use, current config only supports %i.\n", nDevs, MAX_GPUS_USED);
		return false;
	}
	if(halo < 0) { // not reasonable.
		printf((const char *)"Tag halo value is %i < 0 which is absurd. Dumping.\n", halo);
		return false;
	}

	if((permtag < 1) || (permtag > 6)) {
		if(permtag == 0) {
			// meh
		} else {
			printf((const char *)"Permutation tag is %i: Valid values are 1 (XYZ), 2 (XZY), 3 (YXZ), 4 (YZX), 5 (ZXY), 6 (ZYX)\n", permtag);
			return false;
		}
	}

	if((circlebits < 0) || (circlebits > 63)) {
		printf((const char *)"halo sharing bits have value %i, valid range is 0-63!\n", circlebits);
		return false;

	}

	if((vecpart < 0) || (vecpart > 3)) {
		printf((const char *)"vector component has value %i, must be 0 (scalar) or 1/2/3 (x/y/z)!\n", vecpart);
		return false;
	}

	if((partitionDir < 1) || (partitionDir > 3)) {
		printf((const char *)"Indicated partition direction of %i is not 1, 2, or 3.\n", partitionDir);
		return false;
	}

	// Require there be enough additional elements to hold the physical device pointers & cuda device IDs
	int requisiteNumel = GPU_TAG_LENGTH + 2*nDevs;
	if(tagsize != requisiteNumel) {
		printf((const char *)"Tag length is %i: Must be %i base + 2*nDevs = %i\n", tagsize, GPU_TAG_LENGTH, requisiteNumel);
		return false;
	}

	int j;
	x += GPU_TAG_LENGTH;
	// CUDA device #s are nonnegative, and it is nonsensical that there would be over 16 of them.
	for(j = 0; j < nDevs; j++) {
		if((x[2*j] < 0) || (x[2*j] >= MAX_GPUS_USED)) {
			printf((const char *)"Going through .deviceID: Found %i < 0 or > %i is impossible. Dumping.\n", (int)x[2*j], MAX_GPUS_USED);
			return false;
		}
	}

	return true;
}

/* Don't try to understand this. Just... this is how imogen does it */
int decodeMatlabBCs(mxArray *modes, BCSettings *b)
{
	mxArray *bcstr;
	int blen = 32; int bneed;
	char *bs = (char *)malloc((unsigned long)(blen * sizeof(char)));
	int d, i;
	for(d = 0; d < 3; d++) {
		for(i = 0; i < 2; i++) {
			bcstr = mxGetCell(modes, 2*(d) + i);
			bneed = (mxGetNumberOfElements(bcstr)+1);
			if(bneed > blen) {
				bs = (char *)malloc(sizeof(char) * bneed);
				blen = bneed;
			}
			mxGetString(bcstr, bs, mxGetNumberOfElements(bcstr)+1);

			if(strcmp(bs, "circ")    == 0) b->mode[2*d+i] = circular;
			if(strcmp(bs, "const")   == 0) b->mode[2*d+i] = extrapConstant;
			if(strcmp(bs, "linear")  == 0) b->mode[2*d+i] = extrapLinear;
			if(strcmp(bs, "mirror")  == 0) b->mode[2*d+i] = mirror;
			if(strcmp(bs, "bcstatic")== 0) b->mode[2*d+i] = stationary;
			if(strcmp(bs, "wall")    == 0) b->mode[2*d+i] = wall;
			if(strcmp(bs, "outflow") == 0) b->mode[2*d+i] = outflow;
			if(strcmp(bs, "freebalance") == 0) b->mode[2*d+i] = freebalance;
		}

	}
	free(bs);
	return SUCCESSFUL;
}

/* Facilitates access to MGArrays stored in Imogen's Matlab structures:
 * the mxArray pointers prhs[i] for i spanning idxFrom to idxTo inclusive
 * are decoded into mg[i - idxFrom]. Such that if the Matlab call is
 *    matlabFoo(1, 2, gpuA, gpuB, gpuC)
 * then foo(const mxArray *prhs[], ...) should use
 *    MGA_accessMatlabArrays(prhs, 2, 4, x)
 * with the result that x[0] = gpuA, x[1] = gpuB, x[2] = gpuC.
 */
int MGA_accessMatlabArrays(const mxArray *prhs[], int idxFrom, int idxTo, MGArray *mg)
{
	int i;
	int returnCode = SUCCESSFUL;
	prhs += idxFrom;

	int64_t *tag;

	for(i = 0; i < (idxTo + 1 - idxFrom); i++) {
		    if(prhs[i] == NULL) {
		    	PRINT_FAULT_HEADER;
		    	printf("Reading array #%i: prhs[i] was NULL (from=%i, to=%i)\n", i, idxFrom, idxTo);
		    	PRINT_FAULT_FOOTER;
		    	returnCode = ERROR_NULL_POINTER; break;
		    }

			returnCode = getGPUTypeTag(prhs[i], &tag);
			if(returnCode != SUCCESSFUL) break;

			returnCode = deserializeTagToMGArray(tag, &mg[i]);
			if(returnCode != SUCCESSFUL) break;

			mxArray *boundaryData = mxGetProperty(prhs[i], 0, "boundaryData");
			if(boundaryData != NULL) {
				mxArray *bcModes = mxGetField(boundaryData, 0, "bcModes");
				if(bcModes != NULL) {
					returnCode = decodeMatlabBCs(bcModes, &mg[i].boundaryConditions);
				}
			}

			mg[i].boundaryConditions.externalData = (void *)boundaryData;
			mg[i].boundaryConditions.extIndex = 0;

			if(returnCode != SUCCESSFUL) break;
		}

	return CHECK_IMOGEN_ERROR(returnCode);
}

/* Facilitates access to vector GPU array arguments from Matlab. Such that if
 *   matlab>> x = [gpuA gpuB gpuC];
 *   matlab>> matlabFoo(1, x, stuff)
 * Then foo(const mxArray *prhs, ...) should use
 *   MGA_accessMatlabArrayVector(prhs[1], 0, 2, z)
 * with the result that z[0] = gpuA, z[1] = gpuB, z[2] = gpuC.
 */
int MGA_accessMatlabArrayVector(const mxArray *m, int idxFrom, int idxTo, MGArray *mg)
{

	int i;
	int returnCode = SUCCESSFUL;

	int64_t *tag;

	for(i = 0; i < (idxTo + 1 - idxFrom); i++) {
			returnCode = getGPUTypeTagIndexed(m, &tag, i);
			if(returnCode != SUCCESSFUL) break;

			returnCode = deserializeTagToMGArray(tag, &mg[i]);
			if(returnCode != SUCCESSFUL) break;

			// This may be null because this isn't an imogen array, just a normal one
			mxArray *boundaryData = mxGetProperty(m, i, "boundaryData");
			if(boundaryData != NULL) {
				mxArray *bcModes = mxGetField(boundaryData, 0, "bcModes");
				if(bcModes != NULL) {
					returnCode = decodeMatlabBCs(bcModes, &mg[i].boundaryConditions);
				}
			}
			mg[i].boundaryConditions.externalData = (void *)boundaryData;
			mg[i].boundaryConditions.extIndex = i;

			if(returnCode != SUCCESSFUL) break;
		}

	return CHECK_IMOGEN_ERROR(returnCode);
}

/* A convenient wrapper for returning data to Matlab:
 * Creates arrays in the style of MGA_allocArrays, but serializes them into the
 * first N elements of plhs[] as well before returning the C vector pointer.
 */
MGArray *MGA_createReturnedArrays(mxArray *plhs[], int N, MGArray *skeleton)
{

	MGArray *m = NULL;
	int status = MGA_allocArrays(&m, N, skeleton);

	int i;

	mwSize dims[2]; dims[0] = GPU_TAG_LENGTH+2*skeleton->nGPUs; dims[1] = 1;
	int64_t *r;

	// create Matlab arrays holding serialized form,
	for(i = 0; i < N; i++) {
		plhs[i] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);
		r = (int64_t *)mxGetData(plhs[i]);
		serializeMGArrayToTag(m+i, r);
	}

	// send back the MGArray structs.
	return m;
}

/* Does exactly what it says: Accepts one MGArray pointer and
 * writes one serialized representation to plhs[0].
 */
void MGA_returnOneArray(mxArray *plhs[], MGArray *m)
{
	mwSize dims[2]; dims[0] = GPU_TAG_LENGTH+2*m->nGPUs; dims[1] = 1;
	int64_t *r;

	// create Matlab arrays holding serialized form,
	plhs[0] = mxCreateNumericArray(2, dims, mxINT64_CLASS, mxREAL);
	r = (int64_t *)mxGetData(plhs[0]);
	serializeMGArrayToTag(m, r);
}

/* Given a pointer to the Matlab array m, checks that g points to an
 * MGArray of the same size as m:
 * if partitionTo is nonnegative, that that partition's extent equals
 * the size of m
 * if partitionTo is negative, that the g->dim equals the size of m.
 * and then initiates the transfer.
 */
int MGA_uploadMatlabArrayToGPU(const mxArray *m, MGArray *g, int partitionTo)
{

if(m == NULL) return -1;
if(g == NULL) return -1;

mwSize ndims = mxGetNumberOfDimensions(m);
if(ndims > 3) { DROP_MEX_ERROR((const char *)"Input array has more than 3 dimensions!"); }

const mwSize *arraydims = mxGetDimensions(m);

int j;
int failed = 0;

for(j = 0; j < ndims; j++) {
	if(arraydims[j] != g->dim[j]) failed = 1;
}

if(failed) {
	PRINT_FAULT_HEADER;
	printf("Matlab array was %i dimensional, dims [", (int)ndims);
	for(j = 0; j < ndims; j++) { printf("%i ", (int)arraydims[j]); }
	printf("].\nGPU_Type target array was of size [%i %i %i] which is not the same. Not happy :(.\n", g->dim[0], g->dim[1], g->dim[2]);
	PRINT_FAULT_FOOTER;
	return ERROR_INVALID_ARGS;
	}

return CHECK_IMOGEN_ERROR(MGA_uploadArrayToGPU(mxGetPr(m), g, partitionTo));

}

/* Given a pointer to a FluidManager class, accesses the fluids stored
 * within it. */
int MGA_accessFluidCanister(const mxArray *canister, int fluidIdx, MGArray *fluid)
{
	/* Access the FluidManager canisters */
	mxArray *fluidPtrs[3];
	fluidPtrs[0] = mxGetProperty(canister, fluidIdx,(const char *)("mass"));
	if(fluidPtrs[0] == NULL) {
		PRINT_FAULT_HEADER;
		printf("Unable to fetch 'mass' property from canister\nNot a FluidManager class?\n");
		PRINT_FAULT_FOOTER;
		return ERROR_INVALID_ARGS;
	}
	fluidPtrs[1] = mxGetProperty(canister, fluidIdx,(const char *)("ener"));
	if(fluidPtrs[1] == NULL) {
		PRINT_FAULT_HEADER;
		printf("Unable to fetch 'ener' property from canister\nNot a FluidManager class?\n");
		PRINT_FAULT_FOOTER;
		return ERROR_INVALID_ARGS;
	}
	fluidPtrs[2] = mxGetProperty(canister, fluidIdx,(const char *)("mom"));
	if(fluidPtrs[2] == NULL) {
		PRINT_FAULT_HEADER;
		printf("Unable to fetch 'mom' property from canister\nNot a FluidManager class?\n");
		PRINT_FAULT_FOOTER;
		return ERROR_INVALID_ARGS;
	}

	int status = MGA_accessMatlabArrays((const mxArray **)&fluidPtrs[0], 0, 1, &fluid[0]);
	if(status != SUCCESSFUL) return CHECK_IMOGEN_ERROR(status);
    status = MGA_accessMatlabArrayVector(fluidPtrs[2], 0, 2, &fluid[2]);
    if(status != SUCCESSFUL) return CHECK_IMOGEN_ERROR(status);

    return SUCCESSFUL;
}

ThermoDetails accessMatlabThermoDetails(const mxArray *thermstruct)
{
	ThermoDetails thermo;
	thermo.gamma = derefXdotAdotB_scalar(thermstruct, "gamma", NULL);

	thermo.m     = derefXdotAdotB_scalar(thermstruct, "mass", NULL);

	thermo.mu0   = derefXdotAdotB_scalar(thermstruct, "dynViscosity", NULL);
	thermo.muTindex = derefXdotAdotB_scalar(thermstruct, "viscTindex", NULL);
	thermo.sigma0 = derefXdotAdotB_scalar(thermstruct, "sigma", NULL);
	thermo.sigmaTindex= derefXdotAdotB_scalar(thermstruct, "sigmaTindex", NULL);

	thermo.kBolt = derefXdotAdotB_scalar(thermstruct, "kBolt", NULL);
	thermo.Cisothermal = derefXdotAdotB_scalar(thermstruct, "Cisothermal", NULL);

	return thermo;
}

GeometryParams accessMatlabGeometryClass(const mxArray *geoclass)
{
	GeometryParams g;
	double v[3];

	g.Rinner = derefXdotAdotB_scalar(geoclass, "pInnerRadius", NULL);
	derefXdotAdotB_vector(geoclass, "d3h", NULL, &g.h[0], 3);

	derefXdotAdotB_vector(geoclass, "frameRotationCenter", NULL, &g.frameRotateCenter[0], 3);
	g.frameOmega = derefXdotAdotB_scalar(geoclass, "frameRotationOmega", NULL);

	int shapenum = derefXdotAdotB_scalar(geoclass, "pGeometryType", NULL);

	switch(shapenum) {
	case 1: g.shape = SQUARE; break;
	case 2: g.shape = CYLINDRICAL; break;
	// default: ?
	}

	derefXdotAdotB_vector(geoclass, "affine", NULL, &v[0], 3);
	g.x0 = v[0];
	g.y0 = v[1];
	g.z0 = v[2];

	return g;
}

/* A utility to ease access to Matlab structures/classes: fetches in(idx).{fieldA}.{fieldB}
 * or in(idx).{fieldA} if fieldB is NULL and returns the resulting mxArray* */
mxArray *derefXatNdotAdotB(const mxArray *in, int idx, const char *fieldA, const char *fieldB)
{

	if(fieldA == NULL) mexErrMsgTxt("In derefAdotBdotC: fieldA null!");

	mxArray *A; mxArray *B;
	mxClassID t0 = mxGetClassID(in);

	int snum = strlen("Failed to read field fieldA in X.A.B") + (fieldA != NULL ? strlen(fieldA) : 5) + (fieldB != NULL ? strlen(fieldB) : 5) + 10;
	char *estring;

	if(t0 == mxSTRUCT_CLASS) { // Get structure field from A
		A = mxGetField(in, idx, fieldA);

		if(A == NULL) {
			estring = (char *)calloc(snum, sizeof(char));
			sprintf(estring,"Failed to get X.%s", fieldA);
			mexErrMsgTxt(estring);
		}
	} else { // Get field struct A and fail if it doesn't work
		A = mxGetProperty(in, idx, fieldA);

		if(A == NULL) {
			estring = (char *)calloc(snum, sizeof(char));
			sprintf(estring,"Failed to get X.%s", fieldA);
			mexErrMsgTxt(estring);
		}
	}

	if(fieldB != NULL) {
		t0 = mxGetClassID(A);
		if(t0 == mxSTRUCT_CLASS) {
			B = mxGetField(A, idx, fieldB);
		} else {
			B = mxGetProperty(A, idx, fieldB);
		}

		if(B == NULL) {
			estring = (char *)calloc(snum, sizeof(char));
			sprintf(estring,"Failed to get X.%s.%s", fieldA, fieldB);
			mexErrMsgTxt(estring);
		}

		return B;
	} else {
		return A;
	}
}

/* A utility to ease access to Matlab structures/classes: fetches in.{fieldA}.{fieldB}
 * or in.{fieldA} if fieldB is NULL and returns the resulting mxArray* */
mxArray *derefXdotAdotB(const mxArray *in, const char *fieldA, const char *fieldB)
{
	return derefXatNdotAdotB(in, 0, fieldA, fieldB);
}

/* Fetches in.{fieldA}.{fieldB}, or in.{fieldA} if fieldB is NULL,
 * and returns the first double element of this.
 */
double derefXdotAdotB_scalar(const mxArray *in, const char *fieldA, const char *fieldB)
{
	mxArray *u = derefXdotAdotB(in, fieldA, fieldB);

	if(u != NULL) return *mxGetPr(u);

	return NAN;
}

/* Fetches in.{fieldA}.{fieldB}, or in.{fieldA} if fieldB is NULL,
 * and copies the first N elements of this into x[0, ..., N-1] if we get
 * a valid double *, or writes NANs if we do not.
 * If the Matlab array has fewer than N elements, truncates the copy.
 */
void derefXdotAdotB_vector(const mxArray *in, const char *fieldA, const char *fieldB, double *x, int N)
{
	mxArray *u = derefXdotAdotB(in, fieldA, fieldB);

	int Nmax = mxGetNumberOfElements(u);
	N = (N > Nmax) ? Nmax : N;

	double *d = mxGetPr(u);
	int i;

	if(d != NULL) {
		for(i = 0; i < N; i++) { x[i] = d[i]; } // Give it the d.
	} else {
		for(i = 0; i < N; i++) { x[i] = NAN; }
	}

}

/* This function should be used the mexFunction entry points (AND NOWHERE ELSE) to signal Matlab of problems.
 * NOTE: Don't call this directly, use DROP_MEX_ERROR("string") to automatically
 * fill in the file and line numbers correctly.
 */
void dropMexError(const char *excuse, const char *infile, int atline)
{
	static char turd[512];
	snprintf(turd, 511, "Bad news bears:\n\t%s\n\tLocation was %s:%i", excuse, infile, atline);
	mexErrMsgTxt(turd);
}

