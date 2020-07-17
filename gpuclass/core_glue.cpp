#include "stdio.h"
#include "stdlib.h"

#include "cuda.h"
#include "cuda_runtime.h"

#include "../mpi/mpi_common.h"

#include "math.h"

#include "cudaCommon.h"

#include "core_glue.hpp"

bool imRankZero(void)
{
int myrank;
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
if(myrank == 0) { return true; } else { return false; }
}

int lookForArgument(int argc, char **argv, const char *argument)
{
int i;
for(i = 0; i < argc; i++) {
	if(strcmp(argv[i], argument) == 0) return i;
}

return -1;
}

// Looks for the argv[] equal to *key; If this is not the last arg, puts atof(next arg) into *val
int argReadKeyValueDbl(int argc, char **argv, const char *key, double *val)
{
	int q = lookForArgument(argc, argv, key);
	if((q > 0) && (q < (argc-1))) { *val = atof(argv[q+1]); return 0; }
	return -1; // arg not found or invalid
}

// Looks for the argv[] equal to *key; If this is not the last arg, puts atoi(next arg) into *val
int argReadKeyValueInt(int argc, char **argv, const char *key, int *val)
{
	int q = lookForArgument(argc, argv, key);
	if((q > 0) && (q < (argc-1))) { *val = atoi(argv[q+1]); return 0; }

	return -1; // arg not found or invalid
}

// Looks for the argv[] equal to *key; If this is not the last arg, puts * to next arg in *val
int argReadKeyValueStr(int argc, char **argv, const char *key, char **val)
{
	int q = lookForArgument(argc, argv, key);
	if((q > 0) && (q < (argc-1))) { *val = argv[q+1]; return 0; }

	return -1; // arg not found or invalid
}

void printHelpScreed(void)
{
	std::cout << "================================================================================\n";
	std::cout << "This is the imogenCore help screed\n";
	std::cout << "================================================================================\n";
	std::cout << "imogenCore is the all-compiled matlab-dependency-free core of the Imogen\n";
	std::cout << "GPU-Accelerated 3D multiphysics gridded fluid dynamic code. It executes the\n";
	std::cout << "exact same algorithm as GPU-Imogen does, but has no Matlab dependency (and no\n";
	std::cout << "Matlab flexibility, either). It interacts with the world through HDF-5 files.\n";
	std::cout << "\nArguments:\n";
	std::cout << "--help           Prints this and exits\n";
	std::cout << "--initfile foo   Uses the run configuration file foo.h5. This is normally\n";
	std::cout << "                 written by the translateInitializerToH5.m function.\n";
	std::cout << " --devices FOO   Informs imogenCore to utilize the named devices in FOO. FOO is\n";
	std::cout << "                 one of: N (a single integer), 'all' to use all enumerable, or\n";
	std::cout << "                 like A,B,C (comma-separated integers, no spaces) for multi-GPU.\n";
	std::cout << "--frame X        If restarting, attempts to read frame X instead of what the\n";
	std::cout << "                 initfile's /iniFrame parameter says (which is usually zero)\n";
	std::cout << "--show-time      Prints the initfile's current time and iteration limits & exits\n";
	std::cout << "--set-time I T   Resets the the maximum # of iterations to I and time limit to T\n";
	std::cout << "                 and adjusts the save rates to maintain constant iterations or \n";
	std::cout << "                 elapsed time per frame, depending on the save mode, & exits\n";
	std::cout << "                 If I or T < 0, the value is not altered.\n";
	std::cout << "================================================================================" << std::endl;
}


/* Given integer X, computes the prime factorization (sorted ascending), into factors[0] and
 * reports the number of factors in nfactors. factors[0] is alloc'd and must be free'd. */
void factorizeInteger(int X, int **factors, int *nfactors)
{
int nf = 0;

int nmax = 16;
int *f = (int *)malloc(nmax * sizeof(int));

int i;
while(X > 1) {
	for(i = 2; i <= X; i++) { // We are guaranteed to hit a factor by i = sqrt(X) but why calculate that
		int p = X / i; 
		if(i*p == X) { // we found a factor
			//printf("%i is a factor\n", i);
			f[nf] = i;
			nf++;
			if(nf == nmax) { nmax = nmax*2; f = (int *)realloc((void *)f, nmax*sizeof(int)); }
			X = X / i;
			break;
		} else {
			//printf("%i is not a factor\n", i);
		}
	}
}
factors[0] = f;
*nfactors = nf;

}

bool charIsNum(char c)
{
	if((c == '0') || (c == '1') || (c == '2') || (c == '3') || (c == '4') || (c == '5') || (c == '6') || (c == '7') || (c == '8') || (c == '9')) return true;
	return false;
}
/* Processes the --devices argument.
 * Must be one of:
 *   --devices X
 *   --devices all
 *   --devices A,B,C (N comma-delimited integers)
 */
int parseDevicesArgument(char *arg, int *nDevices, int *deviceList)
{

	int retval = SUCCESSFUL;
	if(strcasecmp(arg, "all") == 0) {
		retval = cudaGetDeviceCount(nDevices);
		int i;
		for(i = 0; i < *nDevices; i++) {
			deviceList[i] = i;
		}
	} else {
		int l = strlen(arg);
		int i;
		int ncommas = 0;
		for(i = 0; i < l; i++) {
			if(arg[i] == ',') ncommas++;
		}
		*nDevices = ncommas + 1;

		// go to first number char
		i = 0;
		while((i < l) && (charIsNum(arg[i]) == false)) { i++; }
		//printf("@ numeral 0: remaining str = %s\n", &arg[i]);

		// confirm we actually got a [0-9] character
		if(i == l) {
			PRINT_FAULT_HEADER;
			printf("Fatal device list problem:\n\tNo numeric chars were found after --devices: argument = '%s'\n", arg);
			PRINT_FAULT_FOOTER;
			return ERROR_INVALID_ARGS;
		} else {
			deviceList[0] = atoi(&arg[i]);
		}
		arg = arg + i;

		// read the remainder of numbers, parsing immediately after the ,
		i = 1;
		for(; i <= ncommas; i++) {
			arg = strchr(arg, ',');
			if(arg == NULL) return ERROR_INVALID_ARGS;
			arg++;
			//printf("@ numeral %i: remaining str = %s\n", i, arg);
			deviceList[i] = atoi(arg);
		}

		// Confirm individual device arguments are acceptable
		int ct, j;
		cudaGetDeviceCount(&ct);
		for(i = 0; i < *nDevices; i++) {
			if((deviceList[i] >= ct) || (deviceList[i] < 0)) {
				retval = ERROR_INVALID_ARGS;
				PRINT_FAULT_HEADER;
				std::cout << "Device list: [";
				for(j = 0; j < *nDevices; j++) {
					char endc = (j < ((*nDevices)-1)) ? ' ' : ']';
					std::cout << deviceList[j] << endc;
				}
				std::cout << "\nFatal device list problem:\n\tOne or more arguments are outside acceptable range of [0, " << ct-1 << "]" << std::endl;
				break;
			}
		}

		// Check that all named device IDs are unique
		for(i = 0; i < *nDevices; i++) {
			for(j = i+1; j < *nDevices; j++) {
				if(deviceList[i] == deviceList[j]) {
					PRINT_FAULT_HEADER;
					int k;
					std::cout << "Device list: [";
					for(k = 0; k < *nDevices; k++) {
						char endc = (k < ((*nDevices)-1)) ? ' ' : ']';
						std::cout << deviceList[k] << endc;
					}
					std::cout << "\nFatal device list problem:\n\tEntries " << i << " and " << j << ", and possibly others, are duplicates\nImogenCore does not support this." << std::endl;
					PRINT_FAULT_FOOTER;
					i = *nDevices;
					break;
				}
			}
		}

		if(*nDevices > MAX_GPUS_USED) {
		    PRINT_FAULT_HEADER;
		    std::cout << "Fatal device list problem:\n\t(Total number of devices = " << *nDevices << ") > (MAX_GPUS_USED = " << MAX_GPUS_USED << ")\n\tMAX_GPUS_USED in cudaCommon.h must be changed & the code recompiled." << std::endl;
		    PRINT_FAULT_FOOTER;
		}
	}
	return retval;
}

void describeTopology(ParallelTopology *topo)
{
	std::cout << "Describing topology at ptr=0x" << std::hex << (unsigned long)topo << "\n";
	std::cout << " |- ndim  = " << std::dec << topo->ndim << "\n";
	std::cout << " |- comm  = " << topo->comm << "\n";
	std::cout << " |- nproc = [" << topo->nproc[0] << " " << topo->nproc[1] << " " << topo->nproc[2] << "]\n";
	std::cout << " |- coord = [" << topo->coord[0] << " " << topo->coord[1] << " " << topo->coord[2] << "]\n";
	std::cout << " |- L neighbor = [" << topo->neighbor_left[0] << " " << topo->neighbor_left[1] << " " << topo->neighbor_left[2] << "]\n";
	std::cout << " |- R neighbor = [" << topo->neighbor_right[0] << " " << topo->neighbor_right[1] << " " << topo->neighbor_right[2] << "]\n";
	std::cout << " \\- xyz communicators: " << topo->dimcomm[0] << " " << topo->dimcomm[1] << " " << topo->dimcomm[2] << "]" << std::endl;
}

void describeGeometry(GeometryParams *geo)
{
	std::cout << "Describing geometry at " << std::hex <<  (unsigned long)geo << std::dec << "\n";
	std::cout << " |- GRID PROPERTIES:\n";
	std::cout << " |-- global rez:   [" << geo->globalRez[0] << " " << geo->globalRez[1] << " " << geo->globalRez[2] << "]\n";
	std::cout << " |-- local rez:    [" << geo->localRez[0] << " " << geo->localRez[1] << " " << geo->localRez[2] << "]\n";
	std::cout << " |-- local affine: [" << geo->gridAffine[0] << " " << geo->gridAffine[1] << " " << geo->gridAffine[2] << "]\n";
	std::cout << " |- PHYSICAL PROPERTIES:\n";
	std::cout << " |-- shape = " << geo->shape << "\n";
	std::cout << " |-- h = [" << geo->h[0] << " " << geo->h[1] << " " << geo->h[2] << "]\n";
	std::cout << " |-- x0= [" << geo->x0 << " " << geo->y0 << " " << geo->z0 << "]\n";
	std::cout << " |-- frameOmega        = " << geo->frameOmega << "\n";
	std::cout << " |-- frameRotateCenter = [" << geo->frameRotateCenter[0] << " " << geo->frameRotateCenter[1] << "\n";
	std::cout << " |-- Rinner            = " << geo->Rinner << "\n";
	if(geo->XYVector == NULL) {
		std::cout << " |- XYVector is null\n";
	} else {
		std::cout << " |- XYVector = " << std::hex << (unsigned long)geo->XYVector << std::dec << "\n";
		MGA_debugPrintAboutArray(geo->XYVector);
	}

	std::cout << std::endl;
}

int readFluidDetailModel(ImogenH5IO *cfg, ThermoDetails *td)
{
	cfg->getDblAttr("/fluidDetail1", "gamma", &td->gamma);
	cfg->getDblAttr("/fluidDetail1", "dynViscosity", &td->mu0);
	cfg->getDblAttr("/fluidDetail1", "kBolt", &td->kBolt);
	cfg->getDblAttr("/fluidDetail1", "mass", &td->m);
	cfg->getDblAttr("/fluidDetail1", "sigma", &td->sigma0);
	cfg->getDblAttr("/fluidDetail1", "sigmaTindex", &td->sigmaTindex);
	cfg->getDblAttr("/fluidDetail1", "viscTindex", &td->muTindex);

	td->Cisothermal = -1;
//ATTRIBUTE "minMass" {

	return SUCCESSFUL; // FIXME
}

BCModeTypes num2BCModeType(int x)
{
	switch(x) {
	case 1: return circular;
	case 2: return mirror;
	case 3: return wall;
	case 4: return stationary;
	case 5: return extrapConstant;
	case 6: return extrapLinear;
	case 7: return outflow;
	case 8: return freebalance;
	default:
		std::cout << "ERROR: bc mode corresponding to integer " << x << " is unknown. See gpuImogen:BCManager.m:200\nReturning circular" << std::endl;
		return circular;
	}
}

// ================================================================================================
// Implementation of the ImogenH5IO class that provides all needed functionality for talking to
// H5 files, both as a means of structured grid input/output and for the parameter file, which is
// just a bunch of attributes strapped into an H5 bundle

ImogenH5IO::ImogenH5IO(const char *filename, unsigned int flags)
{
	openUpFile(filename, flags);
	attrhid = -1; spacehid = -1; sethid = -1;
	attrTarg = -1; grouphid = -1;
	attrOverwrite = true;
	attrTargetRoot();
}

ImogenH5IO::ImogenH5IO(const char *filename)
{
	openUpFile(filename, H5F_ACC_RDONLY);
	attrhid = -1; spacehid = -1; sethid = -1;
	attrTarg = -1; grouphid = -1;
	attrOverwrite = true;
	attrTargetRoot();
}

ImogenH5IO::ImogenH5IO(void)
{
	filehid = -1;
	attrhid = -1; spacehid = -1; sethid = -1;
	attrTarg = -1; grouphid = -1;
	attrOverwrite = true;
	attrTargetRoot();
}

ImogenH5IO::~ImogenH5IO(void)
{
	if(haveOpenedFile()) closeOutFile();
}

/* Closes any open savefile and opens either basename_frametype_rankXXX_frameno.h5 if
 * basename != NULL, or frametype_rankXXX_frameno.h5 (the original imogen format) if it is */
int ImogenH5IO::openImogenSavefile(const char *namePrefix, int frameno, int pad, SaveFrameTypes s)
{
	char *x;
	if(namePrefix != NULL) {
		x = (char *)malloc(strlen(namePrefix) + 64);
	} else {
		x = (char *)malloc(128);
	}

	const char *y;
	switch(s) {
	case X: y = "1D_X"; break;
	case Y: y = "1D_Y"; break;
	case Z: y = "1D_Z"; break;
	case XY: y = "2D_XY"; break;
	case XZ: y = "2D_XZ"; break;
	case YZ: y = "2D_YZ"; break;
	case XYZ: y = "3D_XYZ"; break;
	}

	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	if(namePrefix != NULL) {
		sprintf(x, "%s_%s_rank%03i_%0*i.h5", namePrefix, y, myrank, pad, frameno);
	} else {
		sprintf(x, "%s_rank%03i_%0*i.h5", y, myrank, pad, frameno);
	}

	if(ImogenH5IO::haveOpenedFile()) { ImogenH5IO::closeOutFile(); };

	return CHECK_IMOGEN_ERROR(openUpFile(x, H5F_ACC_TRUNC));
}

/* This call opens up the filename indicated by 'filename'. if flags == H5F_ACC_RDONLY
 * it opens for read, if not it uses H5F_Create with the given flags.
 * NOTE: This function does NOT check for concurrent open-for-write!!! */
int ImogenH5IO::openUpFile(const char *filename, unsigned int flags)
{

	FILE *tst = fopen(filename, "r");
	if(tst != NULL) {
		fclose(tst);
		filehid = H5Fopen(filename, flags, H5P_DEFAULT);
	} else {
		if(flags == H5F_ACC_RDONLY) {
			filehid = -1;
			return ERROR_CRASH;
		} else {
			filehid = H5Fcreate(filename, flags, H5P_DEFAULT, H5P_DEFAULT);
		}
	}

	attrTarg = filehid;

	if(filehid > 0) return 0; else; return ERROR_CRASH;
}

bool ImogenH5IO::haveOpenedFile(void)
{ return (filehid >= 0); }

void ImogenH5IO::closeOutFile(void)
{
	// close other stuff before this
	if(filehid > 0) {
		H5Fclose(filehid);
		filehid = -1;
	}
}

int ImogenH5IO::getArrayNDims(const char *name)
{
	int nd;
	herr_t foo = H5LTget_dataset_ndims (filehid, name, &nd);
	return nd;
}

int ImogenH5IO::getArraySize(const char *name, hsize_t *dims)
{

	H5T_class_t a;
	size_t b;
	H5T_class_t classtype;
	unsigned long sizetype;
	herr_t foo = H5LTget_dataset_info (filehid, name, dims, &classtype, &sizetype);

	int nd = getArrayNDims(name);

	reverseVector(dims, nd);
	return foo;
}

int ImogenH5IO::getAttrInfo(const char *location, const char *attrName, hsize_t *dimensions, H5T_class_t *type)
{
	size_t foo;
	return H5LTget_attribute_info(attrTarg, location, attrName, dimensions, type, &foo);
}

// Stores the # of elements in the indicated attribute in *numel
// Returns SUCCESSFUL if successful or ERROR_LIBFAILED if H5 barfs
int ImogenH5IO::chkAttrNumel(const char *location, const char *attrName, int *numel)
{
	// get # of dims
	int ndims;
	herr_t foo = H5LTget_attribute_ndims(attrTarg, location, attrName, &ndims);
	if(foo < 0) { return ERROR_LIBFAILED; }

	// get dims
	hsize_t dimensions[ndims];
	H5T_class_t type;
	size_t nocurr;
	foo = H5LTget_attribute_info(attrTarg, location, attrName, dimensions, &type, &nocurr);
	if(foo < 0) { return ERROR_LIBFAILED; }

	if(type == H5T_STRING) {
		*numel = nocurr; // some1curr after all
	} else {
		int i;
		int ne = 1;
		for(i = 0; i < ndims; i++) { ne *= dimensions[i]; }
		*numel = ne;
	}

	return SUCCESSFUL;
}

//==============================================================================
// Attribute getters

// Fetches the indicated double attribute into *data. Assumes that *data can store at least
// 'nmax' elements (default is 1 & may be ignored for scalar attributes)
// Returns ERROR_NOMEM if numel > nmax, ERROR_CRASH if the read itself barfs, otherwise SUCCESSFUL
int ImogenH5IO::getDblAttr(const char *location, const char *attrName, double *data, int nmax)
{
	int numel;
	int status = chkAttrNumel(location, attrName, &numel);
	if(numel > nmax) { return ERROR_NOMEM; }

	herr_t foo = H5LTget_attribute_double (filehid, location, attrName, data);
	if(foo < 0) { return ERROR_CRASH; } else { return SUCCESSFUL; }
}

// Fetches the indicated double attribute into *data. Assumes that *data can store at least
// 'nmax' elements (default is 1 & may be ignored for scalar attributes)
// Returns ERROR_NOMEM if numel > nmax, ERROR_CRASH if the read itself barfs, otherwise SUCCESSFUL
int ImogenH5IO::getFltAttr(const char *location, const char *attrName, float *data, int nmax)
{
	int numel;
	int status = chkAttrNumel(location, attrName, &numel);
	if(numel > nmax) { return ERROR_NOMEM; }

	herr_t foo = H5LTget_attribute_float (filehid, location, attrName, data);
	if(foo < 0) { return ERROR_CRASH; } else { return SUCCESSFUL; }
}

// Fetches the indicated double attribute into *data. Assumes that *data can store at least
// 'nmax' elements (default is 1 & may be ignored for scalar attributes)
// Returns ERROR_NOMEM if numel > nmax, ERROR_CRASH if the read itself barfs, otherwise SUCCESSFUL
int ImogenH5IO::getInt32Attr(const char *location, const char *attrName, int *data, int nmax)
{
	int numel;
	int status = chkAttrNumel(location, attrName, &numel);
	if(numel > nmax) { return ERROR_NOMEM; }

	herr_t foo = H5LTget_attribute_int (filehid, location, attrName, data);
	if(foo < 0) { return ERROR_CRASH; } else { return SUCCESSFUL; }
}

// Fetches the indicated double attribute into *data. Assumes that *data can store at least
// 'nmax' elements (default is 1 & may be ignored for scalar attributes)
// Returns ERROR_NOMEM if numel > nmax, ERROR_CRASH if the read itself barfs, otherwise SUCCESSFUL
int ImogenH5IO::getInt64Attr(const char *location, const char *attrName, long int *data, int nmax)
{
	int numel;
	int status = chkAttrNumel(location, attrName, &numel);
	if(numel > nmax) { return ERROR_NOMEM; }

	herr_t foo = H5LTget_attribute_long (filehid, location, attrName, data);
	if(foo < 0) { return ERROR_CRASH; } else { return SUCCESSFUL; }
}

int ImogenH5IO::getStrAttr(const char *location, const char *attrName, char **data, int nmax)
{
	int numel;
	int status = chkAttrNumel(location, attrName, &numel);
	if(*data == NULL) {
		nmax = numel;
		*data = (char *)malloc(numel + 1);
	}
	if((numel > nmax) || (*data == (char *)NULL)) { return ERROR_NOMEM; }

	herr_t foo = H5LTget_attribute_string(filehid, location, attrName, data[0]);
	data[0][numel] = 0;

	if(foo < 0) { return ERROR_CRASH; } else { return SUCCESSFUL; }
}

//==============================================================================
// Attribute writers

int ImogenH5IO::checkOverwrite(const char *name)
{
	if(H5Aexists(attrTarg, name)) {
			if(attrOverwrite) {
				H5Adelete_by_name(attrTarg, "/", name, H5P_DEFAULT);
				return SUCCESSFUL;
			} else {
				return ERROR_CRASH;
			}
		}
	return SUCCESSFUL;
}

// Private - Does the setup work for writing an attribute, other than the actual create/write
int ImogenH5IO::prepareAttribute(const char *name, int ndims, int *dimensions, hid_t *attout)
{
	hid_t atttype = H5Screate(H5S_SIMPLE);
	hsize_t foo[ndims];
	int i;
	// We must reverse
	for(i = 0; i < ndims; i++) { foo[ndims-1-i] = (hsize_t)dimensions[i]; }

	if(checkOverwrite(name) < 0) { PRINT_SIMPLE_FAULT("Unable to overwrite attribute!\n"); return ERROR_CRASH; }

	hid_t ret  = H5Sset_extent_simple(atttype, ndims, &foo[0], NULL);
	*attout = atttype;
	return 0;
}

int ImogenH5IO::writeDblAttribute(const char *name, int ndims, int *dimensions, double *x)
{
	hid_t attType;
	int ret = prepareAttribute(name, ndims, dimensions, &attType);
	if(ret < 0) return CHECK_IMOGEN_ERROR(ret);

	hid_t attr1 = H5Acreate2(attrTarg, name, H5T_IEEE_F64LE, attType, H5P_DEFAULT, H5P_DEFAULT);
	ret = H5Awrite(attr1, H5T_IEEE_F64LE, x);
	return 0;
}

int ImogenH5IO::writeFltAttribute(const char *name, int ndims, int *dimensions, float *x)
{
	hid_t attType;
	int ret = prepareAttribute(name, ndims, dimensions, &attType);
	if(ret < 0) return CHECK_IMOGEN_ERROR(ret);

	hid_t attr1 = H5Acreate2(attrTarg, name, H5T_IEEE_F32LE, attType, H5P_DEFAULT, H5P_DEFAULT);
	ret = H5Awrite(attr1, H5T_IEEE_F32LE, x);
	return 0;
}

int ImogenH5IO::writeInt32Attribute(const char *name, int ndims, int *dimensions, int32_t *x)
{
	hid_t attType;
	int ret = prepareAttribute(name, ndims, dimensions, &attType);
	if(ret < 0) return CHECK_IMOGEN_ERROR(ret);

	hid_t attr1 = H5Acreate2(attrTarg, name, H5T_STD_I32LE, attType, H5P_DEFAULT, H5P_DEFAULT);
	ret = H5Awrite(attr1, H5T_STD_I32LE, x);
	return 0;
}

int ImogenH5IO::writeInt64Attribute(const char *name, int ndims, int *dimensions, int64_t *x)
{
	hid_t attType;
	int ret = prepareAttribute(name, ndims, dimensions, &attType);
	if(ret < 0) return CHECK_IMOGEN_ERROR(ret);

	hid_t attr1 = H5Acreate2(attrTarg, name, H5T_STD_I64LE, attType, H5P_DEFAULT, H5P_DEFAULT);
	ret = H5Awrite(attr1, H5T_STD_I64LE, x);
	return 0;
}

//==============================================================================
// Array readers

int ImogenH5IO::readDoubleArray(const char *arrName, double **dataOut)
{
	sethid = H5Dopen(filehid, arrName, H5P_DEFAULT);
	spacehid = H5Dget_space(sethid);

	int ndims = H5Sget_simple_extent_ndims(spacehid);

	//printf("/fluid1/mass has %i dimensions\n", ndims);

	hsize_t dimvals[ndims];
	hsize_t maxvals[ndims];

	H5Sget_simple_extent_dims(spacehid, &dimvals[0], &maxvals[0]);

	//printf("dimensions are: ");
	int i; int numel = 1;
	for(i = 0; i < ndims; i++) {
		//	printf("%i ", (int)dimvals[i]);
		numel *= dimvals[i];
	}
	//printf("\n");

	//herr_t H5Dread( hid_t dataset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t xfer_plist_id, void * buf )
	if(dataOut[0] == NULL) {
		dataOut[0] = (double *)malloc(sizeof(double) * numel);
	}

	return H5Dread (sethid, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataOut[0]);
}

//==============================================================================
// Array writers

int ImogenH5IO::writeDoubleArray(const char *varname, int ndims, int *dims, double *array)
{
	hid_t fid = H5Screate(H5S_SIMPLE);
	hsize_t hd[ndims];
	int i;
	for(i = 0; i < ndims; i++ ) { hd[i] = dims[i]; }
	reverseVector(hd, ndims);

	hid_t ret = H5Sset_extent_simple(fid, ndims, hd, NULL);

	sethid = H5Dcreate2(filehid, varname, H5T_IEEE_F64LE, fid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(sethid, H5T_IEEE_F64LE, H5S_ALL , H5S_ALL, H5P_DEFAULT, (const void *)array);
}

int ImogenH5IO::writeImogenSaveframe(GridFluid *f, int nFluids, GeometryParams *geo, ParallelTopology *pt, ImogenTimeManager *timeManager)
{
	int status = SUCCESSFUL;

	int vectorndim = 1;
	int vecsize = 1;
	hsize_t hvecsize = 1;
	// FIXME this is intended to support h_x = f(idx_x) variable spacing
	status = writeDoubleArray("/dgridx", 1, &vecsize, &geo->h[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = writeDoubleArray("/dgridy", 1, &vecsize, &geo->h[1]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = writeDoubleArray("/dgridz", 1, &vecsize, &geo->h[2]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	double *hostdata = NULL;
	status = MGA_downloadArrayToCPU(&f->data[0], &hostdata, -1);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	H5Gcreate(filehid, "/fluid1", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = writeDoubleArray("/fluid1/mass", 3, &f->data[0].dim[0], hostdata);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = MGA_downloadArrayToCPU(&f->data[1], &hostdata, -1);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = writeDoubleArray("/fluid1/ener", 3, &f->data[1].dim[0], hostdata);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	status = MGA_downloadArrayToCPU(&f->data[2], &hostdata, -1);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = writeDoubleArray("/fluid1/momX", 3, &f->data[2].dim[0], hostdata);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = MGA_downloadArrayToCPU(&f->data[3], &hostdata, -1);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = writeDoubleArray("/fluid1/momY", 3, &f->data[3].dim[0], hostdata);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = MGA_downloadArrayToCPU(&f->data[4], &hostdata, -1);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = writeDoubleArray("/fluid1/momZ", 3, &f->data[4].dim[0], hostdata);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	if(nFluids > 1) {
		f++;
		H5Gcreate(filehid, "/fluid2", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		status = MGA_downloadArrayToCPU(&f[1].data[0], &hostdata, -1);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
		status = writeDoubleArray("/fluid2/mass", 3, &f->data[0].dim[0], hostdata);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
		status = MGA_downloadArrayToCPU(&f[1].data[1], &hostdata, -1);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
		status = writeDoubleArray("/fluid2/ener", 3, &f->data[1].dim[0], hostdata);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
		status = MGA_downloadArrayToCPU(&f[1].data[2], &hostdata, -1);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
		status = writeDoubleArray("/fluid2/momX", 3, &f->data[2].dim[0], hostdata);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
		status = MGA_downloadArrayToCPU(&f[1].data[3], &hostdata, -1);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
		status = writeDoubleArray("/fluid2/momY", 3, &f->data[3].dim[0], hostdata);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
		status = MGA_downloadArrayToCPU(&f[1].data[4], &hostdata, -1);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
		status = writeDoubleArray("/fluid2/momZ", 3, &f->data[4].dim[0], hostdata);
		f--;
	}

	// write null placeholders for B field variables
	double bee = 0;
	H5Gcreate(filehid, "/mag", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = writeDoubleArray("/mag/X", 1, &vecsize, &bee);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = writeDoubleArray("/mag/Y", 1, &vecsize, &bee);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = writeDoubleArray("/mag/Z", 1, &vecsize, &bee);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	// output arrangement of ranks inside our geometry
	double pa[3];
	pa[0] = (double) geo->shape;
	int nranks;
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	double procranks[nranks];
	int qq;
	for(qq = 0; qq < nranks; qq++) { procranks[qq] = qq; }
	status = writeDblAttribute("par_ geometry", 3, &pt->nproc[0], &procranks[0]);

	// output global resolution
	pa[0] = geo->globalRez[0];
	pa[1] = geo->globalRez[1];
	pa[2] = geo->globalRez[2];
	vecsize = 3;
	status = writeDblAttribute("par_ globalDims", 1, &vecsize, &pa[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	pa[0] = f->DataHolder.haloSize;
	vecsize = 1;
	int64_t hb = f->data[0].mpiCircularBoundaryBits;
	status = writeInt64Attribute("par_ haloBits", 1, &vecsize, &hb);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	bee = f->DataHolder.haloSize;
	status = writeDblAttribute("par_ haloAmt", 1, &vecsize, &bee);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	pa[0] = geo->gridAffine[0];
	pa[1] = geo->gridAffine[1];
	pa[2] = geo->gridAffine[2];
	vecsize = 3;
	status = writeDblAttribute("par_ myOffset", 1, &vecsize, &pa[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	vecsize = 1;
	status = writeDblAttribute("gamma", 1, &vecsize, &f->thermo.gamma);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	pa[0] = 1;
	// In the past, timehist[] was a vector of every dt yet taken, now it is just a blank placeholder
	status = writeDoubleArray("/timehist", 1, &vecsize, &pa[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	attrTargetDataset();

	pa[0] = (double)timeManager->iterMax();
	status = writeDblAttribute("iterMax", 1, &vecsize, &pa[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	pa[0] = (double)timeManager->iter();
	status = writeDblAttribute("iteration", 1, &vecsize, &pa[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	pa[0] = (double)timeManager->time();
	status = writeDblAttribute("time", 1, &vecsize, &pa[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	pa[0] = (double)timeManager->timeMax();
	status = writeDblAttribute("timeMax", 1, &vecsize, &pa[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	pa[0] = 1e5;
	status = writeDblAttribute("wallMax", 1, &vecsize, &pa[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	return SUCCESSFUL;
	/* missing:
	 * about (string)
	 * ver (string)
	*/
}

// ================================================================================================
// Implementation of the ImogenTimeManager class functionality
//

ImogenTimeManager::ImogenTimeManager(void)
{
	pIterDigits = 4;
	pTimeMax = 1e9;
	pIterMax = 10;
	int i;
	for(i = 0; i < 3; i++) {
		pTimePerSave[i] = -1;
		pStepsPerSave[i] = -1;
		pSlice[i] = 0;
	}
	pTimePerSave[2] = 1;
	pStepsPerSave[2] = 10;

	pNumPadDigits = 1;
	pIniIterations = 0;

	resetTime();
}

int ImogenTimeManager::readConfigParams(ImogenH5IO *conf)
{
	int status;

	double d0[3];
	int i0[3];
	status = conf->getDblAttr("/", "timeMax", &d0[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = conf->getInt32Attr("/", "iterMax", &i0[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	setLimits(d0[0], i0[0]);

	status = conf->getDblAttr("/save", "percent", &d0[0], 3);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = conf->getInt32Attr("/save", "slice", &i0[0], 3);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	int n;
	for(n = 0; n < 3; n++) {
		pTimePerSave[n]  = pTimeMax * d0[n] / 100.0;
		pStepsPerSave[n] = pIterMax * d0[n] / 100.0;
		pSlice[n] = i0[n];
	}

	status = conf->getInt32Attr("/save", "bytime", &i0[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	if(i0[0]) {
		pSaveBy = elapsedTime;
	} else {
		pSaveBy = iterations;
	}

	return status;
}

/* Sets the save mode to time (saves a frame per given time elapsed), and sets this time equal
 * to t */
void ImogenTimeManager::savePerTime(double t, int dim)
{
	if((dim < 1) || (dim  > 3)) return;

	pSaveBy = elapsedTime;
	pTimePerSave[dim-1] = t;
}

/* Sets the save mode to steps (saves a frame per given #steps elapsed), and sets the number
 * of steps to s */
void ImogenTimeManager::savePerSteps(int s, int dim)
{
	if((dim < 1) || (dim  > 3)) return;

	pSaveBy = iterations;
	pStepsPerSave[dim-1] = s;
}

/* Dumps existing chronometric information, resetting time and iterations to zero */
void ImogenTimeManager::resetTime(void)
{
	pTime = 0;
	pNextDt = 0;
	pKahanTimeC = 0;

	pIterations = 0;
	pIniIterations = 0;
}

int ImogenTimeManager::resumeTime(char *prefix, int frameno)
{
	char *fname = (char *)malloc(strlen(prefix) + 64);
	int pad;
	for(pad = 1; pad < 10; pad++) {
		sprintf(fname, "%s_3D_XYZ_rank000_%0*i.h5", prefix, pad, frameno);
		FILE *ftest = fopen(fname, "r");
		if(ftest != NULL) {
			fclose(ftest);
			break;
		}
	}
	if(pad == 10) {
		PRINT_FAULT_HEADER;
		std::cout << "Attempted to resume time from frame, but accessing file '" << fname << "' failed.\nThis really shouldn't happen since by this point I've successfully read the dataframe...\n";
		PRINT_FAULT_FOOTER;
		return ERROR_CRASH;
	}

	ImogenH5IO reader(fname);

	int status = reader.getDblAttr("/timehist", "time", &pTime);
	status = reader.getInt32Attr("/timehist", "iteration", &pIterations);

	int myrank;

	if(imRankZero()) {
		std::cout << "Resuming from frame " << frameno << " with time=" << pTime << " and iteration=" << pIterations << "." << std::endl; }

	pIniIterations = frameno;

	free(fname);
	return 0;
}


/* Sets the maximum time for the simulation to run & number of iterations to take */
void ImogenTimeManager::setLimits(double timeMax, int iterMax)
{
	if(timeMax > 0) pTimeMax = timeMax;
	if(iterMax > 0) {
		pIterMax = iterMax;
		pNumPadDigits = (int)ceil(log10((double)pIterMax));
	}
}

/* Requests the time manager to "check out" the suggested timestep dt
 * This may return dt, or a lesser value if we are saving at fixed time intervals
 * or the end-of-simulation tims is near */
double ImogenTimeManager::registerTimestep(double dt)
{
	double tNext = pTime + 2*dt;
	if (tNext - pTimeMax > -1e-8) {
		dt = (pTimeMax - pTime) / 2;
	}
	if(pSaveBy == elapsedTime) {
		double persNow = floor(pTime / pTimePerSave[2]);
		double persNext= floor(tNext / pTimePerSave[2]);
		if(persNext > persNow) {
			dt = (persNext * pTimePerSave[2] - pTime) / 2;
		}
	}

	pNextDt = dt;
	return dt;
}

/* Returns true if the current step should be saved.
 * Call immediately after calling applyTimestep after
 * completing the timestep itself. */
bool ImogenTimeManager::saveThisStep(void)
{
	if(pSaveBy == elapsedTime) {
		double pers = floor(pTime / pTimePerSave[2]);
		if( fabs( pers * pTimePerSave[2] - pTime ) < 1e-7 ) return true;
	}
	if(pSaveBy == iterations) {
		if(pIterations % pStepsPerSave[2] == 0) return true;
	}
	return false;
}

/* Applies the stores pNextDt which was updated by registerTimestep()
 * Call after completing the timestep by before checking for save/terminate */
void ImogenTimeManager::applyTimestep(void)
{
	// pTime += pNextDt; // kahan summated
	double y = (2*pNextDt) - pKahanTimeC;
	double t = pTime + y;
	pKahanTimeC = (t - pTime) - y;
	pTime = t;

	pIterations++;
}

/* Returns true if we have either met the maximum time or maximum steps
 * this simulation is set to take */
bool ImogenTimeManager::terminateSimulation(void)
{
if(pIterations >= pIterMax) return true;
if(pTime - pTimeMax > -1e-8) return true;

return false;
}
