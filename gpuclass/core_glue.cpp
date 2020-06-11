#include "stdio.h"
#include "stdlib.h"

#include "cuda.h"
#include "cuda_runtime.h"

#include "../mpi/mpi_common.h"

#include "math.h"

#include "cudaCommon.h"

#include "core_glue.hpp"

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

}

// ================================================================================================
// Implementation of the ImogenH5IO class that provides all needed functionality for talking to
// H5 files, both as a means of structured grid input/output and for the parameter file, which is
// just a bunch of attributes strapped into an H5 bundle

ImogenH5IO::ImogenH5IO(char *filename, unsigned int flags)
{
	openUpFile(filename, flags);
	attrhid = -1; spacehid = -1; sethid = -1;
	attrTarg = -1; grouphid = -1;
	attrTargetRoot();
}

ImogenH5IO::ImogenH5IO(char *filename)
{
	openUpFile(filename, H5F_ACC_RDONLY);
	attrhid = -1; spacehid = -1; sethid = -1;
	attrTarg = -1; grouphid = -1;
	attrTargetRoot();
}

ImogenH5IO::ImogenH5IO(void)
{
	filehid = -1;
	attrhid = -1; spacehid = -1; sethid = -1;
	attrTarg = -1; grouphid = -1;
	attrTargetRoot();
}

ImogenH5IO::~ImogenH5IO(void)
{
	if(haveOpenedFile()) closeOutFile();
}

int ImogenH5IO::openUpFile(char *filename, unsigned int flags)
{
	// read
	if(flags == H5F_ACC_RDONLY) {
		filehid = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
		return 0;
	} else { // write
		int myrank;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
		char myfile[strlen(filename) + 32];
		sprintf(&myfile[0], "%s_rank%03i.h5", filename, myrank);
		filehid = H5Fcreate(&myfile[0], flags, H5P_DEFAULT, H5P_DEFAULT);
	}

	attrTarg = filehid;

	if(filehid > 0) return 0; else; return ERROR_CRASH;
}

bool ImogenH5IO::haveOpenedFile()
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

int ImogenH5IO::readDoubleArray(char *arrName, double **dataOut)
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

	int i;
	int ne = 1;
	for(i = 0; i < ndims; i++) { ne *= dimensions[i]; }

	*numel = ne;

	return SUCCESSFUL;
}

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

int ImogenH5IO::writeDblAttribute(const char *name, int ndims, int *dimensions, double *x)
{
	hid_t atttype = H5Screate(H5S_SIMPLE);
	hsize_t foo[ndims];
	int i;
	for(i = 0; i < ndims; i++) { foo[i] = (hsize_t)dimensions[i]; }

	hid_t ret  = H5Sset_extent_simple(atttype, 1, &foo[0], NULL);
	hid_t attr1 = H5Acreate2(attrTarg, name, H5T_IEEE_F64LE, atttype, H5P_DEFAULT, H5P_DEFAULT);
	ret = H5Awrite(attr1, H5T_IEEE_F64LE, x);
	return 0;
}

int ImogenH5IO::writeFltAttribute(const char *name, int ndims, int *dimensions, float *x)
{
	hid_t atttype = H5Screate(H5S_SIMPLE);
	hsize_t foo[ndims];
	int i;
	for(i = 0; i < ndims; i++) { foo[i] = (hsize_t)dimensions[i]; }

	hid_t ret  = H5Sset_extent_simple(atttype, 1, &foo[0], NULL);
	hid_t attr1 = H5Acreate2(attrTarg, name, H5T_IEEE_F32LE, atttype, H5P_DEFAULT, H5P_DEFAULT);
	ret = H5Awrite(attr1, H5T_IEEE_F32LE, x);
	return 0;
}

int ImogenH5IO::writeInt32Attribute(const char *name, int ndims, int *dimensions, int32_t *x)
{
	hid_t atttype = H5Screate(H5S_SIMPLE);
	hsize_t foo[ndims];
	int i;
	for(i = 0; i < ndims; i++) { foo[i] = (hsize_t)dimensions[i]; }

	hid_t ret  = H5Sset_extent_simple(atttype, 1, &foo[0], NULL);
	hid_t attr1 = H5Acreate2(attrTarg, name, H5T_STD_I32LE, atttype, H5P_DEFAULT, H5P_DEFAULT);
	ret = H5Awrite(attr1, H5T_STD_I32LE, x);
	return 0;
}

int ImogenH5IO::writeInt64Attribute(const char *name, int ndims, int *dimensions, int64_t *x)
{
	hid_t atttype = H5Screate(H5S_SIMPLE);
	hsize_t foo[ndims];
	int i;
	for(i = 0; i < ndims; i++) { foo[i] = (hsize_t)dimensions[i]; }

	hid_t ret  = H5Sset_extent_simple(atttype, 1, &foo[0], NULL);
	hid_t attr1 = H5Acreate2(attrTarg, name, H5T_STD_I64LE, atttype, H5P_DEFAULT, H5P_DEFAULT);
	ret = H5Awrite(attr1, H5T_STD_I64LE, x);
	return 0;
}

int ImogenH5IO::writeImogenSaveframe(GridFluid *f, int nFluids, GeometryParams *geo, ParallelTopology *pt)
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

	double pa[3];
	pa[0] = (double) geo->shape;
	status = writeDblAttribute("par_ geometry", 1, &vecsize, &pa[0]);
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

	status = writeDblAttribute("iterMax", 1, &vecsize, &pa[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = writeDblAttribute("iteration", 1, &vecsize, &pa[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = writeDblAttribute("time", 1, &vecsize, &pa[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = writeDblAttribute("timeMax", 1, &vecsize, &pa[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	status = writeDblAttribute("wallMax", 1, &vecsize, &pa[0]);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

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
	pSaveBy = iterations;
	pTimePerSave = 1;
	pStepsPerSave = 10;

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

	return status;
}

/* Sets the save mode to time (saves a frame per given time elapsed), and sets this time equal
 * to t */
void ImogenTimeManager::savePerTime(double t)
{
	pSaveBy = elapsedTime;
	pTimePerSave = t;
}

/* Sets the save mode to steps (saves a frame per given #steps elapsed), and sets the number
 * of steps to s */
void ImogenTimeManager::savePerSteps(int s)
{
	pSaveBy = iterations;
	pStepsPerSave = s;
}

/* Dumps existing chronometric information, resetting time and iterations to zero */
void ImogenTimeManager::resetTime(void)
{
	pTime = 0;
	pNextDt = 0;
	pKahanTimeC = 0;

	pIterations = 0;
}

/* Sets the maximum time for the simulation to run & number of iterations to take */
void ImogenTimeManager::setLimits(double timeMax, int iterMax)
{
	pTimeMax = timeMax;
	pIterMax = iterMax;
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
		double persNow = floor(pTime / pTimePerSave);
		double persNext= floor(tNext / pTimePerSave);
		if(persNext > persNow) {
			dt = (persNext * pTimePerSave - pTime) / 2;
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
		double pers = floor(pTime / pTimePerSave);
		if( abs( pers * pTimePerSave - pTime ) < 1e-7 ) return true;
	}
	if(pSaveBy == iterations) {
		if(pIterations % pStepsPerSave == 0) return true;
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
if(pTime - pTimeMax > 1e-8) return true;

return false;
}
