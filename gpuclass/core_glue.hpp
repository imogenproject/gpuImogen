void factorizeInteger(int X, int **factors, int *nfactors);

void describeTopology(ParallelTopology *topo);
void describeGeometry(GeometryParams *geo);

BCModeTypes num2BCModeType(int x)
{
	switch(x) {
	case 1: return circular; break;
	case 2: return mirror; break;
	case 3: return wall; break;
	case 4: return stationary; break;
	case 5: return extrapConstant; break;
	case 6: return extrapLinear; break;
	case 7: return outflow; break;
	case 8: return freebalance; break;
	default:
		std::cout << "ERROR: bc mode corresponding to integer " << x << " is unknown. See gpuImogen:BCManager.m:200\nReturning circular" << std::endl;
		return circular;
	}
}

#include "hdf5.h"
#include "hdf5_hl.h"

class ImogenH5IO {
public:
	// Constructor given a g
	ImogenH5IO(char *filename, unsigned int flags);
	ImogenH5IO(char *filename);
	ImogenH5IO(void);
	~ImogenH5IO(void);

	// Open/close functionality
	bool haveOpenedFile();
	int openUpFile(char *filename, unsigned int flags);
	void closeOutFile(void);

	// Array IO functionality
	int getArrayNDims(const char *name);
	int getArraySize(const char *name, hsize_t *dims);
	int readDoubleArray(char *arrName, double **dataOut);
	int writeDoubleArray(const char *varname, int ndims, int *dims, double *array);

	// Attribute aux functions
	void attrTargetRoot(void) { attrTarg = filehid; }
	void attrTargetDataset(void) { attrTarg = sethid; }
	void attrTargetGroup(void) { attrTarg = grouphid; }
	int getAttrInfo(const char *location, const char *attrName, hsize_t *dimensions, H5T_class_t *type);

	// Attribute read access
	int chkAttrNumel(const char *location, const char *attrName, int *numel);

	int getDblAttr(const char *location, const char *attrName, double *data, int nmax = 1);
	int getFltAttr(const char *location, const char *attrName, float *data, int nmax = 1);
	int getInt32Attr(const char *location, const char *attrName, int *data, int nmax = 1);
	int getInt64Attr(const char *location, const char *attrName, long int *data, int nmax = 1);
	int getStrAttr(const char *location, const char *attrName, char *data, int nmax = 1);

	// Attribute write access
	int writeDblAttribute(const char *name, int ndims, int *dimensions, double *x);
	int writeFltAttribute(const char *name, int ndims, int *dimensions, float *x);
	int writeInt32Attribute(const char *name, int ndims, int *dimensions, int32_t *x);
	int writeInt64Attribute(const char *name, int ndims, int *dimensions, int64_t *x);

	int writeImogenSaveframe(GridFluid *f, int nFluids, GeometryParams *geo, ParallelTopology *pt);
private:
	hid_t filehid, attrhid, spacehid, sethid, grouphid;
	hid_t attrTarg;

	/* if the elements of vector x are labelled [1...n],
	 * reorders x in place so that they are now [n...1] */
	template <class V> void reverseVector(V *x, int n)
	{
		int i; V t;
		for(i = 0; i < n/2; i++) {
			t = x[i];
			x[i] = x[n-1-i];
			x[n-1-i] = t;
		}
	}
};

typedef enum { iterations, elapsedTime } ImogenSaveModes;

class ImogenTimeManager {
public:

	ImogenTimeManager(void);

	void setLimits(double timeMax, int iterMax);
	void savePerTime(double t);
	void savePerSteps(int s);
	bool saveThisStep(void);

	int readConfigParams(ImogenH5IO *conf);

	void resetTime(void);
	double registerTimestep(double dt);
	void applyTimestep(void);

	bool terminateSimulation(void);

	double time(void) const { return pTime; }
	double timeMax(void) const { return pTimeMax; }
	int iter(void) const { return pIterations; }
	int iterMax(void) const { return pIterMax; }
private:
	double pTime;  // private - the the total time elapsed; sum of pNextDt every time registerTimestep() is called
	double pNextDt;// private - the next timestep indicated to be taken

	int pIterations;

	double pTimeMax, pIterMax;

	ImogenSaveModes pSaveBy;

	double pTimePerSave; int pStepsPerSave;

	int pIterDigits;
	double pKahanTimeC;
};

