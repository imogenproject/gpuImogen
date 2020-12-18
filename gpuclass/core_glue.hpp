#ifndef CORE_GLUE_H
#define CORE_GLUE_H

bool imRankZero(void);

int lookForArgument(int argc, char **argv, const char *argument);
int argReadKeyValueDbl(int argc, char **argv, const char *key, double *val);
int argReadKeyValueInt(int argc, char **argv, const char *key, int *val);
int argReadKeyValueStr(int argc, char **argv, const char *key, char **val);
void printHelpScreed(void);

int parseDevicesArgument(char *arg, int *nDevices, int *deviceList);

void factorizeInteger(int X, int **factors, int *nfactors);

void describeTopology(ParallelTopology *topo);
void describeGeometry(GeometryParams *geo);

BCModeTypes num2BCModeType(int x);

typedef enum SaveFrameTypes_ { X, Y, Z, XY, XZ, YZ, XYZ } SaveFrameTypes;

#include "hdf5.h"
#include "hdf5_hl.h"

class ImogenH5IO {
public:
	// Constructor given a g
	ImogenH5IO(const char *filename, unsigned int flags);
	ImogenH5IO(const char *filename);
	ImogenH5IO(void);
	~ImogenH5IO(void);

	// Open/close functionality
	bool haveOpenedFile(void);
	int openUpFile(const char *filename, unsigned int flags);
	void closeOutFile(void);
	void closeDataset(void);
	int openImogenSavefile(const char *namePrefix, int frameno, int pad, SaveFrameTypes s);

	// Array IO functionality
	int getArrayNDims(const char *name);
	int getArraySize(const char *name, hsize_t *dims);
	int readDoubleArray(const char *arrName, double **dataOut);
	int writeDoubleArray(const char *varname, int ndims, int *dims, double *array, bool closeoutNow = false);

	// Attribute aux functions
	void attrTargetRoot(void) { attrTarg = filehid; }
	void attrTargetDataset(void) { attrTarg = sethid; }
	void attrTargetGroup(void) { attrTarg = grouphid; }
	int getAttrInfo(const char *location, const char *attrName, hsize_t *dimensions, H5T_class_t *type);
	int checkOverwrite(const char *name);

	// Attribute read access
	int chkAttrNumel(const char *location, const char *attrName, int *numel);

	int getDblAttr(const char *location, const char *attrName, double *data, int nmax = 1);
	int getFltAttr(const char *location, const char *attrName, float *data, int nmax = 1);
	int getInt32Attr(const char *location, const char *attrName, int *data, int nmax = 1);
	int getInt64Attr(const char *location, const char *attrName, long int *data, int nmax = 1);
	int getStrAttr(const char *location, const char *attrName, char **data, int nmax = 1);

	// Attribute write access
	int writeDblAttribute(const char *name, int ndims, int *dimensions, double *x);
	int writeFltAttribute(const char *name, int ndims, int *dimensions, float *x);
	int writeInt32Attribute(const char *name, int ndims, int *dimensions, int32_t *x);
	int writeInt64Attribute(const char *name, int ndims, int *dimensions, int64_t *x);
	int writeStrAttr(const char *name, char *thestring);

	int writeImogenSaveframe(GridFluid *f, int nFluids, GeometryParams *geo, ParallelTopology *pt, class ImogenTimeManager *timeManager);
private:
	hid_t filehid, attrhid, spacehid, sethid, grouphid;
	hid_t attrTarg;
	bool attrOverwrite;

	int prepareAttribute(const char *location, int ndims, int *dimensions, hid_t *attout);

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
	void savePerTime(double t, int dim = 3);
	void savePerSteps(int s, int dim = 3);
	bool saveThisStep(void);

	int readConfigParams(ImogenH5IO *conf);
	int resumeTime(char *prefix, int frameno);

	void resetTime(void);
	double registerTimestep(double dt);
	void applyTimestep(void);

	bool terminateSimulation(void);

	double time(void) const { return pTime; }
	double timeMax(void) const { return pTimeMax; }
	int iter(void) const { return pIterations; }
	int iterMax(void) const { return pIterMax; }
	int iniIterations(void) const { return pIniIterations; }
	int numPadDigits(void) const { return pNumPadDigits; }
private:
	int pNumPadDigits;

	double pTime;  // private - the the total time elapsed; sum of pNextDt every time registerTimestep() is called
	double pNextDt;// private - the next timestep indicated to be taken

	int pIterations; // The number of iterations [ # of calls to applyTimestep since init or resetTime() ]
	int pIniIterations;

	double pTimeMax, pIterMax; // maximum time to elapse or iterations to take before iteration loop exits

	int pSlice[3]; // The [X Y Z] global indexes to take cuts at; If e.g. outputting XY, samples at [:, :, Z].

	ImogenSaveModes pSaveBy; // Either 'iterations' or 'elapsedTime'

	double pTimePerSave[3]; int pStepsPerSave[3]; // Time or steps, depending on pSaveBy, to elapse between saving [1D 2D 3D] frames

	int pIterDigits;
	double pKahanTimeC;
};

#else

#endif
