/*
 * gpuimogen_core.cu
 *
 *  Created on: Apr 24, 2020
 *      Author: Erik Keever
 */

#include <stdio.h>
#include <string.h>

#include <stdint.h>
#include <unistd.h>

#include "mpi.h"

#include "cuda.h"
#include "nvToolsExt.h"

#include "cudaCommon.h"
#include "cudaFluidStep.h"
#include "cudaFreeRadiation.h"
#include "cudaSoundspeed.h"
#include "cflTimestep.h"

#include "sourceStep.h"
#include "flux.h"

#include "core_glue.hpp"

#include "gpuimogen_core.h"

using namespace std;
// Only uncomment this if you plan to debug this file.
// This will cause it to require output arguments to return data in,
// and perturb code behavior by generating writes to the output debug arrays
//#define DEBUGMODE

FluidMethods mlmethodToEnum(int mlmethod);
//int fetchMinDensity(mxArray *mxFluids, int fluidNum, double *rhoMin);

#ifdef DEBUGMODE
    #include "debug_inserts.h"
#endif


int main(int argc, char **argv)
{
	// Step 1 - initialize MPI
	int status = MPI_Init(&argc, &argv);

	// Step 2 - access initializer file
	// The initializer file is an .h5 file, autotranslated by the Matlab GPU-Imogen code,
	// of the (relevant) parts of the IC.ini structure
	ImogenH5IO *conf = new ImogenH5IO("testout.h5");

	int globalResolution[3];
	status = conf->getInt32Attr("/", "globalResolution", &globalResolution[0], 3);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }




	// Step 3 - Setup MPI topology
	ParallelTopology topo = acquireParallelTopology(&globalResolution[0]);

	// Step 2 - initialize CUDA

	// Hardcode this for now
	int deviceList[MAX_GPUS_USED];
	int nCudaDevices = 1;
	deviceList[0] = 0;

	// Set this to make the API run its startup magic
	cudaSetDevice(0);

	int i, j;
	// Default to default stream
	for(i = 0; i < 2*MAX_GPUS_USED; i++) { topo.cudaStreamPtrs[i] = 0; }

	// Create two streams on every device we plan to use
	for(i = 0; i < nCudaDevices; i++) {
		cudaSetDevice(0);
		cudaStreamCreate(&topo.cudaStreamPtrs[i]);
		cudaStreamCreate(&topo.cudaStreamPtrs[nCudaDevices + i]);
	}

	// Step 3 - Begin filling in some hardcoded defaults just for test purposes

	// Default radiation parameters [ off ]
	ParametricRadiation prad;
	prad.exponent       = .5;
	prad.minTemperature = 1.05;
	prad.prefactor      = 0; // DISABLE

	// Default fluid step parameters
	FluidStepParams fsp;
	fsp.dt            = 0;
	fsp.onlyHydro     = 1;
	fsp.stepDirection = 1;
	fsp.stepMethod    = METHOD_HLL; // FIXME HARDCODED
	fsp.stepNumber    = 0;
	status = conf->getDblAttr("/","cfl", &fsp.cflPrefactor);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	// Acquire simulation geometry
	int circularity = 0; // [1 2 4] = [
	fsp.geometry = acquireSimulationGeometry(&globalResolution[0], &topo, circularity);

	status = conf->getDblAttr("/", "d3h", &fsp.geometry.h[0], 3);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	if(0) { // TESTING
		int myrank;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
		//printf("RANK %i:\n", myrank);
		//describeTopology(&topo);
		//describeGeometry(&fsp.geometry);
		//MPI_Finalize();
		//return 0; */
	}

	ImogenTimeManager *timeManager = new ImogenTimeManager();

	timeManager->readConfigParams(conf);
	// FIXME...
	timeManager->savePerSteps(10);

	// Choose # of fluids [ default to 1 for now ]
	int numFluids = 1;
	status = conf->getInt32Attr("/", "multifluidDragMethod", &fsp.multifluidDragMethod); // see cudaSource2FluidDrag.cu ~ 250
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	// Define the gravity field if it exists [ testing default - it doesn't ]
	GravityData gravdat;
	gravdat.phi        = NULL;
	gravdat.spaceOrder = 0;
	gravdat.timeOrder  = 0;
	int haveg = 0;

	// Create the fluids!!!
	GridFluid fluids[numFluids];
	MGArray dataholder[numFluids];

	int fluidct;
	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		int bcnumbers[6];
		// Set the halo buffer size, device count & list of devices
		// These will be duplicated into all generated arrays from the fluid creator
		dataholder[fluidct].haloSize = FLUID_METHOD_HALO_SIZE;
		dataholder[fluidct].nGPUs = nCudaDevices;
		for(i = 0; i < nCudaDevices; i++) dataholder[fluidct].deviceID[i] = deviceList[i];

		// This to be replaced with a call to a real parameter loader one day
		status = readImogenICs(&fluids[fluidct], &dataholder[fluidct], &fsp.geometry, "serial2dbarf");
		//status = generateDummyFluid(&fluids[fluidct], &dataholder[fluidct], &fsp.geometry);
		if(status != SUCCESSFUL) break;

		// FIXME this attaches boundary conditions most naively
		status = conf->getInt32Attr("/fluidDetail1", "/bcmodes", &bcnumbers[0], 6);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
		for(j = 0; j < 5; j ++ ) {
			for(i = 0; i < 6; i++) { fluids[fluidct].data[j].boundaryConditions.mode[i] = num2BCModeType(bcnumbers[i]); }
		}

	}

	if(status != SUCCESSFUL) {
		CHECK_IMOGEN_ERROR(status);
		MPI_Finalize();
		return -1;
	}


	// Fetch the XY vectors
	//MGArray XYvectors;
	//worked = MGA_accessMatlabArrays(&prhs[5], 0, 0, &XYvectors);
	//fsp.geometry.XYVector = &XYvectors;
	fsp.geometry.XYVector = (MGArray *)NULL;
	// Get the global domain rez for doing cfl
//	double *globrez = mxGetPr(derefXdotAdotB(theImogenManager, "geometry", "globalDomainRez"));
	//int i;
	//for(i = 0; i < 3; i++) {
//		fsp.geometry.globalRez[i] = globrez[i];
//	}

	// Determine what kind of source type we're going to do
	int cylcoords = (fsp.geometry.shape == CYLINDRICAL) || (fsp.geometry.shape == RZCYLINDRICAL);
	// sourcerFunction = (fsp.geometry. useCyl + 2*useRF + 4*usePhi + 8*use2F;
	int srcType = 1*(cylcoords == 1) + 2*(fsp.geometry.frameOmega != 0) + 4*(haveg) + 8*(numFluids > 1);

	// Done reading configuration by now one would hope
	delete conf;

	//=================================

	while(1) {
		// This computes the CFL time & also does timeManager->registerTimestep()
		status = performCompleteTimestep(fluids, numFluids, fsp, topo, &gravdat, &prad, srcType, timeManager);
		status = CHECK_IMOGEN_ERROR(status);

		timeManager->applyTimestep();
		if(timeManager->saveThisStep()) {
			std::cout << "TM says to save step " << timeManager->iter() << "." << endl;

		}

		if((status != SUCCESSFUL) | timeManager->terminateSimulation()) break;
	}

	std::cout << "Took " << timeManager->iter() << "steps, Ttotal = " << timeManager->time() << " elapsed." << endl;

	ImogenH5IO frmw("testframe", H5F_ACC_TRUNC);
	frmw.writeImogenSaveframe(fluids, numFluids, &fsp.geometry, &topo);
	frmw.closeOutFile();

	// Tear down Imogen
	delete timeManager;


	// Tear down CUDA

	// Tear down MPI
	topoDestroyDimensionalCommunicators(&topo);

	MPI_Finalize();
}

// Given the global domain resolution (which must point to 3 integers, [Nx Ny Nz],
// assigns available MPI ranks into a cartesian grouping described by the returned
// ParallelTopology.
ParallelTopology acquireParallelTopology(int *globalDomainResolution)
{
	ParallelTopology p;

	// confirm # of space dimensions
	p.ndim = 3;
	if(globalDomainResolution[2] == 1) p.ndim = 2;
	if(globalDomainResolution[1] == 1) p.ndim = 1;

	int nproc, myrank;

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	// Compute prime factorization of the # of ranks in world
	int *factors; int nfactors;
	factorizeInteger(nproc, &factors, &nfactors);

	int nRanks[3] = {1, 1, 1};

	// Apply prime factors to build the most square/cube like domains we can
	// optimization objective, maximize volume over surface area
	int i, t, j;
	for(i = 0; i < nfactors; i++) {
		t = nfactors - 1 - i;
		int nextfact = factors[t];

		j = 0;
		if(globalDomainResolution[1] / nRanks[1] > globalDomainResolution[0] / nRanks[0]) j = 1;
		if(globalDomainResolution[2] / nRanks[2] > globalDomainResolution[1] / nRanks[1]) j = 2;

		nRanks[j] *= nextfact;
	}

	p.comm = MPI_Comm_c2f(MPI_COMM_WORLD);

	// Compute this rank's linear position within the cartesian group
	p.coord[2] = myrank / (nRanks[0]*nRanks[1]);
	int rem = myrank - p.coord[2] * nRanks[0]*nRanks[1];
	p.coord[1] = rem / nRanks[0];
	rem = rem - p.coord[1] * nRanks[0];
	p.coord[0] = rem;

	int neigh[3];
	for(i = 0; i < 3; i++) {
		p.nproc[i] = nRanks[i];

		neigh[0] = p.coord[0]; neigh[1] = p.coord[1]; neigh[2] = p.coord[2];
		neigh[i] = (p.coord[i] - 1 + p.nproc[i]) % p.nproc[i];
		p.neighbor_left[i] = topoNodeToRank(&p, &neigh[0]);

		neigh[i] = (p.coord[i] + 1 + p.nproc[i]) % p.nproc[i];
		p.neighbor_right[i] = topoNodeToRank(&p, &neigh[0]);
	}

	// This return is currently irrelevant
	// FIXME make this return detect errors (file is mpi_common.cpp)
	sleep(myrank);
	int status = topoCreateDimensionalCommunicators(&p);

	return p;
}

// Sets up a basic square grid, filling in correctly the globalRez, localRez and gridAffine components
// but leaving all others defaulted
GeometryParams generateGridGeometry(int *globalResolution, ParallelTopology *topo, int circular)
{
GeometryParams g;

g.Rinner = 0;
g.XYVector = NULL;
g.frameOmega = 0;
// FIXME frame rotation is only supported about the z axis...
g.frameRotateCenter[0] = 0.0;
g.frameRotateCenter[1] = 0.0;
g.frameRotateCenter[2] = 0.0;

g.h[0] = g.h[1] = g.h[2] = 1.0;
g.shape = SQUARE;
g.x0 = 0;
g.y0 = 0;
g.z0 = 0;

// HACK FIXME HACK
int circularBdy = 1;

int i;
int subsize;
int halo = FLUID_METHOD_HALO_SIZE;
int deficit;
for(i = 0; i < 3; i++) {
	if(globalResolution[i] != 1) {
		circularBdy = (circular & (1 << i)) != 0 ? 1 : 0;


		g.globalRez[i] = globalResolution[i];
		subsize = globalResolution[i] / topo->nproc[i];
		deficit = globalResolution[i] - subsize*topo->nproc[i];

		g.gridAffine[i] = subsize * topo->coord[i];

		// extend to add the left halo only if in parallel
		if(topo->nproc[i] > 1) {
			subsize         += halo * ((topo->coord[i] > 0) || (circularBdy) );
			g.gridAffine[i] -= halo * ((topo->coord[i] > 0) || (circularBdy) );
		}

		// extend to add the right halo only if in parallel
		if(topo->nproc[i] > 1) {
			subsize += halo * ((topo->coord[i] < (topo->nproc[i]-1)) || (circularBdy) );
		}

		// Any missing resolution is added to the last set of ranks in the direction
		// So if rez = [5] and nproc = 3, we have sizes [1, 1, 3] for ranks [0, 1, 2] respectively
		if(topo->coord[i] == (topo->nproc[i]-1)) {
			subsize += deficit;
		}
		g.localRez[i] = subsize;
	} else { // size of one: trivial special behavior in this case
		g.globalRez[i] = globalResolution[i];
		g.localRez[i] = globalResolution[i];
		g.gridAffine[i] = 0;
	}
}

return g;
}

// this will need to be passed the input parameters data to fill in the rest of the geometry fields
GeometryParams acquireSimulationGeometry(int *globalResolution, ParallelTopology *topo, int circ)
{
return generateGridGeometry(globalResolution, topo, circ);
}

int generateDummyFluid(GridFluid *g, MGArray *holder, GeometryParams *geo)
{

	double *q = (double *)malloc((unsigned long)(sizeof(double)*geo->localRez[0] * geo->localRez[1] * geo->localRez[2]));

	MGArray m;
	int halosize = holder->haloSize;
	int partDir = 1; // FIXME HACK
	int exteriorHalo = 0; // FIXME HACK AND SOMETIMES WRONG
	int nDevices = holder->nGPUs;
	int *deviceList = &holder->deviceID[0];

	int i;

	long j, ne;
	ne = geo->localRez[0] * geo->localRez[1] * geo->localRez[2];

	for(j = 0; j < ne; j++) { q[j] = 1.0; }
	// upload rho to create first array, m

	int status = uploadHostArray(&m, q, &geo->localRez[0], halosize, partDir, exteriorHalo, 0, nDevices, deviceList);

	// acquire a slab of 5 such arrays stored at the holder
	status = MGA_allocSlab(&m, &holder[0], 5);

	// memcopy rho to holder aka slab 0
	MGA_duplicateArray(&holder, &m);
	g->data[0] = holder[0];
	g->data[0].numSlabs = 0;

	MGA_delete(&m);
	for(j = 0; j < 6; j++) { g->data[0].boundaryConditions.mode[j] = circular; }

	// (fake E here)
	for(j = 0; j < ne; j++) { q[j] = 1.0; }

	// select slab 1
	g->data[1] = g->data[0];
	g->data[1].numSlabs = -1;
	for(i = 0; i < g->data[0].nGPUs; i++) { g->data[1].devicePtr[i] += g->data[0].slabPitch[i]; }
	// upload
	status = uploadHostArray(&g->data[1], q, &geo->localRez[0], halosize, partDir, exteriorHalo, 0, 1, deviceList);
	for(j = 0; j < 6; j++) { g->data[1].boundaryConditions.mode[j] = circular; }

	// (fake px here)
	for(j = 0; j < ne; j++) { q[j] = 0.1; }

	// select slab 2
	g->data[2] = g->data[0];
	g->data[2].numSlabs = -2;
	for(i = 0; i < g->data[0].nGPUs; i++) { g->data[2].devicePtr[i] += 2*g->data[0].slabPitch[i]; }
	// upload
	status = uploadHostArray(&g->data[2], q, &geo->localRez[0], halosize, partDir, exteriorHalo, 0, 1, deviceList);
	for(j = 0; j < 6; j++) { g->data[2].boundaryConditions.mode[j] = circular; }

	// (fake py here)
	for(j = 0; j < ne; j++) { q[j] = 0.2; }

	// select slab 3
	g->data[3] = g->data[0];
	g->data[3].numSlabs = -3;
	for(i = 0; i < g->data[0].nGPUs; i++) { g->data[3].devicePtr[i] += 3*g->data[0].slabPitch[i]; }
	// upload
	status = uploadHostArray(&g->data[3], q, &geo->localRez[0], halosize, partDir, exteriorHalo, 0, 1, deviceList);
	for(j = 0; j < 6; j++) { g->data[3].boundaryConditions.mode[j] = circular; }

	// (fake pz here)
	for(j = 0; j < ne; j++) { q[j] = 0.0; }

	// select slab 4
	g->data[4] = g->data[0];
	g->data[4].numSlabs = -4;
	for(i = 0; i < g->data[0].nGPUs; i++) { g->data[4].devicePtr[i] += 4*g->data[0].slabPitch[i]; }
	// upload
	status = uploadHostArray(&g->data[4], q, &geo->localRez[0], halosize, partDir, exteriorHalo, 0, 1, deviceList);
	for(j = 0; j < 6; j++) { g->data[4].boundaryConditions.mode[j] = circular; }

	g->rhoMin = 1e-7;
	g->thermo = generateDefaultThermo();

	free(q);

	return status;
}

int readImogenICs(GridFluid *g, MGArray *holder, GeometryParams *geo, char *h5dfilebase)
{

	double *q = (double *)malloc((unsigned long)(sizeof(double)*geo->localRez[0] * geo->localRez[1] * geo->localRez[2]));

	MGArray m;
	int halosize = holder->haloSize;
	int partDir = 1; // FIXME HACK
	int exteriorHalo = 0; // FIXME HACK AND SOMETIMES WRONG
	int nDevices = holder->nGPUs;
	int *deviceList = &holder->deviceID[0];

	int i;

	long j;

	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	char *hfile = (char *)malloc(strlen(h5dfilebase) + 16);
	sprintf(hfile,"%s_rank%03i.h5", h5dfilebase, myrank);
	ImogenH5IO IHR(hfile);
	free(hfile);

	// Check data that the initializer stored in the H5 output frames to confirm that this is correct and loadable!!!
	double chkHalo;
	IHR.getDblAttr("/","par_ haloAmt", &chkHalo);
	if((int)chkHalo != holder->haloSize) {
		PRINT_FAULT_HEADER;
		std::cout << "Upon check, it has been found that input file halo size = " << (int)chkHalo << "is not equal to mgarray halo amount of " << holder->haloSize << " and this represents a fatal error." << endl;
		PRINT_FAULT_FOOTER;
		return ERROR_INVALID_ARGS;
	}

	double chkGlobalDims[3];
	IHR.getDblAttr("/", "par_ globalDims", &chkGlobalDims[0], 3);
	for(i = 0; i < 3; i++) {
		if((int)chkGlobalDims[i] != geo->globalRez[i]) {
			PRINT_FAULT_HEADER;
			std::cout << "Upon check, it has been found that input file and mgarray global resolutions differ:\n";
			std::cout << "   file=[" << (int)chkGlobalDims[0] << " " << (int)chkGlobalDims[1] << " " << (int)chkGlobalDims[2] << " vs mgarray [";
			std::cout << geo->globalRez[0] << " " << geo->globalRez[1] << " " << geo->globalRez[2] << "]" << endl;
			PRINT_FAULT_FOOTER;
			return ERROR_INVALID_ARGS;
		}
	}

	/* this appears to differ in meaning, ignore temporarily
	double chkOffset[3];
	IHR.getDblAttr("/", "par_ myOffset", &chkOffset[0], 3);
	for(i = 0; i < 3; i++) {
		if((int)chkOffset[i] != geo->gridAffine[i]) {
			PRINT_FAULT_HEADER;
			std::cout << "Upon check, it has been found that input file and mgarray grid offsets for this rank differ:\n";
			std::cout << "   file=[" << (int)chkOffset[0] << " " << (int)chkOffset[1] << " " << (int)chkOffset[2] << " vs mgarray [";
			std::cout << geo->gridAffine[0] << " " << geo->gridAffine[1] << " " << geo->gridAffine[2] << "]" << endl;
			PRINT_FAULT_FOOTER;
			return ERROR_INVALID_ARGS;
		}
	} */

	hsize_t fasize[3] = {1, 1, 1};
	IHR.getArraySize("/fluid1/mass", &fasize[0]);

	for(i = 0; i < 3; i++) {
		if(fasize[i] != geo->localRez[i]) {
			PRINT_FAULT_HEADER;
			std::cout << "Upon check, it has been found that input file fluid array size and the mgarray size differ:\n";
			std::cout << "   file=[" << fasize[0] << " " << fasize[1] << " " << fasize[2] << " vs mgarray [";
			std::cout << geo->localRez[0] << " " << geo->localRez[1] << " " << geo->localRez[2] << "]" << endl;
			PRINT_FAULT_FOOTER;
			return ERROR_INVALID_ARGS;
		}
	}

	// Read rho
	IHR.readDoubleArray("/fluid1/mass", &q);
	int status = uploadHostArray(&m, q, &geo->localRez[0], halosize, partDir, exteriorHalo, 0, nDevices, deviceList);

	// acquire a slab of 5 such arrays stored at the holder
	status = MGA_allocSlab(&m, &holder[0], 5);

	// memcopy rho to holder aka slab 0
	MGA_duplicateArray(&holder, &m);
	// Adjust numSlabs and device pointers on all 5 slab references
	int vecComps[5] = {0,0,1,2,3};

	for(i = 0; i < 5; i++) {
		g->data[i] = holder[0];
		g->data[i].numSlabs = -i;

		for(j = 0; j < holder->nGPUs; j++) { g->data[i].devicePtr[j] += i*holder->slabPitch[j]/sizeof(double); }
		for(j = 0; j < 6; j++) { g->data[i].boundaryConditions.mode[j] = circular; }

		g->data[i].vectorComponent = vecComps[i];
	}

	MGA_delete(&m);

	// read ener
	IHR.readDoubleArray("/fluid1/ener", &q);
	status = MGA_uploadArrayToGPU(q, &g->data[1], -1);

	// read mom X
	IHR.readDoubleArray("/fluid1/momX", &q);
	status = MGA_uploadArrayToGPU(q, &g->data[2], -1);

	// read mom Y
	IHR.readDoubleArray("/fluid1/momY", &q);
	status = MGA_uploadArrayToGPU(q, &g->data[3], -1);

	// read mom Z
	IHR.readDoubleArray("/fluid1/momZ", &q);
	status = MGA_uploadArrayToGPU(q, &g->data[4], -1);

	g->rhoMin = 1e-7;
	g->thermo = generateDefaultThermo();

	free(q);

	return status;
}

/* Generate default thermo returns thermodynamic parameters for cold molecular hydrogen in SI units */
ThermoDetails generateDefaultThermo(void)
{
	ThermoDetails t;

	double amu = 1.66053904e-27;
	double massH2 = 2.016*amu;

	t.Cisothermal = -1;
	t.gamma = 5.0/3.0;
	t.kBolt = 1.381e-23;
	t.m = massH2;
	t.mu0 = 8.9135e-6;
	t.muTindex = 0.7;
	t.sigma0 = 1.927160224908165e-19;
	t.sigmaTindex = 0.2;

	return t;
}

int performCompleteTimestep(GridFluid *fluids, int numFluids, FluidStepParams fsp, ParallelTopology topo, GravityData *gravdata, ParametricRadiation *rad, int srcType, ImogenTimeManager *itm)
{
	int status = SUCCESSFUL;
	int fluidct;

	MGArray tempStorage;
	tempStorage.nGPUs = -1; // not allocated
	int numarrays;
#ifdef DEBUGMODE
			numarrays = 6 + DBG_NUMARRAYS;
#else
#ifdef USE_RK3
			numarrays = 11;
#else
			numarrays = 6;
#endif
#endif

	if(tempStorage.nGPUs == -1) {
		nvtxMark("flux_multi.cu:131 large malloc 6 arrays");
		status = MGA_allocSlab(&fluids[0].data[0], &tempStorage, numarrays);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }
	}

	double propdt;
	status = calculateMaxTimestep(fluids, numFluids, &fsp, &topo, &tempStorage, &propdt);

	// This may reduce dt if a frame save or end-of-simulation time is less than dt away.
	fsp.dt = itm->registerTimestep(propdt);
	fsp.stepNumber = itm->iter();

// TAKE SOURCE HALF STEP
	fsp.dt *= 0.5;
	status = performSourceFunctions(srcType, fluids, numFluids, fsp, &topo, gravdata, rad, &tempStorage);
	fsp.dt *= 2;

	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	fsp.stepDirection = 1;
// INITIATE FORWARD-ORDERED TRANSPORT STEP
	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		fsp.minimumRho = fluids[fluidct].rhoMin;
		fsp.thermoGamma = fluids[fluidct].thermo.gamma;
		fsp.Cisothermal = fluids[fluidct].thermo.Cisothermal;
		if(fluids[fluidct].thermo.Cisothermal != -1) {
			fsp.thermoGamma = 2;
			// This makes the hydro pressure solver return internal energy when it multiplies eint by (gamma-1)
		}

		status = performFluidUpdate_3D(&fluids[fluidct].data[0], &topo, fsp, &tempStorage);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
	}
	if(status != SUCCESSFUL) { return status; }

// TAKE FULL SOURCE STEP
	status = performSourceFunctions(srcType, fluids, numFluids, fsp, &topo, gravdata, rad, &tempStorage);
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

// INITIATE BACKWARD-ORDERED TRANSPORT STEP
	fsp.stepDirection = -1;

	for(fluidct = 0; fluidct < numFluids; fluidct++) {
		fsp.minimumRho = fluids[fluidct].rhoMin;
		fsp.thermoGamma = fluids[fluidct].thermo.gamma;
		fsp.Cisothermal = fluids[fluidct].thermo.Cisothermal;
		if(fluids[fluidct].thermo.Cisothermal != -1) {
			fsp.thermoGamma = 2;
		}

		status = performFluidUpdate_3D(&fluids[fluidct].data[0], &topo, fsp, &tempStorage);
		if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) break;
	}
	if(status != SUCCESSFUL) { return status; }

// TAKE FINAL SOURCE HALF STEP
	fsp.dt *= 0.5;
	status = performSourceFunctions(srcType, fluids, numFluids, fsp, &topo, gravdata, rad, &tempStorage);
	fsp.dt *= 2;
	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { return status; }

	// CLEANUP
	// This was allocated & re-used many times in performFluidUpdate_3D
	if((tempStorage.nGPUs != -1) && (status == SUCCESSFUL)) {
		#ifdef USE_NVTX
		nvtxMark("Large free flux_ML_iface.cu:144");
		#endif
		status = CHECK_IMOGEN_ERROR(MGA_delete(&tempStorage));
		if(status != SUCCESSFUL) { return status; }
	}

#ifdef SYNCMEX
	MGA_sledgehammerSequentialize(&fluid[0]);
#endif

#ifdef USE_NVTX
	nvtxRangePop();
#endif

	return SUCCESSFUL;
}

/* Computes the maximum permitted timestep allowed by the CFL constraint on the fluid method */
int calculateMaxTimestep(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParallelTopology *topo, MGArray *tempStorage, double *timestep)
{
	int status = SUCCESSFUL;
    double dt = 1e38;

    double currentdt = dt;
    double tau;

    int i, j;
    // compute each fluid's min timestep on this node
    for(i = 0; i < nFluids; i++) {
    	status = CHECK_IMOGEN_ERROR(calculateSoundspeed(&fluids[i].data[0], (MGArray *)NULL, tempStorage, fluids[i].thermo.gamma));

    	double globrez[3];
    	for(j = 0; j < 3; j++) { globrez[j] = fsp->geometry.globalRez[j]; }

    	status = computeLocalCFLTimestep(&fluids[i].data[0], tempStorage, &fsp->geometry, fsp->stepMethod, &globrez[0], &tau);
    	if(CHECK_IMOGEN_ERROR(status) != SUCCESSFUL) { break; }

    	// crash on invalid timesteps
    	if(isnan(tau)) {
    		PRINT_SIMPLE_FAULT("Calculated a timestep that is either infinity or NaN. Crashing\n!");
    		return ERROR_CRASH;
    	}

    	// This is computed globally by computeLocalCFLTimestep
    	currentdt = (currentdt < tau) ? currentdt : tau;
    }

    *timestep = currentdt * fsp->cflPrefactor;
	return SUCCESSFUL;
}

FluidMethods mlmethodToEnum(int mlmethod)
{
	FluidMethods f;
	switch(mlmethod) {
	case 1: f = METHOD_HLL; break;
	case 2: f = METHOD_HLLC; break;
	case 3: f = METHOD_XINJIN; break;
	}

return f;
}


int uploadHostArray(MGArray *gpuArray, double *hostArray, int *dims, int haloSize, int partDir, int exteriorHalo, int forceClone, int nDevices, int *deviceList)
{
	CHECK_CUDA_ERROR("entering uploadHostArray");

	MGArray m;

	// Default to no halo, X partition, add exterior halo
	m.haloSize = 0;
	m.partitionDir = PARTITION_X;
	m.addExteriorHalo = 1;
	m.vectorComponent = 0; // default, poke using GPU_Type.updateVectorComponent(n)

	m.haloSize = haloSize;
	m.partitionDir = partDir;
	m.addExteriorHalo = exteriorHalo;

	// Default to circular boundary conditions
	m.mpiCircularBoundaryBits = 63;

	// With any new upload, assume this is the XYZ orientation
	m.permtag = 1;
	MGA_permtagToNums(m.permtag, &m.currentPermutation[0]);

	m.nGPUs = nDevices;
	int i;
	for(i = 0; i < nDevices; i++) {
		m.deviceID[i] = deviceList[i];
		m.devicePtr[i] = 0x0;
	}

	for(i = 0; i < 3; i++) { m.dim[i] = dims[i]; }

	// If we are already cloning, multiply the size in the partition direct by #GPUs
	if(forceClone) {
		m.dim[m.partitionDir-1] = m.dim[m.partitionDir-1] * m.nGPUs;
	}

	// If the size in the partition direction is 1, clone it instead
	if((m.dim[m.partitionDir-1] == 1) && (m.nGPUs > 1)) {
		m.haloSize = 0;
		m.dim[m.partitionDir-1] = m.nGPUs;
		forceClone = 1;
	}

	// Compute the host # elements & the # elements per partition
	m.numel = m.dim[0]*m.dim[1]*m.dim[2];
	int sub[6];
	for(i = 0; i < m.nGPUs; i++) {
		calcPartitionExtent(&m, i, &sub[0]);
		m.partNumel[i] = sub[3]*sub[4]*sub[5];
	}

	// Set # slabs to 1
	m.numSlabs = 1;

	MGA_allocArrays(&gpuArray, 1, &m);

	int worked;
	if(forceClone) {
		worked = MGA_uploadArrayToGPU(hostArray, gpuArray, 0);
	} else {
		worked = MGA_uploadArrayToGPU(hostArray, gpuArray, -1);
	}
	if(CHECK_IMOGEN_ERROR(worked) != SUCCESSFUL) {
		return worked;
	}
	if(forceClone) {
	    worked = CHECK_IMOGEN_ERROR(MGA_distributeArrayClones(gpuArray, 0));
	}

	return worked;
}
