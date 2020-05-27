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

#include "gpuimogen_core.h"

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

	ParallelTopology topo = acquireParallelTopology((int *)NULL); // FIXME acquires a fake single-node topology

	// Step 2 - initialize CUDA

	// Hardcode this for now
	int deviceList[MAX_GPUS_USED];
	int nCudaDevices = 1;
	deviceList[0] = 0;

	// Set this to make the API run its startup magic
	cudaSetDevice(0);

	cudaStream_t pstream;
	int i;
	// Default to default stream
	for(i = 0; i < 2*MAX_GPUS_USED; i++) { topo.cudaStreamPtrs[i] = 0; }

	// Create two streams on every device we plan to use
	for(i = 0; i < 1; i++) {
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
	fsp.cflPrefactor  = 0.85;

	// Acquire simulation geometry
	fsp.geometry = acquireSimulationGeometry(); // FIXME generates a fake dummy geometry

	// Choose # of fluids [ default to 1 for now ]
	int numFluids = 1;
	fsp.multifluidDragMethod = 0; // see cudaSource2FluidDrag.cu ~ 250

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
		// Set the halo buffer size, device count & list of devices
		// These will be duplicated into all generated arrays from the fluid creator
		dataholder[fluidct].haloSize = FLUID_METHOD_HALO_SIZE;
		dataholder[fluidct].nGPUs = nCudaDevices;
		for(i = 0; i < nCudaDevices; i++) dataholder[fluidct].deviceID[i] = deviceList[i];

		// This to be replaced with a call to a real parameter loader one day
		status = generateDummyFluid(&fluids[fluidct], &dataholder[fluidct], &fsp.geometry.globalRez[0]);
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

	double resultingTimestep;

	status = CHECK_IMOGEN_ERROR(performCompleteTimestep(fluids, numFluids, fsp, topo, &gravdat, &prad, srcType));


}

// Returns a dummy topology for a single rank
ParallelTopology generateDummyTopology(void)
{
	ParallelTopology p;

	p.ndim = 3;
	p.comm = MPI_Comm_c2f(MPI_COMM_WORLD);

	int i;
	for(i = 0; i < 3; i++) {
		p.nproc[i] = 1;
		p.coord[i] = 0;
		p.neighbor_left[i] = 0;
		p.neighbor_right[i] = 0;
		p.dimcomm[i] = MPI_Comm_c2f(MPI_COMM_WORLD);
	}

	return p;
}

ParallelTopology acquireParallelTopology(int *globalDomainResolution)
{
	return generateDummyTopology();
}

// Returns a dummy square grid of size 256x256x1 with unit spacing
GeometryParams generateDummyGeometry(void)
{
GeometryParams g;

g.Rinner = 0;
g.XYVector = NULL;
g.frameOmega = 0;
// FIXME frame rotation is only supported about the z axis...
g.frameRotateCenter[0] = 0.0;
g.frameRotateCenter[1] = 0.0;
g.frameRotateCenter[2] = 0.0;
g.globalRez[0] = g.globalRez[1] = 256;
g.globalRez[2] = 1;
g.h[0] = g.h[1] = g.h[2] = 1.0;
g.shape = SQUARE;
g.x0 = 0;
g.y0 = 0;
g.z0 = 0;

return g;
}

GeometryParams acquireSimulationGeometry(void)
{
return generateDummyGeometry();
}

int generateDummyFluid(GridFluid *g, MGArray *holder, int *localResolution)
{

	double *q = (double *)malloc((unsigned long)(sizeof(double)*localResolution[0]*localResolution[1]*localResolution[2]));

	MGArray m;
	int halosize = holder->haloSize;
	int partDir = 1; // FIXME HACK
	int exteriorHalo = 0; // FIXME HACK AND SOMETIMES WRONG
	int nDevices = holder->nGPUs;
	int *deviceList = &holder->deviceID[0];

	int i;

	long j, ne;
	ne = localResolution[0]*localResolution[1]*localResolution[2];

	for(j = 0; j < ne; j++) { q[j] = 1.0; }
	// upload rho to create first array, m

	int status = uploadHostArray(&m, q, localResolution, halosize, partDir, exteriorHalo, 0, nDevices, deviceList);

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
	status = uploadHostArray(&g->data[1], q, localResolution, halosize, partDir, exteriorHalo, 0, 1, deviceList);
	for(j = 0; j < 6; j++) { g->data[1].boundaryConditions.mode[j] = circular; }

	// (fake px here)
	for(j = 0; j < ne; j++) { q[j] = 0.1; }

	// select slab 2
	g->data[2] = g->data[0];
	g->data[2].numSlabs = -2;
	for(i = 0; i < g->data[0].nGPUs; i++) { g->data[2].devicePtr[i] += 2*g->data[0].slabPitch[i]; }
	// upload
	status = uploadHostArray(&g->data[2], q, localResolution, halosize, partDir, exteriorHalo, 0, 1, deviceList);
	for(j = 0; j < 6; j++) { g->data[2].boundaryConditions.mode[j] = circular; }

	// (fake py here)
	for(j = 0; j < ne; j++) { q[j] = 0.2; }

	// select slab 3
	g->data[3] = g->data[0];
	g->data[3].numSlabs = -3;
	for(i = 0; i < g->data[0].nGPUs; i++) { g->data[3].devicePtr[i] += 3*g->data[0].slabPitch[i]; }
	// upload
	status = uploadHostArray(&g->data[3], q, localResolution, halosize, partDir, exteriorHalo, 0, 1, deviceList);
	for(j = 0; j < 6; j++) { g->data[3].boundaryConditions.mode[j] = circular; }

	// (fake pz here)
	for(j = 0; j < ne; j++) { q[j] = 0.0; }

	// select slab 4
	g->data[4] = g->data[0];
	g->data[4].numSlabs = -4;
	for(i = 0; i < g->data[0].nGPUs; i++) { g->data[4].devicePtr[i] += 4*g->data[0].slabPitch[i]; }
	// upload
	status = uploadHostArray(&g->data[4], q, localResolution, halosize, partDir, exteriorHalo, 0, 1, deviceList);
	for(j = 0; j < 6; j++) { g->data[4].boundaryConditions.mode[j] = circular; }

	g->rhoMin = 1e-7;
	g->thermo = generateDefaultThermo();

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

int performCompleteTimestep(GridFluid *fluids, int numFluids, FluidStepParams fsp, ParallelTopology topo, GravityData *gravdata, ParametricRadiation *rad, int srcType)
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
	fsp.dt = propdt;

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
