/*
 * gpuimogen_core.h
 *
 *  Created on: Apr 24, 2020
 *      Author: erik-k
 */

#ifndef GPUIMOGEN_CORE_H_
#define GPUIMOGEN_CORE_H_

// Sets up a cartesian topology for the simulation
ParallelTopology acquireParallelTopology(int *globalDomainResolution);
void identifyInternalBoundaries(MGArray *phi, ParallelTopology *topo);

// Does the basic setup of assigning grid index coordinates
GeometryParams generateGridGeometry(int *globalResolution, ParallelTopology *topo, int circ);

// Build the g->XYVector MGArray required by frame rotation & cylindrical coordinates
int initializeXYVector(GeometryParams *g, int nDevices, int *deviceList);

// Gets the grid coordinates then sets up the physical coordinates
GeometryParams acquireSimulationGeometry(int *globalResolution, ParallelTopology *topo, ImogenH5IO *conf);

int generateDummyFluid(GridFluid *g, MGArray *holder, GeometryParams *geo);
int readImogenICs(GridFluid *g, MGArray *holder, GeometryParams *geo, char *h5dfilebase, int frameNo = -1);
int readGravityPotential(ImogenH5IO *conf, GravityData *gravdat, MGArray *fluidReference);

ThermoDetails generateDefaultThermo(void);

int calculateMaxTimestep(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParallelTopology *topo, MGArray *tempStorage, double *timestep);
int performCompleteTimestep(GridFluid *fluids, int numFluids, FluidStepParams fsp, ParallelTopology topo, GravityData *gravdata, ParametricRadiation *rad, int srcType, ImogenTimeManager *itm);
//int performCompleteTimestep(GridFluid *fluids, int numFluids, FluidStepParams fsp, ParallelTopology topo, GravityData *gravdata, ParametricRadiation *rad, int srcType);

int uploadHostArray(MGArray *gpuArray, double *hostArray, int *dims, int haloSize, int partDir, int exteriorHalo, int forceClone, int nDevices, int *deviceList);

#endif /* GPUIMOGEN_CORE_H_ */
