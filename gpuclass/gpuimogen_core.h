/*
 * gpuimogen_core.h
 *
 *  Created on: Apr 24, 2020
 *      Author: erik-k
 */

#ifndef GPUIMOGEN_CORE_H_
#define GPUIMOGEN_CORE_H_

ParallelTopology generateDummyTopology(void);
ParallelTopology acquireParallelTopology(int *globalDomainResolution);

GeometryParams generateDummyGeometry(void);
GeometryParams acquireSimulationGeometry(void);

int generateDummyFluid(GridFluid *g, MGArray *holder, int *localResolution);
ThermoDetails generateDefaultThermo(void);

int calculateMaxTimestep(GridFluid *fluids, int nFluids, FluidStepParams *fsp, ParallelTopology *topo, MGArray *tempStorage, double *timestep);
int performCompleteTimestep(GridFluid *fluids, int numFluids, FluidStepParams fsp, ParallelTopology topo, GravityData *gravdata, ParametricRadiation *rad, int srcType);

int uploadHostArray(MGArray *gpuArray, double *hostArray, int *dims, int haloSize, int partDir, int exteriorHalo, int forceClone, int nDevices, int *deviceList);

#endif /* GPUIMOGEN_CORE_H_ */
