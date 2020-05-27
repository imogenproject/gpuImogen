/*
 * flux.h
 *
 *  Created on: Nov 30, 2015
 *      Author: erik
 */

#include "cudaFluidStep.h"

#ifndef FLUX_H_
#define FLUX_H_
// old argument order (fwd or bkwd) is encoded in fsp.sweepDirection
int performFluidUpdate_3D(MGArray *fluid, ParallelTopology* parallelTopo, FluidStepParams fsp, MGArray *tempStorage);
//int performFluidUpdate_3D(MGArray *fluid, ParallelTopology* parallelTopo, FluidStepParams fsp, int stepNumber, int order);
//int performFluidUpdate_3D(MGArray *fluid, ParallelTopology* parallelTopo, int order, int stepNumber, double *lambda, double gamma, double minRho, double stepMethod);

#endif /* FLUX_H_ */
