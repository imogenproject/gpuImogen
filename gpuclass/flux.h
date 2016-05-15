/*
 * flux.h
 *
 *  Created on: Nov 30, 2015
 *      Author: erik
 */

#ifndef FLUX_H_
#define FLUX_H_

int performFluidUpdate_3D(MGArray *fluid, ParallelTopology* parallelTopo, int order, int stepNumber, double *lambda, double gamma, double stepMethod);

#endif /* FLUX_H_ */
