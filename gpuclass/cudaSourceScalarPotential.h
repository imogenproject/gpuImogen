/*
 * cudaSourceScalarPotential.h
 *
 *  Created on: Jan 11, 2016
 *      Author: erik
 */

#ifndef CUDASOURCESCALARPOTENTIAL_H_
#define CUDASOURCESCALARPOTENTIAL_H_

#include "cudaCommon.h"

int sourcefunction_ScalarPotential(MGArray *fluid, MGArray *phi, double dt, GeometryParams geom, double minRho, double rhoFullGravity);
//int sourcefunction_ScalarPotential(MGArray *fluid, double dt, double *d3x, double minRho, double rhoFullGravity);

#endif /* CUDASOURCESCALARPOTENTIAL_H_ */
