/*
 * cudaSourceScalarPotential.h
 *
 *  Created on: Jan 11, 2016
 *      Author: erik
 */

#ifndef CUDASOURCESCALARPOTENTIAL_H_
#define CUDASOURCESCALARPOTENTIAL_H_

int sourcefunction_ScalarPotential(MGArray *fluid, double dt, double *d3x, double minRho, double rhoFullGravity);

#endif /* CUDASOURCESCALARPOTENTIAL_H_ */
