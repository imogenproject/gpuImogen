/*
 * cudaSourceVTO.h
 *
 *  Created on: Sep 8, 2016
 *      Author: erik-k
 */

#ifndef CUDASOURCEVTO_H_
#define CUDASOURCEVTO_H_

int sourcefunction_VacuumTaffyOperator(MGArray *fluid, double dt, double alpha, double beta, double frameOmega, double criticalDensity, GeometryParams geo);

#endif /* CUDASOURCEVTO_H_ */
