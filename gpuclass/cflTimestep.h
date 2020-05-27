/*
 * cflTimestep.h
 *
 *  Created on: Apr 10, 2020
 *      Author: erik-k
 */

#ifndef CFLTIMESTEP_H_
#define CFLTIMESTEP_H_

int computeLocalCFLTimestep(MGArray *fluid, MGArray *csound, GeometryParams *geom, int method, double *globalResolution, double *tstep);

#endif /* CFLTIMESTEP_H_ */
