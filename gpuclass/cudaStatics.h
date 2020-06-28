/*
 * cudaStatics.h
 *
 *  Created on: Dec 9, 2015
 *      Author: erik
 */

#ifndef CUDASTATICS_H_
#define CUDASTATICS_H_
int setFluidBoundary(MGArray *fluid, GeometryParams *geo, int direction);
int setArrayBoundaryConditions(MGArray *phi, GeometryParams *geo, int direction, int side);

int setArrayStaticCells(MGArray *phi);

int doBCForPart(MGArray *fluid, int part, int direct, int rightside);

#endif /* CUDASTATICS_H_ */
