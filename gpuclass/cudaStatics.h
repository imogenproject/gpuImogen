/*
 * cudaStatics.h
 *
 *  Created on: Dec 9, 2015
 *      Author: erik
 */

#ifndef CUDASTATICS_H_
#define CUDASTATICS_H_
int setFluidBoundary(MGArray *fluid, const mxArray *matlabhandle, GeometryParams *geo, int direction);
//int setFluidBoundary(MGArray *fluid, const mxArray *matlabhandle, int direction);
//int setBoundaryConditions(MGArray *array, const mxArray *matlabhandle, int direction);
int setArrayBoundaryConditions(MGArray *array, const mxArray *matlabhandle, GeometryParams *geo, int direction);
int setArrayStaticCells(MGArray *phi, const mxArray *matlabhandle);

int doBCForPart(MGArray *fluid, int part, int direct, int rightside);

#endif /* CUDASTATICS_H_ */
