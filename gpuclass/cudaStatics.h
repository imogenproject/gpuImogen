/*
 * cudaStatics.h
 *
 *  Created on: Dec 9, 2015
 *      Author: erik
 */

#ifndef CUDASTATICS_H_
#define CUDASTATICS_H_

int setFluidBoundary(MGArray *fluid, const mxArray *matlabhandle, int direction);
int setBoundaryConditions(MGArray *array, const mxArray *matlabhandle, int direction);

#endif /* CUDASTATICS_H_ */
