/*
 * cudaGradientKernels.h
 *
 *  Created on: Sep 22, 2017
 *      Author: erik-k
 */

#ifndef CUDAGRADIENTKERNELS_H_
#define CUDAGRADIENTKERNELS_H_

int computeCentralGradient(MGArray *phi, MGArray *gradient, GeometryParams geom, int spaceOrder, double scalingParameter);

#endif /* CUDAGRADIENTKERNELS_H_ */
