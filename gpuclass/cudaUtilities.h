/*
 * cudaUtilities.h
 *
 *  Created on: Apr 7, 2020
 *      Author: erik-k
 */

#ifndef CUDAUTILITIES_H_
#define CUDAUTILITIES_H_

__global__ void cukern_FetchPartitionSubset1D(double *in, int nodeN, double *out, int partX0, int partNX);
__global__ void writeScalarToVector(double *x, long numel, double f);

#endif /* CUDAUTILITIES_H_ */
