/*
 * cudaHaloExchange.h
 *
 *  Created on: Nov 25, 2015
 *      Author: erik
 */

#ifndef CUDAHALOEXCHANGE_H_
#define CUDAHALOEXCHANGE_H_

int exchange_MPI_Halos(MGArray *phi, int nArrays, pParallelTopology topo, int xchgDir);

#endif /* CUDAHALOEXCHANGE_H_ */
