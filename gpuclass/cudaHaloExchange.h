/*
 * cudaHaloExchange.h
 *
 *  Created on: Nov 25, 2015
 *      Author: erik
 */

#ifndef CUDAHALOEXCHANGE_H_
#define CUDAHALOEXCHANGE_H_

void cudahaloSetUseOfRegisteredMemory(int tf);

int exchange_MPI_Halos(MGArray *phi, int nArrays, ParallelTopology* topo, int xchgDir);

#endif /* CUDAHALOEXCHANGE_H_ */
