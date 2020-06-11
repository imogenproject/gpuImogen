#ifndef MPI_COMMON_H
#define MPI_COMMON_H

#include "mpi.h"


// covers various error #defines 
#include "../gpuclass/imogenChecks.h"

bool mpidatatypeIsInteger(MPI_Datatype t);

#define TOPOLOGY_MAXDIMS 3

typedef struct __ParallelTopology {
	int ndim;
	int comm;

	int nproc[TOPOLOGY_MAXDIMS];          /* number of processors in grid */
	int coord[TOPOLOGY_MAXDIMS];          /* our index in the grid        */
	int neighbor_left[TOPOLOGY_MAXDIMS];  /* index of left neighbor [wraps circularly] */
	int neighbor_right[TOPOLOGY_MAXDIMS]; /* index of right neighbor [wraps circularly] */
	MPI_Fint dimcomm[TOPOLOGY_MAXDIMS];       /* Communicators for directional operations */

#ifdef NOMATLAB
	// FIXME yucky hack for the standalone version
	cudaStream_t cudaStreamPtrs[8];
	//int nCudaStreams;
#endif
	} ParallelTopology;

int topoNodeToRank(const ParallelTopology *t, int *node);
int topoCreateDimensionalCommunicators(ParallelTopology *t);
int topoDestroyDimensionalCommunicators(ParallelTopology *t);
int topoCreateVectorComm(ParallelTopology *t, int direction);

int mpi_exchangeHalos(ParallelTopology *topo, int dim, void *sendLeft,
		void *sendRight, void *recvLeft, void *recvRight, size_t numel, MPI_Datatype dt);

#ifndef NOMATLAB
#include "mpi_commonML.h"
#endif

#else

#endif
