#include "stdio.h"
#include "stdlib.h"

#include "unistd.h"

#include "mpi.h"
#include "math.h"

#ifndef NOMATLAB
#include "mex.h"
#endif

//#include "cuda.h"
//#include "cuda_runtime.h"

#include "mpi_common.h"

#ifndef NOMATLAB
#include "mpiCommonMatlab.cpp"
#endif

void printAboutMPIStatus(MPI_Status *s);

//#include "mpiCommonMatlab.cpp"

/* Returns true if the type is suitable for logic AND
 * and false if not (i.e. float/double)
 */
bool mpidatatypeIsInteger(MPI_Datatype t)
{
if(t == MPI_CHAR) return true;
if(t == MPI_BYTE) return true;
if(t == MPI_SHORT) return true;
if(t == MPI_INT) return true;
if(t == MPI_LONG) return true;
if(t == MPI_UNSIGNED_CHAR) return true;
if(t == MPI_UNSIGNED_SHORT) return true;
if(t == MPI_UNSIGNED) return true;
if(t == MPI_UNSIGNED_LONG) return true;

return false;

}

/* Convert a tuple index into a linear index, the node's rank. */
int topoNodeToRank(const ParallelTopology *t, int *node)
{
	int r = 0;
	int stride = 1;

	int i;
	for(i = 0; i < t->ndim; i++) {
		r += stride*node[i];
		stride *= t->nproc[i];
	}

return r;
}

int topoCreateDimensionalCommunicators(ParallelTopology *t)
{
	// Checks if the topology's world communicator is flagged as "don't know"
	// and sets it to the correct value
	if(t->comm < 0) {
		t->comm = MPI_Comm_c2f(MPI_COMM_WORLD);
	}

	int stat;

	stat = topoCreateVectorComm(t, 1);
	if(stat) return stat;
	stat = topoCreateVectorComm(t, 2);
	if(stat) return stat;
	stat = topoCreateVectorComm(t, 3);
	if(stat) return stat;

	return 0;
}

/* Frees the 3 dimension-specific communicators associated with each
 * rectilinear topology
 */
int topoDestroyDimensionalCommunicators(ParallelTopology *t)
{

	if(t->dimcomm[0] > 0) {
		MPI_Comm mc = MPI_Comm_f2c(t->dimcomm[0]);
		if(mc != MPI_COMM_SELF) { MPI_Comm_free(&mc); }
		t->dimcomm[0] = -1;
		}

	if(t->dimcomm[1] > 0) {
		MPI_Comm mc = MPI_Comm_f2c(t->dimcomm[1]);
		if(mc != MPI_COMM_SELF) { MPI_Comm_free(&mc); }
		t->dimcomm[1] = -1;
		}

	if(t->dimcomm[2] > 0) {
		MPI_Comm mc = MPI_Comm_f2c(t->dimcomm[2]);
		if(mc != MPI_COMM_SELF) { MPI_Comm_free(&mc); }
		t->dimcomm[2] = -1;
		}

	return 0;
}

int topoCreateVectorComm(ParallelTopology *t, int direction)
{
	/* Abort on bad direction */
	if(direction < 1) return -1;
	if(direction > TOPOLOGY_MAXDIMS) return -1;

	/* Short circuit to SELF communicator if only 1 rank in direction */
	if(t->nproc[direction-1] == 1) {
		t->dimcomm[direction-1] = MPI_Comm_c2f(MPI_COMM_SELF);
		return 0;
	}

	int *dimprocs = (int *)malloc(t->nproc[direction-1]*sizeof(int));
	//int dimprocs[t->nproc[direction-1]];

	MPI_Comm commune = MPI_Comm_f2c(t->comm);
	MPI_Group worldgroup, dimgroup;
	MPI_Comm dimcomm;
	int r0;
	MPI_Comm_rank(commune, &r0);

	int i;
	int targ[3];
	/* Identify the ranks which are to be in this communicator */
	for(i = 0; i < 3; i++) { targ[i] = t->coord[i]; }
	targ[direction-1] = 0;

	for(i = 0; i < t->nproc[direction-1]; i++) {
		targ[direction-1] = i;
		dimprocs[i] = topoNodeToRank(t, &targ[0]);
	}

	MPI_Comm_group(commune, &worldgroup);
	MPI_Group_incl(worldgroup, t->nproc[direction-1], &dimprocs[0], &dimgroup);
	/* Create communicator for this dimension */
	MPI_Comm_create(commune, dimgroup, &dimcomm);

	MPI_Group_free(&dimgroup);

	t->dimcomm[direction-1] = MPI_Comm_c2f(dimcomm);

	free(dimprocs);
	return 0;
}
/**
 * This is a collective operation and must be called by all processors in the
 * cartesion topology specified by topo.
 */
int mpi_exchangeHalos(ParallelTopology *topo, int dim, void *sendLeft,
		void *sendRight, void *recvLeft, void *recvRight, size_t numel, MPI_Datatype dt)
{
	// Tag our 4 different data blocks uniquely
	int  tagSL = 13 + topo->neighbor_left[dim]; /* The index in this dimension that we send to */
	int  tagSR = 13 + topo->neighbor_right[dim] + topo->nproc[dim];
	int  tagRL = 13 + topo->coord[dim] + topo->nproc[dim]; /* The index in this dimension that receives (L/R swabbed)*/
	int  tagRR = 13 + topo->coord[dim];

	/** MPI variables for asynchronous communication
	 */
	MPI_Request requests[4];
	MPI_Status  howDidItGo[4];

//	MPI_Comm topo_comm = MPI_Comm_f2c(topo->dimcomm[dim]);
	MPI_Comm topo_comm = MPI_Comm_f2c(topo->comm);
// FIXME this dumpster fire should use the dimensional communicator? does it even matter?

	int rank;
	MPI_Comm_rank(topo_comm, &rank);

int debugcrap = 0;
#ifdef DEBUG_OUTPUT
debugcrap = 1;
#endif

	if(debugcrap) printf("[%d]: dim_contig: dim=%d coord=%d neighbor=(%d %d)\n", rank, dim, topo->coord[dim], topo->neighbor_left[dim], topo->neighbor_right[dim]);

	/** Post recvs of data _into_ this processor's halo regions.
	 *
	 * It is a good idea to do this first so that there
	 * is a known storage location for incoming messages.
	 */

int status = MPI_Sendrecv(sendRight, numel, dt, topo->neighbor_right[dim], 13, recvLeft, numel, dt, topo->neighbor_left[dim], 13, topo_comm, &howDidItGo[0]);
int statusB = MPI_Sendrecv(sendLeft, numel, dt, topo->neighbor_left[dim], 14, recvRight, numel, dt, topo->neighbor_right[dim], 14, topo_comm, &howDidItGo[1]);

MPI_Barrier(MPI_COMM_WORLD);

if((status != MPI_SUCCESS) || (statusB != MPI_SUCCESS)) {
	printf("RANK %i: Oh dear, return was unsuccessful!", rank);
	switch(status) { 
		case MPI_ERR_REQUEST: printf("MPI_ERR_REQUEST\n"); return ERROR_INVALID_ARGS; break;
		case MPI_ERR_ARG: printf("MPI_ERR_ARG\n"); return ERROR_INVALID_ARGS; break;
		case MPI_ERR_IN_STATUS:
			printf("MPI_ERR_IN_STATUS\n");
       			printf("RANK %i: status 0 (rx left): ", rank); printAboutMPIStatus(&howDidItGo[0]);
       			printf("RANK %i: status 1 (rx rite): ", rank); printAboutMPIStatus(&howDidItGo[0]);
			return ERROR_CRASH;
			break;
		}
	} else {
		return SUCCESSFUL;
	}



}

void printAboutMPIStatus(MPI_Status *s)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

char shitshitshit[MPI_MAX_ERROR_STRING]; int slen;
MPI_Error_string(s->MPI_ERROR, &shitshitshit[0], &slen);
shitshitshit[slen+1] = 0x00;

printf("RANK %i: mpi status source = %i, tag = %i, error = %i. The MPI says: %s\n", rank, s->MPI_SOURCE, s->MPI_TAG, s->MPI_ERROR, shitshitshit);

}

