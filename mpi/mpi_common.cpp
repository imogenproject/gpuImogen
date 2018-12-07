#include "stdio.h"
#include "stdlib.h"

#include "mpi.h"
#include "math.h"

#include "mex.h"

#include "mpi_common.h"

void printAboutMPIStatus(MPI_Status *s);

MPI_Datatype typeid_ml2mpi(mxClassID id)
{

switch(id) {
  case mxUNKNOWN_CLASS: return MPI_BYTE; break; /* We're going down anyway */
  case mxCELL_CLASS: return MPI_BYTE; break; /* we're boned */
  case mxSTRUCT_CLASS: return MPI_BYTE; break; /* we're boned */
  case mxLOGICAL_CLASS: return MPI_BYTE; break;
  case mxCHAR_CLASS: return MPI_CHAR; break;
  case mxVOID_CLASS: return MPI_BYTE; break;
  case mxDOUBLE_CLASS: return MPI_DOUBLE; break;
  case mxSINGLE_CLASS: return MPI_FLOAT; break;
  case mxINT8_CLASS: return MPI_BYTE; break;
  case mxUINT8_CLASS: return MPI_BYTE; break;
  case mxINT16_CLASS: return MPI_SHORT; break;
  case mxUINT16_CLASS: return MPI_SHORT; break;
  case mxINT32_CLASS: return MPI_INT; break;
  case mxUINT32_CLASS: return MPI_INT; break;
  case mxINT64_CLASS: return MPI_LONG; break;
  case mxUINT64_CLASS: return MPI_LONG; break;
  case mxFUNCTION_CLASS: return MPI_BYTE; break; /* we're boned */
  }

return MPI_BYTE;
}

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

mxArray *cIntegerVectorToMatlab(int *v, int numel)
{
	mwSize m = 1;
	mwSize n = numel;
	mxArray *a = mxCreateDoubleMatrix(m, n, mxREAL);

	double *z = mxGetPr(a);
	int j;
	for(j = 0; j < numel; j++) { z[j] = (double)v[j]; }

	return a;
}

mxArray *cDoubleVectorToMatlab(double *v, int numel)
{
	mwSize m = 1;
	mwSize n = numel;
	mxArray *a = mxCreateDoubleMatrix(m, n, mxREAL);

	double *z = mxGetPr(a);
	int j;
	for(j = 0; j < numel; j++) { z[j] = v[j]; }

	return a;
}

/* Converts the input ParallelTopology structure to a Matlab
 * mxArray structure */
int topoCToStructure(ParallelTopology *topo, mxArray **plhs)
{
	mxArray *mlTopology;
	const char  *  topoFields[7]
	                  = {"ndim", "comm", "coord", "neighbor_left", "neighbor_right", "nproc", "dimcomm"};

	mlTopology = plhs[0] = mxCreateStructMatrix(1, 1, 7, topoFields);

	mxSetFieldByNumber(mlTopology, 0, 0, mxCreateDoubleScalar((double)topo->ndim));
	mxSetFieldByNumber(mlTopology, 0, 1, mxCreateDoubleScalar((double)topo->comm));
	mxSetFieldByNumber(mlTopology, 0, 2, cIntegerVectorToMatlab(topo->coord, 3));
	mxSetFieldByNumber(mlTopology, 0, 3, cIntegerVectorToMatlab(topo->neighbor_left, 3));
	mxSetFieldByNumber(mlTopology, 0, 4, cIntegerVectorToMatlab(topo->neighbor_right, 3));
	mxSetFieldByNumber(mlTopology, 0, 5, cIntegerVectorToMatlab(topo->nproc, 3));
	mxSetFieldByNumber(mlTopology, 0, 6, cIntegerVectorToMatlab(topo->dimcomm, 3));

    return 0; // FIXME: check for errors in all this mex API stuff...
}


int topoStructureToC(const mxArray *prhs, ParallelTopology* pt)
{
	mxArray *a;

	a = mxGetField(prhs,0,(const char *)"ndim"); /* n dimensions */
	pt->ndim = (int)*mxGetPr(a);
	a = mxGetField(prhs,0,(const char *)"comm"); /* MPI_Comm_c2f of world comm */
	pt->comm = (int)*mxGetPr(a); // Prevent invalid comm below
	if(pt->comm < 0) pt->comm = MPI_Comm_c2f(MPI_COMM_WORLD);

	double *val;
	int i;

	val = mxGetPr(mxGetField(prhs,0,(const char *)"coord")); /* My coordinate */
	for(i = 0; i < pt->ndim; i++) pt->coord[i] = (int)val[i];

	val = mxGetPr(mxGetField(prhs,0,(const char *)"neighbor_left")); /* left neighbor per dimension */
	for(i = 0; i < pt->ndim; i++) pt->neighbor_left[i] = (int)val[i];

	val = mxGetPr(mxGetField(prhs,0,(const char *)"neighbor_right")); /* right neighbor per dim */
	for(i = 0; i < pt->ndim; i++) pt->neighbor_right[i] = (int)val[i];

	val = mxGetPr(mxGetField(prhs,0,(const char *)"nproc")); /* Number of nodes/dimension */
	for(i = 0; i < pt->ndim; i++) pt->nproc[i] = (int)val[i];

	val = mxGetPr(mxGetField(prhs,0,(const char *)"dimcomm")); /* MPI_Comm_c2f of directional communicators */
	for(i = 0; i < pt->ndim; i++) pt->dimcomm[i] = (int)val[i];

	for(i = pt->ndim; i < TOPOLOGY_MAXDIMS; i++) {
		pt->coord[i] = 0;
		pt->nproc[i] = 1;
	}

	return 0; // FIXME: Check for errors in all this stuff...
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
		MPI_Comm_free(&mc);
		t->dimcomm[0] = -1;
		}

	if(t->dimcomm[1] > 0) {
		MPI_Comm mc = MPI_Comm_f2c(t->dimcomm[1]);
		MPI_Comm_free(&mc);
		t->dimcomm[1] = -1;
		}

	if(t->dimcomm[2] > 0) {
		MPI_Comm mc = MPI_Comm_f2c(t->dimcomm[2]);
		MPI_Comm_free(&mc);
		t->dimcomm[2] = -1;
		}

	return 0;
}

int topoCreateVectorComm(ParallelTopology *t, int direction)
{
	/* Abort on bad direction */
	if(direction < 1) return -1;
	if(direction > TOPOLOGY_MAXDIMS) return -1;

	int dimprocs[t->nproc[direction-1]];

	int i;
	int targ[3];
	/* Identify the ranks which are to be in this communicator */
	for(i = 0; i < 3; i++) { targ[i] = t->coord[i]; }
	targ[direction-1] = 0;

	for(i = 0; i < t->nproc[direction-1]; i++) {
		targ[direction-1] = i;
		dimprocs[i] = topoNodeToRank(t, &targ[0]);
	}

	MPI_Comm commune = MPI_Comm_f2c(t->comm);
	int r0;
	MPI_Comm_rank(commune, &r0);

	MPI_Group worldgroup, dimgroup;
	MPI_Comm dimcomm;

	MPI_Comm_group(commune, &worldgroup);
	MPI_Group_incl(worldgroup, t->nproc[direction-1], &dimprocs[0], &dimgroup);
	/* Create communicator for this dimension */
	MPI_Comm_create(commune, dimgroup, &dimcomm);

	MPI_Group_free(&dimgroup);

	t->dimcomm[direction-1] = MPI_Comm_c2f(dimcomm);

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

	MPI_Comm topo_comm = MPI_Comm_f2c(topo->dimcomm[dim]);

	int rank;
	MPI_Comm_rank(topo_comm, &rank);


#ifdef DEBUG_OUTPUT
	fprintf(stderr, "[%d]: dim_contig: dim=%d coord=%d neighbor=(%d %d) data=(%p %p %p %p)\n", rank, dim, topo->coord[dim], topo->neighbor_left[dim], topo->neighbor_right[dim], send_lbuf, send_rbuf, recv_lbuf, recv_rbuf);
#endif

	/** Post recvs of data _into_ this processor's halo regions.
	 *
	 * It is a good idea to do this first so that there
	 * is a known storage location for incoming messages.
	 */
	MPI_Irecv(recvLeft,  numel, dt, topo->neighbor_left [dim], 13, topo_comm, &requests[0]);
	MPI_Irecv(recvRight, numel, dt, topo->neighbor_right[dim], 14, topo_comm, &requests[1]);

	/** Start sending data _from_ this processor's interior regions.
	 *
	 * This can be done with blocking sends as we
	 * don't really have anything else to do anyway.
	 */
	MPI_Isend(sendLeft,  numel, dt, topo->neighbor_left [dim], 14, topo_comm, &requests[2]);
	MPI_Isend(sendRight, numel, dt, topo->neighbor_right[dim], 13, topo_comm, &requests[3]);

	// Wait until all transceiving is complete
	int ohdear = MPI_Waitall(4, &requests[0], &howDidItGo[0]);

	if(ohdear != MPI_SUCCESS) {
		printf("RANK %i: Oh dear, return was not success!");
		switch(ohdear) { 
			case MPI_ERR_REQUEST: printf("MPI_ERR_REQUEST\n"); break;
			case MPI_ERR_ARG: printf("MPI_ERR_ARG\n"); break;
			case MPI_ERR_IN_STATUS: printf("MPI_ERR_IN_STATUS\n"); break;
			}
		}
#ifdef DEBUG_OUTPUT
	printf("RANK %i: Emitting excessive debug info just because.\n");
        printf("RANK %i: status 0 (rx left):  ", rank); printAboutMPIStatus(&howDidItGo[0]);
        printf("RANK %i: status 1 (rx right): ", rank); printAboutMPIStatus(&howDidItGo[1]);
        printf("RANK %i: status 2 (tx left):  ", rank); printAboutMPIStatus(&howDidItGo[2]);
        printf("RANK %i: status 3 (tx right): ", rank); printAboutMPIStatus(&howDidItGo[3]);
#endif

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

