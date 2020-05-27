
MPI_Datatype typeid_ml2mpi(mxClassID id);
mxClassID    typeid_mpi2ml(MPI_Datatype md);

mxArray *cIntegerVectorToMatlab(int *v, int numel);
mxArray *cDoubleVectorToMatlab(double *v, int numel);
int topoCToStructure(ParallelTopology *topo, mxArray **plhs);
int topoStructureToC(const mxArray *prhs, ParallelTopology* pt);

