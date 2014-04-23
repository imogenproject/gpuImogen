# Compiles (real) MPI support needed in Matlab that PGW doesn't yet include.

include ../Make.config

# Don't mess with these
NVCC_OPTS = -arch=$(CUDA_ARCH) -O1
NVCC_INCL = -I$(MATLAB_DIR)/extern/include

MEX_OPTS = -DUNIX
MEX_INCL = -I$(MATLAB_DIR)/extern/include
MEX_LIBS = 
MEX_LDIR = -L$(MATLAB_DIR)/runtime/glnxa64

PAR_INCL = -I$(PGW_DIR)/include -I$(MPI_DIR)/include
PAR_LDIR = -L$(PGW_DIR)/lib -L$(MPI_DIR)/lib
PAR_LIBS = -lpgw -lmpi

###############################################################################
all: mpi_amirank0 mpi_allgather mpi_barrier mpi_basicinfo mpi_dimreduce mpi_finalize mpi_init mpi_isinitialized mpi_myrank mpi_scatter mpi_min mpi_max
	echo "Finished!"

mpi_common.o: mpi_common.c
	mpicc $(MEX_OPTS) $(MEX_INCL) $(MEX_LDIR) $(PAR_INCL) -fPIC -c mpi_common.c -o mpi_common.o

mpi_min: mpi_allreduce.c mpi_common.o
	mex $(MEX_OPTS) $(MEX_INCL) $(MEX_LDIR) $(MEX_LIBS) $(PAR_INCL) $(PAR_LDIR) $(PAR_LIBS) -DMPIOPERATION=MPI_MIN mpi_allreduce.c mpi_common.o -o mpi_min

mpi_max: mpi_allreduce.c mpi_common.o
	mex $(MEX_OPTS) $(MEX_INCL) $(MEX_LDIR) $(MEX_LIBS) $(PAR_INCL) $(PAR_LDIR) $(PAR_LIBS) -DMPIOPERATION=MPI_MAX mpi_allreduce.c mpi_common.o -o mpi_max

mpi_amirank0 mpi_allgather mpi_barrier mpi_basicinfo mpi_dimreduce mpi_finalize mpi_init mpi_isinitialized mpi_myrank mpi_scatter: mpi_common.o
	mex $(MEX_OPTS) $(MEX_INCL) $(MEX_LDIR) $(MEX_LIBS) $(PAR_INCL) $(PAR_LDIR) $(PAR_LIBS) $@.c mpi_common.o -o $@

clean:
	rm -f *mexa64 *cpp mpi_common.o

