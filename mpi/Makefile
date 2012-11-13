# Compiles (real) MPI support needed in Matlab that PGW doesn't yet include.

include ../Make.config

# Don't mess with these
NVCC_OPTS = -arch=$(CUDA_ARCH) -O1
NVCC_INCL = -I$(MATLAB_DIR)/extern/include $(GPUMAT_INCL)

MEX_OPTS = -DUNIX
MEX_INCL = -I$(MATLAB_DIR)/extern/include
MEX_LIBS = 
MEX_LDIR = -L$(MATLAB_DIR)/runtime/glnxa64

PAR_INCL = -I$(PGW_DIR)/include -I$(MPI_DIR)/include
PAR_LDIR = -L$(PGW_DIR)/lib -L$(MPI_DIR)/lib
PAR_LIBS = -lpgw -lmpi

###############################################################################
all: mpi_init mpi_finalize mpi_barrier mpi_allgather mpi_scatter mpi_basicinfo mpi_dimreduce
	echo "Finished!"

%::
	mex $(MEX_OPTS) $(MEX_INCL) $(MEX_LDIR) $(MEX_LIBS) $(PAR_INCL) $(PAR_LDIR) $(PAR_LIBS) $@.c -o $@

clean:
	rm -f *mexa64 *cpp

