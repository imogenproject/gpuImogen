# This script compiles the handful of binary modules needed for Imogen to support GPU gravitation.
# These must be rebuilt EVERY TIME GPUMAT IS UPDATED or Matlab will segfault when they are used.

# Change these to suit your system. CUDA_LDIR will be either lib or lib64
#MATLAB_DIR=/opt/Matlab/matlab2010b
MATLAB_DIR =/usr/local/packages/MATLAB/R2011b
CUDA_DIR   = /usr/local/packages/cuda/4.0
CUDA_LDIR  = lib64
CUDA_ARCH  = sm_20

# Don't mess with these
NVCC_OPTS = -arch=$(CUDA_ARCH) -O1
NVCC_INCL = -I$(MATLAB_DIR)/extern/include $(GPUMAT_INCL)

#MPIOPTS   := $(shell mpicc --show | sed -e 's/^.*bin\/gcc //' -e 's/-pthread//')
MPI_DIR   = -I/usr/include/openmpi-x86_64 -L/usr/lib64/openmpi/lib
MPI_LIBS  = -lmpi -ldl

MEX_OPTS  = -DUNIX
MEX_INCL  = -I$(MATLAB_DIR)/extern/include
MEX_LIBS  = 

###############################################################################
all: mpi_init mpi_finalize mpi_barrier mpi_allgather mpi_basicinfo
	echo "Finished!"

%::
	mex $(MEX_OPTS) $(MEX_INCL) $(MEX_LDIR) $(MEX_LIBS) $(MPI_DIR) $(MPI_LIBS) $@.c -o $@

#%::
#	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda $@.cu -o $@.cpp
#	mex $(MEX_OPTS) $(MEX_INCL) $@.cpp cudaCommon.cpp -o $@ $(MEX_LDIR) $(MEX_LIBS)
#	rm -f $@.cpp

clean:
	rm -f *mexa64 *cpp

