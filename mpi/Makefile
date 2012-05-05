# This script compiles the handful of binary modules needed for Imogen to support GPU gravitation.
# These must be rebuilt EVERY TIME GPUMAT IS UPDATED or Matlab will segfault when they are used.

# Change these to suit your system. CUDA_LDIR will be either lib or lib64
#MATLAB_DIR=/opt/Matlab/matlab2010b
MATLAB_DIR=/usr/local/packages/MATLAB/R2011b
CUDA_DIR = /usr/local/packages/cuda/4.0
CUDA_LDIR = lib64
CUDA_ARCH = sm_20

# Don't mess with these
NVCC_OPTS = -arch=$(CUDA_ARCH) -O1
NVCC_INCL = -I$(MATLAB_DIR)/extern/include $(GPUMAT_INCL)

MEX_OPTS  = -DUNIX
MEX_INCL  = -I$(MATLAB_DIR)/extern/include -I/usr/include/openmpi-x86_64
MEX_LDIR  = "-L/usr/lib64/openmpi/lib"
MEX_LIBS  =  "-lmpi"

###############################################################################
all:
	echo "Finished!"

%::
	mex $(MEX_OPTS) $(MEX_INCL) $@.c -o $@ $(MEX_LDIR) $(MEX_LIBS)

#%::
#	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda $@.cu -o $@.cpp
#	mex $(MEX_OPTS) $(MEX_INCL) $@.cpp cudaCommon.cpp -o $@ $(MEX_LDIR) $(MEX_LIBS)
#	rm -f $@.cpp

clean:
	rm -f *mexa64 *cpp

