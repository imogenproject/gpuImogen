# This script compiles the handful of binary modules needed for Imogen to support GPU gravitation.
# These must be rebuilt EVERY TIME GPUMAT IS UPDATED or Matlab will segfault when they are used.

# Change these to suit your system. CUDA_LDIR will be either lib or lib64
MATLAB_DIR=/opt/Matlab/matlab2010b
CUDA_DIR = /usr/local/cuda
CUDA_LDIR = lib64
CUDA_ARCH = sm_13

# Don't mess with these
NVCC_OPTS = -arch=$(CUDA_ARCH) -O1
NVCC_INCL = -I$(MATLAB_DIR)/extern/include $(GPUMAT_INCL)

MEX_OPTS  = -DUNIX
MEX_INCL  = -I$(CUDA_DIR)/include $(GPUMAT_INCL) -I$(MATLAB_DIR)/extern/include
MEX_LDIR  = -L$(CUDA_DIR)/$(CUDA_LDIR)
MEX_LIBS  = -lcuda -lcudart -lcublas

###############################################################################
all: base fluids
	echo "Finished compiling GPU support."

base: kerncommon GPU_cudamemcpy GPU_free GPU_init GPU_memavail cudaBasicOperations
	echo "Done building base operations!"

fluids: kerncommon cudaArrayRotate cudaArrayAtomic cudaBasicOperations cudaShift cudaStatics directionalMaxFinder freezeAndPtot cudaWstep cudaTVDStep cudaMHDKernels cudaFwdAverage cudaFwdDifference cudaMagPrep cudaMagW cudaMagTVD
	echo "Done building fluid kernels."

kerncommon: cudaCommon.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaCommon.cu -o cudaCommon.cpp

%::
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda $@.cu -o $@.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $@.cpp cudaCommon.cpp -o $@ $(MEX_LDIR) $(MEX_LIBS)
	rm -f $@.cpp
clean:
	rm -f *mexa64 *cpp

