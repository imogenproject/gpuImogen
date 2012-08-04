# This makefile builds the GPU kernels and mex files needed to run gpuImogen

include Makefile.config

# Don't mess with any of this. Modify Makefile.config isntead.
NVCC_OPTS = -arch=$(CUDA_ARCH) -O1
NVCC_INCL = -I$(MATLAB_DIR)/extern/include

MEX_OPTS  = -DUNIX
MEX_INCL  = -I$(CUDA_DIR)/include -I$(MATLAB_DIR)/extern/include
MEX_LDIR  = -L$(CUDA_DIR)/$(CUDA_LDIR)
MEX_LIBS  = -lcudart -lcublas

###############################################################################
all: base fluids
	echo "Finished compiling GPU support."

base: GPU_cudamemcpy GPU_free GPU_init GPU_memavail cudaBasicOperations
	echo "Done building base operations!"

fluids: cudaArrayRotate cudaArrayAtomic cudaBasicOperations cudaShift cudaStatics directionalMaxFinder freezeAndPtot cudaWstep cudaTVDStep cudaMHDKernels cudaFwdAverage cudaFwdDifference cudaMagPrep cudaMagW cudaMagTVD cudaSoundspeed
	echo "Done building fluid kernels."
	mv cudaCommon.cpp temp
	rm -f *cpp
	mv temp cudaCommon.cpp

kerncommon:
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaCommon.cu -o cudaCommon.cpp

cuda% GPU% directionalMax% freezeAnd%:: kerncommon
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda $@.cu -o $@.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $@.cpp cudaCommon.cpp -o $@ $(MEX_LDIR) $(MEX_LIBS)

clean:
	rm -f *mexa64 *cpp

