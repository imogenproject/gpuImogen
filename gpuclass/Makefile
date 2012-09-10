# This makefile builds the GPU kernels and mex files needed to run gpuImogen

include Makefile.config

# Don't mess with any of this. Modify Makefile.config isntead.
NVCC_OPTS = -arch=$(CUDA_ARCH) -O1
NVCC_INCL = -I$(MATLAB_DIR)/extern/include

MEX_OPTS  = -DUNIX
MEX_INCL  = -I$(CUDA_DIR)/include -I$(MATLAB_DIR)/extern/include
MEX_LDIR  = -L$(CUDA_DIR)/$(CUDA_LDIR)
MEX_LIBS  = -lcudart -lcublas

MPI = /usr/local/packages/openmpi/1.4_gcc-4.4-tm
PGW = /usr/local/packages/parallel-gateway/pgw-0.1

PAR_INCL = -I$(PGW)/include -I$(MPI)/include
PAR_LDIR = -L$(PGW)/lib -L$(MPI)/lib
PAR_LIBS = -lpgw -lmpi

###############################################################################
all: base fluids
	echo "Finished compiling GPU support."

base: GPU_cudamemcpy GPU_free GPU_init GPU_memavail cudaBasicOperations
	echo "Done building base operations!"

fluids: cudaArrayRotate cudaArrayAtomic cudaBasicOperations cudaShift cudaStatics directionalMaxFinder freezeAndPtot cudaFluidW cudaFluidTVD cudaMHDKernels cudaFwdAverage cudaFwdDifference cudaMagPrep cudaMagW cudaMagTVD cudaSoundspeed cudaApplyScalarPotential
	echo "Done building fluid kernels."
	mv cudaCommon.cpp temp
	rm -f *cpp
	mv temp cudaCommon.cpp

parallel: cudaHaloExchange
	echo "Done building parallel support."

kerncommon:
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaCommon.cu -o cudaCommon.cpp

cuda% GPU% directionalMax% freezeAnd%:: kerncommon
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda $@.cu -o $@.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $@.cpp cudaCommon.cpp -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaHaloExchange:
	nvcc $(NVCC_OPTS) $(NVCC_INCL) $(PAR_INCL) -cuda $@.cu -o $@.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_INCL) $(PAR_LDIR) $(PAR_LIBS) $(MEX_LDIR) $(MEX_LIBS) $@.cpp cudaCommon.cpp -o $@

clean:
	rm -f *mexa64 *cpp

