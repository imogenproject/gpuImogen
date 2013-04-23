# This makefile builds the GPU kernels needed to run gpuImogen

include ../Make.config

# Don't mess with any of this. Modify the above config file instead
GEN_OPTS = 

#NVCC_OPTS = -arch=$(CUDA_ARCH) -O1 $(GEN_OPTS)
NVCC_OPTS = -gencode arch=compute_20,code=sm_20 -O2
NVCC_INCL = -I$(MATLAB_DIR)/extern/include

MEX_OPTS  = -DUNIX $(GEN_OPTS)
MEX_INCL  = -I$(CUDA_DIR)/include -I$(MATLAB_DIR)/extern/include
MEX_LDIR  = -L$(CUDA_DIR)/$(CUDA_LDIR)
MEX_LIBS  = -lcudart -lcublas

PAR_INCL = -I$(PGW_DIR)/include -I$(MPI_DIR)/include
PAR_LDIR = -L$(PGW_DIR)/lib -L$(MPI_DIR)/lib
PAR_LIBS = -lpgw -lmpi

###############################################################################
all: base fluids parallel lambda
	echo "Finished compiling GPU support."
	mv cudaCommon.cpp temp
	rm -f *cpp
	mv temp cudaCommon.cpp

base: GPU_ctrl GPU_free GPU_cudamemcpy GPU_memavail cudaBasicOperations
	echo "Done building base operations!"

fluids: cudaArrayRotate cudaArrayAtomic cudaBasicOperations cudaShift cudaStatics directionalMaxFinder freezeAndPtot cudaFluidW cudaFluidTVD cudaMHDKernels cudaFwdAverage cudaFwdDifference cudaMagPrep cudaMagW cudaMagTVD cudaSoundspeed
	echo "Done building fluid kernels."

lambda: cudaSourceScalarPotential cudaSourceRotatingFrame cudaFreeRadiation cudaAccretingStar
	echo "Done building source/sink term kernels"
  
parallel: cudaHaloExchange
	echo "Done building parallel support."

kerncommon:
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaCommon.cu -o cudaCommon.cpp

cuda% GPU% directionalMax% freezeAnd%:: kerncommon
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -g -cuda $@.cu -o $@.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $@.cpp -g cudaCommon.cpp -o $@ $(MEX_LDIR) $(MEX_LIBS)
#	rm -f $@.cpp

cudaHaloExchange: kerncommon
	nvcc $(NVCC_OPTS) $(NVCC_INCL) $(PAR_INCL) -cuda $@.cu -o $@.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_INCL) $(PAR_LDIR) $(PAR_LIBS) $(MEX_LDIR) $(MEX_LIBS) $@.cpp cudaCommon.cpp -o $@
	rm -f $@.cpp

cudaAccretingStar: kerncommon
	nvcc $(NVCC_OPTS) $(NVCC_INCL) $(PAR_INCL) -cuda $@.cu -o $@.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_INCL) $(PAR_LDIR) $(PAR_LIBS) $(MEX_LDIR) $(MEX_LIBS) $@.cpp cudaCommon.cpp -o $@
	rm -f $@.cpp

clean:
	rm -f *mexa64 *cpp

