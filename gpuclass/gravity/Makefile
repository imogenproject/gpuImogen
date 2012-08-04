# This script compiles the handful of binary modules needed for Imogen to support GPU gravitation.

include ../Makefile.config

# Don't mess with these
NVCC_OPTS = -arch $(CUDA_ARCH)
NVCC_INCL = -I$(MATLAB_DIR)/extern/include -I../

MEX_OPTS  = -DUNIX
MEX_INCL  = -I$(CUDA_DIR)/include -I$(MATLAB_DIR)/extern/include -I../
MEX_LDIR  = -L$(CUDA_DIR)/$(CUDA_LDIR)
MEX_LIBS  = -lcuda -lcudart -lcublas

###############################################################################
all: gravity

gravity: accumulateBterm symmetricLinearOperator integralPoisson_mg interpolate
#wrap_cublasDdot

kernels: cudaKernels.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaKernels.cu -o cudaKernels.cpp

# Build preconditioning operator
accumulateBterm: accumulateBterm.cu kernels
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda accumulateBterm.cu -o accumulateBterm.cpp
	mex $(MEX_OPTS) $(MEX_INCL) accumulateBterm.cpp cudaKernels.cpp ../cudaCommon.cpp -o accumulateBterm $(MEX_LDIR) $(MEX_LIBS)
	rm -f accumulateBterm.cpp

# Build general symmetric linear operator function
symmetricLinearOperator: symmetricLinearOperator.cu kernels
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda symmetricLinearOperator.cu -o symmetricLinearOperator.cpp
	mex $(MEX_OPTS) $(MEX_INCL) symmetricLinearOperator.cpp cudaKernels.cpp ../cudaCommon.cpp -o symmetricLinearOperator $(MEX_LDIR) $(MEX_LIBS)
	rm -f symmetricLinearOperator.cpp

interpolate: interpolateGPUvar.cu kernels
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda interpolateGPUvar.cu -o interpolateGPUvar.cpp
	mex $(MEX_OPTS) $(MEX_INCL) interpolateGPUvar.cpp cudaKernels.cpp ../cudaCommon.cpp -o interpolateGPUvar $(MEX_LDIR) $(MEX_LIBS)
	rm -f interpolateGPUvar.cpp

# Build dot product operator
wrap_cublasDdot: wrap_cublasDdot.cpp
	mex $(MEX_OPTS) $(MEX_INCL) wrap_cublasDdot.cpp $(GPUMAT_CPP) $(MEX_LDIR) $(MEX_LIBS)

# Build boundary condition routine
mgbc_genkern: mgbc_genkern.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda mgbc_genkern.cu -o mgbc_genkern.cpp
	mex $(MEX_OPTS) $(MEX_INCL) mgbc_genkern.cpp $(GPUMAT_CPP) -o mgbc_genkern $(MEX_LDIR) $(MEX_LIBS)
	rm -f mgbc_genkern.cpp

integralPoisson_mg: integralPoisson_mg.cu kernels
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda integralPoisson_mg.cu -o integralPoisson_mg.cpp
	mex $(MEX_OPTS) $(MEX_INCL) integralPoisson_mg.cpp $(GPUMAT_CPP) cudaKernels.cpp ../cudaCommon.cpp -o integralPoisson_mg $(MEX_LDIR) $(MEX_LIBS)
	rm -f integralPoisson_mg.cpp

cudaCommon: cudaCommon.h cudaCommon.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaCommon.cu -o cudaCommon.cpp

clean:
	rm -f *mexa64 cudaKernels.cpp
