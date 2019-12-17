# This makefile builds the GPU kernels needed to run gpuImogen

include ../Make.config

DEFINES = -DUNIX -D_FORCE_INLINES -DRECONSTRUCT_VELOCITY -DUSE_NVTX -DUSE_STATIC_CHALLOC
#-DALLOCFREE_DEBUG      - printf()s to console with every MGA_Alloc/AllocSlab/Free call: A very poor man's Valgrind
#-DFLOATFLUX            - compute fluxes using floats instead of doubles
#-DRECONSTRUCT_VELOCITY - apply PLM reconstruction to velocity instead of momentum
#-DSYNCMEX		- does a complete CUDA device synchronization at the end of MEX functions to
#                         try and help the Matlab profiler be more useful
#-DUSE_NVTX		- inserts a lot of useful nvtx range markers to assist with profiling/debugging
#-DUSE_STATIC_CHALLOC   - Instructs cflTimestep.cu to use cudaHostAlloc once & hold memory in static pointers
#                         This matters for very, very small simulations doing millions of iterations                   

NVCC_OPTS = $(DEFINES) -gencode arch=$(NVARCH),code=$(NVCODE) -Wno-deprecated-gpu-targets -O2 -g -lineinfo
#-Xptxas="-v"

# Debuggable code
#NVCC_OPTS = $(DEFINES) -gencode arch=$(NVARCH),code=$(NVCODE) -O0 -g -G -lineinfo
# Debuggable code with endless verbose statistics from ptxas
#NVCC_OPTS = $(DEFINES) -gencode arch=$(NVARCH),code=$(NVCODE) -O0 -g -G -lineinfo -Xptxas="-v"

NVCC_INCL = -I$(MATLAB_DIR)/extern/include -I../mpi

MEX_OPTS  = -DUNIX $(GEN_OPTS) -g -R2018a
MEX_INCL  = -I$(CUDA_DIR)/include -I$(MATLAB_DIR)/extern/include -I../mpi
MEX_LDIR  = -L$(CUDA_DIR)/$(CUDA_LDIR)
MEX_LIBS  = -lcudart -lcublas

PAR_INCL = -I../mpi -I$(MPI_DIR)/include
PAR_LDIR = -L$(MPI_DIR)/lib
PAR_LIBS = -lmpi -lmpi_cxx

OTHER_LIBS = -lnvToolsExt

OUTPUTCMD = -output

###############################################################################
all: base fluids lambda
	echo "Finished compiling GPU support."
	rm -f *cpp

#base: GPU_ctrl GPU_free GPU_cudamemcpy GPU_memavail cudaBasicOperations
#	echo "Done building base operations!"

base: GPU_free GPU_upload GPU_clone GPU_download GPU_ctrl GPU_dbgtag GPU_dbghalo GPU_getslab GPU_makeslab GPU_setslab GPU_copy cudaBasicOperations GPU_partitionStatics
	

fluids: cudaArrayAtomic cudaBasicOperations directionalMaxFinder freezeAndPtot fluidstep standalones cflTimestep
	echo "Done building fluid kernels."
	gcc -o m2m m2m.c
	./m2m > ../.fluidMethod
	rm m2m

magnets: cudaMHDKernels cudaFwdAverage cudaFwdDifference cudaMagPrep cudaMagW cudaMagTVD
	echo "Done building MHD kernels"

lambda: cudaSourceCylindricalTerms cudaSourceScalarPotential cudaSourceRotatingFrame cudaFreeRadiation cudaAccretingStar cudaSourceVTO cudaTestSourceComposite cudaSource2FluidDrag
	echo "Done building source/sink term kernels"

thermodynamics: cudaSoundspeed freezeAndPtot cudaFluidStep
	echo "Rebuilt pressure/thermo related routines"

# Key point: builds the cudaCommon.o that everything else needs
cudaCommon.o: cudaCommon.cu cudaCommon.h
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) $(PAR_INCL) -cuda cudaCommon.cu -o cudaCommon.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_INCL) -c cudaCommon.cpp


############# Routines that are part of the "compiled core" and have a different
# build procedure from the rest
# Remove all object code once done to prevent standalone/not-standalone cross contamination
standalones: cudaArrayRotateB cudaHaloExchange cudaFluidStep cudaStatics cudaSoundspeed



fluidstep: cudaArrayRotateFcn.o cudaFluidStepFcn.o cudaHaloExchangeFcn.o cudaStaticsFcn.o flux.o iface.o ../mpi/mpi_common.o
	mex $(PAR_INCL) $(MEX_OPTS) $(MEX_INCL) $(PAR_LDIR) -cxx cudaArrayRotateFcn.o cudaFluidStepFcn.o cudaHaloExchangeFcn.o cudaStaticsFcn.o flux.o iface.o cudaCommon.o ../mpi/mpi_common.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

mergedstep: cudaArrayRotateFcn.o cudaFluidStepFcn.o cudaHaloExchangeFcn.o cudaStaticsFcn.o cudaFreeRadiationFcn.o flux.o iface_merged.o ../mpi/mpi_common.o
	mex $(PAR_INCL) $(MEX_OPTS) $(MEX_INCL) $(PAR_LDIR) -cxx cudaArrayRotateFcn.o cudaFluidStepFcn.o cudaHaloExchangeFcn.o cudaStaticsFcn.o cudaFreeRadiationFcn.o flux.o iface_merged.o cudaCommon.o ../mpi/mpi_common.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

flux.o: flux.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda flux.cu -o flux.cpp
	mex $(MEX_OPTS) -c flux.cpp

iface_merged.o: flux_multi.cu cudaCommon.h cudaFluidStep.h cudaFreeRadiation.h
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda flux_multi.cu -o iface_merged.cpp
	mex $(MEX_OPTS) -c iface_merged.cpp

iface.o: flux_ML_iface.cu cudaCommon.h
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda flux_ML_iface.cu -o iface.cpp
	mex $(MEX_OPTS) -c iface.cpp

######## CUDA FLUID ROUTINES

# Array index transposition
cudaArrayRotateFcn.o: cudaArrayRotateB.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaArrayRotateB.cu -o cudaArrayRotateFcn.cpp
	mex $(MEX_OPTS) -c cudaArrayRotateFcn.cpp
	
rotate_standalone: cudaArrayRotateB.cu
	nvcc -DSTANDALONE_MEX_FUNCTION $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) $(PAR_INCL) -cuda cudaArrayRotateB.cu -o cudaArrayRotateB.cpp
	mex $(MEX_OPTS) -c cudaArrayRotateB.cpp

cudaArrayRotateB: rotate_standalone cudaCommon.o
	mex $(MEX_OPTS) $(PAR_LDIR) $(PAR_LIBS) -cxx cudaArrayRotateB.o cudaCommon.o $(OUTPUTCMD) cudaArrayRotateB $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)
	rm cudaArrayRotateB.o
	
# Parallel halo exchanges
cudaHaloExchangeFcn.o: cudaHaloExchange.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL)  -cuda cudaHaloExchange.cu -o cudaHaloExchangeFcn.cpp
	mex $(MEX_OPTS) $(PAR_INCL) -c cudaHaloExchangeFcn.cpp

halo_standalone: cudaHaloExchange.cu
	nvcc -DSTANDALONE_MEX_FUNCTION $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL)  -cuda cudaHaloExchange.cu -o cudaHaloExchange.cpp
	mex $(MEX_OPTS) $(PAR_INCL) -c cudaHaloExchange.cpp

cudaHaloExchange: halo_standalone cudaCommon.o
	mex $(MEX_OPTS) $(PAR_LDIR) $(PAR_LIBS) -cxx ../mpi/nvtx_wrapper.o cudaHaloExchange.o cudaCommon.o ../mpi/mpi_common.o $(OUTPUTCMD) cudaHaloExchange $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)
	rm cudaHaloExchange.o

# Fluid step
cudaFluidStepFcn.o: cudaFluidStep.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) $(PAR_INCL) -cuda cudaFluidStep.cu -o cudaFluidStepFcn.cpp
	mex $(MEX_OPTS) -c cudaFluidStepFcn.cpp

fluid_standalone: cudaFluidStep.cu
	nvcc -DSTANDALONE_MEX_FUNCTION $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) $(PAR_INCL) -cuda cudaFluidStep.cu -o cudaFluidStep.cpp
	mex $(MEX_OPTS) -c cudaFluidStep.cpp

cudaFluidStep: fluid_standalone cudaStaticsFcn.o cudaArrayRotateFcn.o cudaCommon.o
	mex $(MEX_OPTS) $(PAR_LDIR) $(PAR_LIBS) -cxx cudaFluidStep.o cudaCommon.o cudaStaticsFcn.o cudaArrayRotateFcn.o ../mpi/mpi_common.o $(OUTPUTCMD) cudaFluidStep $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)
	rm cudaFluidStep.o

# Statics / boundary condition settings
cudaStaticsFcn.o: cudaStatics.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaStatics.cu -o cudaStaticsFcn.cpp
	mex $(MEX_OPTS) -c cudaStaticsFcn.cpp

statics_standalone: cudaStatics.cu
	nvcc -DSTANDALONE_MEX_FUNCTION $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) $(PAR_INCL) -cuda cudaStatics.cu -o cudaStatics.cpp
	mex $(MEX_OPTS) -c cudaStatics.cpp

cudaStatics: statics_standalone cudaArrayRotateFcn.o cudaCommon.o
	mex $(MEX_OPTS) $(PAR_LDIR) $(PAR_LIBS) -cxx cudaStatics.o cudaArrayRotateFcn.o cudaCommon.o $(OUTPUTCMD) cudaStatics $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)
	rm cudaStatics.o
	
# Standalone EOS solver
cudaSoundspeedFcn.o: cudaSoundspeed.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaSoundspeed.cu -o cudaSoundspeedFcn.cpp
	mex $(MEX_OPTS) -c cudaSoundspeedFcn.cpp

soundspeed_standalone: cudaSoundspeed.cu
	nvcc -DSTANDALONE_MEX_FUNCTION $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) $(PAR_INCL) -cuda cudaSoundspeed.cu -o cudaSoundspeed.cpp
	mex $(MEX_OPTS) -c cudaSoundspeed.cpp

cudaSoundspeed: soundspeed_standalone cudaCommon.o
	mex $(MEX_OPTS) $(PAR_LDIR) $(PAR_LIBS) -cxx cudaSoundspeed.o cudaCommon.o $(OUTPUTCMD) cudaSoundspeed $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)
	rm cudaSoundspeed.o


############# GPU CONTROL ROUTINES

GPU_copy.o: GPU_copy.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_copy.cu -o GPU_copy.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_copy.cpp
GPU_copy: GPU_copy.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_LDIR) -cxx GPU_copy.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)


GPU_dbgtag.o: GPU_dbgtag.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_dbgtag.cu -o GPU_dbgtag.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_dbgtag.cpp
GPU_dbgtag: GPU_dbgtag.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_LDIR) -cxx GPU_dbgtag.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

GPU_dbghalo.o: GPU_dbghalo.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_dbghalo.cu -o GPU_dbghalo.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_dbghalo.cpp
GPU_dbghalo: GPU_dbghalo.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_LDIR) -cxx GPU_dbghalo.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

GPU_getslab.o: GPU_getslab.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_getslab.cu -o GPU_getslab.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_getslab.cpp
GPU_getslab: GPU_getslab.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_LDIR) -cxx GPU_getslab.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)


GPU_setslab.o: GPU_setslab.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_setslab.cu -o GPU_setslab.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_setslab.cpp
GPU_setslab: GPU_setslab.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_LDIR) -cxx GPU_setslab.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)


GPU_makeslab.o: GPU_makeslab.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_makeslab.cu -o GPU_makeslab.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_makeslab.cpp
GPU_makeslab: GPU_makeslab.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_LDIR) -cxx GPU_makeslab.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

GPU_slabs.o: GPU_slabs.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_slabs.cu -o GPU_slabs.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_slabs.cpp

GPU_slabs: GPU_slabs.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_LDIR) -cxx GPU_slabs.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

GPU_ctrl.o: GPU_ctrl.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_ctrl.cu -o GPU_ctrl.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_ctrl.cpp

GPU_ctrl: GPU_ctrl.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_LDIR) -cxx GPU_ctrl.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

GPU_free.o: GPU_free.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_free.cu -o GPU_free.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_free.cpp 
GPU_free: GPU_free.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx cudaCommon.o $@.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

GPU_clone.o: GPU_clone.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_clone.cu -o GPU_clone.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_clone.cpp 
GPU_clone: GPU_clone.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

GPU_cudamemcpy.o: GPU_cudamemcpy.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_cudamemcpy.cu -o GPU_cudamemcpy.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_cudamemcpy.cpp 
GPU_cudamemcpy: GPU_cudamemcpy.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

GPU_download.o: GPU_download.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_download.cu -o GPU_download.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_download.cpp 
GPU_download: GPU_download.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

GPU_memavail.o: GPU_memavail.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_memavail.cu -o GPU_memavail.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_memavail.cpp 
GPU_memavail: GPU_memavail.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

GPU_partitionStatics.o: GPU_partitionStatics.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_partitionStatics.cu -o GPU_partitionStatics.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_partitionStatics.cpp 
GPU_partitionStatics: GPU_partitionStatics.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

GPU_upload.o: GPU_upload.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_upload.cu -o GPU_upload.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_upload.cpp 
GPU_upload: GPU_upload.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

######## CUDA FLUID ROUTINES
cudaArrayAtomic.o: cudaArrayAtomic.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaArrayAtomic.cu -o cudaArrayAtomic.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaArrayAtomic.cpp
cudaArrayAtomic: cudaCommon.o cudaArrayAtomic.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaBasicOperations.o: cudaBasicOperations.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaBasicOperations.cu -o cudaBasicOperations.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaBasicOperations.cpp
cudaBasicOperations: cudaCommon.o cudaBasicOperations.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaShift.o: cudaShift.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaShift.cu -o cudaShift.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaShift.cpp
cudaShift: cudaCommon.o cudaShift.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

directionalMaxFinder.o: directionalMaxFinder.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda directionalMaxFinder.cu -o directionalMaxFinder.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c directionalMaxFinder.cpp
directionalMaxFinder: cudaCommon.o directionalMaxFinder.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o ../mpi/mpi_common.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

freezeAndPtot.o: freezeAndPtot.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda freezeAndPtot.cu -o freezeAndPtot.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c freezeAndPtot.cpp
freezeAndPtot: cudaCommon.o ../mpi/mpi_common.o freezeAndPtot.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o ../mpi/mpi_common.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cflTimestep.o: cflTimestep.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cflTimestep.cu -o cflTimestep.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cflTimestep.cpp

cflTimestep: cflTimestep.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o ../mpi/mpi_common.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaMHDKernels.o: cudaMHDKernels.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaMHDKernels.cu -o cudaMHDKernels.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaMHDKernels.cpp
cudaMHDKernels: cudaCommon.o cudaMHDKernels.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaFwdAverage.o: cudaFwdAverage.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaFwdAverage.cu -o cudaFwdAverage.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaFwdAverage.cpp
cudaFwdAverage: cudaCommon.o cudaFwdAverage.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaFwdDifference.o: cudaFwdDifference.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaFwdDifference.cu -o cudaFwdDifference.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaFwdDifference.cpp
cudaFwdDifference: cudaCommon.o cudaFwdDifference.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaMagPrep.o: cudaMagPrep.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaMagPrep.cu -o cudaMagPrep.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaMagPrep.cpp
cudaMagPrep: cudaCommon.o cudaMagPrep.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaMagW.o: cudaMagW.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaMagW.cu -o cudaMagW.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaMagW.cpp
cudaMagW: cudaCommon.o cudaMagW.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaMagTVD.o: cudaMagTVD.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaMagTVD.cu -o cudaMagTVD.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaMagTVD.cpp
cudaMagTVD: cudaCommon.o cudaMagTVD.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

# SOURCE/SINK ROUTINES
cudaSourceScalarPotential.o: cudaSourceScalarPotential.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaSourceScalarPotential.cu -o cudaSourceScalarPotential.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaSourceScalarPotential.cpp
cudaSourceScalarPotential: cudaCommon.o cudaGradientKernels.o cudaSourceScalarPotential.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaGradientKernels.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaSourceRotatingFrame.o: cudaSourceRotatingFrame.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaSourceRotatingFrame.cu -o cudaSourceRotatingFrame.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaSourceRotatingFrame.cpp
cudaSourceRotatingFrame: cudaCommon.o cudaSourceRotatingFrame.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaSourceCylindricalTerms.o: cudaSourceCylindricalTerms.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaSourceCylindricalTerms.cu -o cudaSourceCylindricalTerms.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaSourceCylindricalTerms.cpp
cudaSourceCylindricalTerms: cudaCommon.o cudaSourceCylindricalTerms.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaFreeRadiationFcn.o: cudaFreeRadiation.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaFreeRadiation.cu -o cudaFreeRadiationFcn.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaFreeRadiationFcn.cpp
cudaFreeRadiation.o: cudaFreeRadiation.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaFreeRadiation.cu -o cudaFreeRadiation.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -DSTANDALONE_MEX_FUNCTION -c cudaFreeRadiation.cpp
cudaFreeRadiation: cudaCommon.o cudaFreeRadiation.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)


cudaAccretingStar.o: cudaAccretingStar.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaAccretingStar.cu -o cudaAccretingStar.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_INCL) -c cudaAccretingStar.cpp
cudaAccretingStar: cudaCommon.o cudaAccretingStar.o
	mex $(MEX_OPTS)  $(PAR_LDIR) $(PAR_LIBS) $(MEX_LDIR) $(MEX_LIBS)  $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@

cudaSourceVTO.o: cudaSourceVTO.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaSourceVTO.cu -o cudaSourceVTO.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_INCL) -c cudaSourceVTO.cpp

cudaSourceVTO: cudaCommon.o cudaSourceVTO.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaTestSourceComposite: cudaCommon.o cudaTestSourceComposite.o cudaGradientKernels.o
	mex $(MEX_OPTS)  $(PAR_LDIR) $(PAR_LIBS) $(MEX_LDIR) $(MEX_LIBS)  $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS) -cxx $@.o cudaGradientKernels.o cudaCommon.o $(OUTPUTCMD) $@

cudaGradientKernels.o: cudaGradientKernels.cu cudaGradientKernels.h
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaGradientKernels.cu -o cudaGradientKernels.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_INCL) -c cudaGradientKernels.cpp

cudaTestSourceComposite.o: cudaTestSourceComposite.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaTestSourceComposite.cu -o cudaTestSourceComposite.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_INCL) -c cudaTestSourceComposite.cpp

cudaSource2FluidDrag.o: cudaSource2FluidDrag.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaSource2FluidDrag.cu -o cudaSource2FluidDrag.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_INCL) -c cudaSource2FluidDrag.cpp

cudaSource2FluidDrag: cudaCommon.o cudaSource2FluidDrag.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

clean:
	rm -f *mexa64 *cpp *o

tidy:
	rm -f *cpp *o
