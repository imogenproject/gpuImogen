# This makefile builds the GPU kernels needed to run gpuImogen

include ../Make.config

DEFINES = -DUNIX -D_FORCE_INLINES -DRECONSTRUCT_VELOCITY -DUSE_NVTX -DUSE_STATIC_CHALLOC -DNOMATLAB
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

CPP       = g++

NVCC_INCL = -I$(MATLAB_DIR)/extern/include -I../mpi

MEX_OPTS  = -DUNIX $(GEN_OPTS) -g
MEX_INCL  = -I$(CUDA_DIR)/include -I$(MATLAB_DIR)/extern/include -I../mpi
MEX_LDIR  = -L$(CUDA_DIR)/$(CUDA_LDIR)
MEX_LIBS  = -lcudart -lcublas

PAR_INCL = -I../mpi -I$(MPI_DIR)/include
PAR_LDIR = -L$(MPI_DIR)/lib
PAR_LIBS = -lmpi -lmpi_cxx

OTHER_LIBS = -lnvToolsExt -lhdf5 -lhdf5_hl

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
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -c -fPIC cudaCommon.cpp
	rm cudaCommon.cpp

cudaUtilities.o: cudaUtilities.cu cudaUtilities.h
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -c cudaUtilities.cu -o cudaUtilities.o

############# Routines that are part of the "compiled core" and have a different
# build procedure from the rest
# Remove all object code once done to prevent standalone/not-standalone cross contamination
standalones: cudaArrayRotateB cudaHaloExchange cudaFluidStep cudaStatics cudaSoundspeed

fluidstep: cudaCommon.o cudaArrayRotateFcn.o  cudaFluidStepFcn.o cudaHaloExchangeFcn.o cudaStaticsFcn.o flux.o iface.o ../mpi/mpi_common.o
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -cxx cudaArrayRotateFcn.o cudaFluidStepFcn.o cudaHaloExchangeFcn.o cudaStaticsFcn.o flux.o cudaCommon.o ../mpi/mpi_common.o iface.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

mergedstep: cudaArrayRotateFcn.o cudaFluidStepFcn.o cudaHaloExchangeFcn.o cudaStaticsFcn.o cudaFreeRadiationFcn.o flux.o iface_merged.o cudaCommon.o ../mpi/mpi_common.o
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -cxx cudaArrayRotateFcn.o cudaFluidStepFcn.o cudaHaloExchangeFcn.o cudaStaticsFcn.o cudaFreeRadiationFcn.o flux.o iface_merged.o cudaCommon.o ../mpi/mpi_common.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

sourceStep.o: sourceStep.cu sourceStep.h
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -DSTANDALONE_MEX_FUNCTION -cuda sourceStep.cu -o sourceStep.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -DSTANDALONE_MEX_FUNCTION -c sourceStep.cpp
	rm sourceStep.cpp

sourceStepFcn.o: sourceStep.cu sourceStep.h
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda sourceStep.cu -o sourceStepFcn.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -fPIC -c sourceStepFcn.cpp
	rm sourceStepFcn.cpp

sourceStep: cudaArrayRotateFcn.o cudaFluidStepFcn.o cudaHaloExchangeFcn.o cudaGradientKernels.o cudaTestSourceCompositeFcn.o cudaSourceScalarPotentialFcn.o cudaSourceCylindricalTermsFcn.o cudaStaticsFcn.o cudaFreeRadiationFcn.o cudaSourceRotatingFrameFcn.o cudaSource2FluidDragFcn.o cudaCommon.o sourceStep.o cudaUtilities.o ../mpi/mpi_common.o
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -cxx cudaArrayRotateFcn.o cudaCommon.o cudaFluidStepFcn.o cudaTestSourceCompositeFcn.o cudaHaloExchangeFcn.o cudaGradientKernels.o cudaSourceCylindricalTermsFcn.o cudaSourceScalarPotentialFcn.o cudaStaticsFcn.o cudaFreeRadiationFcn.o cudaSourceRotatingFrameFcn.o cudaSource2FluidDragFcn.o cudaUtilities.o ../mpi/mpi_common.o sourceStep.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)
	
completeStep: cudaArrayRotateFcn.o cudaFluidStepFcn.o cudaHaloExchangeFcn.o cudaGradientKernels.o cudaTestSourceCompositeFcn.o cudaSourceScalarPotentialFcn.o cudaSourceCylindricalTermsFcn.o cudaStaticsFcn.o cudaFreeRadiationFcn.o cudaSourceRotatingFrameFcn.o cudaSource2FluidDragFcn.o cudaCommon.o sourceStepFcn.o cudaSoundspeedFcn.o iface_merged.o cudaUtilities.o cflTimestepFcn.o flux.o ../mpi/mpi_common.o
	mex $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS) -cxx cudaArrayRotateFcn.o cudaCommon.o cudaFluidStepFcn.o cudaTestSourceCompositeFcn.o cudaHaloExchangeFcn.o cudaGradientKernels.o cudaSourceCylindricalTermsFcn.o cudaSourceScalarPotentialFcn.o cudaStaticsFcn.o cudaFreeRadiationFcn.o cudaSourceRotatingFrameFcn.o cudaSource2FluidDragFcn.o cudaSoundspeedFcn.o flux.o sourceStepFcn.o cflTimestepFcn.o cudaUtilities.o ../mpi/mpi_common.o iface_merged.o $(OUTPUTCMD) $@ 

boundaryConds.o: boundaryConds.cu cudaStatics.h
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda boundaryConds.cu -o boundaryConds.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -fPIC -c boundaryConds.cpp
	rm boundaryConds.cpp

imogenCore: cudaArrayRotateFcn.o cudaFluidStepFcn.o cudaHaloExchangeFcn.o cudaGradientKernels.o cudaTestSourceCompositeFcn.o cudaSourceScalarPotentialFcn.o cudaSourceCylindricalTermsFcn.o boundaryConds.o cudaFreeRadiationFcn.o cudaSourceRotatingFrameFcn.o cudaSource2FluidDragFcn.o cudaCommon.o sourceStepFcn.o cudaSoundspeedFcn.o cudaUtilities.o cflTimestepFcn.o flux.o ../mpi/mpi_common_noml.o gpuimogen_core.o core_glue.o
	$(CPP) $(MEX_LDIR) $(PAR_LIBS) $(OTHER_LIBS) $(MEX_LIBS) -lhdf5 cudaArrayRotateFcn.o cudaCommon.o cudaFluidStepFcn.o cudaTestSourceCompositeFcn.o cudaHaloExchangeFcn.o cudaGradientKernels.o cudaSourceCylindricalTermsFcn.o cudaSourceScalarPotentialFcn.o boundaryConds.o cudaFreeRadiationFcn.o cudaSourceRotatingFrameFcn.o cudaSource2FluidDragFcn.o cudaSoundspeedFcn.o flux.o sourceStepFcn.o cflTimestepFcn.o cudaUtilities.o ../mpi/mpi_common_noml.o core_glue.o gpuimogen_core.o -o $@ 

core_glue.o: core_glue.cpp core_glue.hpp
	nvcc $(DEFINES) $(PAR_INCL) $(MEX_OPTS) -c core_glue.cpp -o core_glue.o

flux.o: flux.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda flux.cu -o flux.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -fPIC -c flux.cpp
	rm flux.cpp

iface_merged.o: flux_multi.cu cudaCommon.h cudaFluidStep.h cudaFreeRadiation.h
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda flux_multi.cu -o iface_merged.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c iface_merged.cpp
	rm iface_merged.cpp

iface.o: flux_ML_iface.cu cudaCommon.h
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda flux_ML_iface.cu -o iface.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c iface.cpp
	rm iface.cpp

gpuimogen_core.o: gpuimogen_core.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda gpuimogen_core.cu -o gpuimogen_core.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -fPIC -c gpuimogen_core.cpp
	rm gpuimogen_core.cpp

######## CUDA FLUID ROUTINES

# Array index transposition
cudaArrayRotateFcn.o: cudaArrayRotateB.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaArrayRotateB.cu -o cudaArrayRotateFcn.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -fPIC -c cudaArrayRotateFcn.cpp
	rm cudaArrayRotateFcn.cpp

cudaArrayRotate.o: cudaArrayRotateB.cu
	nvcc $(PAR_INCL) $(NVCC_INCL) $(PAR_INCL) $(NVCC_OPTS) -DSTANDALONE_MEX_FUNCTION -cuda cudaArrayRotateB.cu -o cudaArrayRotateB.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c cudaArrayRotateB.cpp
	rm cudaArrayRotateB.cpp

cudaArrayRotateB: cudaArrayRotate.o cudaCommon.o cudaUtilities.o
	mex $(PAR_LDIR) $(PAR_LIBS) $(MEX_LDIR) $(MEX_LIBS) $(OTHER_LIBS) $(MEX_OPTS) -cxx cudaArrayRotateB.o cudaUtilities.o cudaCommon.o $(OUTPUTCMD) cudaArrayRotateB 
	
# Parallel halo exchanges
cudaHaloExchangeFcn.o: cudaHaloExchange.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaHaloExchange.cu -o cudaHaloExchangeFcn.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -fPIC  -c cudaHaloExchangeFcn.cpp
	rm cudaHaloExchangeFcn.cpp

halo_standalone: cudaHaloExchange.cu
	nvcc -DSTANDALONE_MEX_FUNCTION $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL)  -cuda cudaHaloExchange.cu -o cudaHaloExchange.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -DSTANDALONE_MEX_FUNCTION -c cudaHaloExchange.cpp
	rm cudaHaloExchange.cpp

cudaHaloExchange: halo_standalone cudaCommon.o
	mex $(PAR_LDIR) $(PAR_LIBS) $(MEX_LDIR) $(MEX_LIBS) $(OTHER_LIBS) $(MEX_OPTS) -cxx ../mpi/nvtx_wrapper.o ../mpi/mpi_common.o cudaHaloExchange.o cudaCommon.o $(OUTPUTCMD) cudaHaloExchange 


# Fluid step
cudaFluidStepFcn.o: cudaFluidStep.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) $(PAR_INCL) -cuda cudaFluidStep.cu -o cudaFluidStepFcn.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -fPIC -c cudaFluidStepFcn.cpp
	rm cudaFluidStepFcn.cpp

fluid_standalone: cudaFluidStep.cu
	nvcc -DSTANDALONE_MEX_FUNCTION $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) $(PAR_INCL) -cuda cudaFluidStep.cu -o cudaFluidStep.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -DSTANDALONE_MEX_FUNCTION -c cudaFluidStep.cpp
	rm cudaFluidStep.cpp

cudaFluidStep: fluid_standalone cudaStaticsFcn.o cudaArrayRotateFcn.o cudaCommon.o
	mex $(PAR_LDIR) $(PAR_LIBS) $(MEX_LDIR) $(MEX_LIBS) $(OTHER_LIBS) $(MEX_OPTS) -cxx cudaFluidStep.o cudaCommon.o cudaStaticsFcn.o cudaArrayRotateFcn.o ../mpi/mpi_common.o $(OUTPUTCMD) cudaFluidStep

# Statics / boundary condition settings
cudaStaticsFcn.o: cudaStatics.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaStatics.cu -o cudaStaticsFcn.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -fPIC -c cudaStaticsFcn.cpp
	rm cudaStaticsFcn.cpp

statics_standalone: cudaStatics.cu
	nvcc -DSTANDALONE_MEX_FUNCTION $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) $(PAR_INCL) -cuda cudaStatics.cu -o cudaStatics.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -DSTANDALONE_MEX_FUNCTION -c cudaStatics.cpp
#	rm cudaStatics.cpp

cudaStatics: statics_standalone cudaArrayRotateFcn.o cudaCommon.o
	mex $(PAR_LDIR) $(PAR_LIBS) $(MEX_LDIR) $(MEX_LIBS) $(OTHER_LIBS) $(MEX_OPTS) -g -cxx cudaArrayRotateFcn.o cudaCommon.o cudaStatics.o $(OUTPUTCMD) cudaStatics

# Standalone EOS solver
cudaSoundspeedFcn.o: cudaSoundspeed.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaSoundspeed.cu -o cudaSoundspeedFcn.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -fPIC -c cudaSoundspeedFcn.cpp
	rm cudaSoundspeedFcn.cpp

soundspeed_standalone: cudaSoundspeed.cu
	nvcc -DSTANDALONE_MEX_FUNCTION $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) $(PAR_INCL) -cuda cudaSoundspeed.cu -o cudaSoundspeed.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -DSTANDALONE_MEX_FUNCTION -c cudaSoundspeed.cpp
	rm cudaSoundspeed.cpp

cudaSoundspeed: soundspeed_standalone cudaCommon.o
	mex $(PAR_LDIR) $(PAR_LIBS) $(MEX_LDIR) $(MEX_LIBS) $(OTHER_LIBS) $(MEX_OPTS) -cxx cudaCommon.o cudaSoundspeed.o $(OUTPUTCMD) cudaSoundspeed

############# GPU CONTROL ROUTINES

GPU_copy.o: GPU_copy.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_copy.cu -o GPU_copy.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c GPU_copy.cpp
	rm GPU_copy.cpp

GPU_copy: GPU_copy.o cudaCommon.o
	mex $(MEX_INCL) $(PAR_LDIR) $(MEX_OPTS) -cxx GPU_copy.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

GPU_dbgtag.o: GPU_dbgtag.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_dbgtag.cu -o GPU_dbgtag.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c GPU_dbgtag.cpp
	rm GPU_dbgtag.cpp

GPU_dbgtag: GPU_dbgtag.o cudaCommon.o
	mex $(MEX_INCL) $(PAR_LDIR) $(MEX_OPTS) -cxx GPU_dbgtag.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

GPU_dbghalo.o: GPU_dbghalo.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_dbghalo.cu -o GPU_dbghalo.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c GPU_dbghalo.cpp
	rm GPU_dbghalo.cpp

GPU_dbghalo: GPU_dbghalo.o cudaCommon.o
	mex $(MEX_INCL) $(PAR_LDIR) $(MEX_OPTS) -cxx GPU_dbghalo.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

GPU_getslab.o: GPU_getslab.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_getslab.cu -o GPU_getslab.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c GPU_getslab.cpp
	rm GPU_getslab.cpp

GPU_getslab: GPU_getslab.o cudaCommon.o
	mex $(MEX_INCL) $(PAR_LDIR) $(MEX_OPTS) -cxx GPU_getslab.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

GPU_setslab.o: GPU_setslab.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_setslab.cu -o GPU_setslab.cpp
	mex $(MEX_INCL) $(MEX_OPTS) -c GPU_setslab.cpp
	rm GPU_setslab.cpp

GPU_setslab: GPU_setslab.o cudaCommon.o
	mex $(MEX_INCL) $(PAR_LDIR) $(MEX_OPTS) -cxx GPU_setslab.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

GPU_makeslab.o: GPU_makeslab.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_makeslab.cu -o GPU_makeslab.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c GPU_makeslab.cpp
	rm GPU_makeslab.cpp

GPU_makeslab: GPU_makeslab.o cudaCommon.o
	mex $(MEX_INCL) $(PAR_LDIR) $(MEX_OPTS) -cxx GPU_makeslab.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

GPU_slabs.o: GPU_slabs.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_slabs.cu -o GPU_slabs.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c GPU_slabs.cpp
	rm GPU_slabs.cpp
	
GPU_slabs: GPU_slabs.o cudaCommon.o
	mex $(MEX_INCL) $(PAR_LDIR) $(MEX_OPTS) -cxx GPU_slabs.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

GPU_ctrl.o: GPU_ctrl.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_ctrl.cu -o GPU_ctrl.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c GPU_ctrl.cpp
	rm GPU_ctrl.cpp

GPU_ctrl: GPU_ctrl.o cudaCommon.o
	mex $(MEX_INCL) $(PAR_LDIR) $(MEX_OPTS) -cxx GPU_ctrl.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LIBS) $(OTHER_LIBS)

GPU_free.o: GPU_free.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_free.cu -o GPU_free.cpp
	mex $(PAR_INCL) $(MEX_INCL) -c GPU_free.cpp
	rm GPU_free.cpp

GPU_free: GPU_free.o cudaCommon.o
	mex $(MEX_INCL) $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS) -cxx cudaCommon.o $@.o $(OUTPUTCMD) $@ 

GPU_clone.o: GPU_clone.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_clone.cu -o GPU_clone.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c GPU_clone.cpp
	rm GPU_clone.cpp

GPU_clone: GPU_clone.o cudaCommon.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

GPU_cudamemcpy.o: GPU_cudamemcpy.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_cudamemcpy.cu -o GPU_cudamemcpy.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c GPU_cudamemcpy.cpp
	rm GPU_cudamemcpy.cpp

GPU_cudamemcpy: GPU_cudamemcpy.o cudaCommon.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

GPU_download.o: GPU_download.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_download.cu -o GPU_download.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c GPU_download.cpp
	rm GPU_download.cpp

GPU_download: GPU_download.o cudaCommon.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

GPU_memavail.o: GPU_memavail.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_memavail.cu -o GPU_memavail.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c GPU_memavail.cpp
	rm GPU_memavail.cpp

GPU_memavail: GPU_memavail.o cudaCommon.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

GPU_partitionStatics.o: GPU_partitionStatics.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_partitionStatics.cu -o GPU_partitionStatics.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c GPU_partitionStatics.cpp
	rm GPU_partitionStatics.cpp

GPU_partitionStatics: GPU_partitionStatics.o cudaCommon.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

GPU_upload.o: GPU_upload.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda GPU_upload.cu -o GPU_upload.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c GPU_upload.cpp
	rm GPU_upload.cpp

GPU_upload: GPU_upload.o cudaCommon.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

######## CUDA FLUID ROUTINES
cudaArrayAtomic.o: cudaArrayAtomic.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaArrayAtomic.cu -o cudaArrayAtomic.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c cudaArrayAtomic.cpp
	rm cudaArrayAtomic.cpp

cudaArrayAtomic: cudaCommon.o cudaArrayAtomic.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaBasicOperations.o: cudaBasicOperations.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaBasicOperations.cu -o cudaBasicOperations.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c cudaBasicOperations.cpp
	rm cudaBasicOperations.cpp

cudaBasicOperations: cudaCommon.o cudaBasicOperations.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaShift.o: cudaShift.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaShift.cu -o cudaShift.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c cudaShift.cpp
	rm cudaShift.cpp

cudaShift: cudaCommon.o cudaShift.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

directionalMaxFinder.o: directionalMaxFinder.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda directionalMaxFinder.cu -o directionalMaxFinder.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c directionalMaxFinder.cpp
	rm directionalMaxFinder.cpp

directionalMaxFinder: cudaCommon.o directionalMaxFinder.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o ../mpi/mpi_common.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

freezeAndPtot.o: freezeAndPtot.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda freezeAndPtot.cu -o freezeAndPtot.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -c freezeAndPtot.cpp
	rm freezeAndPtot.cpp

freezeAndPtot: cudaCommon.o ../mpi/mpi_common.o freezeAndPtot.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o ../mpi/mpi_common.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cflTimestepFcn.o: cflTimestep.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cflTimestep.cu -o cflTimestepFcn.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -fPIC -c cflTimestepFcn.cpp
	rm cflTimestepFcn.cpp

cflTimestep.o: cflTimestep.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -DSTANDALONE_MEX_FUNCTION -cuda cflTimestep.cu -o cflTimestep.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -DSTANDALONE_MEX_FUNCTION  -c cflTimestep.cpp
	rm cflTimestep.cpp

cflTimestep: cflTimestep.o cudaCommon.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o ../mpi/mpi_common.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

# I think these are all dead?
cudaMHDKernels.cpp: cudaMHDKernels.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaMHDKernels.cu -o cudaMHDKernels.cpp
cudaMHDKernels: cudaCommon.o cudaMHDKernels.cpp
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaFwdAverage.cpp: cudaFwdAverage.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaFwdAverage.cu -o cudaFwdAverage.cpp
cudaFwdAverage: cudaCommon.o cudaFwdAverage.cpp
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaFwdDifference.cpp: cudaFwdDifference.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaFwdDifference.cu -o cudaFwdDifference.cpp
cudaFwdDifference: cudaCommon.o cudaFwdDifference.cpp
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaMagPrep.cpp: cudaMagPrep.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaMagPrep.cu -o cudaMagPrep.cpp
cudaMagPrep: cudaCommon.o cudaMagPrep.cpp
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaMagW.cpp: cudaMagW.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaMagW.cu -o cudaMagW.cpp
cudaMagW: cudaCommon.o cudaMagW.cpp
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaMagTVD.cpp: cudaMagTVD.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaMagTVD.cu -o cudaMagTVD.cpp
cudaMagTVD: cudaCommon.o cudaMagTVD.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

# SOURCE/SINK ROUTINES
cudaSourceScalarPotentialFcn.o: cudaSourceScalarPotential.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaSourceScalarPotential.cu -o cudaSourceScalarPotentialFcn.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -fPIC -c cudaSourceScalarPotentialFcn.cpp
	rm cudaSourceScalarPotentialFcn.cpp

cudaSourceScalarPotential.o: cudaSourceScalarPotential.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -DSTANDALONE_MEX_FUNCTION -cuda cudaSourceScalarPotential.cu -o cudaSourceScalarPotential.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -DSTANDALONE_MEX_FUNCTION -c cudaSourceScalarPotential.cpp
	rm cudaSourceScalarPotential.cpp

cudaSourceScalarPotential: cudaCommon.o cudaGradientKernels.o cudaSourceScalarPotential.o cudaUtilities.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx cudaGradientKernels.o cudaCommon.o cudaUtilities.o cudaSourceScalarPotential.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaSourceRotatingFrameFcn.o: cudaSourceRotatingFrame.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaSourceRotatingFrame.cu -o cudaSourceRotatingFrameFcn.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -c cudaSourceRotatingFrameFcn.cpp
	rm cudaSourceRotatingFrameFcn.cpp

cudaSourceRotatingFrame.o: cudaSourceRotatingFrame.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -DSTANDALONE_MEX_FUNCTION -cuda cudaSourceRotatingFrame.cu -o cudaSourceRotatingFrame.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -DSTANDALONE_MEX_FUNCTION -c cudaSourceRotatingFrame.cpp
	rm cudaSourceRotatingFrame.cpp

cudaSourceRotatingFrame: cudaCommon.o cudaSourceRotatingFrame.o cudaUtilities.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx cudaCommon.o cudaUtilities.o $@.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaSourceCylindricalTermsFcn.o: cudaSourceCylindricalTerms.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaSourceCylindricalTerms.cu -o cudaSourceCylindricalTermsFcn.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -c cudaSourceCylindricalTermsFcn.cpp
	rm cudaSourceCylindricalTermsFcn.cpp

cudaSourceCylindricalTerms.o: cudaSourceCylindricalTerms.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -DSTANDALONE_MEX_FUNCTION -cuda cudaSourceCylindricalTerms.cu -o cudaSourceCylindricalTerms.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -DSTANDALONE_MEX_FUNCTION -c cudaSourceCylindricalTerms.cpp
	rm cudaSourceCylindricalTerms.cpp

cudaSourceCylindricalTerms: cudaCommon.o cudaSourceCylindricalTerms.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaFreeRadiationFcn.o: cudaFreeRadiation.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaFreeRadiation.cu -o cudaFreeRadiationFcn.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -c cudaFreeRadiationFcn.cpp
	rm cudaFreeRadiationFcn.cpp

cudaFreeRadiation.o: cudaFreeRadiation.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -DSTANDALONE_MEX_FUNCTION -cuda cudaFreeRadiation.cu -o cudaFreeRadiation.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -DSTANDALONE_MEX_FUNCTION -c cudaFreeRadiation.cpp
	rm cudaFreeRadiation.cpp

cudaFreeRadiation: cudaCommon.o cudaFreeRadiation.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaAccretingStar.o: cudaAccretingStar.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaAccretingStar.cu -o cudaAccretingStar.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS)  -DSTANDALONE_MEX_FUNCTION -c cudaAccretingStar.cpp
	rm cudaAccretingStar.cpp

cudaAccretingStar: cudaCommon.o cudaAccretingStar.o
	mex $(PAR_LDIR) $(PAR_LIBS) $(MEX_OPTS) $(MEX_LDIR) $(MEX_LIBS) $(OTHER_LIBS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@

cudaSourceVTO.o: cudaSourceVTO.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaSourceVTO.cu -o cudaSourceVTO.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) -DSTANDALONE_MEX_FUNCTION -c cudaSourceVTO.cpp
	rm cudaSourceVTO.cpp

cudaSourceVTO: cudaCommon.o cudaSourceVTO.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx $@.o cudaCommon.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

cudaGradientKernels.o: cudaGradientKernels.cu cudaGradientKernels.h
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaGradientKernels.cu -o cudaGradientKernels.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -c cudaGradientKernels.cpp
	rm cudaGradientKernels.cpp

cudaTestSourceCompositeFcn.o: cudaTestSourceComposite.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaTestSourceComposite.cu -o cudaTestSourceCompositeFcn.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -c cudaTestSourceCompositeFcn.cpp
	rm cudaTestSourceCompositeFcn.cpp

cudaTestSourceComposite.o: cudaTestSourceComposite.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -DSTANDALONE_MEX_FUNCTION -cuda cudaTestSourceComposite.cu -o cudaTestSourceComposite.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -DSTANDALONE_MEX_FUNCTION -c cudaTestSourceComposite.cpp

cudaTestSourceComposite: cudaCommon.o cudaTestSourceComposite.o cudaGradientKernels.o cudaUtilities.o
	mex $(MEX_OPTS)  $(PAR_LDIR) $(PAR_LIBS) $(MEX_LDIR) $(MEX_LIBS) $(OTHER_LIBS) -cxx $@.o cudaGradientKernels.o cudaCommon.o cudaUtilities.o $(OUTPUTCMD) $@

cudaSource2FluidDragFcn.o: cudaSource2FluidDrag.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -cuda cudaSource2FluidDrag.cu -o cudaSource2FluidDragFcn.cpp
	$(CPP) $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -c cudaSource2FluidDragFcn.cpp
	rm cudaSource2FluidDragFcn.cpp

cudaSource2FluidDrag.o: cudaSource2FluidDrag.cu
	nvcc $(NVCC_OPTS) $(PAR_INCL) $(NVCC_INCL) -DSTANDALONE_MEX_FUNCTION -cuda cudaSource2FluidDrag.cu -o cudaSource2FluidDrag.cpp
	mex $(PAR_INCL) $(MEX_INCL) $(MEX_OPTS) $(PAR_LDIR) -DSTANDALONE_MEX_FUNCTION -c cudaSource2FluidDrag.cpp

cudaSource2FluidDrag: cudaCommon.o cudaSource2FluidDrag.o cudaUtilities.o
	mex $(MEX_INCL) $(MEX_OPTS) -cxx cudaCommon.o cudaUtilities.o $@.o $(OUTPUTCMD) $@ $(MEX_LDIR) $(MEX_LIBS) $(PAR_LDIR) $(PAR_LIBS) $(OTHER_LIBS)

clean:
	rm -f *mexa64 *cpp *o

tidy:
	rm -f *cpp *o
