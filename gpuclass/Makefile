# This makefile builds the GPU kernels needed to run gpuImogen

include ../Make.config

# Don't mess with any of this. Modify the above config file instead
GEN_OPTS = 

#NVCC_OPTS = -arch=$(CUDA_ARCH) -O1 $(GEN_OPTS)
#NVCC_OPTS = -gencode arch=$(NVARCH),code=$(NVCODE) -O2
# To get debuggability & ptxas chatter...
NVCC_OPTS = -gencode arch=$(NVARCH),code=$(NVCODE) -O2 -g -G -Xptxas="-v"
NVCC_INCL = -I$(MATLAB_DIR)/extern/include

MEX_OPTS  = -DUNIX $(GEN_OPTS) -g 
MEX_INCL  = -I$(CUDA_DIR)/include -I$(MATLAB_DIR)/extern/include
MEX_LDIR  = -L$(CUDA_DIR)/$(CUDA_LDIR)
MEX_LIBS  = -lcudart -lcublas

PAR_INCL = -I$(PGW_DIR)/include -I$(MPI_DIR)/include
PAR_LDIR = -L$(PGW_DIR)/lib -L$(MPI_DIR)/lib
PAR_LIBS = -lpgw -lmpi

###############################################################################
all: base fluids parallel lambda
	echo "Finished compiling GPU support."
	rm -f *cpp

#base: GPU_ctrl GPU_free GPU_cudamemcpy GPU_memavail cudaBasicOperations
#	echo "Done building base operations!"

base: GPU_free GPU_upload GPU_clone GPU_download GPU_ctrl GPU_getslab GPU_makeslab GPU_copy cudaBasicOperations

fluids: cudaArrayRotate2 cudaArrayRotateB cudaArrayAtomic cudaBasicOperations cudaStatics directionalMaxFinder freezeAndPtot cudaFluidStep cudaMHDKernels cudaFwdAverage cudaFwdDifference cudaMagPrep cudaMagW cudaMagTVD cudaSoundspeed
	echo "Done building fluid kernels."

lambda: cudaSourceScalarPotential cudaSourceRotatingFrame cudaFreeRadiation cudaAccretingStar
	echo "Done building source/sink term kernels"

thermodynamics: cudaSoundspeed freezeAndPtot cudaFluidStep
	echo "Rebuilt pressure/thermo related routines"

parallel: cudaHaloExchange
	echo "Done building parallel support."

# Key point: builds the cudaCommon.o that everything else needs
cudaCommon.o: cudaCommon.cu cudaCommon.h
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaCommon.cu -o cudaCommon.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaCommon.cpp
	rm -f cudaCommon.cpp

############# GPU CONTROL ROUTINES

GPU_copy.o: GPU_copy.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda GPU_copy.cu -o GPU_copy.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_copy.cpp
GPU_copy: GPU_copy.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx GPU_copy.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)


GPU_getslab.o: GPU_getslab.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda GPU_getslab.cu -o GPU_getslab.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_getslab.cpp
GPU_getslab: GPU_getslab.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx GPU_getslab.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)


GPU_makeslab.o: GPU_makeslab.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda GPU_makeslab.cu -o GPU_makeslab.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_makeslab.cpp
GPU_makeslab: GPU_makeslab.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx GPU_makeslab.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

GPU_slabs.o: GPU_slabs.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda GPU_slabs.cu -o GPU_slabs.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_slabs.cpp

GPU_slabs: GPU_slabs.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx GPU_slabs.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

GPU_ctrl.o: GPU_ctrl.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda GPU_ctrl.cu -o GPU_ctrl.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_ctrl.cpp

GPU_ctrl: GPU_ctrl.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx GPU_ctrl.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

GPU_free.o: GPU_free.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda GPU_free.cu -o GPU_free.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_free.cpp 
GPU_free: GPU_free.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

GPU_clone.o: GPU_clone.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda GPU_clone.cu -o GPU_clone.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_clone.cpp 
GPU_clone: GPU_clone.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

GPU_download.o: GPU_download.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda GPU_download.cu -o GPU_download.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_download.cpp 
GPU_download: GPU_download.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

GPU_upload.o: GPU_upload.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda GPU_upload.cu -o GPU_upload.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_upload.cpp 
GPU_upload: GPU_upload.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

GPU_cudamemcpy.o: GPU_cudamemcpy.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda GPU_cudamemcpy.cu -o GPU_cudamemcpy.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_cudamemcpy.cpp 
GPU_cudamemcpy: GPU_cudamemcpy.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

GPU_memavail.o: GPU_memavail.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda GPU_memavail.cu -o GPU_memavail.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c GPU_memavail.cpp 
GPU_memavail: GPU_memavail.o cudaCommon.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

######## CUDA FLUID ROUTINES
cudaArrayRotate.o: cudaArrayRotate.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaArrayRotate.cu -o cudaArrayRotate.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaArrayRotate.cpp
cudaArrayRotate: cudaCommon.o cudaArrayRotate.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaArrayRotateB.o: cudaArrayRotateB.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaArrayRotateB.cu -o cudaArrayRotateB.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaArrayRotateB.cpp
cudaArrayRotateB: cudaCommon.o cudaArrayRotateB.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaArrayRotate2_inplace.o: cudaArrayRotate2_inplace.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaArrayRotate2_inplace.cu -o cudaArrayRotate2_inplace.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaArrayRotate2_inplace.cpp
cudaArrayRotate2_inplace: cudaCommon.o cudaArrayRotate2_inplace.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaArrayRotate2.o: cudaArrayRotate2.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaArrayRotate2.cu -o cudaArrayRotate2.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaArrayRotate2.cpp
cudaArrayRotate2: cudaCommon.o cudaArrayRotate2.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaArrayAtomic.o: cudaArrayAtomic.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaArrayAtomic.cu -o cudaArrayAtomic.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaArrayAtomic.cpp
cudaArrayAtomic: cudaCommon.o cudaArrayAtomic.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaBasicOperations.o: cudaBasicOperations.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaBasicOperations.cu -o cudaBasicOperations.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaBasicOperations.cpp
cudaBasicOperations: cudaCommon.o cudaBasicOperations.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaShift.o: cudaShift.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaShift.cu -o cudaShift.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaShift.cpp
cudaShift: cudaCommon.o cudaShift.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaStatics.o: cudaStatics.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaStatics.cu -o cudaStatics.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaStatics.cpp
cudaStatics: cudaCommon.o cudaStatics.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

directionalMaxFinder.o: directionalMaxFinder.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda directionalMaxFinder.cu -o directionalMaxFinder.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c directionalMaxFinder.cpp
directionalMaxFinder: cudaCommon.o directionalMaxFinder.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

freezeAndPtot.o: freezeAndPtot.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda freezeAndPtot.cu -o freezeAndPtot.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c freezeAndPtot.cpp
freezeAndPtot: cudaCommon.o freezeAndPtot.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaFluidW.o: cudaFluidW.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaFluidW.cu -o cudaFluidW.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaFluidW.cpp
cudaFluidW: cudaCommon.o cudaFluidW.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaFluidW_dumb.o: cudaFluidW_dumb.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaFluidW_dumb.cu -o cudaFluidW_dumb.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaFluidW_dumb.cpp
cudaFluidW_dumb: cudaCommon.o cudaFluidW_dumb.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)


cudaFluidTVD.o: cudaFluidTVD.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaFluidTVD.cu -o cudaFluidTVD.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaFluidTVD.cpp
cudaFluidTVD: cudaCommon.o cudaFluidTVD.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaFluidStep.o: cudaFluidStep.cu debug_inserts.h
	nvcc $(NVCC_OPTS) $(NVCC_INCL) $(PAR_INCL) -cuda cudaFluidStep.cu -o cudaFluidStep.cpp
	mex $(MEX_OPTS)  -c cudaFluidStep.cpp
cudaFluidStep: cudaCommon.o cudaFluidStep.o
	mex $(MEX_OPTS) $(PAR_LDIR) $(PAR_LIBS) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)


cudaMHDKernels.o: cudaMHDKernels.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaMHDKernels.cu -o cudaMHDKernels.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaMHDKernels.cpp
cudaMHDKernels: cudaCommon.o cudaMHDKernels.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaFwdAverage.o: cudaFwdAverage.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaFwdAverage.cu -o cudaFwdAverage.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaFwdAverage.cpp
cudaFwdAverage: cudaCommon.o cudaFwdAverage.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaFwdDifference.o: cudaFwdDifference.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaFwdDifference.cu -o cudaFwdDifference.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaFwdDifference.cpp
cudaFwdDifference: cudaCommon.o cudaFwdDifference.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaMagPrep.o: cudaMagPrep.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaMagPrep.cu -o cudaMagPrep.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaMagPrep.cpp
cudaMagPrep: cudaCommon.o cudaMagPrep.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaMagW.o: cudaMagW.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaMagW.cu -o cudaMagW.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaMagW.cpp
cudaMagW: cudaCommon.o cudaMagW.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaMagTVD.o: cudaMagTVD.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaMagTVD.cu -o cudaMagTVD.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaMagTVD.cpp
cudaMagTVD: cudaCommon.o cudaMagTVD.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaSoundspeed.o: cudaSoundspeed.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaSoundspeed.cu -o cudaSoundspeed.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaSoundspeed.cpp
cudaSoundspeed: cudaCommon.o cudaSoundspeed.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

# SOURCE/SINK ROUTINES
cudaNonconservativeTerms.o: cudaNonconservativeTerms.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaNonconservativeTerms.cu -o cudaNonconservativeTerms.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaNonconservativeTerms.cpp
cudaNonconservativeTerms: cudaCommon.o cudaNonconservativeTerms.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaSourceConstantG.o: cudaSourceConstantG.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaSourceConstantG.cu -o cudaSourceConstantG.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaSourceConstantG.cpp
cudaSourceConstantG: cudaCommon.o cudaSourceConstantG.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaSourceScalarPotential.o: cudaSourceScalarPotential.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaSourceScalarPotential.cu -o cudaSourceScalarPotential.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaSourceScalarPotential.cpp
cudaSourceScalarPotential: cudaCommon.o cudaSourceScalarPotential.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaSourceRotatingFrame.o: cudaSourceRotatingFrame.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaSourceRotatingFrame.cu -o cudaSourceRotatingFrame.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaSourceRotatingFrame.cpp
cudaSourceRotatingFrame: cudaCommon.o cudaSourceRotatingFrame.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaFreeRadiation.o: cudaFreeRadiation.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaFreeRadiation.cu -o cudaFreeRadiation.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaFreeRadiation.cpp
cudaFreeRadiation: cudaCommon.o cudaFreeRadiation.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)

cudaAccretingStar.o: cudaAccretingStar.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) $(PAR_INCL) -cuda cudaAccretingStar.cu -o cudaAccretingStar.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_INCL) -c cudaAccretingStar.cpp
cudaAccretingStar: cudaCommon.o cudaAccretingStar.o
	mex $(MEX_OPTS)  $(PAR_LDIR) $(PAR_LIBS) $(MEX_LDIR) $(MEX_LIBS)  -cxx $@.o cudaCommon.o -o $@

cudaSourceAntimach.o: cudaSourceAntimach.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) -cuda cudaSourceAntimach.cu -o cudaSourceAntimach.cpp
	mex $(MEX_OPTS) $(MEX_INCL) -c cudaSourceAntimach.cpp
cudaSourceAntimach: cudaCommon.o cudaSourceAntimach.o
	mex $(MEX_OPTS) $(MEX_INCL) -cxx $@.o cudaCommon.o -o $@ $(MEX_LDIR) $(MEX_LIBS)



# PARALLEL OPERATION ROUTINES **********
cudaHaloExchange.o: cudaHaloExchange.cu
	nvcc $(NVCC_OPTS) $(NVCC_INCL) $(PAR_INCL) -cuda cudaHaloExchange.cu -o cudaHaloExchange.cpp
	mex $(MEX_OPTS) $(MEX_INCL) $(PAR_INCL) -c cudaHaloExchange.cpp
cudaHaloExchange: cudaCommon.o cudaHaloExchange.o
	mex $(MEX_OPTS) $(PAR_LDIR) $(PAR_LIBS) $(MEX_LDIR) $(MEX_LIBS) -cxx $@.o cudaCommon.o -o $@


clean:
	rm -f *mexa64 *cpp *o

tidy:
	rm -f *cpp *o
