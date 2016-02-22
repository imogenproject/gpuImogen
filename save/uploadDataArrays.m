function [mass ener mom mag DataHolder] = uploadDataArrays(FieldSource, run, statics)
% Utility function uploads input data arrays on CPU to GPU

    gm = GPUManager.getInstance();
    iniGPUMem = GPU_ctrl('memory'); iniGPUMem = iniGPUMem(gm.deviceList+1,1);

    memtot = sum(iniGPUMem);
    memneed = numel(FieldSource.mass) * 11 * 8;
    if memneed / memtot > .9
        run.save.logAllPrint('WARNING: Projected GPU memory utilization of %.1g\% exceeds 90% of total device memory.\n', 100*memneed/memtot);
        run.save.logAllPrint('WARNING: Reduction in simulation size or increase in job size may be required.\n');
    end

    DataHolder = GPU_Type(FieldSource.mass);
    DataHolder.createSlabs(5); % [rho E px py pz] slabs

    a = GPU_getslab(DataHolder, 0);

    mass = FluidArray(ENUM.SCALAR, ENUM.MASS, a, run, statics);

    a = GPU_setslab(DataHolder, 1, FieldSource.ener);
    ener = FluidArray(ENUM.SCALAR, ENUM.ENER, a, run, statics);

    mom  = FluidArray.empty(3,0);
    mag  = MagnetArray.empty(3,0);
    fieldnames = {'momX','momY','momZ','magX','magY','magZ'};

    for i = 1:3;
        a = GPU_setslab(DataHolder, 1+i, getfield(FieldSource, fieldnames{i}) );
        mom(i) = FluidArray(ENUM.VECTOR(i), ENUM.MOM, a, run, statics);
        if run.pureHydro == 0
            mag(i) = MagnetArray(ENUM.VECTOR(i), ENUM.MAG, getfield(FieldSource, fieldnames{i+3}), run, statics);
        else
            mag(i) = MagnetArray(ENUM.VECTOR(i), ENUM.MAG, [], run, statics);
        end
     end

    nowGPUMem = GPU_ctrl('memory'); usedGPUMem = sum(iniGPUMem-nowGPUMem(gm.deviceList+1,1))/1048576;
    asize = mass.gridSize();

    run.save.logAllPrint('rank %i: %06.3fMB used by fluid state arrays of size [%i %i %i] partitioned on %i GPUs\n', mpi_myrank(), usedGPUMem, asize(1), asize(2), asize(3), int32(numel(gm.deviceList)) );

end
