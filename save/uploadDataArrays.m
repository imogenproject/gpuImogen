function [fluid, mag] = uploadDataArrays(FieldSource, run, statics)
% Utility function uploads input data arrays on CPU to GPU

    gm = GPUManager.getInstance();
    iniGPUMem = GPU_ctrl('memory'); iniGPUMem = iniGPUMem(gm.deviceList+1,1);

    memtot = sum(iniGPUMem);
    memneed = numel(FieldSource.fluids(1).mass) * 11 * 8 * numel(FieldSource.fluids);
    if memneed / memtot > .9
        run.save.logAllPrint('WARNING: Projected GPU memory utilization of %.1f%c exceeds 9/10 of total device memory.\n', 100*memneed/memtot, 37);
        run.save.logAllPrint('WARNING: Reduction in simulation size or increase in #of GPUs/nodes may be required.\n');
    end

    % Handle magnetic field
    mag  = MagnetArray.empty(3,0);
    fieldnames={'magX','magY','magZ'};
    for i = 1:3;
        if run.pureHydro == 0
            mag(i) = MagnetArray(ENUM.VECTOR(i), ENUM.MAG, FieldSource.(fieldnames{i}), run.magnet, statics);
        else
            mag(i) = MagnetArray(ENUM.VECTOR(i), ENUM.MAG, [], run.magnet, statics);
        end
    end

    fluid = FluidManager.empty(numel(FieldSource.fluids), 0);

    hasNoCFL = 1;

    % Handle each fluid
    for F = 1:numel(FieldSource.fluids)
        fluid(F) = FluidManager();
        % HACK HACK HACK this should be in some other init place
        fluid(F).MASS_THRESHOLD = FieldSource.ini.thresholdMass;
        fluid(F).MINMASS        = FieldSource.ini.minMass;
        fluid(F).parent         = run;

        FluidData = FieldSource.fluids(F);

        DataHolder = GPU_Type(FluidData.mass);
        DataHolder.createSlabs(5);

        a = GPU_getslab(DataHolder, 0);
        mass = FluidArray(ENUM.SCALAR, ENUM.MASS, a, fluid(F), statics);

        a = GPU_setslab(DataHolder, 1, FluidData.ener);
        ener = FluidArray(ENUM.SCALAR, ENUM.ENER, a, fluid(F), statics);

        mom  = FluidArray.empty(3,0);
        fieldnames = {'momX','momY','momZ'};

        for i = 1:3;
            a = GPU_setslab(DataHolder, 1+i, getfield(FluidData, fieldnames{i}) );
            mom(i) = FluidArray(ENUM.VECTOR(i), ENUM.MOM, a, fluid(F), statics);
        end

        fluid(F).processFluidDetails(FluidData.details);
        if fluid(F).checkCFL; hasNoCFL = 0; end
        fluid(F).attachFluid(DataHolder, mass, ener, mom);
    end

    if hasNoCFL;
        error('Fatal error: ALL fluids are marked to not have CFL checked!!!');
    end

    nowGPUMem = GPU_ctrl('memory'); usedGPUMem = sum(iniGPUMem-nowGPUMem(gm.deviceList+1,1))/1048576;
    asize = mass.gridSize();

    run.save.logAllPrint('rank %i: %06.3fMB used by fluid state arrays of size [%i %i %i] partitioned on %i GPUs\n', mpi_myrank(), usedGPUMem, asize(1), asize(2), asize(3), int32(numel(gm.deviceList)) );

end
