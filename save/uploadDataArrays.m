function [fluid, mag] = uploadDataArrays(FieldSource, run, statics, upscaleFactor)
% Utility function uploads input data arrays on CPU to GPU

    SaveManager.logPrint('---------- Transferring arrays to GPU(s)\n');

    gm = GPUManager.getInstance();
    pm = ParallelGlobals();
    
    % Automatically check and fix partitioning.
    if numel(gm.deviceList) > 1
        % FIXME: this also depends on MPI partitioning
        rez = obj.geomgr.globalDomainRez;
        
        if rez(gm.partitionDir) < 6
            SaveManager.logPrint('    NOTICE: Partition direction had to be changed due to incompatibility with domain resolution.\n');
            % Z partition? Try y then x
            if gm.partitionDir == 3
                if rez(2) < 6; gm.partitionDir = 1; else; gm.partitionDir = 2; end
            else
                % Y partition? Try z then x
                if rez(3) > 6; gm.partitionDir = 3; else; gm.partitionDir = 1; end
            end
        else
            if gm.partitionDir == 1; pdc = 'X'; end
            if gm.partitionDir == 2; pdc = 'Y'; end
            if gm.partitionDir == 3; pdc = 'Z'; end
            SaveManager.logPrint('    NOTICE: Checked partition direction (%c) & was OK.\n', pdc);
        end
    end
    
    % The geometric settings and partitioning are now known and it is now time to set the
    % useExternalHalo flag. The Matlab/MPI level partition handler will not add array-edge halos in
    % a direction with only one rank (e.g. single node), but if there are multiple GPUs and
    % partitioning is in that direction, and we are doing circular differencing, a halo at the
    % outer edges is required.
    %
    % The useExteriorHalo flag does this. In all other cases this is not necessary and is zero.
    bcmodes = run.bc.expandBCStruct(run.bc.modes{1});
    bcIsCircular = strcmp(bcmodes{1,gm.partitionDir}, 'circ');
    
    if (numel(gm.deviceList) > 1) && (pm.topology.nproc(gm.partitionDir) == 1) && bcIsCircular
        extHalo = 1;
    else
        extHalo = 0;
    end
    
    gm.useExteriorHalo = extHalo;
    
    iniGPUMem = GPU_ctrl('memory'); iniGPUMem = iniGPUMem(gm.deviceList+1,1);

    % create two sets of cuda event streams
    % This is currently only needed to acheive concurrency in the array rotater code...
    streams = GPU_ctrl('createStreams', gm.deviceList);
    streams = [streams; GPU_ctrl('createStreams', gm.deviceList)];

    memtot = sum(iniGPUMem);
    memneed = numel(FieldSource.fluids(1).mass) * 8 * (5*(numel(FieldSource.fluids)+1)+1);
    mpi_barrier(); % Keep anybody from drinking up available memory before we check

    if memneed / memtot > .9
        run.save.logAllPrint('WARNING: Projected GPU memory utilization of %.1f%c exceeds 9/10 of total device memory.\n', 100*memneed/memtot, 37);
        run.save.logAllPrint('WARNING: Reduction in simulation size or increase in #of GPUs/nodes may be required.\n');
    end

    % Handle magnetic field
    mag  = MagnetArray.empty(3,0);
    fieldnames={'magX','magY','magZ'};
    for i = 1:3
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
    %for F = 1
        SaveManager.logPrint('    Fluid %i: ', int32(F));
        fluid(F) = FluidManager(F);

        % HACK HACK HACK this should be in some other init place
        fluid(F).MINMASS        = FieldSource.ini.minMass;
        fluid(F).MASS_THRESHOLD = FieldSource.ini.thresholdMass;
        fluid(F).parent         = run;

        FluidData = FieldSource.fluids(F);

        DataHolder = GPU_Type(FluidData.mass);
        DataHolder.createSlabs(5);

        a = GPU_getslab(DataHolder, 0);
        SaveManager.logPrint('rho; ');
        mass = FluidArray(ENUM.SCALAR, ENUM.MASS, a, fluid(F), statics);

        a = GPU_setslab(DataHolder, 1, FluidData.ener);
        SaveManager.logPrint('ener; ');
        ener = FluidArray(ENUM.SCALAR, ENUM.ENER, a, fluid(F), statics);

        mom  = FluidArray.empty(3,0);
        fieldnames = {'momX','momY','momZ'};

        for i = 1:3
            a = GPU_setslab(DataHolder, 1+i, FluidData.(fieldnames{i}));
            SaveManager.logPrint('%s; ',fieldnames{i});
            mom(i) = FluidArray(ENUM.VECTOR(i), ENUM.MOM, a, fluid(F), statics);
        end

        SaveManager.logPrint('Processing thermodynamic details; ');
        % Garbage hack
        if ~isfield(FluidData, 'details')
	    run.Save.logPrint('    -->>> WARNING <<<-- This SimInitializer.mat dumped the original FluidData(%i).details struct: replacing with warm_molecular_hydrogen!!!!\n',F);
	    FluidData.details = fluidDetailModel('warm_molecular_hydrogen');
        end

        fluid(F).processFluidDetails(FluidData.details);
        if fluid(F).checkCFL; hasNoCFL = 0; end
        fluid(F).attachFluid(DataHolder, mass, ener, mom);
        fluid(F).attachStreams(streams);

        SaveManager.logPrint('\n');
    end

    if hasNoCFL
        error('Fatal error: ALL fluids are marked to not have CFL checked!!!');
    end

    nowGPUMem = GPU_ctrl('memory'); usedGPUMem = sum(iniGPUMem-nowGPUMem(gm.deviceList+1,1))/1048576;
    asize = mass.gridSize();

    % Try to make text appear in the right order in the logs
    mpi_barrier();
    pause(.05);

    run.save.logAllPrint('%06.3fMB used by fluid state arrays of size [%i %i %i] partitioned on %i GPUs\n', usedGPUMem, asize(1), asize(2), asize(3), int32(numel(gm.deviceList)) );

end
