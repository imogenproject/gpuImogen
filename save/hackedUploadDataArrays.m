function [fluid, mag] = uploadDataArrays_hacked(FieldSource, run, statics)
% This is a rekt version of the uploadDataArrays function that dumps the
%CPU array into the place the ImogenArray class looks for the gpu array,
% It's only for quickly making imogen vomit out test frames

    SaveManager.logPrint('---------- uploadDataArraysHacked in progress....');

    gm = GPUManager.getInstance();

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
        SaveManager.logPrint('Fluid %i: ', int32(F));
        fluid(F) = FluidManager();
        % HACK HACK HACK this should be in some other init place
        fluid(F).MINMASS        = FieldSource.ini.minMass;
        fluid(F).MASS_THRESHOLD = FieldSource.ini.thresholdMass;
        fluid(F).parent         = run;

        FluidData = FieldSource.fluids(F);

        SaveManager.logPrint('rho; ');
        mass = FluidArray(ENUM.SCALAR, ENUM.MASS, FluidData.mass, fluid(F), statics, 'fake');
        ener = FluidArray(ENUM.SCALAR, ENUM.ENER, FluidData.ener, fluid(F), statics, 'fake');

        mom  = FluidArray.empty(3,0);
        fieldnames = {'momX','momY','momZ'};
        for i = 1:3
            mom(i) = FluidArray(ENUM.VECTOR(i), ENUM.MOM, FluidData.(fieldnames{i}), fluid(F), statics, 'fake');
        end

        SaveManager.logPrint('Processing thermodynamic details; ');
        fluid(F).processFluidDetails(FluidData.details);
        if fluid(F).checkCFL; hasNoCFL = 0; end
        fluid(F).attachFluid([], mass, ener, mom);

        SaveManager.logPrint('\n');
    end

    % Try to make text appear in the right order in the logs
    mpi_barrier();
    pause(.05);


end
