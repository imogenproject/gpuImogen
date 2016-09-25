function outdirectory = imogen(srcData, resumeinfo)
% This is the main entry point for the Imogen MHD code. It contains the primary evolution loop and 
% the hooks for writing the results to disk.
%
%> srcData         Either filename from Initializer.saveInitialCondsToFile
%                  or structure from Initializer.saveInitialCondsToStructure
%> resumeinfo      If present, structure as in run/special_Resume.m
%< outdirectory    Path to directory containing output data

    if isstruct(srcData) == 0
        load(srcData);
        if mpi_amirank0(); fprintf('---------- Imogen starting from file'); end
    else
        IC = srcData;
        clear srcData;
        evalin('caller','clear IC'); % Make ML release the memory used above
        if mpi_amirank0(); fprintf('---------- Imogen starting from passed IC structure'); end
    end

    ini     = IC.ini;
    statics = IC.statics;

    collectiveFailure = 0;

    %--- Parse initial parameters from ini input ---%
    %       The initialize function parses the ini structure input and populates all of the manager
    %       classes with the values. From these values the initializeResultsPaths function 
    %       establishes all of the save directories for the run, creating whatever directories are
    %       needed in the process.
    run = initialize(ini);

    if isfield(IC, 'amResuming'); RESTARTING = true; else RESTARTING = false; end

    % Behavior depends if IC.originalPathStruct exists
    initializeResultPaths(run, IC)

    outdirectory = run.paths.save;
    run.save.saveIniSettings(ini);

    mpi_barrier();
    run.save.logPrint('---------- Transferring arrays to GPU(s)\n');

    if RESTARTING
        run.save.logPrint('   Accessing restart data files\n');
        % If reloading from time-evolved point,
        % garner important values from saved files:
        % (1) save paths [above]
        % (2) Q(x,t0) from saved files
        % WARNING - this really, really needs to _know_ which frame type to load
        origpath=pwd(); cd(run.paths.save);
        dframe = util_LoadFrameSegment('2D_XY',run.paths.indexPadding, mpi_myrank(), resumeinfo.frame);
        % (3) serialized time history, from saved data files, except for newly adultered time limits.
        run.time.resumeFromSavedTime(dframe.time, resumeinfo);
        run.image.frame = resumeinfo.imgframe;

        cd(origpath); clear origpath;
        FieldSource = dframe;
    else
        FieldSource = IC;
    end

    try
        [run.fluid, mag] = uploadDataArrays(FieldSource, run, statics);
    catch oops
        prettyprintException(oops, 0, ...
            '    FATAL: Unsuccessful uploading data arrays!\nAborting run...\nAdditional execption will be printed by loader.\n');
        collectiveFailure = 1;
    end
    mpi_errortest(collectiveFailure);

    run.save.logPrint('---------- Preparing physics subsystems\n');

    writeSimInitializer(run, IC);
    %--- Pre-loop actions ---%
    run.initialize(IC, mag);

    clear('IC', 'ini', 'statics');    
    mpi_barrier();
    run.save.logPrint('---------- Entering simulation loop\n');

    if ~RESTARTING
        run.save.logPrint('New simulation: Doing initial save... ');
        try
            resultsHandler([], run, run.fluid, mag);
        catch booboo
            prettyprintException(booboo, 0, ...
                '    FATAL: First resultsHandler() call failed! Data likely unaccessible. Aborting run.\n');
            collectiveFailure = 1;
        end
        mpi_errortest(collectiveFailure); % All ranks will error() if any failed, & make loader abort

        run.save.logPrint('Succeeded.\n');
    else
        run.save.logPrint('Simulation resuming after iteration %i\n',run.time.iteration);
    end

    run.time.recordWallclock();
    backupData = dumpCheckpoint(run);
                               
    %%%=== MAIN ITERATION LOOP ==================================================================%%%
    while run.time.running
        run.time.update(run.fluid, mag);
        %if mod(run.time.iteration, 25) == 24
        %    backupData = dumpCheckpoint(run);
        %end

        fluidstep(run.fluid, mag(1).cellMag, mag(2).cellMag, mag(3).cellMag, [run.time.dTime 1  1 run.time.iteration run.cfdMethod], run.geometry);
        %flux(run, run.fluid, mag, 1);
        source(run, run.fluid, mag, 1);
        fluidstep(run.fluid, mag(1).cellMag, mag(2).cellMag, mag(3).cellMag, [run.time.dTime 1 -1 run.time.iteration run.cfdMethod], run.geometry);
        %flux(run, run.fluid, mag, -1);

        if checkPhysicality(run.fluid)
            restoreCheckpoint(run, backupData);
        end
        run.time.step();
        run.pollEventList(run.fluid, mag);
    end
    %%%=== END MAIN LOOP ========================================================================%%%

% FIXME: This is a terrible hack.
if mpi_amirank0() && numel(run.selfGravity.compactObjects) > 0
  starpath = sprintf('%s/starpath.mat',run.paths.save);
  stardata = run.selfGravity.compactObjects{1}.history;
  save(starpath,'stardata');
end

    run.finalize(run.fluid, mag);

end
