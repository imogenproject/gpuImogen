function outdirectory = imogen(srcData, resumeinfo)
% This is the main entry point for the Imogen MHD code. It contains the primary evolution loop and 
% the hooks for writing the results to disk.
%
%>> IC.massDen     Mass density array (cell-centered).                         double  [nx ny nz]
%>> IC.momDen      Momentum density array (cell-centered).                     double  [3 nx ny nz]
%>> IC.enerDen     Energy density array (cell-centered).                       double  [nx ny nz]
%>> IC.magnet      Magnetic field strength array (face-centered).              double  [3 nx ny nz]
%>> IC.ini         Listing of properties and settings for the run.             struct
%>> IC.statics     Static arrays with lookup to static values.                 struct
%<< outdirectory   Path to directory containing results                        string
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

    %--- Parse initial parameters from ini input ---%
    %       The initialize function parses the ini structure input and populates all of the manager
    %       classes with the values. From these values the initializeResultsPaths function 
    %       establishes all of the save directories for the run, creating whatever directories are
    %       needed in the process.
    run = initialize(ini);

    if isfield(IC, 'amResuming'); RESTARTING = true; else; RESTARTING = false; end

    % Behavior depends if IC.originalPathStruct exists
    initializeResultPaths(run, IC)

    outdirectory = run.paths.save;
    run.save.saveIniSettings(ini);

    mpi_barrier();
    run.save.logPrint('---------- Transferring arrays to GPU(s)\n');

    GIS = GlobalIndexSemantics();

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

        % (4) Recall and give a somewhat late init to indexing semantics.
        GIS.setup(dframe.parallel.globalDims);

        cd(origpath); clear origpath;
        FieldSource = dframe;
    else
        FieldSource = IC;
    end

    try
        [mass ener mom mag DataHolder] = uploadDataArrays(FieldSource, run, statics);
    catch oops
        run.save.logAllPrint('    FATAL: Unsuccessful uploading data arrays!\nAborting run...\n');
        rethrow(oops);
    end

    mpi_barrier();
    run.save.logPrint('---------- Preparing physics subsystems\n');

    writeSimInitializer(run, IC);

    %--- Pre-loop actions ---%
    run.initialize(IC, mass, mom, ener, mag);

    clear('IC', 'ini', 'statics');    

    mpi_barrier();
    run.save.logPrint('---------- Entering simulation loop\n');

    if ~RESTARTING
        run.save.logPrint('New simulation: Doing initial save... ');
        resultsHandler([], run, mass, ener, mom, mag);
        run.save.logPrint('Succeeded.\n');
    else
        run.save.logPrint('Simulation resuming after iteration %i\n',run.time.iteration);
    end

    direction           = [1 -1];
    run.time.recordWallclock();

    %%%=== MAIN ITERATION LOOP ==================================================================%%%
    while run.time.running
        run.time.update(mass, mom, ener, mag, 1);

        fluidstep(mass, ener, mom(1), mom(2), mom(3), mag(1).cellMag, mag(2).cellMag, mag(3).cellMag, [run.time.dTime 1 run.GAMMA 1 run.time.iteration run.cfdMethod], GIS.topology, run.DGRID);
%        flux(run, mass, mom, ener, mag, 1);
        source(run, mass, mom, ener, mag, 1.0);
        fluidstep(mass, ener, mom(1), mom(2), mom(3), mag(1).cellMag, mag(2).cellMag, mag(3).cellMag, [run.time.dTime 1 run.GAMMA -1 run.time.iteration run.cfdMethod], GIS.topology, run.DGRID);

        run.time.step();
        run.pollEventList(mass, ener, mom, mag);
    end
    %%%=== END MAIN LOOP ========================================================================%%%

% FIXME: This is a terrible hack.
if mpi_amirank0() && numel(run.selfGravity.compactObjects) > 0
  starpath = sprintf('%s/starpath.mat',run.paths.save);
  stardata = run.selfGravity.compactObjects{1}.history;
  save(starpath,'stardata');
end

    run.finalize(mass, ener, mom, mag);

end
