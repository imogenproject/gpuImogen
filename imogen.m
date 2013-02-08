%function imogen(massDen, momDen, enerDen, magnet, ini, statics)
function imogen(icfile, resumeinfo)
% This is the main entry point for the Imogen MHD code. It contains the primary evolution loop and 
% the hooks for writing the results to disk.
%
%>> massDen     Mass density array (cell-centered).                         double  [nx ny nz]
%>> momDen      Momentum density array (cell-centered).                     double  [3 nx ny nz]
%>> enerDen     Energy density array (cell-centered).                       double  [nx ny nz]
%>> magnet      Magnetic field strength array (face-centered).              double  [3 nx ny nz]
%>> ini         Listing of properties and settings for the run.             struct
%>> statics     Static arrays with lookup to static values.                 struct
    load(icfile)
    ini     = IC.ini;
    statics = IC.statics;
    
    % Recover memory and disk used to store ICs
%    system(['rm -f ' icfile ]);

    %--- Parse initial parameters from ini input ---%
    %       The initialize function parses the ini structure input and populates all of the manager
    %       classes with the values. From these values the initializeResultsPaths function 
    %       establishes all of the save directories for the run, creating whatever directories are
    %       needed in the process. 
    run = initialize(ini);

    if isfield(IC, 'amResuming'); RESTARTING = true; else; RESTARTING = false; end

    if RESTARTING
        % Recreates our original paths exactly w/o changing
        initializeResultPaths(run, IC.originalPathStruct);
    else
        % Generate unique new paths
        initializeResultPaths(run);
    end

    run.save.saveIniSettings(ini);
    run.preliminary();

    mpi_barrier();
    run.save.logPrint('Creating simulation arrays...\n');

    if RESTARTING
        % If reloading from time-evolved point,
        % garner important values from saved files:
        % (1) save paths [above]
        % (2) Q(x,t0) from saved files
        origpath=pwd(); cd(run.paths.save);
        dframe = util_LoadFrameSegment('3D_XYZ',run.paths.indexPadding, mpi_myrank(), resumeinfo.frame);
        % (3) serialized time history, from saved data files, except for newly adultered time limits.
        run.time.resumeFromSavedTime(dframe.time, resumeinfo);

        % (4) Recall and give a somewhat late init to indexing semantics.
        GIS = GlobalIndexSemantics(); GIS.setup(dframe.parallel.globalDims);

        cd(origpath); clear origpath;

        mass = FluidArray(ENUM.SCALAR, ENUM.MASS, dframe.mass, run, statics);
        ener = FluidArray(ENUM.SCALAR, ENUM.ENER, dframe.ener, run, statics);
        mom  = FluidArray.empty(3,0); mag  = MagnetArray.empty(3,0);
        fieldnames = {'momX','momY','momZ','magX','magY','magZ'};

        for i = 1:3;
            mom(i) = FluidArray(ENUM.VECTOR(i), ENUM.MOM, getfield(dframe, fieldnames{i}), run, statics);
            if run.pureHydro == 0
                mag(i) = MagnetArray(ENUM.VECTOR(i), ENUM.MAG, getfield(dframe,fieldnames{i}), run, statics);
            else
                mag(i) = MagnetArray(ENUM.VECTOR(i), ENUM.MAG, [], run, statics);
            end
        end

     else % IC contains Q(x,t0)
        mass = FluidArray(ENUM.SCALAR, ENUM.MASS, IC.mass, run, statics);
        ener = FluidArray(ENUM.SCALAR, ENUM.ENER, IC.ener, run, statics);
        mom  = FluidArray.empty(3,0);
        mag  = MagnetArray.empty(3,0);
        for i=1:3
            mom(i) = FluidArray(ENUM.VECTOR(i), ENUM.MOM, IC.mom(i,:,:,:), run, statics); 
            if run.pureHydro == 0
                mag(i) = MagnetArray(ENUM.VECTOR(i), ENUM.MAG, IC.magnet(i,:,:,:), run, statics);
            else
                mag(i) = MagnetArray(ENUM.VECTOR(i), ENUM.MAG, [], run, statics);
            end
        end
    end    
%IC.selfGravity.compactObjectStates
    run.selfGravity.initialize(IC.selfGravity, mass);
%run.selfGravity.compactObjects
    run.potentialField.initialize(IC.potentialField);
    %--- Store everything but Q(x,t0) in a new IC file in the save directory ---%
    IC.mass = []; IC.ener = [];
    IC.mom = [];  IC.mag  = [];
    IC.amResuming = 1;
    IC.originalPathStruct = run.paths.serialize();

    GIS = GlobalIndexSemantics();
    save(sprintf('%s/SimInitializer_rank%i.mat',run.paths.save,GIS.context.rank),'IC');

    %--- Pre-loop actions ---%
    clear('IC', 'ini', 'statics');    
    run.initialize(mass, mom, ener, mag);
   
    mpi_barrier();
    if ~RESTARTING
        run.save.logPrint('Running initial save...\n');
        resultsHandler(run, mass, mom, ener, mag);
        run.time.iteration  = 1;
    else
        run.save.logPrint(sprintf('Simulation resuming at iteration %i.\n',run.time.iteration));
    end
    direction           = [1 -1];


%starState = GIS.domainResolution*run.DGRID{1}/2;
%starState(4) = .1;
%starState(5:10) = 0;
%starState(11) = 1;
%starState(12) = run.fluid.MINMASS;
%starState(13) = 5*run.fluid.MINMASS;
%starState(14) = run.fluid.MINMASS*.02*3/10
%run.selfGravity.addCompactObject(starState);


    run.save.logPrint('Beginning simulation loop...\n');
    clockA = clock;
    %%%=== MAIN ITERATION LOOP ==================================================================%%%
    while run.time.running
        run.time.update(mass, mom, ener, mag, i);

        for i=1:2 % Two timesteps per iteration, forward & backward
            flux(run, mass, mom, ener, mag, direction(i));
            treadmillGrid(run, mass, mom, ener, mag);
            if i == 1; source(run, mass, mom, ener, mag); end
        end

        %--- Intermediate file saves ---%
        resultsHandler(run, mass, mom, ener, mag);
        run.time.step();
    end
    %%%=== END MAIN LOOP ========================================================================%%%
    run.save.logPrint(sprintf('%gh %gs in main sim loop\n', floor(etime(clock, clockA)/3600), ...
                                     etime(clock, clockA)-3600*floor(etime(clock, clockA)/3600) ));
%    error('development: error to prevent matlab exiting at end-of-run')

starpath = sprintf('%s/starpath.mat',run.paths.save);
save(starpath,run.selfGravity.compactObjects{1}.history);

    run.postliminary();
end
