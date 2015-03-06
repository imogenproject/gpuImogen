%function imogen(massDen, momDen, enerDen, magnet, ini, statics)
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
        if mpi_amirank0(); disp('Resuming from file'); end
    else
        IC = srcData;
        clear srcData;
        evalin('caller','clear IC'); % Make ML release the memory used above
        if mpi_amirank0(); disp('Starting for first time'); end
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
    run.preliminary();

    mpi_barrier();
    run.save.logPrint('Creating simulation arrays...\n');
%    mf0 = GPU_memavail();

    if RESTARTING
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
        GIS = GlobalIndexSemantics(); GIS.setup(dframe.parallel.globalDims);

        cd(origpath); clear origpath;
	FieldSource = dframe;
    else
	FieldSource = IC;
    end

    DataHolder = GPU_Type(FieldSource.mass); % This is a sort of dumb way of shoehorning the data in but it works
    DataHolder.createSlabs(5); % [rho E px py pz] slabs

    a = GPU_getslab(DataHolder, 0); % access mass
    mass = FluidArray(ENUM.SCALAR, ENUM.MASS, a, run, statics);
    a = GPU_getslab(DataHolder, 1); % access E slab
    b = GPU_Type(FieldSource.ener);
    GPU_copy(a, b);
    ener = FluidArray(ENUM.SCALAR, ENUM.ENER, a, run, statics);

    mom  = FluidArray.empty(3,0);
    mag  = MagnetArray.empty(3,0);
    fieldnames = {'momX','momY','momZ','magX','magY','magZ'};

    for i = 1:3;
	a = GPU_getslab(DataHolder, 1+i);
	b.array = getfield(FieldSource, fieldnames{i});
	GPU_copy(a, b);
        mom(i) = FluidArray(ENUM.VECTOR(i), ENUM.MOM, a, run, statics);
        if run.pureHydro == 0
            mag(i) = MagnetArray(ENUM.VECTOR(i), ENUM.MAG, getfield(FieldSource, fieldnames{i+3}), run, statics);
        else
            mag(i) = MagnetArray(ENUM.VECTOR(i), ENUM.MAG, [], run, statics);
        end
     end

    clear b; 

%    mf1 = GPU_memavail();
    asize = mass.gridSize();
% FIXME: re-do this but since we use multiple GPUs now it's not so trivial
%    run.save.logAllPrint(sprintf('rank %i: GPU reports %06.3fMB used by fluid state arrays\narray dimensions: [%i %i %i]\n', mpi_myrank(), (mf0-mf1)/1048576,asize(1), asize(2), asize(3) ) );

    run.selfGravity.initialize(IC.selfGravity, mass);
    run.potentialField.initialize(IC.potentialField);

    %--- Store everything but Q(x,t0) in a new IC file in the save directory ---%
    IC.mass = []; IC.ener   = [];
    IC.mom = [];  IC.magnet = [];
    IC.amResuming = 1;
    IC.originalPathStruct = run.paths.serialize();

    GIS = GlobalIndexSemantics();
    if run.save.FSAVE; save(sprintf('%s/SimInitializer_rank%i.mat',run.paths.save,GIS.context.rank),'IC'); end

    doInSitu = ini.useInSituAnalysis;
    if doInSitu; inSituAnalyzer = ini.inSituHandle(run, mass, mom, ener, mag); inSituSteps = ini.stepsPerInSitu; end

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

    run.save.logPrint('Beginning simulation loop...\n');
    clockA = clock;

    %%%=== MAIN ITERATION LOOP ==================================================================%%%
    while run.time.running
        run.time.update(mass, mom, ener, mag, i);
        for i=1:2 % Two timesteps per iteration, forward & backward
            flux(run, mass, mom, ener, mag, direction(i));
            treadmillGrid(run, mass, mom, ener, mag);
            if i == 1; source(run, mass, mom, ener, mag, 1); end
        end

        %--- Intermediate file saves ---%
        resultsHandler(run, mass, mom, ener, mag);

        %--- Analysis done as simulation runs ---%
	if doInSitu && (mod(run.time.iteration, inSituSteps) == 0);
	    inSituAnalyzer.FrameAnalyzer(run, mass, mom, ener, mag);
	end

        run.time.step();
    end
    %%%=== END MAIN LOOP ========================================================================%%%
    run.save.logPrint(sprintf('%gh %gs in main sim loop\n', floor(etime(clock, clockA)/3600), ...
                                     etime(clock, clockA)-3600*floor(etime(clock, clockA)/3600) ));
%    error('development: error to prevent matlab exiting at end-of-run')

if doInSitu; inSituAnalyzer.finish(run); end

% This is a bit hackish but it works
if mpi_amirank0() && numel(run.selfGravity.compactObjects) > 0
  starpath = sprintf('%s/starpath.mat',run.paths.save);
  stardata = run.selfGravity.compactObjects{1}.history;
  save(starpath,'stardata');
end

    % Delete GPU arrays to make Matlab happy
    % Though I think mexlocking the master file to keep it from losing its context
    % solved the actual problem...
    mass.cleanup(); ener.cleanup();
    for i = 1:3; mom(i).cleanup(); end
    if run.pureHydro == 0;
        for i = 1:3; mag(i).array = 1; end;
    end

    run.postliminary();
end
