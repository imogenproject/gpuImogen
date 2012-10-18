%function imogen(massDen, momDen, enerDen, magnet, ini, statics)
function imogen(icfile)
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
    system(['rm -f ' icfile ]);

    %--- Parse initial parameters from ini input ---%
    %       The initialize function parses the ini structure input and populates all of the manager
    %       classes with the values. From these values the initializeResultsPaths function 
    %       establishes all of the save directories for the run, creating whatever directories are
    %       needed in the process. 
    run = initialize(ini);
    initializeResultPaths(run);
    run.save.saveIniSettings(ini);
    run.preliminary();

    mpi_barrier();
    run.save.logPrint('Creating simulation arrays...\n');
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

    run.selfGravity.initialize(IC.selfGravity, mass);
    run.potentialField.initialize(IC.potentialField);
    %--- Pre-loop actions ---%
    clear('IC', 'ini', 'statics');    
    run.initialize(mass, mom, ener, mag);
   
    mpi_barrier();
    run.save.logPrint('Running initial save...\n');

    resultsHandler(run, mass, mom, ener, mag);
    run.time.iteration  = 1;
    direction           = [1 -1];

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
    %error('development: error to prevent matlab exiting at end-of-run')

    run.postliminary();
end
