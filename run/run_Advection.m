
grid = [256 256 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run             = AdvectionInitializer(grid);
run.iterMax     = 999999;
run.info        = 'Advection test.';
run.notes       = 'Simple advection test in the x-direction.';

%run.image.interval = 100;
%run.image.mass = true;

run.activeSlices.x = false;
run.activeSlices.xy = false;
run.activeSlices.xyz = false;

run.ppSave.dim1 = 100;
run.ppSave.dim2 = 100;

% Set a background speed at which the fluid is advected
run.backgroundMach = -1;

% Set the type of wave to be run.
% One of 'entropy', 'sound', 'alfven', 'slow ma', 'fast ma'
% The MHD waves require a B to be set; Setting one is optional for the Entropy wave.
% Any nonzero B will automatically activate magnetic fluxing
run.waveType = 'sonic';
run.amplitude = .05;
% FWIW an amplitude of .0001 corresponds to a roughly 100dB sound in air
%                      .01                    roughly 140dB

% number of transverse wave periods in Y and Z directions
run.wavenumber = [5 7 0];
% 1st method of setting run duration: normalized by cycle time
%run.cycles = 5;
% 2nd method of setting run duration: normalized by steepening critical time t*
run.forCriticalTimes(0.2);

run.alias= 'sonic';

run.ppSave.dim3 = 10;

        run.useInSituAnalysis = 0;
        run.stepsPerInSitu = 10;
        run.inSituHandle = @RealtimePlotter;
instruct.plotmode = 4;
instruct.plotDifference = 0;
instruct.pause = 1;
        run.inSituInstructions = instruct;

run.waveLinearity(0);
run.waveStationarity(0);

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    outpath = imogen(IC);
    AdvectionAnalysis(outpath, 1);
    if mpi_amirank0(); fprintf('RUN STORED AT: %s\n', outpath); end
end

