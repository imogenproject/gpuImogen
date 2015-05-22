% Run Advection test.

%-- Initialize Imogen directory ---%
starterRun();

grid = [256 2 1];
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
run.activeSlices.xyz = true;

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

% number of transverse wave periods in Y and Z directions
run.wavenumber = [1 0 0];
%run.cycles = 5;
run.forCriticalTimes(10);

%run.alias = sprintf('ADVECT_N%i_%i_%i',16,0,0);
run.alias= '10TC';

% Store 8 steps for each time a sound wave goes past a given point
%run.ppSave.dim3 =  100;
run.ppSave.dim3 = 1;

run.waveLinearity(0);
run.waveStationarity(0);

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    outpath = imogen(IC);
    AdvectionAutoanalyze(outpath);
end

