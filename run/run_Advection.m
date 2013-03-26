% Run Advection test.

%-- Initialize Imogen directory ---%
starterRun();

grid = [512 512 1];
GIS = GlobalIndexSemantics(); GIS.setup(grid);

%--- Initialize test ---%
run             = AdvectionInitializer(grid);
run.iterMax     = 10000;
run.info        = 'Advection test.';
run.notes       = 'Simple advection test in the x-direction.';
run.alias = 'ADVECT';

run.image.interval = 20;
run.image.mass = true;;

run.activeSlices.x = false;
run.activeSlices.xy = false;
run.activeSlices.xyz = true;

run.ppSave.dim1 = 100;
run.ppSave.dim2 = 100;

% Set a background speed at which the fluid is advected
run.waveDirection = 1;
run.backgroundMach = 0;

% Set the type of wave to be run.
% One of 'entropy', 'sound', 'alfven', 'slow ma', 'fast ma'
% The MHD waves require a B to be set; Setting one is optional for the Entropy wave.
% Any nonzero B will automatically activate magnetic fluxing
run.waveType = 'sound';
run.waveAmplitude = .0001;
% FWIW an amplitude of .0001 corresponds to a roughly 100dB sound

% number of transverse wave periods in Y and Z directions
run.waveK    = [5 5 0];
run.numWavePeriods = 2;

% Store 8 steps for each time a sound wave goes past a given point
run.ppSave.dim3 =  12.5/(run.numWavePeriods*norm(run.waveK) );

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    imogen(IC);
end

