% Run Advection test.

%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
run             = AdvectionInitializer([1024 1024 1]);
run.iterMax     = 10;
run.info        = 'Advection test.';
run.notes       = 'Simple advection test in the x-direction.';
run.alias = 'TEST';

run.image.interval = 25;
%run.image.mass = true;

run.activeSlices.x = false;
run.activeSlices.xyz = false;

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
run.waveAmplitude = .001;

% number of transverse wave periods in Y and Z directions
run.waveK    = [1 1 0];

run.numWavePeriods = 4;

%--- Run tests ---%
if (true)
    icfile = run.saveInitialCondsToFile();
    imogen(icfile);
end

