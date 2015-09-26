% Run Advection test.

%--- Initialize test ---%
grid = [1024 64 1];
run             = AdvectionInitializer(grid);
run.iterMax     = 5;
run.info        = 'Dry run: Start, run 25 steps, bug out.';
run.notes       = 'Simple advection test in the x-direction.';

%run.image.interval = 100;
%run.image.mass = true;

run.activeSlices.x = false;
run.activeSlices.xy = false;
run.activeSlices.xyz = true;

run.ppSave.dim1 = 100;
run.ppSave.dim2 = 100;

% Set a background speed at which the fluid is advected
run.backgroundMach = 0;

% Set the type of wave to be run.
% One of 'entropy', 'sound', 'alfven', 'slow ma', 'fast ma'
% The MHD waves require a B to be set; Setting one is optional for the Entropy wave.
% Any nonzero B will automatically activate magnetic fluxing
run.waveType = 'sonic';
run.amplitude = .0001;
% FWIW an amplitude of .0001 corresponds to a roughly 100dB sound in air

% number of transverse wave periods in Y and Z directions
run.wavenumber  = [16 0 0];
run.cycles      = 16;

run.alias = 'DRYRUN_SONIC';

run.saveFormat = ENUM.FORMAT_MAT;

% Store 8 steps for each time a sound wave goes past a given point
run.ppSave.dim3 =  100;

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    outpath = imogen(IC);
%    fprintf('Dry hydrodynamic run at %s\n', outpath);
end

run.backgroundB = [.5 .3 0];
run.waveType = 'fast ma';

%if(true)
%    IC = run.saveInitialCondsToStructure();
%    outpath = imogen(IC);
%    fprintf('Dry MHD run at %s\n', outpath);
%end
