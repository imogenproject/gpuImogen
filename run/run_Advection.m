% Run Advection test.

%-- Initialize Imogen directory ---%
starterRun();

%--- Initialize test ---%
run             = AdvectionInitializer([512 512 1]);
run.iterMax     = 10000;
run.info        = 'Advection test.';
run.notes       = 'Simple advection test in the x-direction.';
run.alias = 'TEST'

run.image.interval = 25;
%run.image.mass = true;

run.ppSave.dim1 = 100;
run.ppSave.dim2 = 12.5;

run.gpuDeviceNumber = 0;

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
run.waveK    = [0 1 0];

run.numWavePeriods = 4;

%--- Run tests ---%
if (true)
    [mass, mom, ener, magnet, statics, ini] = run.getInitialConditions();
    IC.mass = mass;
    IC.mom = mom;
    IC.ener = ener;
    IC.magnet = magnet;
    IC.statics = statics;
    IC.ini = ini;
    icfile = [tempname '.mat'];

    save(icfile, 'IC');
    clear IC mass mom ener magnet statics ini run;
    imogen(icfile);
end

