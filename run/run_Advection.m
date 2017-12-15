% Wave advection simulation

%grid = [512 512 1];
grid = [512 1 1];

%--- Initialize test ---%
run             = AdvectionInitializer(grid);
run.iterMax     = 1000;
run.info        = 'Advection test.';
run.notes       = 'Simple advection test in the x-direction.';

%run.image.interval = 100;
%run.image.mass = true;

run.activeSlices.x = false;
run.activeSlices.xy = false;
run.activeSlices.xyz = false;

run.ppSave.dim1 = 100;
run.ppSave.dim2 = 10;

% Set a background speed at which the fluid is advected
run.backgroundMach = .45;

% Set the type of wave to be run.
% One of 'entropy', 'sound', 'alfven', 'slow ma', 'fast ma'
% The MHD waves require a B to be set; Setting one is optional for the Entropy wave.
% Any nonzero B will automatically activate magnetic fluxing
run.waveType = 'sonic';
run.amplitude = .25;
% FWIW an amplitude of .0001 corresponds to a roughly 100dB sound in air
%                      .01                    roughly 140dB

% number of transverse wave periods in Y and Z directions
run.wavenumber = [3 0 0];
% 1st method of setting run duration: normalized by cycle time
%run.cycles = 1;
% 2nd method of setting run duration: normalized by steepening critical time t*
run.forCriticalTimes(1.2);

run.alias= 'sonic';

run.ppSave.dim3 = 100;

rp = RealtimePlotter();
  rp.plotmode = 7;
  rp.plotDifference = 0;
  rp.insertPause = 1;
  rp.iterationsPerCall = 100;
  rp.firstCallIteration = 1;
%run.peripherals{end+1} = rp;
fm = FlipMethod(); % 1 = HLL, 2 = HLLC, 3 = XJ
  fm.iniMethod = 2; 
%  fm.toMethod = 2;
%  fm.atstep = -1;
run.peripherals{end+1} = fm;

run.waveLinearity(0);
run.waveStationarity(0);

%--- Run tests ---%
if (true)
    IC = run.saveInitialCondsToStructure();
    outpath = imogen(IC);
    SonicAdvectionAnalysis(outpath, 1);
    if mpi_amirank0(); fprintf('RUN STORED AT: %s\n', outpath); end
end

