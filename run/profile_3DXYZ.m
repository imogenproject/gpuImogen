% Wave advection simulation

grid = [256 256 32];
%grid = [32 64 72];

%--- Initialize test ---%
run             = AdvectionInitializer(grid);
run.iterMax     = 4;
run.info        = '1-fluid advection test';
run.notes       = 'Most basic linear correctness test';

%run.image.interval = 100;
%run.image.mass = true;

run.activeSlices.x = false;
run.activeSlices.xy = false;
run.activeSlices.xyz = true;

%run.ppSave.dim1 = 100;
%run.ppSave.dim2 = 100;
run.ppSave.dim3 = 25;

% Set a background speed at which the fluid is advected
run.backgroundMach = -0;

% Set the type of wave to be run.
% One of 'entropy', 'sound', 'alfven', 'slow ma', 'fast ma'
% The MHD waves require a B to be set; Setting one is optional for the Entropy wave.
% Any nonzero B will automatically activate magnetic fluxing
run.waveType = 'sonic';
run.amplitude = .05;

% FWIW an amplitude of .0001 corresponds to a roughly 100dB sound in air
%                      .01                    roughly 140dB

% number of transverse wave periods in Y and Z directions
run.wavenumber = [4 3 1];
% 1st method of setting run duration: normalized by cycle time
run.cycles = 50;

run.fluidDetails(1) = fluidDetailModel('cold_molecular_hydrogen');

run.alias= 'sontest';

run.ppSave.dim3 = 100;
  
fm = FlipMethod();
  fm.iniMethod = ENUM.CFD_HLLC;
%  fm.toMethod = ENUM.CFD_HLLC;
%  fm.atstep = -1;
run.peripherals{end+1} = fm;

run.waveLinearity(0);
run.waveStationarity(0);

%--- Run tests ---%
if true
    IC = run.saveInitialCondsToStructure();
    outpath = imogen(IC);
    if mpi_amirank0(); fprintf('RUN STORED AT: %s\n', outpath); end
end
