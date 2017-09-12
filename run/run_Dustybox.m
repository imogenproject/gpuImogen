% Wave advection simulation

%grid = [512 512 1];
grid = [64 1 1];

%--- Initialize test ---%
run             = AdvectionInitializer(grid);
run.iterMax     = 20000;
run.info        = '2-fluid advection test';
run.notes       = 'basic testbed for 2-fluid drag code';

%run.image.interval = 100;
%run.image.mass = true;

run.activeSlices.x = false;
run.activeSlices.xy = false;
run.activeSlices.xyz = false;

run.ppSave.dim1 = 100;
run.ppSave.dim2 = 10;

% Set a background speed at which the fluid is advected
run.backgroundMach = .12;

% Set the type of wave to be run.
% One of 'entropy', 'sound', 'alfven', 'slow ma', 'fast ma'
% The MHD waves require a B to be set; Setting one is optional for the Entropy wave.
% Any nonzero B will automatically activate magnetic fluxing
run.waveType = 'sonic';
run.amplitude = .0;

% FWIW an amplitude of .0001 corresponds to a roughly 100dB sound in air
%                      .01                    roughly 140dB

% number of transverse wave periods in Y and Z directions
run.wavenumber = [1 0 0];
% 1st method of setting run duration: normalized by cycle time
run.cycles = 25;

run.addNewFluid(1);

run.fluidDetails(1) = fluidDetailModel('cold_molecular_hydrogen');
run.fluidDetails(2) = fluidDetailModel('10um_iron_balls');

run.peripherals{end+1} = DustyBoxAnalyzer();

run.writeFluid = 2;
  run.amplitude = 0;
  run.backgroundMach = 0;
  run.setBackground(0.1, .001);

run.alias= 'dustybox';

run.ppSave.dim3 = 100;
  
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
    AdvectionAnalysis(outpath, 1)
    if mpi_amirank0(); fprintf('RUN STORED AT: %s\n', outpath); end
end

