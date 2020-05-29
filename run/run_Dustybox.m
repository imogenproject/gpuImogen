% Wave advection simulation

%grid = [512 512 1];
grid = [16 1 1];

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
run.backgroundMach = .0001;

% Set the type of wave to be run.
% One of 'entropy', 'sound'
% Zero amplitude just tests the drag code
run.waveType = 'sonic';
run.amplitude = 0;
% FWIW an amplitude of .0001 corresponds to a roughly 100dB sound in air
%                      .01                    roughly 140dB

% number of transverse wave periods in Y and Z directions
run.wavenumber = [1 0 0];
% 1st method of setting run duration: normalized by cycle time
run.cycles = 5;

% Hydrogen at room temp and density
run.setBackground(.084, 101325);
run.fluidDetails(1) = fluidDetailModel('warm_molecular_hydrogen');

run.addNewFluid(1);

run.fluidDetails(1) = fluidDetailModel('cold_molecular_hydrogen');
run.fluidDetails(2) = fluidDetailModel('10um_iron_balls');
run.fluidDetails(2).sigma = run.fluidDetails(2).sigma * 1e1;
run.fluidDetails(2).mass = run.fluidDetails(2).mass * 1e-3; %

run.writeFluid = 2;
  run.amplitude = 0;
  run.backgroundMach = 0;
  run.setBackground(.0084, .00001*101325);

run.peripherals{end+1} = DustyBoxAnalyzer();

run.alias= 'dustybox';

run.ppSave.dim3 = 100;
  
fm = FlipMethod();
  fm.iniMethod = ENUM.CFD_HLLC; 
%  fm.toMethod = 2;
%  fm.atstep = -1;
run.peripherals{end+1} = fm;

run.multifluidDragMethod = ENUM.MULTIFLUID_LOGTRAP3;
%run.multifluidDragMethod = ENUM.MULTIFLUID_ETDRK1;

run.waveLinearity(0);
run.waveStationarity(0);

%--- Run tests ---%
if true
    IC = run.saveInitialCondsToStructure();
    outpath = imogen(IC);
    SaveManager.logPrint('RUN STORED AT: %s\n', outpath);
end

if run.numFluids == 1
    SonicAdvectionAnalysis(outpath, 1)
end

% This was emitted by the dustyboxanalyzer
cd(outpath);
load('drag_analysis.mat');

figure(1); 
hold on;
plot(result.time, result.dvExact,'b-');
hold on;
plot(result.time, result.dvImogen,'rx');

figure(2);
hold on
plot(result.time, 1-result.dvImogen./result.dvExact);
