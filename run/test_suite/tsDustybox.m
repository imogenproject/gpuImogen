function result = tsDustybox(N0, gamma, M, doublings, prettyPictures, methodPicker)
% result = tsDustybox(N0, gamma, M, doublings, prettyPictures, methodPicker) runs a
% number of tests of the DustyBox gas-dust code test (Laibe & Price 2011b),
% checking proper convergence of the numeric solution of nonlinear drag.

if nargin < 4;
    disp('Number of doublings not given; Defaulted to 3.');
    doublings = 3;
end
if nargin < 5
    prettyPictures = 0;
end

grid = [N0 1 1];
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
run.backgroundMach = M;

run.waveType = 'sonic';
run.amplitude = .0;

run.gamma = gamma;

% FWIW an amplitude of .0001 corresponds to a roughly 100dB sound in air
%                      .01                    roughly 140dB

% number of transverse wave periods in Y and Z directions
run.wavenumber = [1 0 0];
% 1st method of setting run duration: normalized by cycle time
%run.cycles = 25;

run.addNewFluid(1);

run.fluidDetails(1) = fluidDetailModel('cold_molecular_hydrogen');
run.fluidDetails(2) = fluidDetailModel('10um_iron_balls');

run.peripherals{end+1} = DustyBoxAnalyzer();
run.peripherals{end}.stepsPerPoint = 10;

if prettyPictures
    rp = RealtimePlotter();
    rp.plotmode = 1;
    rp.plotDifference = 0;
    rp.insertPause = 0;
    rp.firstCallIteration = 1;
    rp.iterationsPerCall = 25;
    run.peripherals{end+1} = rp;
end
if nargin == 6
    run.peripherals{end+1} = methodPicker;
end

run.timeMax = 5; % several drag times

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

result.N = [];
result.L2 = [];
result.paths = {};

%--- Run tests ---%
for R = 1:doublings;
    % Set resolution and go
    grid(1) = N0 * 2^(R-1);
    run.geomgr.setup(grid);
    icfile = run.saveInitialCondsToFile();
    dirout = imogen(icfile);
    enforceConsistentView(dirout);

    x = load([dirout '/drag_analysis.mat']);
    
    result.N(end+1) = grid(1);
    result.L2(end+1) = x.result.error(end);
    result.paths{end+1} = dirout;
end


end